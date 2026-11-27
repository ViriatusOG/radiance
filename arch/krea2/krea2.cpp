/* krea2 -- Krea 2 (Raw and Turbo): text to an RGB image.
 *
 * ============================== THE MODEL ==============================
 *
 * Three networks behind one prompt (diffusers' Krea2Pipeline):
 *
 *   Conditioning network: the language model of Qwen3-VL-4B over the prompt in Qwen-Image's
 *   template. The generator reads twelve of its hidden states per token (after layers 1, 4, ...,
 *   34, before any final norm) and drops the template's fixed system turn.
 *
 *   Denoiser: a 12.9B single-stream diffusion transformer. A small text-fusion transformer first
 *   turns each token's twelve tapped states into one row: two blocks attending across the twelve
 *   (a token's taps are twelve consecutive rows here), a learned Linear(12, 1) over them
 *   (`mix_rows`), two blocks attending across the tokens, then `txt_in` to the DiT's width. The
 *   28 blocks run over one joint sequence, the text and then the image, every row of it modulated
 *   by the step's timestep: `h += gate1 attn(norm1(h)(1 + scale1) + shift1)`, `h += gate2
 *   mlp(norm2(h)(1 + scale2) + shift2)`, where the six modulation rows are one projection of the
 *   timestep embedding shared by every block plus a learned table of each block's own
 *   (`table_add`), gates without tanh. Attention is grouped (48 query heads, 12 key heads) with a
 *   sigmoid output gate, q/k RMS norms and a three-axis rotary (text at the origin, an image token
 *   at (0, row, column)). Every RMS norm is zero-centred, its gain 1 + w in f32.
 *
 *   Since the text rows change with the step, nothing is cached: the conditioning pass leaves the
 *   text rows `txt_in` makes in each lane's store, and every denoising step runs them with the
 *   image (the core's JOINT_STEP layout, abi/rad_builder.h).
 *
 *   Decoder: Qwen-Image's VAE (Wan 2.1's, its causal 3-D convolutions on one frame), 16 latent
 *   channels at 8x; a latent token is 2 x 2 of its pixels (64 values), so 16 pixels a token side.
 *
 * Raw samples 28 steps with guidance 4.5 in the card's terms (cond + 4.5 (cond - uncond), an
 * empty negative prompt) on a resolution-shifted schedule; Turbo 8 steps without guidance at a
 * fixed shift. Both come from the container's scheduler configuration and pipeline name.
 *
 * ============================== WHAT IS QUANTISED ==============================
 *
 * One plugin serves bf16, block-fp8 and rotated int8 (IQ8R) containers: every linear of the
 * language model, of the text fusion and of the DiT's blocks is a LinearQ (rad_linear.h), bf16
 * throughout when the first DiT projection is plain. That projection also says whether the
 * linears are rotated int8: the activations they read then carry int8 codes in place of the E4M3
 * pair, written by the rotating twins of the fused ops where a library serves them
 * (rms_mod_had_quant_i8g for a modulated norm, gate_had_quant_i8g for an attention's output
 * gate); elsewhere the norm, or the sigmoid and multiply pair, is followed by the rotating
 * quantiser. txt_in, img_in, the timestep MLP and modulation, the final layer and the VAE stay
 * bf16.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * The language model, the text fusion and the DiT blocks split by heads and feed-forward width,
 * an all-reduce after each output projection; everything else is replicated. The decoder splits
 * by rows after its mid-block, as Qwen-Image 2.1's does (rad_vae2d.h).
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_vit.h>
#include <arch/rad_linear.h>
#include <arch/rad_vae2d.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace krea2 {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* The largest image: 2048 x 2048 pixels, 16384 latent tokens. 1024 x 1024 is the default. */
static constexpr int64_t kMaxImageTokens = 16384;
static int64_t image_tokens(int64_t pixels) {
    return pixels > 0 ? std::min<int64_t>(kMaxImageTokens, pixels / 256) : kMaxImageTokens;
}
/* The longest conditioning in rows: the pipeline's max_sequence_length. The prompt the language
 * model reads is that plus the template's system turn. */
enum { kCondRows = 512, kEditCondRows = 1024, kTemplateRows = 64 };
static constexpr int64_t kGroundingPatches = 2304;  /* one 768×768 reference */
static constexpr int64_t kMaxReferences = 2;

static int64_t reference_count(int64_t requested) {
    return requested > 0 ? requested : 1;
}

static int64_t edit_cond_rows(int64_t refs, int64_t merge) {
    const int64_t group = merge * merge;
    return kEditCondRows + (refs - 1) * ((kGroundingPatches + group - 1) / group);
}

static bool has_reference_encoder(const RadModelMeta* meta) {
    return rad_meta_geti(meta, "pipeline.krea_reference_encoder", 0) != 0;
}

static int check_reference_encoder(const RadModelMeta* meta) {
    if (!has_reference_encoder(meta) || VisionTower::present(meta)) return RAD_OK;
    std::fprintf(stderr, "radiance: krea2: edit conversion was requested, but the Qwen3-VL "
                         "vision tower metadata is missing; convert the complete pipeline\n");
    return RAD_E_INVAL;
}
/* Images in flight, and so the lanes of latent state. */
enum { kMaxLanes = 8 };
/* The language model's hidden states the generator reads: hidden_states[i] for these i, the
 * output of layer i - 1. Layers past the last tap are not run. */
static const int kTaps[] = { 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35 };
enum { kNTaps = 12 };

/* The rebalance presets a request may ask for (`rebalance`, RAD_GEN_COND_REBALANCE with the row in
 * RAD_GEN_COND_REBALANCE_PRESET): every tap times a fixed gain before the text fusion.
 *   legacy    ComfyUI's ConditioningKrea2Rebalance (nova452) at its defaults -- multiplier 4 times
 *             the per-layer weights 1, 1, 1, 1, 1, 1, 1, 2.5, 5, 1.1, 4, 1, about 9x the taps' RMS.
 *   balanced, detail, subtle
 *             comfyui-krea2-conditioning's presets (huwhitememes) with renormalize on, multiplier 1:
 *             the weights times s = sqrt(sum m_t / sum w_t^2 m_t), m_t the mean square of tap t over
 *             the prompt's rows. The node measures s per prompt; here it is a constant, m_t averaged
 *             over twenty prompts through krea/Krea-2-Turbo's text encoder
 *             (scripts/krea2_rebalance_calib.py), and per prompt s stayed within 1.7 % of it.
 * A preset row the plugin does not know scales nothing: the plain model. */
enum { kNPresets = 4 };
static const double kRebalance[kNPresets][kNTaps] = {
    { 4, 4, 4, 4, 4, 4, 4, 10, 20, 4.4, 16, 4 },
    /* balanced: 1, 1, 1, 1, 1, 1, 1, 2.5, 5, 1.1, 4, 1 at s 0.44109 */
    { 0.44109, 0.44109, 0.44109, 0.44109, 0.44109, 0.44109, 0.44109, 1.1027, 2.2055, 0.4852,
      1.7644, 0.44109 },
    /* detail: 0.8, 0.8, 0.9, 0.9, 1, 1, 1.2, 3, 6, 1.5, 5, 1.2 at s 0.35702 */
    { 0.28562, 0.28562, 0.32132, 0.32132, 0.35702, 0.35702, 0.42842, 1.0711, 2.1421, 0.53553,
      1.7851, 0.42842 },
    /* subtle: 1, 1, 1, 1, 1, 1, 1, 1.5, 2, 1, 1.5, 1 at s 0.84759 */
    { 0.84759, 0.84759, 0.84759, 0.84759, 0.84759, 0.84759, 0.84759, 1.2714, 1.6952, 0.84759,
      1.2714, 0.84759 },
};

static const char* kSysPrompt =
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:";

/* ================================================================== the model */
struct TeLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ  qkv{}, o{}, gate_up{}, down{};
    QuantAct   q_h{}, q_attn{}, q_h2{};
    GateQuant gq{};
    rad_op     op_n1 = 0, op_qkr = 0, op_attn = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_ar2 = 0, op_add2 = 0, op_tap = 0;
    rad_op     op_gain[kNPresets] = {};   /* the tap's rebalance gain into te.h2, a preset each */
    rad_op     op_tap_g = 0;  /* the scatter from te.h2, when rebalanced */
    int        tap = -1;      /* this layer's output is tap `tap`, or none */
};

/* A text-fusion block: pre-norm attention (q/k norms, no rotary, sigmoid output gate) and a
 * SwiGLU, each added back to its residual `x`. */
struct FuseBlock {
    rad_buf    x = 0;
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ  qkv{}, gate{}, o{}, gate_up{}, down{};
    QuantAct   q_h{}, q_h2{}, q_attn{};
    GateQuant gq{};
    rad_op     op_n1 = 0, op_qkn = 0, op_attn = 0, op_gq = 0, op_sig = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_ar2 = 0, op_add2 = 0;
};

struct DitLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0, w_tab = 0;
    rad_op     op_tab = 0;    /* the shared modulation plus this block's table, into its columns */
    LinearQ  qkv{}, gate{}, o{}, gate_up{}, down{};
    QuantAct   q_attn{};
    /* The two modulated norms' quantisers for a rotated int8 model whose norms are the
     * unquantised form (Model::norm_i8 false). */
    QuantAct   q_n1{}, q_n2{};
    GateQuant gq{};
    rad_op     op_n1 = 0, op_n2 = 0, op_qkr = 0, op_attn = 0, op_gq = 0, op_sig = 0;
    rad_op     op_ar1 = 0, op_ar2 = 0;
};

/* Weight groups (movement units) by network. */
enum : int {
    kGrpFuse  = 1000,   /* the text fusion's blocks, kGrpFuse + i */
    kGrpTxtIn = 1100,
    kGrpVaeDec = 1200,
    kGrpVaeEnc = 1300,
    kGrpVision = 1400,
    kGrpDit   = 2000,   /* the DiT's model-level weights; its blocks kGrpDit + 1.. */
};

struct Model {
    Names nm{""};
    Geom  gt{}, gf{}, gd{};    /* the language model's, the text fusion's and the DiT's geometry */
    int64_t max_image_tokens = kMaxImageTokens;
    bool  turbo = false;
    std::string te_pre;        /* the language model's tensor prefix */

    int64_t cond_rows = kCondRows;
    int64_t te_rows = kCondRows + kTemplateRows;
    int64_t store_rows = kCondRows;
    int64_t max_refs = 0;
    /* Lanes of latent state and conditioning store: the configured sequences, at most kMaxLanes. */
    int64_t lanes = kMaxLanes;
    VisionTower vision{};
    rad_op op_vision_embed = 0;
    rad_op op_deepstack[VisionTower::MAX_DS] = {};

    /* ---- the language model */
    rad_weight w_tok = 0;
    rad_op  op_embed = 0, op_iota = 0;
    rad_buf b_tx = 0, b_tkv = 0, b_th2 = 0, b_tgu = 0, b_tidx = 0;
    ActQ  a_th{}, a_tattn{}, a_tffn{};
    std::vector<TeLayer> te;
    rad_buf b_taps = 0;        /* [T * 12, n_embd]: token t's tap i at row 12 t + i */
    std::string tap_mul[kNPresets][kNTaps];   /* chan_affine's `mul` for each preset's tap gain */
    std::string tap_add;           /* chan_affine's `add`: zeros */

    /* ---- the text fusion: the layer-wise blocks over the taps themselves, the refiner blocks
     * over the mixed rows `b_ft` */
    rad_buf b_fqkv = 0, b_fg = 0, b_fo = 0, b_fgu = 0, b_fcu = 0;
    ActQ  a_fh{}, a_fattn{}, a_fffn{};
    rad_op  op_fcu = 0, op_mix = 0;
    rad_weight w_mix = 0;
    std::vector<FuseBlock> lw, rf;   /* the layer-wise and the refiner blocks */
    rad_buf b_ft = 0;          /* [T, n_embd] the mixed rows, the refiner's residual */

    /* ---- txt_in: zero-centred RMSNorm, a linear, gelu_tanh, a linear, into the lanes */
    rad_weight w_txn = 0, w_tx1 = 0, b_tx1 = 0, w_tx2 = 0, b_tx2 = 0;
    rad_op  op_txn = 0, op_tx1 = 0, op_txg = 0, op_tx2 = 0, op_store = 0;
    rad_buf b_txa = 0, b_txb = 0, b_txt = 0;

    /* ---- the timestep: embedding, MLP, the shared modulation and every block's tables */
    rad_weight w_t1 = 0, b_t1 = 0, w_t2 = 0, b_t2 = 0, w_mod = 0, b_mod = 0, w_ftab = 0;
    rad_op  op_temb = 0, op_t1 = 0, op_tg = 0, op_t2 = 0, op_tg2 = 0, op_mod = 0, op_ftab = 0;
    rad_buf b_tproj = 0, b_te1 = 0, b_te1g = 0, b_temb = 0, b_tembg = 0, b_tmod = 0, b_mods = 0,
            b_fmod = 0;

    /* ---- the DiT */
    int64_t C = 64;            /* latent channels a token */
    rad_weight w_imgin = 0, b_imgin = 0;
    rad_op  op_randn = 0, op_lgather = 0, op_imgin = 0, op_join = 0, op_g2 = 0, op_gimg = 0;
    rad_buf b_lat = 0, b_himg = 0, b_h = 0, b_qkv = 0, b_g = 0, b_ao = 0, b_gu = 0, b_hi = 0;
    ActQ  a_hm{}, a_attn{}, a_ffn{};
    std::vector<DitLayer> dit;
    LinearQ qkv0{};
    /* The linears of every network are rotated int8 (IQ8R, or the 128-group int8 before it), read
     * off qkv0's probe: the activations then carry int8 code pairs with one scale per (token, `g8`
     * columns) in place of the E4M3 pair. */
    bool      i8 = false;
    int64_t   g8 = RAD_SCALE_GROUP;
    /* Where a library serves the rotating twins, a DiT block's modulated norms write those int8
     * codes in their own pass (rms_mod_had_quant_i8g: norm_i8) and an attention's output gate
     * writes the out projection's (gate_had_quant_i8g: gate_i8, and fgate_i8 for the fusion blocks'). */
    bool      norm_i8 = false, gate_i8 = false, fgate_i8 = false;
    rad_weight w_fn = 0, w_pout = 0, b_pout = 0;
    rad_op  op_fn = 0, op_pout = 0, op_euler = 0;
    rad_buf b_fz = 0, b_v = 0;

    /* ---- the per-image state and the conditioning store (PERSIST) */
    rad_buf b_state = 0, b_prev = 0, b_store = 0;

    /* ---- the VAE */
    Vae     enc{}, dec{};
    VImg    enc_out{};
    rad_buf b_enc_cu = 0, b_enc_norm = 0, b_enc_pack = 0, b_enc_embed = 0;
    rad_op  op_enc_cu = 0, op_enc_norm = 0, op_enc_zero = 0, op_enc_pack = 0;
    rad_op  op_enc_project = 0, op_enc_store = 0;
    std::string enc_mean, enc_std;
    VImg    dec_out{};
    rad_buf b_dl = 0, b_dl2 = 0, b_decin = 0, b_cu4 = 0, b_img = 0, b_rgbf = 0;
    rad_op  op_dinv = 0, op_dadd = 0, op_dzero = 0, op_unpack = 0, op_cu4 = 0, op_rgb = 0;
    rad_op  op_bands = 0, op_rgb_bands = 0;
    std::string inv_std, mean, ones, zeros;
};

static Model g_model[MAX_SLOTS];

/* A bf16 linear with a bias, read whole on every rank. */
static rad_op dense_bias(RadBuilder* b, Names& nm, const char* dn, const char* src, int64_t N,
                         int64_t K, int64_t rows, rad_buf in, rad_buf out, rad_weight* w,
                         rad_weight* bias, int access, int group) {
    *w = wcopy(b, nm.f("%s.weight", dn), nm.ckpt("%s.weight", src), RAD_BF16, { N, K }, access,
               group);
    *bias = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), RAD_BF16, { N }, access, group);
    if (!*w || !*bias) return 0;
    return gemm_bias_op(b, rows, N, K, *w, *bias, in, out);
}

/* A zero-centred RMSNorm (gain 1 + w in f32, the norm and gain rounded once): rmsnorm's
 * `wadd`. */
static rad_op zc_norm(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight w,
                      rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_STR("dtype", "bf16"), RAD_F64("wadd", 1.0)),
                        RAD_WEIGHTS(w)),
              { x }, { y });
}

/* The attention's sigmoid output gate: `gate_quant_fp8` (gate, multiply and the out projection's
 * quantiser in one pass) over fp8 linears, rows (token, head) pairs of `head_dim` columns; over
 * rotated int8 ones (`pair`) its rotating twin `gate_had_quant_i8g`, a token a row, since one
 * scale covers the codes' group across heads (`fused_i8` says so); or `sigmoid` then `mul` over
 * bf16 ones and wherever no loaded library declares the fused op (the out projection's quantiser
 * then runs on its own). Returns the gate op; `sig` is the pair's sigmoid, written into `g`
 * itself. */
static rad_op out_gate(RadBuilder* b, const Geom& g, int64_t rows, int64_t width, rad_buf gate,
                       const ActQ& attn, rad_op* sig, bool pair, bool* fused_i8) {
    const int64_t hd = g.head_dim;
    *fused_i8 = false;
    if (pair && !g.w_bf16 && attn.q8_fed && attn.q8_rot && rad_op_declared(b, "gate_had_quant_i8g")) {
        *sig = 0;
        *fused_i8 = true;
        return rw(b, RAD_OP(b, "gate_had_quant_i8g",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                       RAD_INT("group", attn.cg()), RAD_STR("dtype", "bf16"),
                                       RAD_STR("act", "sigmoid")),
                            RAD_NOWEIGHTS),
                  { gate, attn.x }, { attn.cq(), attn.cs(), attn.x });
    }
    if (g.w_bf16 || pair || !rad_op_declared(b, "gate_quant_fp8")) {
        *sig = rw(b, RAD_OP(b, "sigmoid",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { gate }, { gate });
        return rw(b, RAD_OP(b, "mul",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { attn.x, gate }, { attn.x });
    }
    *sig = 0;
    return rw(b, RAD_OP(b, "gate_quant_fp8",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows * (width / hd)), RAD_INT("n", hd),
                                   RAD_INT("group", RAD_SCALE_GROUP), RAD_STR("dtype", "bf16"),
                                   RAD_STR("act", "sigmoid")),
                        RAD_NOWEIGHTS),
              { gate, attn.x }, { attn.cq(), attn.cs(), attn.x });
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    const char* arch = rad_meta_gets(meta, "pipeline._class_name", "");
    if (std::strncmp(arch, "Krea2", 5) != 0) {
        fprintf(stderr, "radiance: krea2: the container's pipeline is '%s', not Krea 2's\n", arch);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "transformer.num_text_layers", kNTaps) != kNTaps ||
        rad_meta_geti(meta, "vae.z_dim", 16) != 16 ||
        rad_meta_geti(meta, "transformer.in_channels", 64) != 64) {
        fprintf(stderr, "radiance: krea2: the DiT must read twelve tapped layers and 64-value "
                        "tokens of a 16-channel VAE\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== adapter key names */
/* LoRA files for Krea 2 name the DiT and the text fusion by the reference code's modules
 * (krea-ai/krea-2 mmdit.py): `blocks.N.attn.wq`, `txtfusion.refiner_blocks.N.mlp.up`, `tproj.1`;
 * the container keeps diffusers' names. The language model takes no adapters. */
static int declare_lora_names(RadBuilder* b) {
    std::vector<LoraName> n;
    static const char* const kInner[][2] = {
        { "attn.wq", "attn.to_q" },   { "attn.wk", "attn.to_k" },  { "attn.wv", "attn.to_v" },
        { "attn.wo", "attn.to_out.0" }, { "attn.gate", "attn.to_gate" },
        { "mlp.gate", "ff.gate" },    { "mlp.up", "ff.up" },       { "mlp.down", "ff.down" },
    };
    static const char* const kBlocks[][2] = {
        { "blocks.#.", "transformer_blocks.#." },
        { "txtfusion.layerwise_blocks.#.", "text_fusion.layerwise_blocks.#." },
        { "txtfusion.refiner_blocks.#.", "text_fusion.refiner_blocks.#." },
    };
    for (const auto& bl : kBlocks)
        for (const auto& in : kInner)
            n.push_back({ std::string(bl[0]) + in[0], std::string(bl[1]) + in[1] });
    n.push_back({ "txtfusion.projector", "text_fusion.projector" });
    n.push_back({ "txtmlp.1", "txt_in.linear_1" });
    n.push_back({ "txtmlp.3", "txt_in.linear_2" });
    n.push_back({ "first", "img_in" });
    n.push_back({ "last_linear", "final_layer.linear" });
    n.push_back({ "last.linear", "final_layer.linear" });
    n.push_back({ "tmlp.0", "time_embed.linear_1" });
    n.push_back({ "tmlp.2", "time_embed.linear_2" });
    n.push_back({ "tproj.1", "time_mod_proj" });
    return lora_names(b, "transformer.", kLoraDiffusionSpellings, n);
}

/* ================================================================== declare */
static int declare_te(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gt;
    Names& nm = m.nm;
    const int64_t T = m.te_rows;   /* the prompt with its system turn */
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const char* pre = m.te_pre.c_str();

    RAD_ARCH_TRY(map_copy(b, nm.f("te.token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", pre)));
    m.w_tok = decl_w(b, nm.f("te.token_embd.weight"), RAD_BF16, { g.n_vocab_all, g.n_embd },
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.b_tx   = decl_b(b, nm.f("te.x"), RAD_BF16, { T, g.n_embd });
    m.a_th.x = decl_b(b, nm.f("te.h"), RAD_BF16, { T, g.n_embd });
    RAD_ARCH_TRY(m.a_th.declare_pair(b, nm, g, "te.h", g.n_embd, T, m.i8, m.g8,
                                 "had_quant_act_i8g"));
    m.b_tkv  = decl_b(b, nm.f("te.qkv"), RAD_BF16, { T, q_dim + 2 * kv_dim });
    m.a_tattn.x = decl_b(b, nm.f("te.attn"), RAD_BF16, { T, q_dim });
    RAD_ARCH_TRY(m.a_tattn.declare_pair(b, nm, g, "te.attn", q_dim, T, m.i8, m.g8,
                                 "had_quant_act_i8g"));
    m.b_th2  = decl_b(b, nm.f("te.h2"), RAD_BF16, { T, g.n_embd });
    m.b_tgu  = decl_b(b, nm.f("te.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_tffn.x = decl_b(b, nm.f("te.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_tffn.declare_pair(b, nm, g, "te.ffn", g.n_ff, T, m.i8, m.g8,
                                 "gated_had_quant_i8g"));
    m.b_taps = decl_b(b, nm.f("te.taps"), RAD_BF16, { (int64_t)m.te_rows * kNTaps, g.n_embd });
    m.b_tidx = decl_b(b, nm.f("te.tap_idx"), RAD_I32, { T });

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", g.n_embd),
                                         RAD_INT("n_vocab", g.n_vocab_all), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", 0)),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_tx });
    if (m.vision.on) {
        m.op_vision_embed = rw(b, RAD_OP(b, "scatter_rows",
                                         RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                                    RAD_STR("dtype", "bf16")),
                                         RAD_NOWEIGHTS),
                               { m.vision.b_out, m.b_tx }, { m.b_tx });
        if (!m.op_vision_embed) return RAD_E_INVAL;
        for (int i = 0; i < m.vision.n_ds; ++i) {
            m.op_deepstack[i] = rw(b, RAD_OP(b, "scatter_add_rows",
                                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                                       RAD_STR("dtype", "bf16")),
                                            RAD_NOWEIGHTS),
                                  { m.vision.b_ds[i], m.b_tx }, { m.b_tx });
            if (!m.op_deepstack[i]) return RAD_E_INVAL;
        }
    }
    /* Row 12 t of the taps for token t; a tap i is scattered at row offset i. */
    m.op_iota = rw(b, RAD_OP(b, "iota",
                             RAD_PARAMS(RAD_RANGE("M", 1, m.te_rows), RAD_INT("start", 0),
                                        RAD_INT("step", kNTaps)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_tidx });

    const std::vector<double> sec = meta_list(meta, "text_config.rope_scaling.mrope_section");
    char sections[48];
    if (sec.size() == 3)
        std::snprintf(sections, sizeof sections, "%d %d %d", (int)sec[0], (int)sec[1], (int)sec[2]);
    else
        std::snprintf(sections, sizeof sections, "24 20 20");

    /* The rebalance gains, as chan_affine lists: the tap's row seen as rows of 64 channels. The
     * scaled copy goes through te.h2, free between a layer's add and the next layer's projections,
     * so the residual stream itself is never scaled. */
    enum { kGainCh = 64 };
    if (g.n_embd % kGainCh) {
        fprintf(stderr, "radiance: krea2: the text encoder's hidden size %lld is not a multiple of "
                "%d, the rows the rebalance gains are applied in\n", (long long)g.n_embd,
                (int)kGainCh);
        return RAD_E_UNSUPPORTED;
    }
    for (int p = 0; p < kNPresets; ++p)
        for (int t = 0; t < kNTaps; ++t) {
            char v[32];
            std::snprintf(v, sizeof v, "%.9g", kRebalance[p][t]);
            std::string& s = m.tap_mul[p][t];
            s.clear();
            for (int k = 0; k < kGainCh; ++k) {
                if (k) s += ' ';
                s += v;
            }
        }
    m.tap_add.clear();
    for (int k = 0; k < kGainCh; ++k) m.tap_add += k ? " 0" : "0";

    /* Layers up to the last tap only. */
    const int n_run = kTaps[kNTaps - 1];
    if (g.n_layer < n_run) {
        fprintf(stderr, "radiance: krea2: the text encoder has %lld layers; its last tap is after "
                        "layer %d\n", (long long)g.n_layer, n_run);
        return RAD_E_INVAL;
    }
    m.te.resize((size_t)n_run);
    for (int l = 0; l < n_run; ++l) {
        TeLayer& L = m.te[(size_t)l];
        for (int t = 0; t < kNTaps; ++t) if (kTaps[t] == l + 1) L.tap = t;
        const char* lp = nm.f("%s.layers.%d", pre, l);
        L.w_n1 = wcopy(b, nm.f("te.blk.%d.attn_norm.weight", l),
                       nm.ckpt("%s.input_layernorm.weight", lp), RAD_F32, { g.n_embd },
                       RAD_ACCESS_PER_TOKEN, l);
        L.w_n2 = wcopy(b, nm.f("te.blk.%d.ffn_norm.weight", l),
                       nm.ckpt("%s.post_attention_layernorm.weight", lp), RAD_F32, { g.n_embd },
                       RAD_ACCESS_PER_TOKEN, l);
        L.w_qn = wcopy(b, nm.f("te.blk.%d.q_norm.weight", l), nm.ckpt("%s.self_attn.q_norm.weight", lp),
                       RAD_BF16, { g.head_dim }, RAD_ACCESS_PER_TOKEN, l);
        L.w_kn = wcopy(b, nm.f("te.blk.%d.k_norm.weight", l), nm.ckpt("%s.self_attn.k_norm.weight", lp),
                       RAD_BF16, { g.head_dim }, RAD_ACCESS_PER_TOKEN, l);
        if (!L.w_n1 || !L.w_n2 || !L.w_qn || !L.w_kn) return RAD_E_INVAL;
        L.op_n1 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n1)),
                     { m.b_tx }, { m.a_th.x });
        RAD_ARCH_TRY(L.q_h.declare(b, g, m.a_th, g.n_embd));
        RAD_ARCH_TRY(L.qkv.declare(b, nm, g, nm.f("te.blk.%d.attn_qkv", l), q_dim + 2 * kv_dim,
                                   g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                   { nm.ckpt("%s.self_attn.q_proj.weight", lp),
                                     nm.ckpt("%s.self_attn.k_proj.weight", lp),
                                     nm.ckpt("%s.self_attn.v_proj.weight", lp) },
                                   m.a_th, m.b_tkv, T, { q_dim, kv_dim, kv_dim }));
        L.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
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
        RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("te.blk.%d.attn_output", l), g.n_embd, q_dim,
                                 RAD_SHARD_COL, grp_layer(l),
                                 { nm.ckpt("%s.self_attn.o_proj.weight", lp) }, m.a_tattn, m.b_th2,
                                 T));
        L.op_ar1 = decl_all_reduce_tp(b, g, m.b_th2, T);
        L.op_add1 = add_op(b, T, g.n_embd, m.b_tx, m.b_th2, m.b_tx);
        L.op_n2 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n2)),
                     { m.b_tx }, { m.a_th.x });
        RAD_ARCH_TRY(L.q_h2.declare(b, g, m.a_th, g.n_embd));
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("te.blk.%d.ffn_gate_up", l), 2 * g.n_ff,
                                       g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                       { nm.ckpt("%s.mlp.gate_proj.weight", lp),
                                         nm.ckpt("%s.mlp.up_proj.weight", lp) },
                                       m.a_th, m.b_tgu, T, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(L.gq.declare(b, g, m.b_tgu, m.a_tffn, g.n_ff, "silu", T, true));
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("te.blk.%d.ffn_down", l), g.n_embd, g.n_ff,
                                    RAD_SHARD_COL, grp_layer(l),
                                    { nm.ckpt("%s.mlp.down_proj.weight", lp) }, m.a_tffn, m.b_th2,
                                    T));
        L.op_ar2 = decl_all_reduce_tp(b, g, m.b_th2, T);
        L.op_add2 = add_op(b, T, g.n_embd, m.b_tx, m.b_th2, m.b_tx);
        if (L.tap >= 0) {
            /* f32 arithmetic, rounded once into bf16 (`round` 0) */
            for (int p = 0; p < kNPresets; ++p) {
                L.op_gain[p] = rw(b, RAD_OP(b, "chan_affine",
                                            RAD_PARAMS(RAD_RANGE("M", 1, m.te_rows),
                                                       RAD_INT("n", kGainCh),
                                                       RAD_STR("mul", m.tap_mul[p][L.tap].c_str()),
                                                       RAD_STR("add", m.tap_add.c_str()),
                                                       RAD_STR("mode", "fma"), RAD_STR("dtype", "bf16"),
                                                       RAD_INT("round", 0)),
                                            RAD_NOWEIGHTS),
                                  { m.b_tx }, { m.b_th2 });
                if (!L.op_gain[p]) return RAD_E_INVAL;
            }
            L.op_tap_g = rw(b, RAD_OP(b, "scatter_rows",
                                      RAD_PARAMS(RAD_RANGE("M", 1, m.te_rows),
                                                 RAD_INT("n", g.n_embd), RAD_STR("dtype", "bf16")),
                                      RAD_NOWEIGHTS),
                            { m.b_th2, m.b_tidx }, { m.b_taps });
        }
        if (L.tap >= 0)
            L.op_tap = rw(b, RAD_OP(b, "scatter_rows",
                                    RAD_PARAMS(RAD_RANGE("M", 1, m.te_rows), RAD_INT("n", g.n_embd),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                          { m.b_tx, m.b_tidx }, { m.b_taps });
        if (!L.op_n1 || !L.op_qkr || !L.op_attn || !L.op_add1 || !L.op_n2 || !L.op_add2 ||
            (L.tap >= 0 && (!L.op_tap || !L.op_tap_g)))
            return RAD_E_INVAL;
    }
    return m.op_embed && m.op_iota ? RAD_OK : RAD_E_INVAL;
}

/* One text-fusion block over at most `rows` rows of the residual `x`. */
static int declare_fuse_block(RadBuilder* b, Model& m, FuseBlock& F, const char* pre, int grp,
                              int64_t rows, rad_buf x) {
    F.x = x;
    Geom& g = m.gf;
    Names& nm = m.nm;
    const int64_t D = g.n_embd, qd = g.q_dim(), hd = g.head_dim;
    const double eps = g.eps;
    const char* dn = nm.f("fuse.%d", grp - kGrpFuse);
    F.w_n1 = wcopy(b, nm.f("%s.norm1.weight", dn), nm.ckpt("%s.norm1.weight", pre), RAD_F32, { D },
                   RAD_ACCESS_PER_REQUEST, grp);
    F.w_n2 = wcopy(b, nm.f("%s.norm2.weight", dn), nm.ckpt("%s.norm2.weight", pre), RAD_F32, { D },
                   RAD_ACCESS_PER_REQUEST, grp);
    F.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), nm.ckpt("%s.attn.norm_q.weight", pre), RAD_F32,
                   { hd }, RAD_ACCESS_PER_REQUEST, grp);
    F.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), nm.ckpt("%s.attn.norm_k.weight", pre), RAD_F32,
                   { hd }, RAD_ACCESS_PER_REQUEST, grp);
    if (!F.w_n1 || !F.w_n2 || !F.w_qn || !F.w_kn) return RAD_E_INVAL;
    F.op_n1 = zc_norm(b, rows, D, eps, F.w_n1, F.x, m.a_fh.x);
    RAD_ARCH_TRY(F.q_h.declare(b, g, m.a_fh, D));
    RAD_ARCH_TRY(F.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 3 * qd, D, RAD_SHARD_ROW,
                               grp_layer(grp),
                               { nm.ckpt("%s.attn.to_q.weight", pre),
                                 nm.ckpt("%s.attn.to_k.weight", pre),
                                 nm.ckpt("%s.attn.to_v.weight", pre) },
                               m.a_fh, m.b_fqkv, rows, { qd, qd, qd }));
    RAD_ARCH_TRY(F.gate.declare(b, nm, g, nm.f("%s.attn_gate", dn), qd, D, RAD_SHARD_ROW,
                                grp_layer(grp), { nm.ckpt("%s.attn.to_gate.weight", pre) }, m.a_fh,
                                m.b_fg, rows));
    /* q/k norms alone (no rotary in the fusion), zero-centred. */
    F.op_qkn = rw(b, RAD_OP(b, "qk_norm_rope",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                       RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head),
                                       RAD_F64("theta", 10000.0), RAD_F64("eps", eps),
                                       RAD_INT("rotary_dim", 0), RAD_F64("wadd", 1.0)),
                            RAD_WEIGHTS(F.w_qn, F.w_kn)),
                  { m.b_fqkv }, { m.b_fqkv });
    F.op_attn = rw(b, RAD_OP(b, "attn_dense",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                        RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                        RAD_STR("dtype", "bf16"),
                                        RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                             RAD_NOWEIGHTS),
                   { m.b_fqkv }, { m.a_fattn.x });
    F.op_gq = out_gate(b, g, rows, qd, m.b_fg, m.a_fattn, &F.op_sig, m.i8, &m.fgate_i8);
    if (F.op_sig) RAD_ARCH_TRY(F.q_attn.declare(b, g, m.a_fattn, qd));
    RAD_ARCH_TRY(F.o.declare(b, nm, g, nm.f("%s.attn_output", dn), D, qd, RAD_SHARD_COL,
                             grp_layer(grp), { nm.ckpt("%s.attn.to_out.0.weight", pre) },
                             m.a_fattn, m.b_fo, rows));
    F.op_ar1 = decl_all_reduce_tp(b, g, m.b_fo, rows);
    F.op_add1 = add_op(b, rows, D, F.x, m.b_fo, F.x);
    F.op_n2 = zc_norm(b, rows, D, eps, F.w_n2, F.x, m.a_fh.x);
    RAD_ARCH_TRY(F.q_h2.declare(b, g, m.a_fh, D));
    RAD_ARCH_TRY(F.gate_up.declare(b, nm, g, nm.f("%s.ffn_gate_up", dn), 2 * g.n_ff, D,
                                   RAD_SHARD_ROW, grp_layer(grp),
                                   { nm.ckpt("%s.ff.gate.weight", pre),
                                     nm.ckpt("%s.ff.up.weight", pre) },
                                   m.a_fh, m.b_fgu, rows, { g.n_ff, g.n_ff }));
    RAD_ARCH_TRY(F.gq.declare(b, g, m.b_fgu, m.a_fffn, g.n_ff, "silu", rows, true));
    RAD_ARCH_TRY(F.down.declare(b, nm, g, nm.f("%s.ffn_down", dn), D, g.n_ff, RAD_SHARD_COL,
                                grp_layer(grp), { nm.ckpt("%s.ff.down.weight", pre) }, m.a_fffn,
                                m.b_fo, rows));
    F.op_ar2 = decl_all_reduce_tp(b, g, m.b_fo, rows);
    F.op_add2 = add_op(b, rows, D, F.x, m.b_fo, F.x);
    if (!F.op_n1 || !F.op_qkn || !F.op_attn || !F.op_gq || !F.op_add1 || !F.op_n2 || !F.op_add2)
        return RAD_E_INVAL;
    return RAD_OK;
}

static int declare_fusion(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gf;
    Names& nm = m.nm;
    const int64_t D = g.n_embd, qd = g.q_dim();
    const int64_t RL = (int64_t)m.cond_rows * kNTaps;   /* the layer-wise blocks' rows */
    m.a_fh.x = decl_b(b, nm.f("fuse.h"), RAD_BF16, { RL, D });
    RAD_ARCH_TRY(m.a_fh.declare_pair(b, nm, g, "fuse.h", D, RL, m.i8, m.g8,
                                 "had_quant_act_i8g"));
    m.b_fqkv = decl_b(b, nm.f("fuse.qkv"), RAD_BF16, { RL, 3 * qd });
    m.b_fg   = decl_b(b, nm.f("fuse.gate"), RAD_BF16, { RL, qd });
    m.a_fattn.x = decl_b(b, nm.f("fuse.attn"), RAD_BF16, { RL, qd });
    RAD_ARCH_TRY(m.a_fattn.declare_pair(b, nm, g, "fuse.attn", qd, RL, m.i8, m.g8,
                                 "had_quant_act_i8g"));
    m.b_fo   = decl_b(b, nm.f("fuse.o"), RAD_BF16, { RL, D });
    m.b_fgu  = decl_b(b, nm.f("fuse.gate_up"), RAD_BF16, { RL, 2 * g.n_ff });
    m.a_fffn.x = decl_b(b, nm.f("fuse.ffn"), RAD_BF16, { RL, g.n_ff });
    /* The feed-forward product's scale group is its reader's, the first block's down projection,
     * which a recipe may keep narrower than the other linears: the fusion's 6912-wide product is
     * 3456 a rank at two, which whole 128s tile and whole 256s do not (krea-2-iq8r.recipe). The
     * probe maps the weight through the block's own LinearQ, which then does not map it again;
     * the resizes here keep the blocks the loops below declare. */
    const int nlw = (int)rad_meta_geti(meta, "transformer.num_layerwise_text_blocks", 2);
    const int nrf = (int)rad_meta_geti(meta, "transformer.num_refiner_text_blocks", 2);
    m.lw.resize((size_t)nlw);
    m.rf.resize((size_t)nrf);
    int64_t ffn_g8 = m.g8;
    if (m.i8 && nlw + nrf > 0) {
        LinearQ& d0 = nlw > 0 ? m.lw[0].down : m.rf[0].down;
        RAD_ARCH_TRY(d0.probe(b, nm, g, nm.f("fuse.0.ffn_down"),
                              { nm.ckpt("transformer.text_fusion.%s.0.ff.down.weight",
                                        nlw > 0 ? "layerwise_blocks" : "refiner_blocks") }));
        ffn_g8 = d0.probed_g8;
    }
    RAD_ARCH_TRY(m.a_fffn.declare_pair(b, nm, g, "fuse.ffn", g.n_ff, RL, m.i8, ffn_g8,
                                 "gated_had_quant_i8g"));
    m.b_fcu  = decl_b(b, nm.f("fuse.cu"), RAD_I32, { m.cond_rows + 1 });
    m.b_ft   = decl_b(b, nm.f("fuse.out"), RAD_BF16, { m.cond_rows, D });

    /* The layer-wise blocks attend within each token's twelve rows: cu 0, 12, 24, ... */
    m.op_fcu = rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, m.cond_rows + 1), RAD_INT("start", 0),
                                       RAD_INT("step", kNTaps)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_fcu });
    for (int i = 0; i < nlw; ++i) {
        RAD_ARCH_TRY(declare_fuse_block(b, m, m.lw[(size_t)i],
                                        nm.f("transformer.text_fusion.layerwise_blocks.%d", i),
                                        kGrpFuse + i, RL, m.b_taps));
        /* Their attention reads the twelve-row bounds. */
        RAD_ARCH_TRY(rad_op_reads(b, m.lw[(size_t)i].op_attn, &m.b_fcu, 1));
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("fuse.mix.weight"),
                          nm.ckpt("transformer.text_fusion.projector.weight")));
    /* The release stores the projector, embedders, modulations and output layer f32, which are
     * served bf16 as diffusers runs them: a recipe casts them. */
    if (weight_codes_dtype(b, "fuse.mix.weight", RAD_BF16) != RAD_BF16) {
        fprintf(stderr, "radiance: krea2: 'fuse.mix.weight' is stored %s; convert with "
                        "data/recipes/krea-2-bf16.recipe or krea-2-iq8r.recipe, which cast the "
                        "release's f32 tensors\n",
                rad_dtype_name(weight_codes_dtype(b, "fuse.mix.weight", RAD_BF16)));
        return RAD_E_UNSUPPORTED;
    }
    m.w_mix = decl_w(b, nm.f("fuse.mix.weight"), RAD_BF16, { 1, kNTaps }, RAD_ACCESS_PER_REQUEST,
                     RAD_SHARD_NONE, grp_layer(kGrpFuse));
    m.op_mix = rw(b, RAD_OP(b, "mix_rows",
                            RAD_PARAMS(RAD_RANGE("M", 1, m.cond_rows), RAD_INT("n", D),
                                       RAD_INT("k", kNTaps), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.w_mix)),
                  { m.b_taps }, { m.b_ft });
    for (int i = 0; i < nrf; ++i)
        RAD_ARCH_TRY(declare_fuse_block(b, m, m.rf[(size_t)i],
                                        nm.f("transformer.text_fusion.refiner_blocks.%d", i),
                                        kGrpFuse + nlw + i, m.cond_rows, m.b_ft));
    if (!m.op_fcu || !m.w_mix || !m.op_mix) return RAD_E_INVAL;

    /* ---- txt_in, into each lane's conditioning store */
    const int64_t Dd = m.gd.n_embd;
    m.b_txa = decl_b(b, nm.f("txt.a"), RAD_BF16, { m.cond_rows, D });
    m.b_txb = decl_b(b, nm.f("txt.b"), RAD_BF16, { m.cond_rows, Dd });
    m.b_txt = decl_b(b, nm.f("txt.out"), RAD_BF16, { m.cond_rows, Dd });
    m.w_txn = wcopy(b, nm.f("txt.norm.weight"), nm.ckpt("transformer.txt_in.norm.weight"), RAD_F32,
                    { D }, RAD_ACCESS_PER_REQUEST, kGrpTxtIn);
    m.op_txn = zc_norm(b, m.cond_rows, D, g.eps, m.w_txn, m.b_ft, m.b_txa);
    m.op_tx1 = dense_bias(b, nm, "txt.l1", "transformer.txt_in.linear_1", Dd, D, m.cond_rows, m.b_txa,
                          m.b_txb, &m.w_tx1, &m.b_tx1, RAD_ACCESS_PER_REQUEST, kGrpTxtIn);
    m.op_txg = unary(b, "gelu_tanh", m.cond_rows, Dd, m.b_txb, m.b_txb);
    m.op_tx2 = dense_bias(b, nm, "txt.l2", "transformer.txt_in.linear_2", Dd, Dd, m.cond_rows,
                          m.b_txb, m.b_txt, &m.w_tx2, &m.b_tx2, RAD_ACCESS_PER_REQUEST, kGrpTxtIn);
    /* Lane l's branch b at rows (2 l + b) m.cond_rows: the store's `stride` is two branches. */
    m.op_store = rw(b, RAD_OP(b, "lane_store",
                              RAD_PARAMS(RAD_RANGE("M", 1, m.cond_rows), RAD_INT("n", Dd),
                                         RAD_INT("stride", 2 * m.store_rows),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_txt, m.b_store }, { m.b_store });
    return m.w_txn && m.op_txn && m.op_tx1 && m.op_txg && m.op_tx2 && m.op_store ? RAD_OK
                                                                                 : RAD_E_INVAL;
}

/* The DiT's six modulation rows, reordered to rms_mod's [shift | scale] pairs: the checkpoint
 * holds (prescale, preshift, pregate, postscale, postshift, postgate). */
static int map_mod_rows(RadBuilder* b, const char* declared, const char* src, int64_t D) {
    return map_rows(b, declared, { { src, (int)D, D }, { src, 0, D }, { src, (int)(2 * D), D },
                                   { src, (int)(4 * D), D }, { src, (int)(3 * D), D },
                                   { src, (int)(5 * D), D } });
}

static int declare_dit(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, S = g.max_seqs * (m.max_refs ? 3 : 1);
    const int64_t D = g.n_embd, hd = g.head_dim, qd = g.q_dim(), kvd = g.kv_dim();
    const int64_t C = m.C, L = g.n_layer;
    const double eps = g.eps;
    const int gx = kGrpDit;
    /* Image rows a pass: a denoising branch an entry (one for an unguided image, two under
     * classifier-free guidance), as many entries as max-num-seqs allows, within the step's rows. */
    const int64_t NI = g.max_seqs > T / m.max_image_tokens ? T : g.max_seqs * m.max_image_tokens;

    /* ---- the timestep: [cos | sin] of 1000 bf16(t) over 256, linear, gelu_tanh, linear (temb);
     * the shared modulation from gelu_tanh(temb), then each block's table added (table_add), and
     * the final layer's from temb itself. */
    const int64_t te = rad_meta_geti(meta, "transformer.timestep_embed_dim", 256);
    m.b_tproj = decl_b(b, nm.f("dit.tproj"), RAD_BF16, { S, te });
    m.b_te1   = decl_b(b, nm.f("dit.te1"), RAD_BF16, { S, D });
    m.b_te1g  = decl_b(b, nm.f("dit.te1g"), RAD_BF16, { S, D });
    m.b_temb  = decl_b(b, nm.f("dit.temb"), RAD_BF16, { S, D });
    m.b_tembg = decl_b(b, nm.f("dit.tembg"), RAD_BF16, { S, D });
    m.b_tmod  = decl_b(b, nm.f("dit.tmod"), RAD_BF16, { S, 6 * D });
    m.b_mods  = decl_b(b, nm.f("dit.mods"), RAD_BF16, { S, L * 6 * D });
    m.b_fmod  = decl_b(b, nm.f("dit.fmod"), RAD_BF16, { S, 2 * D });
    m.op_temb = rw(b, RAD_OP(b, "timestep_embed",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", te),
                                        RAD_F64("scale", 1000.0), RAD_F64("max_period", 10000.0),
                                        RAD_INT("round", 2), RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   {}, { m.b_tproj });
    m.op_t1 = dense_bias(b, nm, "dit.t1", "transformer.time_embed.linear_1", D, te, S, m.b_tproj,
                         m.b_te1, &m.w_t1, &m.b_t1, RAD_ACCESS_PER_TOKEN, gx);
    m.op_tg = unary(b, "gelu_tanh", S, D, m.b_te1, m.b_te1g);
    m.op_t2 = dense_bias(b, nm, "dit.t2", "transformer.time_embed.linear_2", D, D, S, m.b_te1g,
                         m.b_temb, &m.w_t2, &m.b_t2, RAD_ACCESS_PER_TOKEN, gx);
    m.op_tg2 = unary(b, "gelu_tanh", S, D, m.b_temb, m.b_tembg);
    RAD_ARCH_TRY(map_mod_rows(b, nm.f("dit.mod.weight"), nm.ckpt("transformer.time_mod_proj.weight"),
                              D));
    m.w_mod = decl_w(b, nm.f("dit.mod.weight"), RAD_BF16, { 6 * D, D }, RAD_ACCESS_PER_TOKEN,
                     RAD_SHARD_NONE, grp_layer(gx));
    RAD_ARCH_TRY(map_mod_rows(b, nm.f("dit.mod.bias"), nm.ckpt("transformer.time_mod_proj.bias"),
                              D));
    m.b_mod = decl_w(b, nm.f("dit.mod.bias"), RAD_BF16, { 6 * D }, RAD_ACCESS_PER_TOKEN,
                     RAD_SHARD_NONE, grp_layer(gx));
    if (!m.w_mod || !m.b_mod) return RAD_E_INVAL;
    m.op_mod = rw(b, RAD_OP(b, "gemm_nt_bias",
                            RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("N", 6 * D), RAD_INT("K", D),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.w_mod, m.b_mod)),
                  { m.b_tembg }, { m.b_tmod });
    /* Each block's table, its six rows reordered as the projection's, added to the shared
     * modulation into the block's 6D columns of `mods` (declared with the blocks). */
    /* The final layer's (scale, shift), reordered to (shift, scale). */
    {
        const char* src = nm.ckpt("transformer.final_layer.scale_shift_table");
        RAD_ARCH_TRY(map_rows(b, nm.f("dit.final.table"), { { src, 1, 1 }, { src, 0, 1 } }));
        m.w_ftab = decl_w(b, nm.f("dit.final.table"), RAD_BF16, { 2, D }, RAD_ACCESS_PER_TOKEN,
                          RAD_SHARD_NONE, grp_layer(gx));
    }
    m.op_ftab = rw(b, RAD_OP(b, "table_add",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", D), RAD_INT("L", 2),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.w_ftab)),
                   { m.b_temb }, { m.b_fmod });
    if (!m.op_temb || !m.op_t1 || !m.op_tg || !m.op_t2 || !m.op_tg2 || !m.op_mod || !m.w_ftab ||
        !m.op_ftab)
        return RAD_E_INVAL;

    /* ---- the input rows: the image's latents a lane at a time through img_in, joined with the
     * lanes' text rows into the pass's [text | image] entries */
    m.b_lat  = decl_b(b, nm.f("dit.lat"), RAD_BF16, { NI, C });
    m.b_himg = decl_b(b, nm.f("dit.himg"), RAD_BF16, { NI, D });
    m.b_h    = decl_b(b, nm.f("dit.h"), RAD_BF16, { T, D });
    m.op_randn = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                         RAD_INT("stride", (int64_t)m.max_image_tokens),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_state }, { m.b_state });
    m.op_lgather = rw(b, RAD_OP(b, "lane_gather",
                                RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                           RAD_INT("stride", (int64_t)m.max_image_tokens),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_state }, { m.b_lat });
    m.op_imgin = dense_bias(b, nm, "dit.img_in", "transformer.img_in", D, C, NI, m.b_lat, m.b_himg,
                            &m.w_imgin, &m.b_imgin, RAD_ACCESS_PER_TOKEN, gx);
    m.op_join = rw(b, RAD_OP(b, "gather_rows2",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_store, m.b_himg }, { m.b_h });
    if (!m.op_randn || !m.op_lgather || !m.op_imgin || !m.op_join) return RAD_E_INVAL;

    /* ---- the blocks */
    m.a_hm.x = decl_b(b, nm.f("dit.hm"), RAD_BF16, { T, D });
    RAD_ARCH_TRY(m.a_hm.declare_pair(b, nm, g, "dit.hm", D, 0, m.i8, m.g8,
                                 "had_quant_act_i8g"));
    m.b_qkv = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { T, qd + 2 * kvd });
    m.b_g   = decl_b(b, nm.f("dit.gate"), RAD_BF16, { T, qd });
    m.a_attn.x = decl_b(b, nm.f("dit.attn"), RAD_BF16, { T, qd });
    RAD_ARCH_TRY(m.a_attn.declare_pair(b, nm, g, "dit.attn", qd, 0, m.i8, m.g8,
                                 "had_quant_act_i8g"));
    m.b_ao = decl_b(b, nm.f("dit.ao"), RAD_BF16, { T, D });
    m.b_gu = decl_b(b, nm.f("dit.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_ffn.x = decl_b(b, nm.f("dit.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_ffn.declare_pair(b, nm, g, "dit.ffn", g.n_ff, 0, m.i8, m.g8,
                                 "gated_had_quant_i8g"));

    const std::vector<double> ax = meta_list(meta, "transformer.axes_dims_rope");
    const int64_t a0 = ax.size() == 3 ? (int64_t)ax[0] : 32, a1 = ax.size() == 3 ? (int64_t)ax[1] : 48,
                  a2 = ax.size() == 3 ? (int64_t)ax[2] : 48;
    if (a0 + a1 + a2 != hd) {
        fprintf(stderr, "radiance: krea2: transformer.axes_dims_rope must sum to the head\n");
        return RAD_E_INVAL;
    }
    const char* sections = nm.f("%lld %lld %lld", (long long)a0 / 2, (long long)a1 / 2,
                                (long long)a2 / 2);
    const double theta = rad_meta_getf(meta, "transformer.rope_theta", 1000.0);

    /* The modulated norm, gated by the branch before it when `gated`, into the next linear's
     * input: codes for fp8 and, where the rotating twin is served, for rotated int8; the bf16
     * plane for bf16 and for rotated int8 without the twin, whose codes the rotating quantiser
     * then takes from it (DitLayer::q_n1 / q_n2). */
    m.norm_i8 = m.i8 && m.a_hm.q8_fed && rad_op_declared(b, "rms_mod_had_quant_i8g") != 0;
    auto mod_norm = [&](rad_weight w, bool gated) {
        const bool q = (!g.w_bf16 && !m.i8) || m.norm_i8;
        rad_op op = q ? RAD_OP(b, m.norm_i8 ? "rms_mod_had_quant_i8g" : "rms_mod_quant_fp8",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_F64("eps", eps),
                                          RAD_INT("group",
                                                  m.norm_i8 ? m.g8 : (int64_t)RAD_SCALE_GROUP),
                                          RAD_INT("gated", gated ? 1 : 0), RAD_STR("dtype", "bf16"),
                                          RAD_F64("wadd", 1.0), RAD_INT("norm_once", 1)),
                               RAD_WEIGHTS(w))
                      : RAD_OP(b, "rms_mod",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_F64("eps", eps),
                                          RAD_INT("gated", gated ? 1 : 0), RAD_STR("dtype", "bf16"),
                                          RAD_F64("wadd", 1.0), RAD_INT("norm_once", 1)),
                               RAD_WEIGHTS(w));
        if (!op) return op;
        rad_buf rd[3] = { m.b_mods, m.b_h, m.b_ao };
        rad_buf wq[3] = { m.a_hm.cq(), m.a_hm.cs(), m.b_h };
        rad_buf wb[2] = { m.a_hm.x, m.b_h };
        rad_op_reads(b, op, rd, gated ? 3 : 2);
        if (q) rad_op_writes(b, op, wq, 3);
        else   rad_op_writes(b, op, wb, 2);
        return op;
    };

    m.dit.resize((size_t)L);
    for (int64_t l = 0; l < L; ++l) {
        DitLayer& B = m.dit[(size_t)l];
        const int li = (int)l;
        const int grp = kGrpDit + 1 + li;
        const char* pre = nm.ckpt("transformer.transformer_blocks.%d", li);
        const char* dn = nm.f("dit.blk.%d", li);
        B.w_n1 = wcopy(b, nm.f("%s.norm1.weight", dn), nm.ckpt("%s.norm1.weight", pre), RAD_BF16,
                       { D }, RAD_ACCESS_PER_TOKEN, grp);
        B.w_n2 = wcopy(b, nm.f("%s.norm2.weight", dn), nm.ckpt("%s.norm2.weight", pre), RAD_BF16,
                       { D }, RAD_ACCESS_PER_TOKEN, grp);
        B.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), nm.ckpt("%s.attn.norm_q.weight", pre),
                       RAD_F32, { hd }, RAD_ACCESS_PER_TOKEN, grp);
        B.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), nm.ckpt("%s.attn.norm_k.weight", pre),
                       RAD_F32, { hd }, RAD_ACCESS_PER_TOKEN, grp);
        if (!B.w_n1 || !B.w_n2 || !B.w_qn || !B.w_kn) return RAD_E_INVAL;
        {
            const char* src = nm.ckpt("%s.scale_shift_table", pre);
            RAD_ARCH_TRY(map_rows(b, nm.f("%s.table", dn),
                                  { { src, 1, 1 }, { src, 0, 1 }, { src, 2, 1 }, { src, 4, 1 },
                                    { src, 3, 1 }, { src, 5, 1 } }));
            B.w_tab = decl_w(b, nm.f("%s.table", dn), RAD_BF16, { 6, D }, RAD_ACCESS_PER_TOKEN,
                             RAD_SHARD_NONE, grp_layer(grp));
            B.op_tab = rw(b, RAD_OP(b, "table_add",
                                    RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", 6 * D),
                                               RAD_INT("L", 1), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(B.w_tab)),
                          { m.b_tmod }, { m.b_mods });
            if (!B.w_tab || !B.op_tab) return RAD_E_INVAL;
        }
        B.op_n1 = mod_norm(B.w_n1, l > 0);
        if (m.i8 && m.a_hm.q8_fed && !m.norm_i8) RAD_ARCH_TRY(B.q_n1.declare(b, g, m.a_hm, D));
        if (l == 0) B.qkv = m.qkv0;
        RAD_ARCH_TRY(B.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), qd + 2 * kvd, D,
                                   RAD_SHARD_ROW, grp_layer(grp),
                                   { nm.ckpt("%s.attn.to_q.weight", pre),
                                     nm.ckpt("%s.attn.to_k.weight", pre),
                                     nm.ckpt("%s.attn.to_v.weight", pre) },
                                   m.a_hm, m.b_qkv, 0, { qd, kvd, kvd }));
        RAD_ARCH_TRY(B.gate.declare(b, nm, g, nm.f("%s.attn_gate", dn), qd, D, RAD_SHARD_ROW,
                                    grp_layer(grp), { nm.ckpt("%s.attn.to_gate.weight", pre) },
                                    m.a_hm, m.b_g));
        B.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", g.n_head),
                                           RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_F64("theta", theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "gptj_blocks"),
                                           RAD_INT("rotary_dim", hd), RAD_STR("sections", sections),
                                           RAD_F64("wadd", 1.0)),
                                RAD_WEIGHTS(B.w_qn, B.w_kn)),
                      { m.b_qkv }, { m.b_qkv });
        B.op_attn = rw(b, RAD_OP(b, m.max_refs ? "attn_dense_ref" : "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", g.gqa()), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                 RAD_NOWEIGHTS),
                       { m.b_qkv }, { m.a_attn.x });
        B.op_gq = out_gate(b, g, T, qd, m.b_g, m.a_attn, &B.op_sig, m.i8, &m.gate_i8);
        if (B.op_sig) RAD_ARCH_TRY(B.q_attn.declare(b, g, m.a_attn, qd));
        RAD_ARCH_TRY(B.o.declare(b, nm, g, nm.f("%s.attn_output", dn), D, qd, RAD_SHARD_COL,
                                 grp_layer(grp), { nm.ckpt("%s.attn.to_out.0.weight", pre) },
                                 m.a_attn, m.b_ao));
        B.op_ar1 = decl_all_reduce_tp(b, g, m.b_ao, T);
        B.op_n2 = mod_norm(B.w_n2, true);
        if (m.i8 && m.a_hm.q8_fed && !m.norm_i8) RAD_ARCH_TRY(B.q_n2.declare(b, g, m.a_hm, D));
        RAD_ARCH_TRY(B.gate_up.declare(b, nm, g, nm.f("%s.ffn_gate_up", dn), 2 * g.n_ff, D,
                                       RAD_SHARD_ROW, grp_layer(grp),
                                       { nm.ckpt("%s.ff.gate.weight", pre),
                                         nm.ckpt("%s.ff.up.weight", pre) },
                                       m.a_hm, m.b_gu, 0, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(B.gq.declare(b, g, m.b_gu, m.a_ffn, g.n_ff, "silu", 0, true));
        RAD_ARCH_TRY(B.down.declare(b, nm, g, nm.f("%s.ffn_down", dn), D, g.n_ff, RAD_SHARD_COL,
                                    grp_layer(grp), { nm.ckpt("%s.ff.down.weight", pre) }, m.a_ffn,
                                    m.b_ao));
        B.op_ar2 = decl_all_reduce_tp(b, g, m.b_ao, T);
        if (!B.op_n1 || !B.op_qkr || !B.op_attn || !B.op_gq || !B.op_n2) return RAD_E_INVAL;
    }

    /* ---- the output: the last block's feed-forward gated in, the image's rows picked out, the
     * final adaptive norm and proj_out, and the sampler's step */
    m.b_hi = decl_b(b, nm.f("dit.h_img"), RAD_BF16, { NI, D });
    m.b_fz = decl_b(b, nm.f("dit.fz"), RAD_BF16, { NI, D });
    m.b_v  = decl_b(b, nm.f("dit.v"), RAD_BF16, { NI, C });
    m.op_g2 = rw(b, RAD_OP(b, "gate_add",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                      RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                           RAD_NOWEIGHTS),
                 { m.b_ao, m.b_mods, m.b_h }, { m.b_h });
    m.op_gimg = rw(b, RAD_OP(b, "gather_rows",
                             RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", D),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_h }, { m.b_hi });
    m.w_fn = wcopy(b, nm.f("dit.final.norm.weight"), nm.ckpt("transformer.final_layer.norm.weight"),
                   RAD_BF16, { D }, RAD_ACCESS_PER_TOKEN, gx);
    m.op_fn = rw(b, RAD_OP(b, "rms_mod",
                           RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", D), RAD_F64("eps", eps),
                                      RAD_INT("gated", 0), RAD_STR("dtype", "bf16"),
                                      RAD_F64("wadd", 1.0), RAD_INT("norm_once", 1)),
                           RAD_WEIGHTS(m.w_fn)),
                 { m.b_fmod, m.b_hi }, { m.b_hi, m.b_fz });
    m.op_pout = dense_bias(b, nm, "dit.proj_out", "transformer.final_layer.linear", C, D, NI,
                           m.b_fz, m.b_v, &m.w_pout, &m.b_pout, RAD_ACCESS_PER_TOKEN, gx);
    m.op_euler = rw(b, RAD_OP(b, "flow_euler",
                              RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                         RAD_INT("stride", (int64_t)m.max_image_tokens),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_v, m.b_state, m.b_prev }, { m.b_state, m.b_prev });
    return m.op_g2 && m.op_gimg && m.w_fn && m.op_fn && m.op_pout && m.op_euler ? RAD_OK
                                                                               : RAD_E_INVAL;
}

/* The Qwen-Image VAE encoder for a single reference frame. The temporal downsample
 * convolutions are bypassed on the first frame. Spatial downsamples still run. */
static int declare_reference_encoder(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Vae& v = m.enc;
    Names& nm = m.nm;
    v.nm = &nm;
    v.b = b;
    v.scope = "venc";
    v.down = true;
    v.first_group = kGrpVaeEnc;
    v.base_px = m.max_image_tokens * 256;
    v.form = VAE_CONV_CAUSAL;
    const int64_t base = rad_meta_geti(meta, "vae.base_dim", 96);
    const int64_t channels = rad_meta_geti(meta, "vae.z_dim", 16);
    const int64_t blocks = rad_meta_geti(meta, "vae.num_res_blocks", 2);
    const auto mult = meta_list(meta, "vae.dim_mult");
    if (mult.size() != 4 || blocks != 2 || channels * 4 != m.C) {
        fprintf(stderr, "radiance: krea2: reference encoding requires the Qwen-Image VAE\n");
        return RAD_E_UNSUPPORTED;
    }

    m.b_enc_cu = decl_b(b, nm.f("venc.cu"), RAD_I32, {2});
    if (!m.b_enc_cu) return RAD_E_INVAL;
    m.op_enc_cu = rw(b, RAD_OP(b, "cu_repeat",
                               RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", 4)),
                               RAD_NOWEIGHTS), {}, {m.b_enc_cu});
    if (!m.op_enc_cu) return RAD_E_INVAL;

    const VImg input{0, rad_meta_geti(meta, "vae.input_channels", 3), 0};
    VImg x = v.conv(input, base, 3, 1, 1, "same", "vae.encoder.conv_in");
    int index = 0;
    for (size_t level = 0; level < mult.size(); ++level) {
        const int64_t width = base * (int64_t)mult[level];
        for (int64_t block = 0; block < blocks; ++block)
            x = v.resblock(x, width, nm.ckpt("vae.encoder.down_blocks.%d", index++));
        if (level + 1 == mult.size()) break;
        v.form = VAE_CONV_2D;
        x = v.conv(x, width, 3, 2, 1, "down",
                   nm.ckpt("vae.encoder.down_blocks.%d.resample.1", index++));
        v.form = VAE_CONV_CAUSAL;
    }
    x = v.resblock(x, x.ch, "vae.encoder.mid_block.resnets.0");
    v.form = VAE_CONV_2D;
    x = v.attention(x, "vae.encoder.mid_block.attentions.0");
    v.form = VAE_CONV_CAUSAL;
    x = v.resblock(x, x.ch, "vae.encoder.mid_block.resnets.1");
    x = v.norm(x, true, "vae.encoder.norm_out", 4);
    x = v.conv(x, 2 * channels, 3, 1, 1, "same", "vae.encoder.conv_out");
    m.enc_out = v.conv(x, 2 * channels, 1, 1, 1, "same", "vae.quant_conv");
    if (!m.enc_out.buf) return RAD_E_INVAL;
    return v.attn_reads(m.b_enc_cu);
}

static int declare_reference_tokens(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const int64_t D = m.gd.n_embd, C = m.C, Z = C / 4, N = m.max_image_tokens;
    const auto mean = meta_list(meta, "vae.latents_mean"), sd = meta_list(meta, "vae.latents_std");
    if ((int64_t)mean.size() != Z || (int64_t)sd.size() != Z) return RAD_E_SHAPE;
    for (int64_t i = 0; i < Z; ++i) {
        if (!std::isfinite(mean[i]) || !(sd[i] > 0) || !std::isfinite(sd[i])) return RAD_E_INVAL;
        m.enc_mean += (i ? " " : "") + std::to_string(mean[i]);
        m.enc_std += (i ? " " : "") + std::to_string(sd[i]);
    }
    m.b_enc_norm = decl_b(b, nm.f("venc.mode"), RAD_BF16, {4 * N, Z});
    m.b_enc_pack = decl_b(b, nm.f("venc.pack"), RAD_BF16, {N, C});
    m.b_enc_embed = decl_b(b, nm.f("venc.embed"), RAD_BF16, {N, D});
    m.op_enc_norm = rw(b, RAD_OP(b, "chan_affine",
                                 RAD_PARAMS(RAD_RANGE("M", 1, 4 * N), RAD_INT("n", Z),
                                            RAD_STR("mul", m.enc_std.c_str()),
                                            RAD_STR("add", m.enc_mean.c_str()),
                                            RAD_STR("mode", "inv"), RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS), {m.enc_out.buf}, {m.b_enc_norm});
    m.op_enc_zero = rw(b, RAD_OP(b, "fill",
                                 RAD_PARAMS(RAD_RANGE("M", 1, N), RAD_INT("n", C),
                                            RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                                 RAD_NOWEIGHTS), {}, {m.b_enc_pack});
    m.op_enc_pack = rw(b, RAD_OP(b, "s2d_avg_add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, N), RAD_INT("cin", Z),
                                            RAD_INT("cout", C), RAD_INT("ft", 1),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS), {m.b_enc_norm, m.b_enc_pack}, {m.b_enc_pack});
    m.op_enc_project = rw(b, RAD_OP(b, "gemm_nt_bias",
                                    RAD_PARAMS(RAD_RANGE("M", 1, N), RAD_INT("N", D),
                                               RAD_INT("K", C), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(m.w_imgin, m.b_imgin)),
                            {m.b_enc_pack}, {m.b_enc_embed});
    m.op_enc_store = rw(b, RAD_OP(b, "lane_store",
                                  RAD_PARAMS(RAD_RANGE("M", 1, N), RAD_INT("n", D),
                                             RAD_INT("stride", 2 * m.store_rows),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS), {m.b_enc_embed, m.b_store}, {m.b_store});
    return m.op_enc_cu && m.op_enc_norm && m.op_enc_zero && m.op_enc_pack &&
           m.op_enc_project && m.op_enc_store
               ? RAD_OK : RAD_E_INVAL;
}

/* The decoder: the latent denormalised, the 2 x 2 tokens unpacked to the VAE's 16 channels, then
 * Wan 2.1's decoder on one frame. Its resblock, conv_in, conv_out and post_quant_conv convolutions
 * are causal 3-D ones read at their last tap; the attention's and the upsamplers' are 2-D. */
static int declare_decoder(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Vae& v = m.dec;
    Names& nm = m.nm;
    v.nm = &nm;
    v.b = b;
    v.scope = "vdec";
    v.down = false;
    v.first_group = kGrpVaeDec;
    v.base_px = (int64_t)m.max_image_tokens * 4;
    v.form = VAE_CONV_CAUSAL;
    const int64_t base = rad_meta_geti(meta, "vae.base_dim", 96);
    const int64_t zd = rad_meta_geti(meta, "vae.z_dim", 16);
    const int64_t nres = rad_meta_geti(meta, "vae.num_res_blocks", 2);
    std::vector<double> mult = meta_list(meta, "vae.dim_mult");
    std::vector<double> td = meta_list(meta, "vae.temperal_downsample");
    if (mult.empty()) mult = { 1, 2, 4, 4 };
    if (td.empty()) td = { 0, 1, 1 };
    const std::vector<double> mean = meta_list(meta, "vae.latents_mean");
    const std::vector<double> sd = meta_list(meta, "vae.latents_std");
    if ((int64_t)mean.size() != zd || (int64_t)sd.size() != zd || mult.size() != 4) {
        fprintf(stderr, "radiance: krea2: the VAE's latents_mean / latents_std / dim_mult are not "
                        "Wan 2.1's\n");
        return RAD_E_INVAL;
    }
    /* The packed channel c 4 + k reads latent channel c: the statistics repeated four times. The
     * reference divides by bf16(1 / std) and adds the mean, each rounded. */
    for (int64_t k = 0; k < 4 * zd; ++k) {
        char t[64];
        std::snprintf(t, sizeof t, "%s%.9g", k ? " " : "", 1.0 / sd[(size_t)(k / 4)]);
        m.inv_std += t;
        std::snprintf(t, sizeof t, "%s%.9g", k ? " " : "", mean[(size_t)(k / 4)]);
        m.mean += t;
        m.ones += k ? " 1" : "1";
        m.zeros += k ? " 0" : "0";
    }
    const int64_t NT = m.max_image_tokens;
    m.b_dl  = decl_b(b, nm.f("vdec.lat"), RAD_BF16, { NT, 4 * zd });
    m.b_dl2 = decl_b(b, nm.f("vdec.lat2"), RAD_BF16, { NT, 4 * zd });
    m.b_decin = decl_b(b, nm.f("vdec.in"), RAD_BF16, { 4 * NT, zd });
    m.b_cu4 = decl_b(b, nm.f("vdec.cu"), RAD_I32, { 2 });
    m.op_dinv = rw(b, RAD_OP(b, "chan_affine",
                             RAD_PARAMS(RAD_RANGE("M", 1, NT), RAD_INT("n", 4 * zd),
                                        RAD_STR("mul", m.inv_std.c_str()),
                                        RAD_STR("add", m.zeros.c_str()), RAD_STR("mode", "inv"),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_state, m.b_prev }, { m.b_dl });
    m.op_dadd = rw(b, RAD_OP(b, "chan_affine",
                             RAD_PARAMS(RAD_RANGE("M", 1, NT), RAD_INT("n", 4 * zd),
                                        RAD_STR("mul", m.ones.c_str()),
                                        RAD_STR("add", m.mean.c_str()), RAD_STR("mode", "fma"),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_dl }, { m.b_dl2 });
    m.op_dzero = rw(b, RAD_OP(b, "fill",
                              RAD_PARAMS(RAD_RANGE("M", 1, 4 * NT), RAD_INT("n", zd),
                                         RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                              RAD_NOWEIGHTS),
                    {}, { m.b_decin });
    m.op_unpack = rw(b, RAD_OP(b, "d2s_add",
                               RAD_PARAMS(RAD_RANGE("M", 1, 4 * NT), RAD_INT("cin", 4 * zd),
                                          RAD_INT("cout", zd), RAD_INT("ft", 1),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_dl2, m.b_decin }, { m.b_decin });
    m.op_cu4 = rw(b, RAD_OP(b, "cu_repeat",
                            RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", 4)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_cu4 });
    if (!m.op_dinv || !m.op_dadd || !m.op_dzero || !m.op_unpack || !m.op_cu4) return RAD_E_INVAL;

    /* dims = base * [mult[-1]] + mult reversed; up block i reads dims[i] (halved after an
     * upsampler) and writes dims[i + 1]. */
    std::vector<int64_t> dims{ base * (int64_t)mult.back() };
    for (int i = 3; i >= 0; --i) dims.push_back(base * (int64_t)mult[(size_t)i]);
    VImg x{ m.b_decin, zd, 0 };
    x = v.conv(x, zd, 1, 1, 1, "same", "vae.post_quant_conv");
    x = v.conv(x, dims[0], 3, 1, 1, "same", "vae.decoder.conv_in");
    x = v.resblock(x, dims[0], "vae.decoder.mid_block.resnets.0");
    v.form = VAE_CONV_2D;
    x = v.attention(x, "vae.decoder.mid_block.attentions.0");
    v.form = VAE_CONV_CAUSAL;
    x = v.resblock(x, dims[0], "vae.decoder.mid_block.resnets.1");
    v.band_from = (int)v.ops.size();
    v.whole.push_back(m.b_decin);
    for (const VOp& o : v.ops) v.whole.push_back(o.y.buf);
    for (int i = 0; i < 4; ++i) {
        for (int r = 0; r < nres + 1; ++r)
            x = v.resblock(x, dims[(size_t)i + 1], nm.ckpt("vae.decoder.up_blocks.%d.resnets.%d", i, r));
        if (i != 3) {
            v.form = VAE_CONV_2D;
            x = v.conv(x, dims[(size_t)i + 1] / 2, 3, 1, 2, "same",
                       nm.ckpt("vae.decoder.up_blocks.%d.upsamplers.0.resample.1", i));
            v.form = VAE_CONV_CAUSAL;
        }
    }
    x = v.norm(x, true, "vae.decoder.norm_out", 4);
    m.dec_out = v.conv(x, 3, 3, 1, 1, "same", "vae.decoder.conv_out");
    if (!m.dec_out.buf) return RAD_E_INVAL;
    RAD_ARCH_TRY(v.attn_reads(m.b_cu4));
    const int64_t px = (int64_t)m.max_image_tokens * 256;
    m.b_img = decl_b(b, nm.f("image_out"), RAD_U8, { px * 3 }, RAD_BUF_PERSIST);
    m.op_rgb = rw(b, RAD_OP(b, "to_rgba8",
                            RAD_PARAMS(RAD_RANGE("M", 1, px), RAD_INT("n", 3),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { m.dec_out.buf }, { m.b_img });
    if (!m.op_rgb) return RAD_E_INVAL;
    v.halo = v.band_halo();
    if (v.halo <= 0) {
        fprintf(stderr, "radiance: krea2: the decoder after its mid-block is not row-local\n");
        return RAD_E_INVAL;
    }
    /* More than one rank: each runs a band of the tail and the bands are gathered (qwenimage21's
     * scheme: a band is ceil(H / world) latent rows plus a halo each side). Level 0 is the
     * unpacked latent, two rows a token row, eight pixels a latent row. */
    const int world = m.gt.world;
    if (world > 1) {
        const int64_t H0 = 4 * (int64_t)m.max_image_tokens;
        const int64_t bpx = (H0 / world + H0 / (2 * v.halo) + 1) * 64;
        m.b_rgbf = decl_b(b, nm.f("vdec.bands"), RAD_BF16, { world * bpx, 3 });
        m.op_bands = rw(b, RAD_OP(b, "all_gather",
                                  RAD_PARAMS(RAD_INT("world_size", world),
                                             RAD_RANGE("numel", 1, bpx * 3),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS),
                        { m.dec_out.buf }, { m.b_rgbf });
        m.op_rgb_bands = rw(b, RAD_OP(b, "to_rgba8",
                                      RAD_PARAMS(RAD_RANGE("M", 1, px), RAD_INT("n", 3),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_NOWEIGHTS),
                            { m.b_rgbf }, { m.b_img });
        if (!m.op_bands || !m.op_rgb_bands) return RAD_E_INVAL;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->max_seqs <= 0) return RAD_E_INVAL;
    RAD_ARCH_TRY(check_reference_encoder(meta));
    /* Grounded edits reserve text/reference/target modulation rows per entry. */
    if (has_reference_encoder(meta) && ctx->max_seqs > INT64_MAX / 3) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    m.lanes = std::clamp<int64_t>(ctx->max_seqs, 1, kMaxLanes);
    m.max_image_tokens = image_tokens(ctx->gen_max_pixels);
    if (m.max_image_tokens == 0) return RAD_E_INVAL;
    RAD_ARCH_TRY(check_format(meta));
    RAD_ARCH_TRY(geom_from(m.gt, meta, ctx, "bf16", RAD_BF16));
    Names& nm = m.nm;
    /* Qwen3VLModel's tensors (language_model.*) or a ...ForConditionalGeneration's
     * (model.language_model.*). */
    m.te_pre = std::strstr(rad_meta_gets(meta, "architectures", ""), "ForConditionalGeneration")
                   ? "model.language_model" : "language_model";
    m.turbo = std::strstr(rad_meta_gets(meta, "pipeline._class_name", ""), "Turbo") != nullptr ||
              rad_meta_geti(meta, "pipeline.is_distilled", 0) != 0;

    /* ---- the text fusion's and the DiT's geometry, off the transformer's config */
    Geom& gf = m.gf;
    gf = m.gt;
    gf.n_embd = rad_meta_geti(meta, "transformer.text_hidden_dim", 2560);
    gf.n_head_all = rad_meta_geti(meta, "transformer.text_num_attention_heads", 20);
    gf.n_head_kv_all = rad_meta_geti(meta, "transformer.text_num_key_value_heads", gf.n_head_all);
    gf.head_dim = gf.n_embd / gf.n_head_all;
    gf.n_ff_all = rad_meta_geti(meta, "transformer.text_intermediate_size", 6912);
    gf.eps = (float)rad_meta_getf(meta, "transformer.norm_eps", 1e-5);
    gf.max_tok = (int64_t)m.cond_rows * kNTaps;
    RAD_ARCH_TRY(divide_or_fail(gf.n_head_all, gf.world, "the text fusion's heads", &gf.n_head));
    RAD_ARCH_TRY(divide_or_fail(gf.n_head_kv_all, gf.world, "the text fusion's KV heads",
                                &gf.n_head_kv));
    RAD_ARCH_TRY(divide_or_fail(gf.n_ff_all, gf.world, "the text fusion's feed-forward width",
                                &gf.n_ff));
    if (gf.n_head != gf.n_head_kv || gf.head_dim != 128) {
        fprintf(stderr, "radiance: krea2: the text fusion must be multi-head at head_dim 128\n");
        return RAD_E_UNSUPPORTED;
    }
    Geom& gd = m.gd;
    gd = m.gt;
    gd.n_layer    = rad_meta_geti(meta, "transformer.num_layers", 28);
    gd.head_dim   = rad_meta_geti(meta, "transformer.attention_head_dim", 128);
    gd.n_head_all = rad_meta_geti(meta, "transformer.num_attention_heads", 48);
    gd.n_head_kv_all = rad_meta_geti(meta, "transformer.num_key_value_heads", 12);
    gd.n_embd     = gd.head_dim * gd.n_head_all;
    gd.n_ff_all   = rad_meta_geti(meta, "transformer.intermediate_size", 16384);
    gd.eps        = (float)rad_meta_getf(meta, "transformer.norm_eps", 1e-5);
    RAD_ARCH_TRY(divide_or_fail(gd.n_head_all, gd.world, "transformer.num_attention_heads",
                                &gd.n_head));
    RAD_ARCH_TRY(divide_or_fail(gd.n_head_kv_all, gd.world, "transformer.num_key_value_heads",
                                &gd.n_head_kv));
    RAD_ARCH_TRY(divide_or_fail(gd.n_ff_all, gd.world, "the DiT's feed-forward width", &gd.n_ff));
    gd.kv_off = 0;
    if (gd.n_layer >= 1000 || gf.n_layer > kGrpTxtIn - kGrpFuse) return RAD_E_UNSUPPORTED;
    m.C = rad_meta_geti(meta, "transformer.in_channels", 64);

    /* ---- bf16 or block fp8: what the container holds for the first DiT projection decides it
     * for every linear of every network (rad_linear.h, Geom::w_bf16). */
    {
        LinearQ& probe = m.qkv0;
        RAD_ARCH_TRY(probe.probe(b, nm, gd, "dit.blk.0.attn_qkv",
                                 { "transformer.transformer_blocks.0.attn.to_q.weight",
                                   "transformer.transformer_blocks.0.attn.to_k.weight",
                                   "transformer.transformer_blocks.0.attn.to_v.weight" }));
        RadEncoding e{};
        const bool known = weight_enc(b, nm.f("dit.blk.0.attn_qkv.weight"), &e);
        const bool plain = known && rad_enc_is(&e, "plain");
        gd.w_bf16 = gf.w_bf16 = m.gt.w_bf16 = plain;
        /* Or rotated int8 (IQ8R): the activations then carry int8 codes at the weights' scale
         * group in place of the E4M3 pair (ActQ::declare_pair). */
        m.i8 = known && (linear_is_w8h(e) || linear_is_wnh(e));
        if (m.i8) m.g8 = linear_w8h_group(e);
    }

    const bool references = has_reference_encoder(meta);
    if (references && VisionTower::present(meta) && ctx->max_enc_patches > 0) {
        RAD_ARCH_TRY(m.vision.declare(b, meta, ctx, nm, m.gt.n_embd, kGrpVision, true, false,
                                      m.te_pre == "language_model" ? "visual" : "model.visual"));
        m.max_refs = reference_count(ctx->gen_max_refs);
        if (m.max_refs > kMaxReferences) {
            std::fprintf(stderr, "radiance: krea2: --image-gen-max-refs supports at most two references\n");
            return RAD_E_INVAL;
        }
        m.cond_rows = edit_cond_rows(m.max_refs, m.vision.merge);
        m.te_rows = m.cond_rows + kTemplateRows;
        gf.max_tok = m.cond_rows * kNTaps;
        m.store_rows = m.cond_rows + m.max_refs * m.max_image_tokens;
    }

    /* ---- the per-image state, the conditioning store, and in issue order the language model,
     * the text fusion, the DiT and the decoder */
    const int64_t lane_rows = m.lanes * m.max_image_tokens;
    m.b_state = decl_b(b, nm.f("lat_state"), RAD_BF16, { lane_rows, m.C }, RAD_BUF_PERSIST);
    m.b_prev  = decl_b(b, nm.f("lat_preview"), RAD_BF16, { lane_rows, m.C }, RAD_BUF_PERSIST);
    m.b_store = decl_b(b, nm.f("cond_store"), RAD_BF16,
                       { 2 * m.lanes * m.store_rows, gd.n_embd }, RAD_BUF_PERSIST);
    RAD_ARCH_TRY(declare_lora_names(b));
    RAD_ARCH_TRY(declare_te(b, meta, m));
    RAD_ARCH_TRY(declare_fusion(b, meta, m));
    RAD_ARCH_TRY(declare_dit(b, meta, m));
    if (references) RAD_ARCH_TRY(declare_reference_encoder(b, meta, m));
    if (m.max_refs) RAD_ARCH_TRY(declare_reference_tokens(b, meta, m));
    RAD_ARCH_TRY(declare_decoder(b, meta, m));

    /* ---- what the core drives */
    if (!ctx->shape_probe) {
        static thread_local std::string tmpl, edit, drop;
        tmpl = std::string("<|im_start|>system\n") + kSysPrompt +
               "<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
        drop = std::string("<|im_start|>system\n") + kSysPrompt + "<|im_end|>\n<|im_start|>user\n";
        RadGeneratorDecl d{};
        d.name = m.turbo ? "krea-2-turbo" : "krea-2";
        d.layout = RAD_GEN_LAYOUT_JOINT_STEP;
        d.tmpl_text = tmpl.c_str();
        edit = drop + "{images}{prompt}<|im_end|>\n<|im_start|>assistant\n";
        d.tmpl_edit = m.max_refs ? edit.c_str() : tmpl.c_str();
        d.tmpl_image = m.max_refs ? "{image}" : nullptr;
        d.drop_prefix = drop.c_str();
        d.latent_channels = m.C;
        d.latent_scale = 16;
        d.size_multiple = 16;
        d.pixel_channels = 3;
        d.slot_rows = 1;
        d.max_tokens = m.max_image_tokens;
        d.max_refs = m.max_refs;
        d.ref_area = m.max_refs ? m.max_image_tokens * 256 : 1;
        const int64_t side = (int64_t)std::sqrt(std::min<int64_t>(m.max_image_tokens, 4096)) * 16;
        d.default_width = side;
        d.default_height = side;
        /* Raw: 28 steps on the resolution-shifted schedule, guidance 4.5 in the card's terms.
         * Turbo: 8 steps at a fixed mu of 1.15 (base and max shift equal), no guidance. */
        d.default_steps = m.turbo ? 8 : 28;
        d.n_sigmas = 0;
        d.sigmas = nullptr;
        d.shift = rad_meta_getf(meta, "scheduler.shift", 1.0);
        d.dyn_shift = (int32_t)rad_meta_geti(meta, "scheduler.use_dynamic_shifting", 1);
        d.shift_exp = std::strcmp(rad_meta_gets(meta, "scheduler.time_shift_type", "exponential"),
                                  "exponential") == 0;
        d.base_shift = m.turbo ? 1.15 : rad_meta_getf(meta, "scheduler.base_shift", 0.5);
        d.max_shift = m.turbo ? 1.15 : rad_meta_getf(meta, "scheduler.max_shift", 1.15);
        d.base_seq = rad_meta_geti(meta, "scheduler.base_image_seq_len", 256);
        d.max_seq = rad_meta_geti(meta, "scheduler.max_image_seq_len", 6400);
        d.shift_terminal = rad_meta_getf(meta, "scheduler.shift_terminal", 0.0);
        d.cfg_original = 1;
        d.default_cfg = m.turbo ? 0.0 : 4.5;
        d.default_negative = "";
        d.max_lanes = m.lanes;
        d.image_out = m.b_img;
        d.cond_rows = m.store_rows;
        d.rope_hw_scale = 1.0;
        d.rope_text_l = 0;
        d.cond_opts = RAD_GEN_COND_REBALANCE | RAD_GEN_COND_REBALANCE_PRESET;
        RAD_ARCH_TRY(rad_declare_generator(b, &d));
    }
    rad_note(b, "rank %d/%d: Krea 2%s -- text encoder %d of %lld layers, text fusion %zu + %zu "
                "blocks, DiT %lld blocks (%lld/%lld heads a rank, ff %lld), %s linears",
             m.gt.rank, m.gt.world, m.turbo ? " Turbo" : "", kTaps[kNTaps - 1],
             (long long)m.gt.n_layer, m.lw.size(), m.rf.size(), (long long)gd.n_layer,
             (long long)gd.n_head, (long long)gd.n_head_kv, (long long)gd.n_ff,
             gd.w_bf16 ? "bf16" : m.i8 ? "rotated int8" : "block-fp8");
    return RAD_OK;
}

/* ================================================================== step */
/* The language model over the prompt (the system turn included, as the reference runs it), each
 * tapped layer's rows scattered to the taps at row 12 t + i. */
static void step_te(RadCtx* c, const RadBatch* batch, const Model& m) {
    const Geom& g = m.gt;
    const int64_t T = batch->n_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const int32_t opts = batch->gen_b ? batch->gen_b->cond_opts : 0;
    const int preset = (opts & RAD_GEN_COND_REBALANCE_PRESET) >> RAD_GEN_COND_REBALANCE_PRESET_SHIFT;
    const bool rebalance = (opts & RAD_GEN_COND_REBALANCE) && preset < kNPresets;
    RAD_ISSUE_N(c, m.op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok),
                brows(m.b_tx, T));
    const int64_t media_rows = m.vision.on ? batch->n_mm_rows : 0;
    if (media_rows > 0)
        RAD_ISSUE_N(c, m.op_vision_embed, media_rows, brows(m.vision.b_out, media_rows),
                    praw(batch->mm_rows, RAD_I32, media_rows), brows(m.b_tx, T));
    RAD_ISSUE_N(c, m.op_iota, T, brows(m.b_tidx, T));
    const RadOperand pos = praw2(batch->rope_pos, RAD_I32, 3, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    for (size_t layer = 0; layer < m.te.size(); ++layer) {
        const TeLayer& L = m.te[layer];
        RAD_ISSUE_N(c, L.op_n1, T, brows(m.b_tx, T), RAD_W(L.w_n1), brows(m.a_th.x, T));
        L.q_h.step(c, T);
        L.qkv.step(c, m.a_th, m.b_tkv, T);
        RAD_ISSUE_N(c, L.op_qkr, T, brows(m.b_tkv, T), pos, RAD_W(L.w_qn), RAD_W(L.w_kn));
        RAD_ISSUE_N(c, L.op_attn, T, bcol(m.b_tkv, 0, q_dim, T), bcol(m.b_tkv, q_dim, kv_dim, T),
                    bcol(m.b_tkv, q_dim + kv_dim, kv_dim, T), cu, brows(m.a_tattn.x, T));
        L.q_attn.step(c, T);
        L.o.step(c, m.a_tattn, m.b_th2, T);
        if (L.op_ar1) RAD_ISSUE_N(c, L.op_ar1, T * g.n_embd, brows(m.b_th2, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_add1, T, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        RAD_ISSUE_N(c, L.op_n2, T, brows(m.b_tx, T), RAD_W(L.w_n2), brows(m.a_th.x, T));
        L.q_h2.step(c, T);
        L.gate_up.step(c, m.a_th, m.b_tgu, T);
        L.gq.step(c, m.b_tgu, T);
        L.down.step(c, m.a_tffn, m.b_th2, T);
        if (L.op_ar2) RAD_ISSUE_N(c, L.op_ar2, T * g.n_embd, brows(m.b_th2, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_add2, T, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        if (media_rows > 0 && layer < (size_t)m.vision.n_ds)
            RAD_ISSUE_N(c, m.op_deepstack[layer], media_rows, brows(m.vision.b_ds[layer], media_rows),
                        praw(batch->mm_rows, RAD_I32, media_rows), brows(m.b_tx, T));
        if (L.tap >= 0 && rebalance)
            RAD_ISSUE_N(c, L.op_gain[preset], T, brows(m.b_tx, T), brows(m.b_th2, T));
        if (L.tap >= 0)
            RAD_ISSUE_N(c, rebalance ? L.op_tap_g : L.op_tap, T,
                        brows(rebalance ? m.b_th2 : m.b_tx, T), brows(m.b_tidx, T),
                        brow_slice(m.b_taps, L.tap, (int64_t)m.te_rows * kNTaps - L.tap,
                                   g.n_embd));
    }
}

/* One text-fusion block over `R` rows of its residual from row `r0`, bounded by `cu`. */
static void step_fuse(RadCtx* c, const Model& m, const FuseBlock& F, int64_t r0, int64_t R,
                      const RadOperand& cu) {
    const Geom& g = m.gf;
    const int64_t D = g.n_embd, qd = g.q_dim(), hd = g.head_dim;
    const RadOperand x = brow_slice(F.x, r0, R, D);
    RAD_ISSUE_N(c, F.op_n1, R, x, RAD_W(F.w_n1), brows(m.a_fh.x, R));
    F.q_h.step(c, R);
    F.qkv.step(c, m.a_fh, m.b_fqkv, R);
    F.gate.step(c, m.a_fh, m.b_fg, R);
    RAD_ISSUE_N(c, F.op_qkn, R, brows(m.b_fqkv, R), RAD_NONE, RAD_W(F.w_qn), RAD_W(F.w_kn));
    RAD_ISSUE_N(c, F.op_attn, R, bcol(m.b_fqkv, 0, qd, R), bcol(m.b_fqkv, qd, qd, R),
                bcol(m.b_fqkv, 2 * qd, qd, R), cu, brows(m.a_fattn.x, R));
    if (F.op_sig) {
        RAD_ISSUE_N(c, F.op_sig, R, brows(m.b_fg, R), brows(m.b_fg, R));
        RAD_ISSUE_N(c, F.op_gq, R, brows(m.a_fattn.x, R), brows(m.b_fg, R), brows(m.a_fattn.x, R));
        F.q_attn.step(c, R);
    } else if (m.fgate_i8) {
        RAD_ISSUE_N(c, F.op_gq, R, brows(m.b_fg, R), brows(m.a_fattn.x, R),
                    brows(m.a_fattn.cq(), R), brows(m.a_fattn.cs(), R), brows(m.a_fattn.x, R));
    } else {
        const int64_t rh = R * (qd / hd), nb = hd / RAD_SCALE_GROUP;
        RAD_ISSUE_N(c, F.op_gq, rh, RAD_P_T2(rad_buf_ptr(c, m.b_fg), RAD_BF16, rh, hd),
                    RAD_P_T2(rad_buf_ptr(c, m.a_fattn.x), RAD_BF16, rh, hd),
                    RAD_P_T2(rad_buf_ptr(c, m.a_fattn.cq()), RAD_F8E4M3, rh, hd),
                    RAD_P_T2(rad_buf_ptr(c, m.a_fattn.cs()), RAD_F32, rh, nb),
                    RAD_P_T2(rad_buf_ptr(c, m.a_fattn.x), RAD_BF16, rh, hd));
    }
    F.o.step(c, m.a_fattn, m.b_fo, R);
    if (F.op_ar1) RAD_ISSUE_N(c, F.op_ar1, R * D, brows(m.b_fo, R), RAD_NONE);
    RAD_ISSUE_N(c, F.op_add1, R, x, brows(m.b_fo, R), x);
    RAD_ISSUE_N(c, F.op_n2, R, x, RAD_W(F.w_n2), brows(m.a_fh.x, R));
    F.q_h2.step(c, R);
    F.gate_up.step(c, m.a_fh, m.b_fgu, R);
    F.gq.step(c, m.b_fgu, R);
    F.down.step(c, m.a_fffn, m.b_fo, R);
    if (F.op_ar2) RAD_ISSUE_N(c, F.op_ar2, R * D, brows(m.b_fo, R), RAD_NONE);
    RAD_ISSUE_N(c, F.op_add2, R, x, brows(m.b_fo, R), x);
}

/* The conditioning: the language model's taps, the text fusion over the tokens after the system
 * turn (the layer-wise blocks over each token's twelve taps, the mix, the refiner over the
 * tokens), txt_in, and the rows into every lane's store at this branch. */
static void step_cond(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t P = G->n_px ? G->seg_h[1] : G->rows, drop = batch->n_tok - P;
    if (P > m.cond_rows || drop < 0 || batch->n_tok > m.te_rows) {
        rad_step_fail(c, "krea2: the prompt exceeds the conditioning buffer");
        return;
    }
    const int64_t D = m.gf.n_embd, Dd = m.gd.n_embd, RL = P * kNTaps;
    if (m.vision.on && batch->enc_n_patch > 0) m.vision.step(c, batch);
    step_te(c, batch, m);
    RAD_ISSUE_N(c, m.op_fcu, P + 1, brows(m.b_fcu, P + 1));
    for (const FuseBlock& F : m.lw) step_fuse(c, m, F, drop * kNTaps, RL, brows(m.b_fcu, P + 1));
    RAD_ISSUE_N(c, m.op_mix, P, brow_slice(m.b_taps, drop * kNTaps, RL, D), RAD_W(m.w_mix),
                brows(m.b_ft, P));
    const RadOperand cu = praw(G->seg_cu, RAD_I32, 2);
    for (const FuseBlock& F : m.rf) step_fuse(c, m, F, 0, P, cu);
    RAD_ISSUE_N(c, m.op_txn, P, brows(m.b_ft, P), RAD_W(m.w_txn), brows(m.b_txa, P));
    RAD_ISSUE_N(c, m.op_tx1, P, brows(m.b_txa, P), RAD_W(m.w_tx1), RAD_W(m.b_tx1), RAD_NONE,
                brows(m.b_txb, P));
    RAD_ISSUE_N(c, m.op_txg, P, brows(m.b_txb, P), brows(m.b_txb, P));
    RAD_ISSUE_N(c, m.op_tx2, P, brows(m.b_txb, P), RAD_W(m.w_tx2), RAD_W(m.b_tx2), RAD_NONE,
                brows(m.b_txt, P));
    const int64_t base = (int64_t)G->branch * m.store_rows;
    const int64_t rows = 2 * m.lanes * m.store_rows;
    const RadOperand lanes = praw(G->lane, RAD_I32, G->n_lane);
    RAD_ISSUE_N(c, m.op_store, P, brows(m.b_txt, P), lanes,
                brow_slice(m.b_store, base, rows - base, Dd));
    int64_t row = P;
    for (int64_t i = 0; i < G->n_px; ++i) {
        const int64_t h = G->px_h[3 * i + 1], w = G->px_h[3 * i + 2];
        const int64_t H = h / 8, W = w / 8, Z = m.C / 4, n = H * W / 4;
        const auto* px = (const uint16_t*)G->px + 3LL * G->px_h[3 * i];
        RAD_ISSUE_N(c, m.op_enc_cu, 2, praw(G->px_cu + 2 * i + 1, RAD_I32, 1), brows(m.b_enc_cu, 2));
        m.enc.issue(c, h, w, brows(m.b_enc_cu, 2), px);
        RAD_ISSUE_N(c, m.op_enc_norm, H * W, bcol(m.enc_out.buf, 0, Z, H * W),
                    brows(m.b_enc_norm, H * W));
        RAD_ISSUE_N(c, m.op_enc_zero, n, brows(m.b_enc_pack, n));
        RAD_ISSUE_N(c, m.op_enc_pack, n,
                    RAD_P_T2(rad_buf_ptr(c, m.b_enc_norm), RAD_BF16, H, W * Z),
                    RAD_P_T2(rad_buf_ptr(c, m.b_enc_pack), RAD_BF16, H / 2, (W / 2) * m.C));
        RAD_ISSUE_N(c, m.op_enc_project, n, brows(m.b_enc_pack, n), RAD_W(m.w_imgin), RAD_W(m.b_imgin),
                    RAD_NONE, brows(m.b_enc_embed, n));
        RAD_ISSUE_N(c, m.op_enc_store, n, brows(m.b_enc_embed, n), lanes,
                    brow_slice(m.b_store, base + row, rows - base - row, Dd));
        row += n;
    }
}

/* One denoising step: the entries' modulation, the joint rows, the blocks, and the image rows'
 * output through the sampler. */
static void step_denoise(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const Geom& g = m.gd;
    const int64_t N = batch->n_tok, S = batch->n_seq, NI = G->n_img;
    const int64_t D = g.n_embd, qd = g.q_dim(), kvd = g.kv_dim(), hd = g.head_dim;
    const int64_t L = g.n_layer, W = L * 6 * D;
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const int64_t MS = m.max_refs ? 3 * S : S;
    const RadOperand mod_cu = m.max_refs ? praw(G->mode_cu, RAD_I32, MS + 1) : cu;
    const RadOperand mod_sigma = praw(m.max_refs ? G->mode_sigma : G->sigma, RAD_F32, MS);
    const RadOperand icu = praw(G->img_cu, RAD_I32, S + 1);
    const RadOperand lanes = praw(G->lane, RAD_I32, S);
    const int64_t LR = m.lanes * m.max_image_tokens;

    /* ---- modulation: the timestep MLP, the shared projection, each block's table, the final */
    RAD_ISSUE_N(c, m.op_temb, MS, mod_sigma, brows(m.b_tproj, MS));
    RAD_ISSUE_N(c, m.op_t1, MS, brows(m.b_tproj, MS), RAD_W(m.w_t1), RAD_W(m.b_t1), RAD_NONE,
                brows(m.b_te1, MS));
    RAD_ISSUE_N(c, m.op_tg, MS, brows(m.b_te1, MS), brows(m.b_te1g, MS));
    RAD_ISSUE_N(c, m.op_t2, MS, brows(m.b_te1g, MS), RAD_W(m.w_t2), RAD_W(m.b_t2), RAD_NONE,
                brows(m.b_temb, MS));
    RAD_ISSUE_N(c, m.op_tg2, MS, brows(m.b_temb, MS), brows(m.b_tembg, MS));
    RAD_ISSUE_N(c, m.op_mod, MS, brows(m.b_tembg, MS), RAD_W(m.w_mod), RAD_W(m.b_mod), RAD_NONE,
                brows(m.b_tmod, MS));
    if (m.max_refs) {
        for (int64_t e = 0; e < S; ++e)
            RAD_ISSUE_N(c, m.op_ftab, 1, brow_slice(m.b_temb, 3 * e + 2, 1, D), RAD_W(m.w_ftab),
                        brow_slice(m.b_fmod, e, 1, 2 * D));
    } else {
        RAD_ISSUE_N(c, m.op_ftab, S, brows(m.b_temb, S), RAD_W(m.w_ftab), brows(m.b_fmod, S));
    }

    /* ---- the rows: the images' latents (noise on a first step), img_in, joined with the text */
    RAD_ISSUE_N(c, m.op_randn, NI, praw(G->flags, RAD_I32, S), praw(G->seed, RAD_U32, 2 * S),
                lanes, icu, brows(m.b_state, LR));
    RAD_ISSUE_N(c, m.op_lgather, NI, brows(m.b_state, LR), lanes, icu, brows(m.b_lat, NI));
    RAD_ISSUE_N(c, m.op_imgin, NI, brows(m.b_lat, NI), RAD_W(m.w_imgin), RAD_W(m.b_imgin), RAD_NONE,
              brows(m.b_himg, NI));
    RAD_ISSUE_N(c, m.op_join, N, brows(m.b_store, 2 * m.lanes * m.store_rows),
                brows(m.b_himg, NI), praw(G->src, RAD_I32, N), brows(m.b_h, N));

    /* ---- the blocks; block l's modulation is columns [6 l D, 6 (l + 1) D) of `mods`:
     * (preshift, prescale), pregate, (postshift, postscale), postgate */
    const RadOperand pos = praw2(G->rope_f, RAD_F32, 4, N);
    auto mod = [&](int64_t l, int k, int width) { return bcol_at(m.b_mods, 0, W, (6 * l + k) * D,
                                                                 width * D, MS); };
    const bool q8 = (!g.w_bf16 && !m.i8) || m.norm_i8;
    for (int64_t l = 0; l < L; ++l) {
        const DitLayer& B = m.dit[(size_t)l];
        /* The block's modulation: the shared projection plus its own table, into its columns. */
        RAD_ISSUE_N(c, B.op_tab, MS, brows(m.b_tmod, MS), RAD_W(B.w_tab),
                    bcol_at(m.b_mods, 0, W, l * 6 * D, 6 * D, MS));
        if (q8)
            RAD_ISSUE_N(c, B.op_n1, N, l > 0 ? brows(m.b_ao, N) : RAD_NONE,
                        l > 0 ? mod(l - 1, 5, 1) : RAD_NONE, mod(l, 0, 2), RAD_W(B.w_n1), mod_cu,
                        brows(m.b_h, N), brows(m.a_hm.cq(), N), brows(m.a_hm.cs(), N),
                        RAD_NONE);
        else
            RAD_ISSUE_N(c, B.op_n1, N, l > 0 ? brows(m.b_ao, N) : RAD_NONE,
                        l > 0 ? mod(l - 1, 5, 1) : RAD_NONE, mod(l, 0, 2), RAD_W(B.w_n1), mod_cu,
                        brows(m.b_h, N), brows(m.a_hm.x, N));
        B.q_n1.step(c, N);
        B.qkv.step(c, m.a_hm, m.b_qkv, N);
        B.gate.step(c, m.a_hm, m.b_g, N);
        RAD_ISSUE_N(c, B.op_qkr, N, brows(m.b_qkv, N), pos, RAD_W(B.w_qn), RAD_W(B.w_kn));
        if (m.max_refs)
            RAD_ISSUE_N(c, B.op_attn, N, bcol(m.b_qkv, 0, qd, N), bcol(m.b_qkv, qd, kvd, N),
                        bcol(m.b_qkv, qd + kvd, kvd, N), cu, brows(m.a_attn.x, N),
                        praw(G->mode_cu, RAD_I32, 3 * S + 1), praw(G->ref_log_bias, RAD_F32, S));
        else
            RAD_ISSUE_N(c, B.op_attn, N, bcol(m.b_qkv, 0, qd, N), bcol(m.b_qkv, qd, kvd, N),
                        bcol(m.b_qkv, qd + kvd, kvd, N), cu, brows(m.a_attn.x, N));
        if (B.op_sig) {
            RAD_ISSUE_N(c, B.op_sig, N, brows(m.b_g, N), brows(m.b_g, N));
            RAD_ISSUE_N(c, B.op_gq, N, brows(m.a_attn.x, N), brows(m.b_g, N), brows(m.a_attn.x, N));
            B.q_attn.step(c, N);
        } else if (m.gate_i8) {
            RAD_ISSUE_N(c, B.op_gq, N, brows(m.b_g, N), brows(m.a_attn.x, N),
                        brows(m.a_attn.cq(), N), brows(m.a_attn.cs(), N), brows(m.a_attn.x, N));
        } else {
            const int64_t rh = N * (qd / hd), nb = hd / RAD_SCALE_GROUP;
            RAD_ISSUE_N(c, B.op_gq, rh, RAD_P_T2(rad_buf_ptr(c, m.b_g), RAD_BF16, rh, hd),
                        RAD_P_T2(rad_buf_ptr(c, m.a_attn.x), RAD_BF16, rh, hd),
                        RAD_P_T2(rad_buf_ptr(c, m.a_attn.cq()), RAD_F8E4M3, rh, hd),
                        RAD_P_T2(rad_buf_ptr(c, m.a_attn.cs()), RAD_F32, rh, nb),
                        RAD_P_T2(rad_buf_ptr(c, m.a_attn.x), RAD_BF16, rh, hd));
        }
        B.o.step(c, m.a_attn, m.b_ao, N);
        if (B.op_ar1) RAD_ISSUE_N(c, B.op_ar1, N * D, brows(m.b_ao, N), RAD_NONE);
        if (q8)
            RAD_ISSUE_N(c, B.op_n2, N, brows(m.b_ao, N), mod(l, 2, 1), mod(l, 3, 2), RAD_W(B.w_n2),
                        mod_cu, brows(m.b_h, N), brows(m.a_hm.cq(), N), brows(m.a_hm.cs(), N),
                        RAD_NONE);
        else
            RAD_ISSUE_N(c, B.op_n2, N, brows(m.b_ao, N), mod(l, 2, 1), mod(l, 3, 2), RAD_W(B.w_n2),
                        mod_cu, brows(m.b_h, N), brows(m.a_hm.x, N));
        B.q_n2.step(c, N);
        B.gate_up.step(c, m.a_hm, m.b_gu, N);
        B.gq.step(c, m.b_gu, N);
        B.down.step(c, m.a_ffn, m.b_ao, N);
        if (B.op_ar2) RAD_ISSUE_N(c, B.op_ar2, N * D, brows(m.b_ao, N), RAD_NONE);
    }
    /* ---- the last block's feed-forward gated in, the image rows out, the final layer */
    RAD_ISSUE_N(c, m.op_g2, N, brows(m.b_ao, N), mod(L - 1, 5, 1), mod_cu, brows(m.b_h, N));
    RAD_ISSUE_N(c, m.op_gimg, NI, brows(m.b_h, N), praw(G->img_rows, RAD_I32, NI),
                brows(m.b_hi, NI));
    RAD_ISSUE_N(c, m.op_fn, NI, RAD_NONE, RAD_NONE, brows(m.b_fmod, S), RAD_W(m.w_fn), icu,
                brows(m.b_hi, NI), brows(m.b_fz, NI));
    RAD_ISSUE_N(c, m.op_pout, NI, brows(m.b_fz, NI), RAD_W(m.w_pout), RAD_W(m.b_pout), RAD_NONE,
                brows(m.b_v, NI));
    RAD_ISSUE_N(c, m.op_euler, NI, brows(m.b_v, NI), praw(G->sigma, RAD_F32, S),
                praw(G->sigma_next, RAD_F32, S), praw(G->cfg, RAD_F32, S),
                praw(G->pair, RAD_I32, S), lanes, praw(G->flags, RAD_I32, S),
                praw(G->seed, RAD_U32, 2 * S), icu, RAD_NONE, RAD_NONE, brows(m.b_state, LR),
                brows(m.b_prev, LR));
}

/* One image's latent to RGB: denormalised, unpacked to the VAE's grid, decoded (in row bands
 * with more than one rank). */
static void step_decode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t H = G->dec_height / 16, W = G->dec_width / 16, n = H * W, C = m.C;
    const rad_buf src = G->dec_preview ? m.b_prev : m.b_state;
    RAD_ISSUE_N(c, m.op_dinv, n, brow_slice(src, (int64_t)G->dec_lane * m.max_image_tokens, n, C),
                brows(m.b_dl, n));
    RAD_ISSUE_N(c, m.op_dadd, n, brows(m.b_dl, n), brows(m.b_dl2, n));
    const int64_t zd = C / 4, H0 = 2 * H, W0 = 2 * W;
    RAD_ISSUE_N(c, m.op_dzero, H0 * W0, brows(m.b_decin, H0 * W0));
    RAD_ISSUE_N(c, m.op_unpack, H0 * W0,
                RAD_P_T2(rad_buf_ptr(c, m.b_dl2), RAD_BF16, H, W * C),
                RAD_P_T2(rad_buf_ptr(c, m.b_decin), RAD_BF16, H0, W0 * zd));
    RAD_ISSUE_N(c, m.op_cu4, 2, praw(batch->cu_seqlens + 1, RAD_I32, 1), brows(m.b_cu4, 2));
    const RadOperand cu = brows(m.b_cu4, 2);
    const int64_t npx = (int64_t)G->dec_height * G->dec_width;
    const int world = m.gt.world, rank = m.gt.rank;
    const int64_t band = (H0 + world - 1) / world, halo = m.dec.halo;
    if (!m.op_bands || band + 2 * halo >= H0) {
        m.dec.issue(c, H0, W0, cu, nullptr);
        RAD_ISSUE_N(c, m.op_rgb, npx, RAD_P_T2(rad_buf_ptr(c, m.dec_out.buf), RAD_BF16, npx, 3),
                    RAD_P_T2(rad_buf_ptr(c, m.b_img), RAD_U8, npx, 3));
        return;
    }
    m.dec.issue(c, H0, W0, cu, nullptr, 0, (size_t)m.dec.band_from);
    const int64_t a = rank * band;
    const int64_t lo = std::max<int64_t>(0, a - halo), hi = std::min<int64_t>(H0, a + band + halo);
    m.dec.issue(c, hi - lo, W0, cu, nullptr, (size_t)m.dec.band_from, (size_t)-1, lo);
    const int64_t bpx = band * 8 * W0 * 8;
    const char* out = (const char*)rad_buf_ptr(c, m.dec_out.buf) +
                      (a - lo) * 8 * W0 * 8 * 3 * (int64_t)sizeof(uint16_t);
    RAD_ISSUE_N(c, m.op_bands, bpx * 3, RAD_P_T2(out, RAD_BF16, bpx, 3),
                RAD_P_T2(rad_buf_ptr(c, m.b_rgbf), RAD_BF16, world * bpx, 3));
    RAD_ISSUE_N(c, m.op_rgb_bands, npx, RAD_P_T2(rad_buf_ptr(c, m.b_rgbf), RAD_BF16, npx, 3),
                RAD_P_T2(rad_buf_ptr(c, m.b_img), RAD_U8, npx, 3));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->gen_b) {
        rad_step_fail(c, "krea2 is an image generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_COND:    step_cond(c, batch, m); return;
    case RAD_GEN_PASS_DENOISE: step_denoise(c, batch, m); return;
    case RAD_GEN_PASS_DECODE:  step_decode(c, batch, m); return;
    default: rad_step_fail(c, "krea2: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows a pass carries: one image at the largest size with guidance, its text both ways. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    RAD_ARCH_TRY(check_reference_encoder(meta));
    const int64_t tokens = image_tokens(out->gen_max_pixels);
    if (tokens == 0) return RAD_E_INVAL;
    const bool vision = has_reference_encoder(meta) && VisionTower::present(meta);
    if (!vision) {
        out->gen_rows = 2 * (tokens + kCondRows);
        out->gen_patches = 0;
        return RAD_OK;
    }
    const int64_t refs = reference_count(out->gen_max_refs);
    const int64_t merge = rad_meta_geti(meta, "vision_config.spatial_merge_size", 0);
    if (refs > kMaxReferences || merge < 1 || merge > 16) return RAD_E_INVAL;
    out->gen_rows = 2 * (tokens + edit_cond_rows(refs, merge) + refs * tokens);
    const int64_t group = merge * merge;
    out->gen_patches = refs * ((kGroundingPatches + group - 1) / group * group);
    return RAD_OK;
}

}  /* namespace krea2 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(krea2)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(krea2, "krea2", "", "0.1.0",
                "Krea 2 / Krea 2 Turbo: Qwen3-VL-4B taps through a text-fusion transformer, a 12.9B "
                "single-stream diffusion transformer over the joint text and image, and Wan 2.1's "
                "VAE; bf16 or block-fp8 linears")
#endif
