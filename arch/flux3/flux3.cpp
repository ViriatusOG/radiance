/* flux3 -- FLUX.3 [klein] (the step-distilled release and its base): text and up to ten
 * reference pictures to an RGB image.
 *
 * ============================== THE MODEL ==============================
 *
 * Three networks behind one prompt (diffusers' FLUX.3 modular pipeline):
 *
 *   Conditioning network: Qwen3-VL-4B-Instruct's language model over the prompt in its chat
 *   template, right-padded to a multiple of 80 tokens. The padding takes part: each padding token
 *   sits at position 1 and attends to the prompt alone (`attn_cross`, the core's `valid_cu`). The
 *   generator reads eight hidden states per token (after layers 3, 7, ..., 31) side by side,
 *   20480 values, through `context_embedder`.
 *
 *   Denoiser: a 5.6B diffusion transformer over three modes, the text, the reference pictures'
 *   latents (VAE-encoded, the generator's own encoder; the conditioning network never sees them)
 *   and the image. Every block is parallel: `h += gate (attn(n) + mlp(n))`, `n` a LayerNorm with
 *   no affine times (1 + scale) plus shift. Five blocks per mode run each mode on its own, then
 *   28 joint blocks run all three together; each mode takes its own shift, scale and gate from
 *   its own projection of the timestep embedding (the text and pictures at time 0, the image at
 *   the step's), separately for the early and the joint blocks. The step-distilled release is a
 *   flow map: the embedding of the time the step goes to is added to every mode's. A four-axis
 *   rotary (t, h, w, l) over the joint sequence: text along l, the image at (0, h/2, w/2, 0) and
 *   reference k at l = -(1 + k).
 *
 *   Since every row changes with the step, nothing is cached as keys: the conditioning pass
 *   leaves each lane's text and reference rows (after their embedders) in its store, and every
 *   denoising step runs the three modes' early blocks over them and the image (mode-major rows)
 *   and the joint blocks over the entries ([text | pictures | image] each, the core's JOINT_STEP
 *   layout).
 *
 *   Decoder: FLUX.2's VAE, 32 channels at 8x with GroupNorm and a single-head mid-block
 *   attention; a latent token is 2 x 2 of its pixels (128 values), normalised by the VAE's
 *   BatchNorm running statistics.
 *
 * The distilled release samples 16 steps without guidance, the base 50 with guidance 3.5 and an
 * empty negative prompt; both a constant shift of 6.93.
 *
 * ============================== WHAT IS QUANTISED ==============================
 *
 * Every linear of the language model and of the DiT's blocks is a LinearFP8 (rad_fp8.h), bf16
 * throughout when the first joint block's projection is plain; the embedders, the timestep MLPs,
 * the modulations, the output layer and the VAE stay bf16.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * The language model and the DiT blocks split by heads and feed-forward width. A parallel
 * block's two branch outputs are added before their one all-reduce (the sum of the partial sums).
 * Everything else, the VAE included (its GroupNorm reads the whole picture), runs on every rank.
 */
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>
#include <arch/rad_vae2d.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef FLUX3_NS
#define FLUX3_NS flux3
#define FLUX3_ID "flux3"
#endif

namespace FLUX3_NS {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* The largest image: 2048 x 2048 pixels, 16384 latent tokens; 1024 x 1024 is the default.
 * --image-gen-max-pixels lowers it (Model::max_img), and the buffers sized for it with it. */
enum { kMaxImageTokens = 16384 };
/* Reference pictures: ten at most, each shrunk to 1024 x 1024 pixels of area when larger;
 * --image-gen-max-refs lowers the count (Model::max_refs). */
enum { kMaxRefs = 10, kRefArea = 1024 * 1024 };
/* The prompt with its padding, at most; the pipeline pads to a multiple of 80. */
enum { kMaxText = 1040, kPadMultiple = 80 };
/* A reference picture's latent tokens at most: its area over 16 x 16 and the slack of its sides'
 * rounding (the driver's bound). */
static inline int64_t ref_tokens_max() { return (int64_t)kRefArea / 256 * 11 / 10 + 64; }
/* The largest image's latent tokens under --image-gen-max-pixels `px` (0: the model's own). */
static inline int64_t image_tokens(int64_t px) {
    return px > 0 ? std::clamp<int64_t>(px / 256, 1, kMaxImageTokens) : (int64_t)kMaxImageTokens;
}
/* The longest conditioning with `refs` reference pictures: the text and every picture's tokens. */
static inline int64_t cond_rows(int64_t refs) { return kMaxText + refs * ref_tokens_max(); }
/* Images in flight, at most; --max-num-seqs lowers it (Model::lanes). Each lane keeps both
 * branches' conditioning (cond_rows() rows apiece). */
enum { kMaxLanes = 4 };
/* The language model's hidden states read: hidden_states[i], the output of layer i - 1. */
static const int kTaps[] = { 4, 8, 12, 16, 20, 24, 28, 32 };
enum { kNTaps = 8 };
/* Modes, in the order the joint layout lays an entry out. */
enum { kText = 0, kRefs = 1, kImage = 2 };

/* ================================================================== the model */
struct TeLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearFP8  qkv{}, o{}, gate_up{}, down{};
    QuantFP8   q_h{}, q_attn{}, q_h2{};
    GateQuantFP8 gq{};
    rad_op     op_n1 = 0, op_qkr = 0, op_attn = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_ar2 = 0, op_add2 = 0, op_tap = 0;
    int        tap = -1;
};

/* A parallel block: the modulated LayerNorm (the previous block's gated sum added first), then
 * attention and the SwiGLU from the same normed rows, their outputs added. */
struct Block {
    rad_weight w_qn = 0, w_kn = 0;
    LinearFP8  qkv{}, o{}, ff_in{}, ff_out{};
    GateQuantFP8 gq{};
    QuantFP8   q_attn{};
    rad_op     op_n = 0, op_qkr = 0, op_attn = 0, op_sum = 0, op_ar = 0;
};

enum : int {
    kGrpCtx    = 1000,
    kGrpVaeEnc = 1100,
    kGrpVaeDec = 1200,
    kGrpDit    = 2000,   /* model-level weights; the blocks kGrpDit + 1.. */
};

struct Model {
    Names nm{""};
    Geom  gt{}, gd{};
    bool  flow_map = false;
    /* The deployment's bounds (declare, from RadBuildCtx): the largest image in latent tokens, the
     * reference pictures a request and the images in flight. Every per-image, conditioning and
     * lane buffer is sized for them. */
    int64_t max_img = kMaxImageTokens, max_refs = kMaxRefs, lanes = kMaxLanes;
    std::string te_pre;

    /* ---- the language model and context_embedder */
    rad_weight w_tok = 0, w_ctx = 0;
    rad_op  op_embed = 0, op_iota = 0, op_ctx = 0, op_store_t = 0;
    rad_buf b_tx = 0, b_tkv = 0, b_th2 = 0, b_tgu = 0, b_tidx = 0, b_taps = 0, b_ctx = 0;
    ActFP8  a_th{}, a_tattn{}, a_tffn{};
    std::vector<TeLayer> te;

    /* ---- the reference pictures: the encoder, the mode, 2 x 2 tokens, BatchNorm, the embedder */
    Vae     enc{};
    VImg    enc_q{};
    rad_weight w_bnm = 0, w_bnv = 0, w_xc = 0;
    rad_op  op_enc_mode = 0, op_enc_zero = 0, op_enc_s2d = 0, op_enc_bn = 0, op_xc = 0,
            op_store_r = 0, op_lstore_lat = 0, op_lstore_mask = 0, op_ecu = 0;
    rad_buf b_emode = 0, b_epack = 0, b_elat = 0, b_eemb = 0, b_ecu = 0;
    double  bn_eps = 1e-4;

    /* ---- the timestep and every modulation */
    rad_weight w_t1 = 0, w_t2 = 0, w_tt1 = 0, w_tt2 = 0, w_fmod = 0;
    rad_weight w_early[3] = {}, w_single[3] = {};
    rad_op  op_tp = 0, op_tpt = 0, op_one = 0, op_zero = 0, op_iota3 = 0;
    rad_op  op_t1 = 0, op_ts = 0, op_t2 = 0, op_tt1 = 0, op_tts = 0, op_tt2 = 0;
    rad_op  op_addt = 0, op_actimg = 0, op_actzero = 0, op_fmod = 0;
    rad_op  op_early[3] = {}, op_single[3] = {}, op_ilv[3] = {};
    rad_buf b_tp = 0, b_e1 = 0, b_e1s = 0, b_temb = 0, b_tpt = 0, b_tt1 = 0, b_tt1s = 0;
    rad_buf b_ttgt = 0;
    rad_buf b_aimg = 0, b_azero = 0, b_idx3 = 0, b_fmod = 0;
    rad_buf b_early[3] = {}, b_smod[3] = {}, b_single = 0;

    /* ---- the DiT */
    int64_t C = 128;
    int64_t t_ch = 256;        /* the timestep sinusoid's width */
    rad_weight w_xi = 0;
    rad_op  op_randn = 0, op_lgather = 0, op_xi = 0, op_gtxt = 0, op_join = 0, op_g2 = 0, op_g3 = 0,
            op_gimg = 0, op_nout = 0, op_pout = 0, op_euler = 0;
    rad_buf b_lat = 0, b_xc = 0, b_h = 0, b_qkv = 0, b_o = 0, b_gu = 0, b_f = 0, b_sum = 0,
            b_hi = 0, b_fz = 0, b_v = 0;
    ActFP8  a_hm{}, a_attn{}, a_ffn{};
    std::vector<Block> mode[3], single;
    LinearFP8 qkv0{};
    rad_weight w_pout = 0;

    /* ---- per-image state and conditioning store (PERSIST) */
    rad_buf b_state = 0, b_prev = 0, b_inp = 0, b_mask = 0, b_store = 0;

    /* ---- the decoder */
    Vae     dec{};
    VImg    dec_out{};
    rad_op  op_dbn = 0, op_dzero = 0, op_unpack = 0, op_cu4 = 0, op_rgb = 0;
    rad_buf b_dl = 0, b_decin = 0, b_cu4 = 0, b_img = 0;
};

static Model g_model[MAX_RANKS];

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

static rad_weight wcopy(RadBuilder* b, const char* dn, const char* src, uint32_t dtype,
                        std::initializer_list<int64_t> shape, int access, int group) {
    if (map_copy(b, dn, src) < 0) return 0;
    return decl_w(b, dn, dtype, shape, access, RAD_SHARD_NONE, grp_layer(group));
}

/* A bf16 linear without a bias, read whole on every rank. */
static rad_op dense(RadBuilder* b, Names& nm, const char* dn, const char* src, int64_t N, int64_t K,
                    int64_t rows, rad_buf in, rad_buf out, rad_weight* w, int access, int group) {
    *w = wcopy(b, nm.f("%s.weight", dn), nm.ckpt("%s.weight", src), RAD_BF16, { N, K }, access,
               group);
    if (!*w) return 0;
    return rw(b, RAD_OP(b, "gemm_nt",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(*w)),
              { in }, { out });
}

static rad_op unary(RadBuilder* b, const char* op, int64_t rows, int64_t n, rad_buf in,
                    rad_buf out) {
    return rw(b, RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                          RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
              { in }, { out });
}

static rad_op add_op(RadBuilder* b, int64_t rows, int64_t n, rad_buf a, rad_buf c, rad_buf y) {
    return rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
              { a, c }, { y });
}

static rad_op all_reduce(RadBuilder* b, const Geom& g, rad_buf x, int64_t rows) {
    if (g.world <= 1) return 0;
    return rw(b, RAD_OP(b, "all_reduce",
                        RAD_PARAMS(RAD_INT("world_size", g.world),
                                   RAD_RANGE("numel", g.n_embd, rows * g.n_embd),
                                   RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                                   RAD_INT("min_bytes", g.wire_min_bytes)),
                        RAD_NOWEIGHTS),
              { x }, { x });
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    const char* arch = rad_meta_gets(meta, "pipeline._class_name", "");
    if (std::strncmp(arch, "Flux3", 5) != 0) {
        fprintf(stderr, "radiance: flux3: the container's pipeline is '%s', not FLUX.3's\n", arch);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "transformer.in_channels", 128) != 128 ||
        rad_meta_geti(meta, "vae.latent_channels", 32) != 32) {
        fprintf(stderr, "radiance: flux3: the DiT must read 128-value tokens of a 32-channel "
                        "VAE\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== declare */
/* The language model up to its last tap; a tapped layer's output goes to its 2560 columns of the
 * stacked row, which context_embedder reads. */
static int declare_te(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gt;
    Names& nm = m.nm;
    const int64_t T = kMaxText;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim(), E = g.n_embd;
    const char* pre = m.te_pre.c_str();

    RAD_ARCH_TRY(map_copy(b, nm.f("te.token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", pre)));
    m.w_tok = decl_w(b, nm.f("te.token_embd.weight"), RAD_BF16, { g.n_vocab_all, E },
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.b_tx   = decl_b(b, nm.f("te.x"), RAD_BF16, { T, E });
    m.a_th.x = decl_b(b, nm.f("te.h"), RAD_BF16, { T, E });
    RAD_ARCH_TRY(m.a_th.declare_qs(b, nm, g, "te.h", E, T));
    m.b_tkv  = decl_b(b, nm.f("te.qkv"), RAD_BF16, { T, q_dim + 2 * kv_dim });
    m.a_tattn.x = decl_b(b, nm.f("te.attn"), RAD_BF16, { T, q_dim });
    RAD_ARCH_TRY(m.a_tattn.declare_qs(b, nm, g, "te.attn", q_dim, T));
    m.b_th2  = decl_b(b, nm.f("te.h2"), RAD_BF16, { T, E });
    m.b_tgu  = decl_b(b, nm.f("te.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_tffn.x = decl_b(b, nm.f("te.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_tffn.declare_qs(b, nm, g, "te.ffn", g.n_ff, T));
    m.b_taps = decl_b(b, nm.f("te.taps"), RAD_BF16, { T, kNTaps * E });
    m.b_tidx = decl_b(b, nm.f("te.rows"), RAD_I32, { T });

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                         RAD_INT("n_vocab", g.n_vocab_all), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", 0)),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_tx });
    m.op_iota = rw(b, RAD_OP(b, "iota",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("start", 0),
                                        RAD_INT("step", 1)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_tidx });
    const std::vector<double> sec = meta_list(meta, "text_config.rope_scaling.mrope_section");
    char sections[48];
    if (sec.size() == 3)
        std::snprintf(sections, sizeof sections, "%d %d %d", (int)sec[0], (int)sec[1], (int)sec[2]);
    else
        std::snprintf(sections, sizeof sections, "24 20 20");
    const int n_run = kTaps[kNTaps - 1];
    if (g.n_layer < n_run) {
        fprintf(stderr, "radiance: flux3: the text encoder has %lld layers; its last tap is after "
                        "layer %d\n", (long long)g.n_layer, n_run);
        return RAD_E_INVAL;
    }
    m.te.resize((size_t)n_run);
    for (int l = 0; l < n_run; ++l) {
        TeLayer& L = m.te[(size_t)l];
        for (int t = 0; t < kNTaps; ++t) if (kTaps[t] == l + 1) L.tap = t;
        const char* lp = nm.f("%s.layers.%d", pre, l);
        L.w_n1 = wcopy(b, nm.f("te.blk.%d.attn_norm.weight", l),
                       nm.ckpt("%s.input_layernorm.weight", lp), RAD_F32, { E }, RAD_ACCESS_PER_TOKEN, l);
        L.w_n2 = wcopy(b, nm.f("te.blk.%d.ffn_norm.weight", l),
                       nm.ckpt("%s.post_attention_layernorm.weight", lp), RAD_F32, { E },
                       RAD_ACCESS_PER_TOKEN, l);
        L.w_qn = wcopy(b, nm.f("te.blk.%d.q_norm.weight", l), nm.ckpt("%s.self_attn.q_norm.weight", lp),
                       RAD_BF16, { g.head_dim }, RAD_ACCESS_PER_TOKEN, l);
        L.w_kn = wcopy(b, nm.f("te.blk.%d.k_norm.weight", l), nm.ckpt("%s.self_attn.k_norm.weight", lp),
                       RAD_BF16, { g.head_dim }, RAD_ACCESS_PER_TOKEN, l);
        if (!L.w_n1 || !L.w_n2 || !L.w_qn || !L.w_kn) return RAD_E_INVAL;
        L.op_n1 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n1)),
                     { m.b_tx }, { m.a_th.x });
        RAD_ARCH_TRY(L.q_h.declare(b, g, m.a_th, E));
        RAD_ARCH_TRY(L.qkv.declare(b, nm, g, nm.f("te.blk.%d.attn_qkv", l), q_dim + 2 * kv_dim, E,
                                   RAD_SHARD_ROW, grp_layer(l),
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
        /* The padding attends to the prompt only: its keys are the prompt's (valid_cu). */
        L.op_attn = rw(b, RAD_OP(b, "attn_cross",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                            RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)g.head_dim))),
                                 RAD_NOWEIGHTS),
                       { m.b_tkv }, { m.a_tattn.x });
        RAD_ARCH_TRY(L.q_attn.declare(b, g, m.a_tattn, q_dim));
        RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("te.blk.%d.attn_output", l), E, q_dim,
                                 RAD_SHARD_COL, grp_layer(l),
                                 { nm.ckpt("%s.self_attn.o_proj.weight", lp) }, m.a_tattn, m.b_th2,
                                 T));
        L.op_ar1 = all_reduce(b, g, m.b_th2, T);
        L.op_add1 = add_op(b, T, E, m.b_tx, m.b_th2, m.b_tx);
        L.op_n2 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n2)),
                     { m.b_tx }, { m.a_th.x });
        RAD_ARCH_TRY(L.q_h2.declare(b, g, m.a_th, E));
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("te.blk.%d.ffn_gate_up", l), 2 * g.n_ff, E,
                                       RAD_SHARD_ROW, grp_layer(l),
                                       { nm.ckpt("%s.mlp.gate_proj.weight", lp),
                                         nm.ckpt("%s.mlp.up_proj.weight", lp) },
                                       m.a_th, m.b_tgu, T, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(L.gq.declare(b, g, m.b_tgu, m.a_tffn, g.n_ff, "silu", T, true));
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("te.blk.%d.ffn_down", l), E, g.n_ff,
                                    RAD_SHARD_COL, grp_layer(l),
                                    { nm.ckpt("%s.mlp.down_proj.weight", lp) }, m.a_tffn, m.b_th2,
                                    T));
        L.op_ar2 = all_reduce(b, g, m.b_th2, T);
        L.op_add2 = add_op(b, T, E, m.b_tx, m.b_th2, m.b_tx);
        if (L.tap >= 0)
            L.op_tap = rw(b, RAD_OP(b, "gather_rows",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                          { m.b_tx, m.b_tidx, m.b_taps }, { m.b_taps });
        if (!L.op_n1 || !L.op_qkr || !L.op_attn || !L.op_add1 || !L.op_n2 || !L.op_add2 ||
            (L.tap >= 0 && !L.op_tap))
            return RAD_E_INVAL;
    }
    /* ---- context_embedder over the stacked taps, into every lane's store */
    const int64_t D = m.gd.n_embd;
    m.b_ctx = decl_b(b, nm.f("ctx.out"), RAD_BF16, { T, D });
    m.op_ctx = dense(b, nm, "ctx", "transformer.context_embedder", D, kNTaps * E, T, m.b_taps,
                     m.b_ctx, &m.w_ctx, RAD_ACCESS_PER_REQUEST, kGrpCtx);
    m.op_store_t = rw(b, RAD_OP(b, "lane_store",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                           RAD_INT("stride", 2 * cond_rows(m.max_refs)),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_ctx, m.b_store }, { m.b_store });
    return m.op_embed && m.op_iota && m.op_ctx && m.op_store_t ? RAD_OK : RAD_E_INVAL;
}

/* FLUX.2's VAE encoder for the reference pictures (and a repainted picture), its mode, the 2 x 2
 * tokens, the BatchNorm, and the reference embedder into the store. */
static int declare_encoder(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Vae& v = m.enc;
    Names& nm = m.nm;
    v.nm = &nm;
    v.b = b;
    v.scope = "venc";
    v.down = true;
    v.first_group = kGrpVaeEnc;
    v.base_px = ref_tokens_max() * 256 > m.max_img * 256
                    ? ref_tokens_max() * 256 : m.max_img * 256;
    /* The mid-block attends over the 8x grid: four of the pass's 16x tokens a 16x token. */
    m.b_ecu = decl_b(b, nm.f("venc.cu"), RAD_I32, { 2 });
    m.op_ecu = rw(b, RAD_OP(b, "cu_repeat", RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", 4)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_ecu });
    if (!m.op_ecu) return RAD_E_INVAL;
    std::vector<double> ch = meta_list(meta, "vae.block_out_channels");
    if (ch.empty()) ch = { 128, 256, 512, 512 };
    const int64_t nres = rad_meta_geti(meta, "vae.layers_per_block", 2);
    const int64_t groups = rad_meta_geti(meta, "vae.norm_num_groups", 32);
    const int64_t zc = rad_meta_geti(meta, "vae.latent_channels", 32);
    const double eps = 1e-6;
    const int64_t nb = (int64_t)ch.size();
    VImg x = v.conv(VImg{ 0, 3, 0 }, (int64_t)ch[0], 3, 1, 1, "same", "vae.encoder.conv_in");
    for (int64_t i = 0; i < nb; ++i) {
        for (int64_t r = 0; r < nres; ++r)
            x = v.resblock_gn(x, (int64_t)ch[(size_t)i],
                              nm.f("vae.encoder.down_blocks.%lld.resnets.%lld", (long long)i,
                                   (long long)r), groups, eps);
        if (i + 1 < nb)
            x = v.conv(x, x.ch, 3, 2, 1, "down",
                       nm.f("vae.encoder.down_blocks.%lld.downsamplers.0.conv", (long long)i));
    }
    x = v.resblock_gn(x, x.ch, "vae.encoder.mid_block.resnets.0", groups, eps);
    x = v.attention_gn(x, "vae.encoder.mid_block.attentions.0", groups, eps);
    x = v.resblock_gn(x, x.ch, "vae.encoder.mid_block.resnets.1", groups, eps);
    x = v.gnorm(x, true, "vae.encoder.conv_norm_out", groups, eps);
    x = v.conv(x, 2 * zc, 3, 1, 1, "same", "vae.encoder.conv_out");
    m.enc_q = v.conv(x, 2 * zc, 1, 1, 1, "same", "vae.quant_conv");
    if (!m.enc_q.buf) return RAD_E_INVAL;
    RAD_ARCH_TRY(v.attn_reads(m.b_ecu));
    /* The mode (the first zc channels) as a dense image, its 2 x 2 tokens (s2d_avg_add over one
     * value a group is a plain space-to-depth into c 4 + 2 i + j), then the BatchNorm. */
    const int64_t px8 = v.px_at(m.enc_q.lvl), tok = (px8 + 3) / 4;
    m.b_emode = decl_b(b, nm.f("venc.mode"), RAD_BF16, { px8, zc });
    m.b_epack = decl_b(b, nm.f("venc.pack"), RAD_BF16, { tok, 4 * zc });
    m.b_elat  = decl_b(b, nm.f("venc.lat"), RAD_BF16, { tok, 4 * zc });
    m.b_eemb  = decl_b(b, nm.f("venc.emb"), RAD_BF16, { tok, m.gd.n_embd });
    std::string ones, zeros;
    for (int64_t k = 0; k < zc; ++k) { ones += k ? " 1" : "1"; zeros += k ? " 0" : "0"; }
    static thread_local std::string s_ones, s_zeros;
    s_ones = ones;
    s_zeros = zeros;
    m.op_enc_mode = rw(b, RAD_OP(b, "chan_affine",
                                 RAD_PARAMS(RAD_RANGE("M", 1, px8), RAD_INT("n", zc),
                                            RAD_STR("mul", s_ones.c_str()),
                                            RAD_STR("add", s_zeros.c_str()), RAD_STR("mode", "fma"),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.enc_q.buf }, { m.b_emode });
    m.op_enc_zero = rw(b, RAD_OP(b, "fill",
                                 RAD_PARAMS(RAD_RANGE("M", 1, tok), RAD_INT("n", 4 * zc),
                                            RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                                 RAD_NOWEIGHTS),
                       {}, { m.b_epack });
    m.op_enc_s2d = rw(b, RAD_OP(b, "s2d_avg_add",
                                RAD_PARAMS(RAD_RANGE("M", 1, tok), RAD_INT("cin", zc),
                                           RAD_INT("cout", 4 * zc), RAD_INT("ft", 1),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_emode, m.b_epack }, { m.b_epack });
    m.w_bnm = wcopy(b, nm.f("vae.bn.mean"), nm.ckpt("vae.bn.running_mean"), RAD_BF16, { 4 * zc },
                    RAD_ACCESS_PER_REQUEST, kGrpVaeEnc);
    m.w_bnv = wcopy(b, nm.f("vae.bn.var"), nm.ckpt("vae.bn.running_var"), RAD_BF16, { 4 * zc },
                    RAD_ACCESS_PER_REQUEST, kGrpVaeEnc);
    m.bn_eps = rad_meta_getf(meta, "vae.batch_norm_eps", 1e-4);
    m.op_enc_bn = rw(b, RAD_OP(b, "chan_stats",
                               RAD_PARAMS(RAD_RANGE("M", 1, tok), RAD_INT("n", 4 * zc),
                                          RAD_STR("mode", "norm"), RAD_INT("var", 1),
                                          RAD_F64("eps", m.bn_eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(m.w_bnm, m.w_bnv)),
                     { m.b_epack }, { m.b_elat });
    /* A reference's tokens through its embedder into the store after the text; a repainted
     * picture's into its lanes with the mask. */
    m.op_xc = dense(b, nm, "dit.x_cond", "transformer.x_embedders.image_cond", m.gd.n_embd, 4 * zc,
                    tok, m.b_elat, m.b_eemb, &m.w_xc, RAD_ACCESS_PER_REQUEST, kGrpDit);
    m.op_store_r = rw(b, RAD_OP(b, "lane_store",
                                RAD_PARAMS(RAD_RANGE("M", 1, tok), RAD_INT("n", m.gd.n_embd),
                                           RAD_INT("stride", 2 * cond_rows(m.max_refs)),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_eemb, m.b_store }, { m.b_store });
    m.op_lstore_lat = rw(b, RAD_OP(b, "lane_store",
                                   RAD_PARAMS(RAD_RANGE("M", 1, m.max_img),
                                              RAD_INT("n", 4 * zc),
                                              RAD_INT("stride", m.max_img),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS),
                         { m.b_elat, m.b_inp }, { m.b_inp });
    m.op_lstore_mask = rw(b, RAD_OP(b, "lane_store",
                                    RAD_PARAMS(RAD_RANGE("M", 1, m.max_img),
                                               RAD_INT("n", 1),
                                               RAD_INT("stride", m.max_img),
                                               RAD_STR("dtype", "f32")),
                                    RAD_NOWEIGHTS),
                          { m.b_mask }, { m.b_mask });
    return m.op_enc_mode && m.op_enc_zero && m.op_enc_s2d && m.w_bnm && m.w_bnv && m.op_enc_bn &&
                   m.op_xc && m.op_store_r && m.op_lstore_lat && m.op_lstore_mask
               ? RAD_OK : RAD_E_INVAL;
}

/* One parallel block's weights and ops over at most `rows` rows. `gated`: the norm adds the
 * previous block's gated sum first. */
static int declare_block(RadBuilder* b, Model& m, Block& B, const char* pre, const char* dn,
                         int grp, int64_t rows, bool gated, const char* sections, rad_buf mod) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const int64_t D = g.n_embd, qd = g.q_dim(), hd = g.head_dim;
    const double eps = g.eps;
    B.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), nm.ckpt("%s.attn.norm_q.weight", pre), RAD_F32,
                   { hd }, RAD_ACCESS_PER_TOKEN, grp);
    B.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), nm.ckpt("%s.attn.norm_k.weight", pre), RAD_F32,
                   { hd }, RAD_ACCESS_PER_TOKEN, grp);
    if (!B.w_qn || !B.w_kn) return RAD_E_INVAL;
    const bool q = !g.w_bf16;
    auto norm = [&](bool gt) {
        rad_op op = q ? RAD_OP(b, "rms_mod_quant_fp8",
                               RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", D),
                                          RAD_F64("eps", eps), RAD_INT("group", RAD_FP8_BLOCK),
                                          RAD_INT("gated", gt ? 1 : 0), RAD_STR("dtype", "bf16"),
                                          RAD_STR("norm", "layer")),
                               RAD_NOWEIGHTS)
                      : RAD_OP(b, "rms_mod",
                               RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", D),
                                          RAD_F64("eps", eps), RAD_INT("gated", gt ? 1 : 0),
                                          RAD_STR("dtype", "bf16"), RAD_STR("norm", "layer")),
                               RAD_NOWEIGHTS);
        if (!op) return op;
        /* The residual, the modulation table and, gated, the branch sum. */
        rad_buf rd[4] = { m.b_xc, m.b_h, mod, m.b_sum };
        rad_buf wq[3] = { m.a_hm.q, m.a_hm.s, m.b_h };
        rad_buf wb[2] = { m.a_hm.x, m.b_h };
        rad_op_reads(b, op, rd, gt ? 4 : 3);
        if (q) rad_op_writes(b, op, wq, 3);
        else   rad_op_writes(b, op, wb, 2);
        return op;
    };
    B.op_n = norm(gated);
    RAD_ARCH_TRY(B.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 3 * qd, D, RAD_SHARD_ROW,
                               grp_layer(grp),
                               { nm.ckpt("%s.attn.to_q.weight", pre),
                                 nm.ckpt("%s.attn.to_k.weight", pre),
                                 nm.ckpt("%s.attn.to_v.weight", pre) },
                               m.a_hm, m.b_qkv, rows, { qd, qd, qd }));
    B.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                       RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head),
                                       RAD_F64("theta", g.theta), RAD_F64("eps", eps),
                                       RAD_STR("mode", "gptj_blocks"), RAD_INT("rotary_dim", hd),
                                       RAD_STR("sections", sections)),
                            RAD_WEIGHTS(B.w_qn, B.w_kn)),
                  { m.b_qkv }, { m.b_qkv });
    B.op_attn = rw(b, RAD_OP(b, "attn_dense",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                        RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                        RAD_STR("dtype", "bf16"),
                                        RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                             RAD_NOWEIGHTS),
                   { m.b_qkv }, { m.a_attn.x });
    RAD_ARCH_TRY(B.q_attn.declare(b, g, m.a_attn, qd));
    RAD_ARCH_TRY(B.o.declare(b, nm, g, nm.f("%s.attn_output", dn), D, qd, RAD_SHARD_COL,
                             grp_layer(grp), { nm.ckpt("%s.attn.to_out.0.weight", pre) }, m.a_attn,
                             m.b_o, rows));
    RAD_ARCH_TRY(B.ff_in.declare(b, nm, g, nm.f("%s.ffn_in", dn), 2 * g.n_ff, D, RAD_SHARD_ROW,
                                 grp_layer(grp), { nm.ckpt("%s.ff.linear_in.weight", pre) }, m.a_hm,
                                 m.b_gu, rows, { g.n_ff, g.n_ff }));
    RAD_ARCH_TRY(B.gq.declare(b, g, m.b_gu, m.a_ffn, g.n_ff, "silu", rows, true));
    RAD_ARCH_TRY(B.ff_out.declare(b, nm, g, nm.f("%s.ffn_out", dn), D, g.n_ff, RAD_SHARD_COL,
                                  grp_layer(grp), { nm.ckpt("%s.ff.linear_out.weight", pre) },
                                  m.a_ffn, m.b_f, rows));
    B.op_sum = add_op(b, rows, D, m.b_o, m.b_f, m.b_sum);
    B.op_ar = all_reduce(b, g, m.b_sum, rows);
    return B.op_n && B.op_qkr && B.op_attn && B.op_sum ? RAD_OK : RAD_E_INVAL;
}

static int declare_dit(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, S = g.max_seqs, D = g.n_embd, C = m.C;
    const int gx = kGrpDit;
    const int64_t NI = std::min<int64_t>(T, 2 * m.max_img);
    const int64_t te = rad_meta_geti(meta, "transformer.timestep_channels", 256);
    m.t_ch = te;

    /* ---- timestep embeddings, the MLP over 2 S rows at once: rows [0, S) the image's time, rows
     * [S, 2 S) time 0's, whose sinusoid is [cos 0 | sin 0], ones then zeros. With a flow map the
     * target time's embedding is added to both halves. silu of each half feeds the modulations.
     * Every op is issued once a step: an op issued again over other buffers would outrun the
     * arena's plan, which orders buffers by declaration. */
    m.b_tp   = decl_b(b, nm.f("dit.tp"), RAD_BF16, { 2 * S, te });
    m.b_e1   = decl_b(b, nm.f("dit.e1"), RAD_BF16, { 2 * S, D });
    m.b_e1s  = decl_b(b, nm.f("dit.e1s"), RAD_BF16, { 2 * S, D });
    m.b_temb = decl_b(b, nm.f("dit.temb"), RAD_BF16, { 2 * S, D });
    m.b_aimg = decl_b(b, nm.f("dit.a_img"), RAD_BF16, { S, D });
    m.b_azero = decl_b(b, nm.f("dit.a_zero"), RAD_BF16, { S, D });
    auto tstep = [&](rad_buf out) {
        return rw(b, RAD_OP(b, "timestep_embed",
                            RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", te),
                                       RAD_F64("scale", 1000.0), RAD_F64("max_period", 10000.0),
                                       RAD_INT("round", 3), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  {}, { out });
    };
    m.op_tp = tstep(m.b_tp);
    auto fill = [&](double v) {
        return rw(b, RAD_OP(b, "fill",
                            RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", te / 2),
                                       RAD_STR("dtype", "bf16"), RAD_F64("value", v)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_tp });
    };
    m.op_one = fill(1.0);
    m.op_zero = fill(0.0);
    m.op_t1 = dense(b, nm, "dit.t1", "transformer.timestep_embedder.linear_1", D, te, 2 * S,
                    m.b_tp, m.b_e1, &m.w_t1, RAD_ACCESS_PER_TOKEN, gx);
    m.op_ts = unary(b, "silu", 2 * S, D, m.b_e1, m.b_e1s);
    m.op_t2 = dense(b, nm, "dit.t2", "transformer.timestep_embedder.linear_2", D, D, 2 * S, m.b_e1s,
                    m.b_temb, &m.w_t2, RAD_ACCESS_PER_TOKEN, gx);
    if (m.flow_map) {
        m.b_tpt = decl_b(b, nm.f("dit.tp_tgt"), RAD_BF16, { S, te });
        m.b_tt1 = decl_b(b, nm.f("dit.tt1"), RAD_BF16, { S, D });
        m.b_tt1s = decl_b(b, nm.f("dit.tt1s"), RAD_BF16, { S, D });
        m.b_ttgt = decl_b(b, nm.f("dit.t_tgt"), RAD_BF16, { S, D });
        m.op_tpt = tstep(m.b_tpt);
        m.op_tt1 = dense(b, nm, "dit.tt1", "transformer.target_timestep_embedder.linear_1", D, te,
                         S, m.b_tpt, m.b_tt1, &m.w_tt1, RAD_ACCESS_PER_TOKEN, gx);
        m.op_tts = unary(b, "silu", S, D, m.b_tt1, m.b_tt1s);
        m.op_tt2 = dense(b, nm, "dit.tt2", "transformer.target_timestep_embedder.linear_2", D, D,
                         S, m.b_tt1s, m.b_ttgt, &m.w_tt2, RAD_ACCESS_PER_TOKEN, gx);
        /* Issued on each half of the same buffers. */
        m.op_addt = add_op(b, S, D, m.b_temb, m.b_ttgt, m.b_temb);
        if (!m.op_tpt || !m.op_tt1 || !m.op_tts || !m.op_tt2 || !m.op_addt) return RAD_E_INVAL;
    }
    m.op_actimg = unary(b, "silu", S, D, m.b_temb, m.b_aimg);
    m.op_actzero = unary(b, "silu", S, D, m.b_temb, m.b_azero);
    if (!m.op_tp || !m.op_one || !m.op_zero || !m.op_t1 || !m.op_ts || !m.op_t2 || !m.op_actimg ||
        !m.op_actzero)
        return RAD_E_INVAL;

    /* ---- every mode's modulations: the early ones by mode (each mode's blocks find an entry's
     * row by their own cu), the joint ones interleaved as [3 S, 3 D], entry e's mode k at row
     * 3 e + k (the joint blocks find a row's segment by mode_cu) */
    static const char* kModeName[3] = { "txt", "image_cond", "image" };
    m.b_idx3 = decl_b(b, nm.f("dit.idx3"), RAD_I32, { S });
    m.op_iota3 = rw(b, RAD_OP(b, "iota",
                              RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("start", 0), RAD_INT("step", 3)),
                              RAD_NOWEIGHTS),
                    {}, { m.b_idx3 });
    m.b_single = decl_b(b, nm.f("dit.mod_single"), RAD_BF16, { 3 * S, 3 * D });
    for (int k = 0; k < 3; ++k) {
        const rad_buf act = k == kImage ? m.b_aimg : m.b_azero;
        m.b_early[k] = decl_b(b, nm.f("dit.mod_early.%d", k), RAD_BF16, { S, 3 * D });
        m.b_smod[k]  = decl_b(b, nm.f("dit.mod_s.%d", k), RAD_BF16, { S, 3 * D });
        m.op_early[k] = dense(b, nm, nm.f("dit.early.%s", kModeName[k]),
                              nm.ckpt("transformer.early_stream_modulations.%s.linear", kModeName[k]),
                              3 * D, D, S, act, m.b_early[k], &m.w_early[k], RAD_ACCESS_PER_TOKEN, gx);
        m.op_single[k] = dense(b, nm, nm.f("dit.single.%s", kModeName[k]),
                               nm.ckpt("transformer.single_stream_modulations.%s.linear",
                                       kModeName[k]),
                               3 * D, D, S, act, m.b_smod[k], &m.w_single[k], RAD_ACCESS_PER_TOKEN,
                               gx);
        m.op_ilv[k] = rw(b, RAD_OP(b, "scatter_rows",
                                   RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", 3 * D),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS),
                         { m.b_smod[k], m.b_idx3, m.b_single }, { m.b_single });
        if (!m.op_early[k] || !m.op_single[k] || !m.op_ilv[k]) return RAD_E_INVAL;
    }
    /* The output layer's (scale, shift) from silu(image temb), reordered to (shift, scale). */
    {
        const char* src = nm.ckpt("transformer.norm_out.linear.weight");
        RAD_ARCH_TRY(map_rows(b, nm.f("dit.norm_out.weight"), { { src, (int)D, D }, { src, 0, D } }));
        m.w_fmod = decl_w(b, nm.f("dit.norm_out.weight"), RAD_BF16, { 2 * D, D },
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gx));
        m.b_fmod = decl_b(b, nm.f("dit.mod_out"), RAD_BF16, { S, 2 * D });
        m.op_fmod = rw(b, RAD_OP(b, "gemm_nt",
                                 RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("N", 2 * D),
                                            RAD_INT("K", D), RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(m.w_fmod)),
                       { m.b_aimg }, { m.b_fmod });
        if (!m.w_fmod || !m.op_fmod) return RAD_E_INVAL;
    }

    /* ---- the rows: the image's latents through its embedder, the store's text and pictures,
     * all mode-major in `xc` (text, pictures, image), then joined entry by entry into `h` */
    m.b_lat = decl_b(b, nm.f("dit.lat"), RAD_BF16, { NI, C });
    m.b_xc  = decl_b(b, nm.f("dit.xc"), RAD_BF16, { T, D });
    m.b_h   = decl_b(b, nm.f("dit.h"), RAD_BF16, { T, D });
    m.op_randn = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                         RAD_INT("stride", m.max_img),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_state }, { m.b_state });
    m.op_lgather = rw(b, RAD_OP(b, "lane_gather",
                                RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                           RAD_INT("stride", m.max_img),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_state }, { m.b_lat });
    m.op_xi = dense(b, nm, "dit.x_img", "transformer.x_embedders.image", D, C, NI, m.b_lat, m.b_xc,
                    &m.w_xi, RAD_ACCESS_PER_TOKEN, gx);
    m.op_gtxt = rw(b, RAD_OP(b, "gather_rows",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_store }, { m.b_xc });
    if (!m.op_randn || !m.op_lgather || !m.op_xi || !m.op_gtxt) return RAD_E_INVAL;

    /* ---- the blocks' buffers */
    const int64_t qd = g.q_dim();
    m.a_hm.x = decl_b(b, nm.f("dit.hm"), RAD_BF16, { T, D });
    RAD_ARCH_TRY(m.a_hm.declare_qs(b, nm, g, "dit.hm", D));
    m.b_qkv = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { T, 3 * qd });
    m.a_attn.x = decl_b(b, nm.f("dit.attn"), RAD_BF16, { T, qd });
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "dit.attn", qd));
    m.b_o   = decl_b(b, nm.f("dit.o"), RAD_BF16, { T, D });
    m.b_gu  = decl_b(b, nm.f("dit.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_ffn.x = decl_b(b, nm.f("dit.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "dit.ffn", g.n_ff));
    m.b_f   = decl_b(b, nm.f("dit.f"), RAD_BF16, { T, D });
    m.b_sum = decl_b(b, nm.f("dit.sum"), RAD_BF16, { T, D });

    const std::vector<double> ax = meta_list(meta, "transformer.axes_dims_rope");
    if (!ax.empty() && (ax.size() != 4 || (int64_t)(ax[0] + ax[1] + ax[2] + ax[3]) != g.head_dim)) {
        fprintf(stderr, "radiance: flux3: transformer.axes_dims_rope must be four widths summing "
                        "to the head\n");
        return RAD_E_INVAL;
    }
    const char* sections = ax.empty() ? "16 16 16 16"
                                      : nm.f("%d %d %d %d", (int)ax[0] / 2, (int)ax[1] / 2,
                                             (int)ax[2] / 2, (int)ax[3] / 2);

    /* The early blocks, five a mode, then the joint blocks; the ops are declared in issue
     * order: text, pictures, image, joint. */
    const int64_t nl = rad_meta_geti(meta, "transformer.num_layers", 5);
    const int64_t ns = rad_meta_geti(meta, "transformer.num_single_layers", 28);
    int grp = kGrpDit + 1;
    for (int k = 0; k < 3; ++k) {
        m.mode[k].resize((size_t)nl);
        for (int64_t i = 0; i < nl; ++i) {
            const char* pre = k == kText
                ? nm.ckpt("transformer.txt_mode_blocks.%d", (int)i)
                : nm.ckpt("transformer.mode_blocks.%s.%d", kModeName[k], (int)i);
            RAD_ARCH_TRY(declare_block(b, m, m.mode[k][(size_t)i], pre,
                                       nm.f("dit.mode.%d.%d", k, (int)i), grp++, T, i > 0,
                                       sections, m.b_early[k]));
        }
    }
    /* The early stacks' last sums gated in where each mode's rows lie in the mode-major order,
     * then the rows joined. */
    m.op_g2 = rw(b, RAD_OP(b, "gate_add",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                      RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                           RAD_NOWEIGHTS),
                 { m.b_sum, m.b_early[0], m.b_early[1], m.b_early[2], m.b_xc }, { m.b_xc });
    m.op_join = rw(b, RAD_OP(b, "scatter_rows",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_xc, m.b_h }, { m.b_h });
    m.single.resize((size_t)ns);
    for (int64_t i = 0; i < ns; ++i)
        RAD_ARCH_TRY(declare_block(b, m, m.single[(size_t)i],
                                   nm.ckpt("transformer.single_transformer_blocks.%d", (int)i),
                                   nm.f("dit.single.%d", (int)i), grp++, T, i > 0, sections,
                                   m.b_single));
    /* The joint stack's last sum gated in. */
    m.op_g3 = rw(b, RAD_OP(b, "gate_add",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                      RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                           RAD_NOWEIGHTS),
                 { m.b_sum, m.b_single, m.b_h }, { m.b_h });
    if (!m.op_g3) return RAD_E_INVAL;

    /* ---- the output: the image rows, the final adaptive LayerNorm, proj_out, the step */
    m.b_hi = decl_b(b, nm.f("dit.h_img"), RAD_BF16, { NI, D });
    m.b_fz = decl_b(b, nm.f("dit.fz"), RAD_BF16, { NI, D });
    m.b_v  = decl_b(b, nm.f("dit.v"), RAD_BF16, { NI, C });
    m.op_gimg = rw(b, RAD_OP(b, "gather_rows",
                             RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", D),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_h }, { m.b_hi });
    m.op_nout = rw(b, RAD_OP(b, "rms_mod",
                             RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", D), RAD_F64("eps", g.eps),
                                        RAD_INT("gated", 0), RAD_STR("dtype", "bf16"),
                                        RAD_STR("norm", "layer")),
                             RAD_NOWEIGHTS),
                   { m.b_fmod, m.b_hi }, { m.b_hi, m.b_fz });
    m.op_pout = dense(b, nm, "dit.proj_out", "transformer.proj_out", C, D, NI, m.b_fz, m.b_v,
                      &m.w_pout, RAD_ACCESS_PER_TOKEN, gx);
    m.op_euler = rw(b, RAD_OP(b, "flow_euler",
                              RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                         RAD_INT("stride", m.max_img),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_v, m.b_inp, m.b_mask, m.b_state, m.b_prev }, { m.b_state, m.b_prev });
    return m.op_g2 && m.op_join && m.op_gimg && m.op_nout && m.op_pout && m.op_euler
               ? RAD_OK : RAD_E_INVAL;
}

/* FLUX.2's VAE decoder: the latent denormalised by the BatchNorm, unpacked to 32 channels, then
 * GroupNorm residual blocks, the mid-block attention and nearest-neighbour upsampling, run whole
 * on every rank. */
static int declare_decoder(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Vae& v = m.dec;
    Names& nm = m.nm;
    v.nm = &nm;
    v.b = b;
    v.scope = "vdec";
    v.down = false;
    v.first_group = kGrpVaeDec;
    v.base_px = m.max_img * 4;
    std::vector<double> ch = meta_list(meta, "vae.decoder_block_out_channels");
    if (ch.empty()) ch = meta_list(meta, "vae.block_out_channels");
    if (ch.empty()) ch = { 128, 256, 512, 512 };
    const int64_t nres = rad_meta_geti(meta, "vae.layers_per_block", 2);
    const int64_t groups = rad_meta_geti(meta, "vae.norm_num_groups", 32);
    const int64_t zc = rad_meta_geti(meta, "vae.latent_channels", 32);
    const double eps = 1e-6;
    const int64_t NT = m.max_img;
    m.b_dl = decl_b(b, nm.f("vdec.lat"), RAD_BF16, { NT, 4 * zc });
    m.b_decin = decl_b(b, nm.f("vdec.in"), RAD_BF16, { 4 * NT, zc });
    m.b_cu4 = decl_b(b, nm.f("vdec.cu"), RAD_I32, { 2 });
    m.op_dbn = rw(b, RAD_OP(b, "chan_stats",
                            RAD_PARAMS(RAD_RANGE("M", 1, NT), RAD_INT("n", 4 * zc),
                                       RAD_STR("mode", "denorm"), RAD_INT("var", 1),
                                       RAD_F64("eps", m.bn_eps), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.w_bnm, m.w_bnv)),
                  { m.b_state, m.b_prev }, { m.b_dl });
    m.op_dzero = rw(b, RAD_OP(b, "fill",
                              RAD_PARAMS(RAD_RANGE("M", 1, 4 * NT), RAD_INT("n", zc),
                                         RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                              RAD_NOWEIGHTS),
                    {}, { m.b_decin });
    m.op_unpack = rw(b, RAD_OP(b, "d2s_add",
                               RAD_PARAMS(RAD_RANGE("M", 1, 4 * NT), RAD_INT("cin", 4 * zc),
                                          RAD_INT("cout", zc), RAD_INT("ft", 1),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_dl, m.b_decin }, { m.b_decin });
    m.op_cu4 = rw(b, RAD_OP(b, "cu_repeat", RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", 4)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_cu4 });
    if (!m.op_dbn || !m.op_dzero || !m.op_unpack || !m.op_cu4) return RAD_E_INVAL;
    const std::vector<double> rch(ch.rbegin(), ch.rend());
    VImg x{ m.b_decin, zc, 0 };
    if (rad_meta_geti(meta, "vae.use_post_quant_conv", 1))
        x = v.conv(x, zc, 1, 1, 1, "same", "vae.post_quant_conv");
    x = v.conv(x, (int64_t)rch[0], 3, 1, 1, "same", "vae.decoder.conv_in");
    x = v.resblock_gn(x, x.ch, "vae.decoder.mid_block.resnets.0", groups, eps);
    x = v.attention_gn(x, "vae.decoder.mid_block.attentions.0", groups, eps);
    x = v.resblock_gn(x, x.ch, "vae.decoder.mid_block.resnets.1", groups, eps);
    for (size_t i = 0; i < rch.size(); ++i) {
        for (int64_t r = 0; r < nres + 1; ++r)
            x = v.resblock_gn(x, (int64_t)rch[i],
                              nm.f("vae.decoder.up_blocks.%d.resnets.%d", (int)i, (int)r), groups,
                              eps);
        if (i + 1 < rch.size())
            x = v.conv(x, x.ch, 3, 1, 2, "same", nm.f("vae.decoder.up_blocks.%d.upsamplers.0.conv",
                                                      (int)i));
    }
    x = v.gnorm(x, true, "vae.decoder.conv_norm_out", groups, eps);
    m.dec_out = v.conv(x, 3, 3, 1, 1, "same", "vae.decoder.conv_out");
    if (!m.dec_out.buf) return RAD_E_INVAL;
    RAD_ARCH_TRY(v.attn_reads(m.b_cu4));
    const int64_t px = m.max_img * 256;
    m.b_img = decl_b(b, nm.f("image_out"), RAD_U8, { px * 3 }, RAD_BUF_PERSIST);
    m.op_rgb = rw(b, RAD_OP(b, "to_rgba8",
                            RAD_PARAMS(RAD_RANGE("M", 1, px), RAD_INT("n", 3),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { m.dec_out.buf }, { m.b_img });
    return m.op_rgb ? RAD_OK : RAD_E_INVAL;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    RAD_ARCH_TRY(geom_from(m.gt, meta, ctx, "bf16", RAD_BF16));
    Names& nm = m.nm;
    m.te_pre = std::strstr(rad_meta_gets(meta, "architectures", ""), "ForConditionalGeneration")
                   ? "model.language_model" : "language_model";
    m.flow_map = rad_meta_geti(meta, "transformer.flow_map", 0) != 0;
    /* The deployment's bounds: --image-gen-max-pixels in latent tokens of 16 x 16 pixels,
     * --image-gen-max-refs, and --max-num-seqs's images in flight (0: the model's own). */
    m.max_img = image_tokens(ctx->gen_max_pixels);
    if (ctx->gen_max_refs > 0) m.max_refs = std::min<int64_t>(ctx->gen_max_refs, kMaxRefs);
    if (ctx->max_seqs > 0) m.lanes = std::min<int64_t>(ctx->max_seqs, kMaxLanes);

    Geom& gd = m.gd;
    gd = m.gt;
    gd.head_dim   = rad_meta_geti(meta, "transformer.attention_head_dim", 128);
    gd.n_head_all = rad_meta_geti(meta, "transformer.num_attention_heads", 24);
    gd.n_head_kv_all = gd.n_head_all;
    gd.n_embd     = gd.head_dim * gd.n_head_all;
    gd.n_ff_all   = (int64_t)(gd.n_embd * rad_meta_getf(meta, "transformer.mlp_ratio", 3.0));
    gd.eps        = (float)rad_meta_getf(meta, "transformer.eps", 1e-6);
    gd.theta      = (float)rad_meta_getf(meta, "transformer.rope_theta", 10000.0);
    gd.n_layer    = rad_meta_geti(meta, "transformer.num_single_layers", 28);
    RAD_ARCH_TRY(divide_or_fail(gd.n_head_all, gd.world, "transformer.num_attention_heads",
                                &gd.n_head));
    gd.n_head_kv = gd.n_head;
    gd.kv_off = 0;
    RAD_ARCH_TRY(divide_or_fail(gd.n_ff_all, gd.world, "the DiT's feed-forward width", &gd.n_ff));
    if (m.gt.n_embd * kNTaps != rad_meta_geti(meta, "transformer.joint_attention_dim", 20480)) {
        fprintf(stderr, "radiance: flux3: the DiT reads %lld-wide conditioning and %d taps of the "
                        "text encoder are %lld\n",
                (long long)rad_meta_geti(meta, "transformer.joint_attention_dim", 0), kNTaps,
                (long long)(m.gt.n_embd * kNTaps));
        return RAD_E_INVAL;
    }
    m.C = rad_meta_geti(meta, "transformer.in_channels", 128);
    {
        LinearFP8& probe = m.qkv0;
        RAD_ARCH_TRY(probe.probe(b, nm, gd, "dit.single.0.attn_qkv",
                                 { "transformer.single_transformer_blocks.0.attn.to_q.weight",
                                   "transformer.single_transformer_blocks.0.attn.to_k.weight",
                                   "transformer.single_transformer_blocks.0.attn.to_v.weight" }));
        RadEncoding e{};
        const bool plain = weight_enc(b, nm.f("dit.single.0.attn_qkv.weight"), &e) &&
                           rad_enc_is(&e, "plain");
        gd.w_bf16 = m.gt.w_bf16 = plain;
    }

    const int64_t lane_rows = m.lanes * m.max_img;
    m.b_state = decl_b(b, nm.f("lat_state"), RAD_BF16, { lane_rows, m.C }, RAD_BUF_PERSIST);
    m.b_prev  = decl_b(b, nm.f("lat_preview"), RAD_BF16, { lane_rows, m.C }, RAD_BUF_PERSIST);
    m.b_inp   = decl_b(b, nm.f("lat_repaint"), RAD_BF16, { lane_rows, m.C }, RAD_BUF_PERSIST);
    m.b_mask  = decl_b(b, nm.f("lat_mask"), RAD_F32, { lane_rows, 1 }, RAD_BUF_PERSIST);
    m.b_store = decl_b(b, nm.f("cond_store"), RAD_BF16,
                       { 2 * m.lanes * cond_rows(m.max_refs), gd.n_embd }, RAD_BUF_PERSIST);
    /* ---- adapter key names. LoRA files name the blocks by Black Forest Labs' modules, as
     * ComfyUI loads them (comfy/ldm/flux3): `single_blocks.N.q_proj` or a fused `qkv`,
     * `attn_out`, `mlp_in`, `mlp_out`, and the mode blocks under `content_mode_blocks`. */
    {
        const int64_t Q = gd.n_head_all * gd.head_dim;
        std::vector<LoraName> n;
        static const char* const kBlocks[][2] = {
            { "single_blocks.#.", "single_transformer_blocks.#." },
            { "txt_mode_blocks.#.", "txt_mode_blocks.#." },
            { "content_mode_blocks.image.#.", "mode_blocks.image.#." },
            { "content_mode_blocks.image_cond.#.", "mode_blocks.image_cond.#." },
        };
        for (const auto& bl : kBlocks) {
            const std::string f = bl[0], t = bl[1];
            n.push_back({ f + "q_proj", t + "attn.to_q" });
            n.push_back({ f + "k_proj", t + "attn.to_k" });
            n.push_back({ f + "v_proj", t + "attn.to_v" });
            n.push_back({ f + "qkv", t + "attn.to_q", 0, Q, 0 });
            n.push_back({ f + "qkv", t + "attn.to_k", Q, 2 * Q, 0 });
            n.push_back({ f + "qkv", t + "attn.to_v", 2 * Q, 3 * Q, 0 });
            n.push_back({ f + "attn_out", t + "attn.to_out.0" });
            n.push_back({ f + "mlp_in", t + "ff.linear_in" });
            n.push_back({ f + "mlp_out", t + "ff.linear_out" });
        }
        RAD_ARCH_TRY(lora_names(b, "transformer.", kLoraDiffusionSpellings, n));
    }
    RAD_ARCH_TRY(declare_te(b, meta, m));
    RAD_ARCH_TRY(declare_encoder(b, meta, m));
    RAD_ARCH_TRY(declare_dit(b, meta, m));
    RAD_ARCH_TRY(declare_decoder(b, meta, m));

    if (!ctx->shape_probe) {
        RadGeneratorDecl d{};
        d.name = m.flow_map ? "flux-3-image-klein" : "flux-3-image-klein-base";
        d.layout = RAD_GEN_LAYOUT_JOINT_STEP;
        d.tmpl_text = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
        d.tmpl_edit = d.tmpl_text;
        d.drop_prefix = "";
        d.latent_channels = m.C;
        d.latent_scale = 16;
        d.size_multiple = 16;
        d.pixel_channels = 3;
        d.slot_rows = 1;
        d.max_tokens = m.max_img;
        d.max_refs = m.max_refs;
        d.ref_area = kRefArea;
        d.ref_shrink_only = 1;
        /* 1024 x 1024, or the largest square under --image-gen-max-pixels. */
        const int64_t side = std::min<int64_t>(64, (int64_t)std::sqrt((double)m.max_img));
        d.default_width = side * 16;
        d.default_height = side * 16;
        d.default_steps = m.flow_map ? 16 : 50;
        d.shift = rad_meta_getf(meta, "scheduler.shift", 6.93);
        d.dyn_shift = (int32_t)rad_meta_geti(meta, "scheduler.use_dynamic_shifting", 0);
        d.shift_exp = std::strcmp(rad_meta_gets(meta, "scheduler.time_shift_type", "exponential"),
                                  "exponential") == 0;
        d.base_shift = rad_meta_getf(meta, "scheduler.base_shift", 0.5);
        d.max_shift = rad_meta_getf(meta, "scheduler.max_shift", 1.15);
        d.base_seq = rad_meta_geti(meta, "scheduler.base_image_seq_len", 256);
        d.max_seq = rad_meta_geti(meta, "scheduler.max_image_seq_len", 4096);
        d.shift_terminal = rad_meta_getf(meta, "scheduler.shift_terminal", 0.0);
        d.default_cfg = m.flow_map ? 1.0 : 3.5;
        d.default_negative = "";
        d.max_lanes = m.lanes;
        d.image_out = m.b_img;
        d.cond_rows = cond_rows(m.max_refs);
        d.rope_hw_scale = 0.5;
        d.rope_text_l = 1;
        d.pad_multiple = kPadMultiple;
        d.pad_token = (int32_t)rad_meta_geti(meta, "pad_token_id",
                                            rad_meta_geti(meta, "text_config.pad_token_id", 151643));
        RAD_ARCH_TRY(rad_declare_generator(b, &d));
    }
    rad_note(b, "rank %d/%d: FLUX.3 [klein]%s -- text encoder %d of %lld layers, DiT %zu x 3 early "
                "+ %zu joint blocks (%lld heads a rank, ff %lld), %s linears",
             m.gt.rank, m.gt.world, m.flow_map ? " (flow map)" : " base", kTaps[kNTaps - 1],
             (long long)m.gt.n_layer, m.mode[0].size(), m.single.size(), (long long)gd.n_head,
             (long long)gd.n_ff, gd.w_bf16 ? "bf16" : "block-fp8");
    return RAD_OK;
}

/* ================================================================== step */
static void step_te(RadCtx* c, const RadBatch* batch, const Model& m) {
    const Geom& g = m.gt;
    const RadGenBatch* G = batch->gen_b;
    const int64_t T = batch->n_tok, E = g.n_embd;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    RAD_ISSUE_N(c, m.op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok),
                brows(m.b_tx, T));
    RAD_ISSUE_N(c, m.op_iota, T, brows(m.b_tidx, T));
    const RadOperand pos = praw2(batch->rope_pos, RAD_I32, 3, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    const RadOperand cuk = praw(G->valid_cu, RAD_I32, 2);
    for (const TeLayer& L : m.te) {
        RAD_ISSUE_N(c, L.op_n1, T, brows(m.b_tx, T), RAD_W(L.w_n1), brows(m.a_th.x, T));
        L.q_h.step(c, T);
        L.qkv.step(c, m.a_th, m.b_tkv, T);
        RAD_ISSUE_N(c, L.op_qkr, T, brows(m.b_tkv, T), pos, RAD_W(L.w_qn), RAD_W(L.w_kn));
        RAD_ISSUE_N(c, L.op_attn, T, bcol(m.b_tkv, 0, q_dim, T), bcol(m.b_tkv, q_dim, kv_dim, T),
                    bcol(m.b_tkv, q_dim + kv_dim, kv_dim, T), cu, cuk, brows(m.a_tattn.x, T));
        L.q_attn.step(c, T);
        L.o.step(c, m.a_tattn, m.b_th2, T);
        if (L.op_ar1) RAD_ISSUE_N(c, L.op_ar1, T * E, brows(m.b_th2, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_add1, T, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        RAD_ISSUE_N(c, L.op_n2, T, brows(m.b_tx, T), RAD_W(L.w_n2), brows(m.a_th.x, T));
        L.q_h2.step(c, T);
        L.gate_up.step(c, m.a_th, m.b_tgu, T);
        L.gq.step(c, m.b_tgu, T);
        L.down.step(c, m.a_tffn, m.b_th2, T);
        if (L.op_ar2) RAD_ISSUE_N(c, L.op_ar2, T * E, brows(m.b_th2, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_add2, T, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        if (L.tap >= 0)
            RAD_ISSUE_N(c, L.op_tap, T, brows(m.b_tx, T), brows(m.b_tidx, T),
                        bcol(m.b_taps, L.tap * E, E, T));
    }
}

/* One picture through the encoder to its normalised tokens in `elat` (n tokens). */
static int64_t step_encode(RadCtx* c, const Model& m, const uint16_t* px, int64_t h, int64_t w,
                           const int32_t* px_cu) {
    RAD_ISSUE_N(c, m.op_ecu, 2, praw(px_cu + 1, RAD_I32, 1), brows(m.b_ecu, 2));
    m.enc.issue(c, h, w, brows(m.b_ecu, 2), px);
    const int64_t h8 = h / 8, w8 = w / 8, n8 = h8 * w8, n = n8 / 4, zc = m.C / 4;
    RAD_ISSUE_N(c, m.op_enc_mode, n8, bcol(m.enc_q.buf, 0, zc, n8), brows(m.b_emode, n8));
    RAD_ISSUE_N(c, m.op_enc_zero, n, brows(m.b_epack, n));
    RAD_ISSUE_N(c, m.op_enc_s2d, n, RAD_P_T2(rad_buf_ptr(c, m.b_emode), RAD_BF16, h8, w8 * zc),
                RAD_P_T2(rad_buf_ptr(c, m.b_epack), RAD_BF16, h8 / 2, (w8 / 2) * m.C));
    RAD_ISSUE_N(c, m.op_enc_bn, n, brows(m.b_epack, n), RAD_W(m.w_bnm), RAD_W(m.w_bnv),
              brows(m.b_elat, n));
    return n;
}

/* The conditioning: the text through the language model and context_embedder, each reference
 * picture through the encoder and its embedder, all into every lane's store at this branch; a
 * repainted picture's tokens and its mask into the lanes. */
static void step_cond(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t T = batch->n_tok, D = m.gd.n_embd;
    const int64_t base = (int64_t)G->branch * cond_rows(m.max_refs);
    const int64_t store_rows = 2 * m.lanes * cond_rows(m.max_refs);
    const RadOperand lanes = praw(G->lane, RAD_I32, G->n_lane);
    step_te(c, batch, m);
    RAD_ISSUE_N(c, m.op_ctx, T, brows(m.b_taps, T), RAD_W(m.w_ctx), brows(m.b_ctx, T));
    RAD_ISSUE_N(c, m.op_store_t, T, brows(m.b_ctx, T), lanes,
                brow_slice(m.b_store, base, store_rows - base, D));
    int64_t row = T;
    const int64_t n_ref = G->n_px - (G->inpaint ? 1 : 0);
    for (int64_t i = 0; i < G->n_px; ++i) {
        const int32_t off = G->px_h[3 * i], ph = G->px_h[3 * i + 1], pw = G->px_h[3 * i + 2];
        const uint16_t* px = (const uint16_t*)G->px + (int64_t)off * 3;
        const int64_t n = step_encode(c, m, px, ph, pw, G->px_cu + 2 * i);
        if (i < n_ref) {
            RAD_ISSUE_N(c, m.op_xc, n, brows(m.b_elat, n), RAD_W(m.w_xc), brows(m.b_eemb, n));
            RAD_ISSUE_N(c, m.op_store_r, n, brows(m.b_eemb, n), lanes,
                        brow_slice(m.b_store, base + row, store_rows - base - row, D));
            row += n;
        } else {
            const int64_t L = m.lanes * m.max_img;
            RAD_ISSUE_N(c, m.op_lstore_lat, n, brows(m.b_elat, n), lanes, brows(m.b_inp, L));
            RAD_ISSUE_N(c, m.op_lstore_mask, n, praw2(G->inp_mask, RAD_F32, n, 1), lanes,
                        brows(m.b_mask, L));
        }
    }
}

/* A stack of parallel blocks over `R` rows of `x` from row `r0`: `acu` bounds the attention,
 * `mcu` the modulation's segments, `s` the [shift | scale] rows and `gt` the gate rows. */
static void step_blocks(RadCtx* c, const Model& m, const std::vector<Block>& stack, rad_buf x,
                        int64_t r0, int64_t R, const RadOperand& acu, const RadOperand& mcu,
                        const RadOperand& s, const RadOperand& gt, const RadOperand& pos) {
    const Geom& g = m.gd;
    const int64_t D = g.n_embd, qd = g.q_dim(), nD = D / RAD_FP8_BLOCK;
    const bool q8 = !g.w_bf16;
    const RadOperand xv = brow_slice(x, r0, R, D);
    for (size_t i = 0; i < stack.size(); ++i) {
        const Block& B = stack[i];
        const bool gated = i > 0;
        if (q8)
            RAD_ISSUE_N(c, B.op_n, R, gated ? brows(m.b_sum, R) : RAD_NONE, gated ? gt : RAD_NONE,
                        s, RAD_NONE, mcu, xv, brows(m.a_hm.q, R), brows(m.a_hm.s, R), RAD_NONE);
        else
            RAD_ISSUE_N(c, B.op_n, R, gated ? brows(m.b_sum, R) : RAD_NONE, gated ? gt : RAD_NONE,
                        s, RAD_NONE, mcu, xv, brows(m.a_hm.x, R));
        B.qkv.step(c, m.a_hm, m.b_qkv, R);
        RAD_ISSUE_N(c, B.op_qkr, R, brows(m.b_qkv, R), pos, RAD_W(B.w_qn), RAD_W(B.w_kn));
        RAD_ISSUE_N(c, B.op_attn, R, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                    bcol(m.b_qkv, 2 * qd, qd, R), acu, brows(m.a_attn.x, R));
        B.q_attn.step(c, R);
        B.o.step(c, m.a_attn, m.b_o, R);
        B.ff_in.step(c, m.a_hm, m.b_gu, R);
        B.gq.step(c, m.b_gu, R);
        B.ff_out.step(c, m.a_ffn, m.b_f, R);
        RAD_ISSUE_N(c, B.op_sum, R, brows(m.b_o, R), brows(m.b_f, R), brows(m.b_sum, R));
        if (B.op_ar) RAD_ISSUE_N(c, B.op_ar, R * D, brows(m.b_sum, R), RAD_NONE);
    }
    (void)nD;
}

static void step_denoise(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const Geom& g = m.gd;
    const int64_t N = batch->n_tok, S = batch->n_seq, NI = G->n_img, C = m.C, D = g.n_embd;
    const int64_t nt = G->n_mode_h[kText], nr = G->n_mode_h[kRefs];
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const RadOperand icu = praw(G->img_cu, RAD_I32, S + 1);
    const RadOperand lanes = praw(G->lane, RAD_I32, S);
    const int64_t LR = m.lanes * m.max_img;

    /* ---- the timestep embeddings and every modulation */
    RAD_ISSUE_N(c, m.op_tp, S, praw(G->sigma, RAD_F32, S), brows(m.b_tp, S));
    RAD_ISSUE_N(c, m.op_one, S, bcol_at(m.b_tp, S, m.t_ch, 0, m.t_ch / 2, S));
    RAD_ISSUE_N(c, m.op_zero, S, bcol_at(m.b_tp, S, m.t_ch, m.t_ch / 2, m.t_ch / 2, S));
    RAD_ISSUE_N(c, m.op_t1, 2 * S, brows(m.b_tp, 2 * S), RAD_W(m.w_t1), brows(m.b_e1, 2 * S));
    RAD_ISSUE_N(c, m.op_ts, 2 * S, brows(m.b_e1, 2 * S), brows(m.b_e1s, 2 * S));
    RAD_ISSUE_N(c, m.op_t2, 2 * S, brows(m.b_e1s, 2 * S), RAD_W(m.w_t2), brows(m.b_temb, 2 * S));
    if (m.flow_map) {
        RAD_ISSUE_N(c, m.op_tpt, S, praw(G->sigma_next, RAD_F32, S), brows(m.b_tpt, S));
        RAD_ISSUE_N(c, m.op_tt1, S, brows(m.b_tpt, S), RAD_W(m.w_tt1), brows(m.b_tt1, S));
        RAD_ISSUE_N(c, m.op_tts, S, brows(m.b_tt1, S), brows(m.b_tt1s, S));
        RAD_ISSUE_N(c, m.op_tt2, S, brows(m.b_tt1s, S), RAD_W(m.w_tt2), brows(m.b_ttgt, S));
        for (int64_t h = 0; h < 2; ++h)
            RAD_ISSUE_N(c, m.op_addt, S, brow_slice(m.b_temb, h * S, S, D), brows(m.b_ttgt, S),
                        brow_slice(m.b_temb, h * S, S, D));
    }
    RAD_ISSUE_N(c, m.op_actimg, S, brows(m.b_temb, S), brows(m.b_aimg, S));
    RAD_ISSUE_N(c, m.op_actzero, S, brow_slice(m.b_temb, S, S, D), brows(m.b_azero, S));
    RAD_ISSUE_N(c, m.op_iota3, S, brows(m.b_idx3, S));
    for (int k = 0; k < 3; ++k) {
        const rad_buf act = k == kImage ? m.b_aimg : m.b_azero;
        RAD_ISSUE_N(c, m.op_early[k], S, brows(act, S), RAD_W(m.w_early[k]),
                    brows(m.b_early[k], S));
        RAD_ISSUE_N(c, m.op_single[k], S, brows(act, S), RAD_W(m.w_single[k]),
                    brows(m.b_smod[k], S));
        RAD_ISSUE_N(c, m.op_ilv[k], S, brows(m.b_smod[k], S), brows(m.b_idx3, S),
                    brow_slice(m.b_single, k, 3 * S - k, 3 * D));
    }
    RAD_ISSUE_N(c, m.op_fmod, S, brows(m.b_aimg, S), RAD_W(m.w_fmod), brows(m.b_fmod, S));

    /* ---- the rows, mode-major in `xc`: the store's text and pictures, the image's latents */
    RAD_ISSUE_N(c, m.op_randn, NI, praw(G->flags, RAD_I32, S), praw(G->seed, RAD_U32, 2 * S),
                lanes, icu, brows(m.b_state, LR));
    RAD_ISSUE_N(c, m.op_lgather, NI, brows(m.b_state, LR), lanes, icu, brows(m.b_lat, NI));
    RAD_ISSUE_N(c, m.op_xi, NI, brows(m.b_lat, NI), RAD_W(m.w_xi),
                brow_slice(m.b_xc, nt + nr, NI, D));
    if (nt + nr > 0)
        RAD_ISSUE_N(c, m.op_gtxt, nt + nr, brows(m.b_store, 2 * m.lanes * cond_rows(m.max_refs)),
                    praw(G->mode_src, RAD_I32, nt + nr), brows(m.b_xc, nt + nr));

    /* ---- each mode's early blocks over its own rows, the last sum gated in */
    const int64_t r0[3] = { 0, nt, nt + nr }, rn[3] = { nt, nr, NI };
    const float* mpos = G->mode_rope_f;
    for (int k = 0; k < 3; ++k) {
        if (rn[k] <= 0) continue;
        const RadOperand mc = praw(G->mode_ccu + k * (S + 1), RAD_I32, S + 1);
        const RadOperand s = bcol(m.b_early[k], 0, 2 * D, S), gt = bcol(m.b_early[k], 2 * D, D, S);
        /* This mode's positions, a dense [4, rn] block at 4 r0. */
        step_blocks(c, m, m.mode[k], m.b_xc, r0[k], rn[k], mc, mc, s, gt,
                    praw2(mpos + 4 * r0[k], RAD_F32, 4, rn[k]));
        RAD_ISSUE_N(c, m.op_g2, rn[k], brows(m.b_sum, rn[k]), gt, mc,
                    brow_slice(m.b_xc, r0[k], rn[k], D));
    }
    /* ---- joined entry by entry, the joint blocks, the last sum gated in */
    RAD_ISSUE_N(c, m.op_join, N, brows(m.b_xc, N), praw(G->mode_rows, RAD_I32, N),
                brows(m.b_h, N));
    const RadOperand mcu = praw(G->mode_cu, RAD_I32, 3 * S + 1);
    const RadOperand ss = bcol(m.b_single, 0, 2 * D, 3 * S), sg = bcol(m.b_single, 2 * D, D, 3 * S);
    step_blocks(c, m, m.single, m.b_h, 0, N, cu, mcu, ss, sg, praw2(G->rope_f, RAD_F32, 4, N));
    RAD_ISSUE_N(c, m.op_g3, N, brows(m.b_sum, N), sg, mcu, brows(m.b_h, N));

    /* ---- the image rows out */
    RAD_ISSUE_N(c, m.op_gimg, NI, brows(m.b_h, N), praw(G->img_rows, RAD_I32, NI),
                brows(m.b_hi, NI));
    RAD_ISSUE_N(c, m.op_nout, NI, RAD_NONE, RAD_NONE, brows(m.b_fmod, S), RAD_NONE, icu,
                brows(m.b_hi, NI), brows(m.b_fz, NI));
    RAD_ISSUE_N(c, m.op_pout, NI, brows(m.b_fz, NI), RAD_W(m.w_pout), brows(m.b_v, NI));
    RAD_ISSUE_N(c, m.op_euler, NI, brows(m.b_v, NI), praw(G->sigma, RAD_F32, S),
                praw(G->sigma_next, RAD_F32, S), praw(G->cfg, RAD_F32, S),
                praw(G->pair, RAD_I32, S), lanes, praw(G->flags, RAD_I32, S),
                praw(G->seed, RAD_U32, 2 * S), icu, brows(m.b_inp, LR), brows(m.b_mask, LR),
                brows(m.b_state, LR), brows(m.b_prev, LR));
    (void)C;
}

static void step_decode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t H = G->dec_height / 16, W = G->dec_width / 16, n = H * W, C = m.C;
    const rad_buf src = G->dec_preview ? m.b_prev : m.b_state;
    RAD_ISSUE_N(c, m.op_dbn, n, brow_slice(src, (int64_t)G->dec_lane * m.max_img, n, C),
              RAD_W(m.w_bnm), RAD_W(m.w_bnv), brows(m.b_dl, n));
    const int64_t zc = C / 4, H0 = 2 * H, W0 = 2 * W;
    RAD_ISSUE_N(c, m.op_dzero, H0 * W0, brows(m.b_decin, H0 * W0));
    RAD_ISSUE_N(c, m.op_unpack, H0 * W0, RAD_P_T2(rad_buf_ptr(c, m.b_dl), RAD_BF16, H, W * C),
                RAD_P_T2(rad_buf_ptr(c, m.b_decin), RAD_BF16, H0, W0 * zc));
    RAD_ISSUE_N(c, m.op_cu4, 2, praw(batch->cu_seqlens + 1, RAD_I32, 1), brows(m.b_cu4, 2));
    m.dec.issue(c, H0, W0, brows(m.b_cu4, 2), nullptr);
    const int64_t npx = (int64_t)G->dec_height * G->dec_width;
    RAD_ISSUE_N(c, m.op_rgb, npx, RAD_P_T2(rad_buf_ptr(c, m.dec_out.buf), RAD_BF16, npx, 3),
                RAD_P_T2(rad_buf_ptr(c, m.b_img), RAD_U8, npx, 3));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_rank(c)];
    if (!batch->gen || !batch->gen_b) {
        rad_step_fail(c, "flux3 is an image generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_COND:    step_cond(c, batch, m); return;
    case RAD_GEN_PASS_DENOISE: step_denoise(c, batch, m); return;
    case RAD_GEN_PASS_DECODE:  step_decode(c, batch, m); return;
    default: rad_step_fail(c, "flux3: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows a pass carries: an image at the largest size (under --image-gen-max-pixels) with one
 * reference picture and its longest text, both branches of guidance. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->gen_rows = 2 * (image_tokens(out->gen_max_pixels) + kMaxText + ref_tokens_max());
    out->gen_patches = 0;
    return RAD_OK;
}

}  /* namespace FLUX3_NS */

#ifndef RAD_ARCH_NO_EXPORTS
/* One more level of expansion so `#NS` sees the expanded namespace. */
#define FLUX3_PROBE_X(NS)        RAD_ARCH_PROBE(NS)
#define FLUX3_PLUGIN_X(NS, ...)  RAD_ARCH_PLUGIN(NS, __VA_ARGS__)
FLUX3_PROBE_X(FLUX3_NS)
FLUX3_PLUGIN_X(FLUX3_NS, FLUX3_ID, "", "0.1.0",
               "FLUX.3 [klein] and its base: Qwen3-VL-4B's stacked layers, a 5.6B three-mode "
               "diffusion transformer over the text, up to ten reference pictures and the image, "
               "and FLUX.2's VAE; bf16 or block-fp8 linears")
#endif
