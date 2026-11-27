/* qwenimage21 -- Qwen-Image 2.1 and 2.1-Turbo: text and reference pictures to an RGBA image.
 *
 * ============================== THE MODEL ==============================
 *
 * Three networks behind one prompt (diffusers' QwenImage21Pipeline):
 *
 *   Conditioning network: Qwen3-VL-8B-Instruct, unchanged. Its vision tower (with the three
 *   deepstack taps) reads the reference pictures and its 36-layer language model reads the
 *   prompt. The generator conditions on the last layer's output before the final norm, minus the
 *   template's fixed system turn.
 *
 *   Denoiser: a 7.1B single-stream diffusion transformer over one joint sequence: the
 *   conditioning's text rows (through `txt_in`), each reference picture's latent tokens in place of
 *   its vision slots (through `img_in`, four latent rows per slot), and the image being made. 32
 *   blocks of `h += tanh(g1) attn(LN(h)(1 + s1)); h += tanh(g2) mlp(LN(h)(1 + s2))`, the four
 *   modulation rows coming from one projection of the timestep embedding shared by every block.
 *   Attention is block-causal: text causal, a picture's rows bidirectional, every row sees all
 *   earlier ones. The conditioning is modulated at t = 0, so its keys and values do not depend on
 *   the step: they are computed once (a COND pass) and cached in this plugin's KV group, and each
 *   DENOISE pass runs only the image's rows against them.
 *
 *   Decoder: a 2-D VAE, 64 latent channels, 16x down, RGBA in and out. Its encoder turns the
 *   reference pictures (and a picture being repainted) into latents on the COND pass; its decoder
 *   turns a finished latent into pixels on a DECODE pass.
 *
 * The passes, their batches and the lanes an image's latent lives in are the core's generator
 * (abi/rad_builder.h, rad_declare_generator; core/gen). Turbo is the same graph with different
 * DiT weights and a fixed 8-step schedule, which the container's `pipeline.sample_sigmas` says.
 *
 * ============================== WHAT IS QUANTISED ==============================
 *
 * One plugin serves bf16, block-fp8 and rotated int8 (IQ8R) containers: every linear of the
 * language model and of the DiT's blocks is a LinearQ, which reads whatever the container holds
 * (rad_linear.h), and the model is bf16 throughout when the first DiT projection is plain. That
 * projection also says whether the linears are rotated int8, and then the activations they read
 * carry int8 codes in place of the E4M3 pair (declare_act). The small linears around them
 * (txt_in, img_in, the timestep MLP, the shared modulation, norm_out, proj_out), the vision tower
 * and the VAE stay bf16 in both, since they are read once per pass or less.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * The language model and the DiT blocks are split by heads and by feed-forward width, an
 * all-reduce after each output projection. Everything else is replicated and runs on every rank
 * (the embedding table, the vision tower, the small linears, the sampler and the VAE encoder),
 * since each runs once per pass and costs less than the collective a split would add. The decoder
 * is split by rows after its mid-block (see `Vae::band_from`); every rank ends with the same
 * pixels, and the core reads rank 0's.
 */
#include <arch/rad_arch.h>
#include <arch/rad_linear.h>
#include <arch/rad_block_vit.h>
#include <arch/rad_vae2d.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwenimage21 {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* The largest image: 2400 x 1792 pixels, the widest preset the model card lists, 16800 latent
 * tokens. 2048 x 2048 (16384) is the native size and the default. */
enum { kMaxImageTokens = 16800 };
/* Reference pictures: at most ten per request, each resized to 1024 x 1024 pixels of area. */
enum { kMaxRefs = 10, kRefArea = 1024 * 1024 };
/* Images a request may ask for, and so the lanes of latent state. */
enum { kMaxLanes = 16 };
/* The most latent tokens of a reference picture: its area over 16 x 16, plus slack for its sides'
 * rounding (the driver's bound, core/gen/driver.cpp). */
static inline int64_t ref_tokens_max() { return (int64_t)kRefArea / 256 * 11 / 10 + 64; }

/* Every output side is a multiple of 32, so attainable areas are multiples of four
 * latent tokens. Keep the model's original bounds when the operator supplies none. */
static int64_t image_tokens(int64_t pixels) {
    return pixels > 0 ? std::min<int64_t>(kMaxImageTokens, pixels / 1024 * 4)
                      : (int64_t)kMaxImageTokens;
}

static int64_t reference_count(int64_t refs) {
    return refs > 0 ? std::min<int64_t>(refs, kMaxRefs) : (int64_t)kMaxRefs;
}

static const char* kSysPrompt = "Comprehend and analyze the provided prompt.";


/* ================================================================== the model */
struct TeLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ  qkv{}, o{}, gate_up{}, down{};
    QuantAct   q_h{}, q_attn{}, q_h2{};
    GateQuant gq{};
    rad_op     op_n1 = 0, op_qkr = 0, op_attn = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_ar2 = 0, op_add2 = 0;
};

struct DitLayer {
    rad_weight w_qn = 0, w_kn = 0;
    LinearQ  qkv{}, o{}, gate_up{}, down{};
    QuantAct   q_h{}, q_attn{}, q_h2{};
    GateQuant gq{};
    rad_op     op_ln1 = 0, op_qkr = 0, op_kv = 0, op_attn_c = 0, op_attn_b = 0, op_ar1 = 0;
    rad_op     op_g1 = 0, op_ln2 = 0, op_ar2 = 0, op_g2 = 0;
    /* The two all-reduces as a split pass issues them, on the second lane (step_blocks). */
    rad_op     op_ar1h = 0, op_ar2h = 0;
    /* The fp8 DiT's two norms with the gated residual in front of each folded in
     * (ln_mod_quant_fp8): the first takes the previous block's MLP branch, the second this
     * block's attention branch. */
    rad_op     op_lnq1 = 0, op_lnq2 = 0;
    /* The fp8 DiT's denoising attention in its int8 Q.K form, and the quantiser that gives it the
     * keys. */
    rad_op     op_kq = 0, op_attn_q = 0;
};

/* Weight groups (movement units) by network. The text encoder's layers take 0.. and the DiT's
 * blocks kGrpDit + 1..; declare refuses a network deep enough to run into the next range. */
enum : int {
    kGrpTxtIn  = 1000,   /* txt_in */
    kGrpVaeEnc = 1100,
    kGrpVaeDec = 1200,
    kGrpDit    = 2000,   /* the DiT's model-level weights */
    kGrpVision = 3000,
};

struct Model {
    Names nm{""};
    Geom  gt{}, gd{};          /* the language model's and the DiT's geometry, per rank */
    int64_t max_image_tokens = kMaxImageTokens;
    int64_t max_refs = kMaxRefs;
    int64_t max_ref_rows = 0;  /* every reference picture's latent tokens together */
    bool  media = false;
    VisionTower vit{};
    rad_kvgroup kv = 0;

    /* ---- the language model */
    rad_weight w_tok = 0;
    rad_op  op_embed = 0, op_mm = 0, op_ds[VisionTower::MAX_DS] = {};
    rad_buf b_tx = 0, b_tkv = 0, b_th2 = 0, b_tgu = 0;
    ActQ  a_th{}, a_tattn{}, a_tffn{};
    std::vector<TeLayer> te;

    /* ---- txt_in: ZeroCenter RMSNorm (1 + w), a linear, gelu_tanh, a linear */
    rad_weight w_txn = 0, w_tx1 = 0, w_tx2 = 0;
    rad_op  op_txn = 0, op_tx1 = 0, op_txg = 0, op_tx2 = 0;
    rad_buf b_txa = 0, b_txb = 0, b_txt = 0;

    /* ---- the timestep and the shared modulation */
    rad_weight w_t1 = 0, w_t2 = 0, w_mod = 0, w_nout = 0;
    rad_op  op_temb = 0, op_t1 = 0, op_ts = 0, op_t2 = 0, op_ts2 = 0, op_mod = 0, op_nout = 0;
    rad_buf b_tproj = 0, b_te1 = 0, b_te1s = 0, b_temb = 0, b_tembs = 0, b_mod = 0, b_modo = 0;

    /* ---- the DiT's input */
    rad_weight w_imgin = 0;
    rad_op  op_gather = 0, op_imgin = 0, op_randn = 0, op_lgather = 0;
    rad_buf b_lat = 0;
    /* ---- the DiT */
    rad_buf b_h = 0, b_qkv = 0, b_ao = 0, b_gu = 0;
    rad_buf b_k8 = 0, b_ks = 0;   /* the keys at int8 codes and their per-head scales */
    ActQ  a_hm{}, a_attn{}, a_ffn{};
    std::vector<DitLayer> dit;
    LinearQ qkv0{};          /* layer 0's qkv, as the probe that chose bf16 or fp8 mapped it */
    /* The linears of both networks are rotated int8 (IQ8R, or the 128-group int8 before it), read
     * off that same probe: the activations then carry int8 code pairs with one scale per (token,
     * `g8` columns) in place of the E4M3 pair. */
    bool      i8 = false;
    int64_t   g8 = RAD_SCALE_GROUP;
    int64_t   C = 64;          /* latent channels */
    rad_weight w_pout = 0;
    rad_op  op_nout_ln = 0, op_pout = 0, op_euler = 0;
    rad_buf b_v = 0;

    /* ---- the per-image state (PERSIST) and what crosses passes */
    rad_buf b_state = 0, b_prev = 0, b_inp = 0, b_mask = 0, b_img = 0, b_reflat = 0;
    rad_op  op_lstore_lat = 0, op_lstore_mask = 0, op_enc_aff = 0, op_dec_aff = 0, op_rgba = 0;
    /* More than one rank: the decoder's bands gathered, and the bytes made from the gather. */
    rad_op  op_bands = 0, op_rgba_bands = 0;

    /* ---- the VAE */
    Vae     enc{}, dec{};
    VImg    enc_in{}, enc_q{}, dec_z{}, dec_out{};
    rad_buf b_encz = 0, b_decin = 0, b_rgbaf = 0;
    std::string lat_mean, lat_std;
};

static Model g_model[MAX_SLOTS];

/* A bf16 linear read whole on every rank: `gemm_nt`, no bias. */
static rad_op dense(RadBuilder* b, const char* dn, const char* src, int64_t N, int64_t K,
                    int64_t rows, rad_buf in, rad_buf out, rad_weight* w, int group) {
    *w = wcopy(b, dn, src, RAD_BF16, { N, K }, RAD_ACCESS_PER_TOKEN, group);
    if (!*w) return 0;
    return gemm_op(b, rows, N, K, *w, in, out);
}

/* `overlap`: issued on the second lane beside compute, which picks a narrow launch. */
static rad_op all_reduce(RadBuilder* b, const Geom& g, rad_buf x, int64_t rows,
                         bool overlap = false) {
    if (g.world <= 1) return 0;
    return decl_all_reduce_overlap(b, g, x, g.n_embd, rows, "bf16", overlap ? 1 : 0);
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    const char* arch = rad_meta_gets(meta, "pipeline._class_name", "");
    if (std::strcmp(arch, "QwenImage21Pipeline") != 0) {
        fprintf(stderr, "radiance: qwenimage21: the container's pipeline is '%s', not "
                        "QwenImage21Pipeline\n", arch);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "transformer.patch_size", 1) != 1 ||
        rad_meta_geti(meta, "transformer.causal_condition", 1) != 1 ||
        rad_meta_geti(meta, "vae.is_residual", 0) != 1 ||
        rad_meta_geti(meta, "vae.z_dim", 0) != 64) {
        fprintf(stderr, "radiance: qwenimage21: the DiT must be patch 1 with causal conditioning and "
                        "the VAE the residual 64-channel one\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== declare */
/* An activation's code pair, `x` already declared: E4M3, or int8 for a rotated int8 model
 * (ActQ::declare_pair). */
static int declare_act(RadBuilder* b, Names& nm, const Geom& g, const Model& m, ActQ& a,
                       const char* name, int64_t n, const char* fed_op) {
    return a.declare_pair(b, nm, g, name, n, 0, m.i8, m.g8, fed_op);
}

static int declare_te(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gt;
    Names& nm = m.nm;
    const int64_t T = g.max_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();

    /* The embedding is replicated: each rank looks up every token itself, so no collective sums
     * it, and the vision tower's rows overwrite it identically on every rank. */
    RAD_ARCH_TRY(map_copy(b, nm.f("te.token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("te.token_embd.weight"), RAD_BF16, { g.n_vocab_all, g.n_embd },
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.b_tx   = decl_b(b, nm.f("te.x"), RAD_BF16, { T, g.n_embd });
    m.a_th.x = decl_b(b, nm.f("te.h"), RAD_BF16, { T, g.n_embd });
    RAD_ARCH_TRY(declare_act(b, nm, g, m, m.a_th, "te.h", g.n_embd, "had_quant_act_i8g"));
    m.b_tkv  = decl_b(b, nm.f("te.qkv"), RAD_BF16, { T, q_dim + 2 * kv_dim });
    m.a_tattn.x = decl_b(b, nm.f("te.attn"), RAD_BF16, { T, q_dim });
    RAD_ARCH_TRY(declare_act(b, nm, g, m, m.a_tattn, "te.attn", q_dim, "had_quant_act_i8g"));
    m.b_th2  = decl_b(b, nm.f("te.h2"), RAD_BF16, { T, g.n_embd });
    m.b_tgu  = decl_b(b, nm.f("te.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_tffn.x = decl_b(b, nm.f("te.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(declare_act(b, nm, g, m, m.a_tffn, "te.ffn", g.n_ff, "gated_had_quant_i8g"));

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
                     { m.vit.b_out, m.b_tx }, { m.b_tx });
        for (int t = 0; t < m.vit.n_ds; ++t)
            m.op_ds[t] = rw(b, RAD_OP(b, "scatter_add_rows",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_NOWEIGHTS),
                            { m.vit.b_ds[t], m.b_tx }, { m.b_tx });
    }

    /* Qwen3-VL's text attention: plain RMSNorm gains (no `1 +`), q/k head norms, interleaved
     * M-RoPE over three position components. */
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
        RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.attn_norm.weight", li),
                              nm.ckpt("%s.%d.input_layernorm.weight", pre, li)));
        L.w_n1 = decl_w(b, nm.f("te.blk.%d.attn_norm.weight", li), RAD_F32, { g.n_embd },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
        RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.ffn_norm.weight", li),
                              nm.ckpt("%s.%d.post_attention_layernorm.weight", pre, li)));
        L.w_n2 = decl_w(b, nm.f("te.blk.%d.ffn_norm.weight", li), RAD_F32, { g.n_embd },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
        RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.q_norm.weight", li),
                              nm.ckpt("%s.%d.self_attn.q_norm.weight", pre, li)));
        L.w_qn = decl_w(b, nm.f("te.blk.%d.q_norm.weight", li), RAD_BF16, { g.head_dim },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
        RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.k_norm.weight", li),
                              nm.ckpt("%s.%d.self_attn.k_norm.weight", pre, li)));
        L.w_kn = decl_w(b, nm.f("te.blk.%d.k_norm.weight", li), RAD_BF16, { g.head_dim },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));

        L.op_n1 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n1)),
                     { m.b_tx }, { m.a_th.x });
        RAD_ARCH_TRY(L.q_h.declare(b, g, m.a_th, g.n_embd));
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
        RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("te.blk.%d.attn_output", li), g.n_embd, q_dim,
                                 RAD_SHARD_COL, grp_layer(li),
                                 { nm.ckpt("%s.%d.self_attn.o_proj.weight", pre, li) },
                                 m.a_tattn, m.b_th2));
        L.op_ar1 = all_reduce(b, g, m.b_th2, T);
        L.op_add1 = rw(b, RAD_OP(b, "add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.b_tx, m.b_th2 }, { m.b_tx });
        L.op_n2 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n2)),
                     { m.b_tx }, { m.a_th.x });
        RAD_ARCH_TRY(L.q_h2.declare(b, g, m.a_th, g.n_embd));
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("te.blk.%d.ffn_gate_up", li), 2 * g.n_ff,
                                       g.n_embd, RAD_SHARD_ROW, grp_layer(li),
                                       { nm.ckpt("%s.%d.mlp.gate_proj.weight", pre, li),
                                         nm.ckpt("%s.%d.mlp.up_proj.weight", pre, li) },
                                       m.a_th, m.b_tgu, 0, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(L.gq.declare(b, g, m.b_tgu, m.a_tffn, g.n_ff, "silu", 0, true));
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("te.blk.%d.ffn_down", li), g.n_embd, g.n_ff,
                                    RAD_SHARD_COL, grp_layer(li),
                                    { nm.ckpt("%s.%d.mlp.down_proj.weight", pre, li) },
                                    m.a_tffn, m.b_th2));
        L.op_ar2 = all_reduce(b, g, m.b_th2, T);
        L.op_add2 = rw(b, RAD_OP(b, "add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.b_tx, m.b_th2 }, { m.b_tx });
    }

    /* ---- txt_in, over every prompt row. Its output must survive across the conditioning's
     * segments, each of which runs the DiT's blocks over the same buffers, so it is PERSIST. */
    const int gti = kGrpTxtIn;
    const int64_t D = m.gd.n_embd;
    m.b_txa = decl_b(b, nm.f("txt.a"), RAD_BF16, { T, D });
    m.b_txb = decl_b(b, nm.f("txt.b"), RAD_BF16, { T, D });
    m.b_txt = decl_b(b, nm.f("txt.out"), RAD_BF16, { T, D }, RAD_BUF_PERSIST);
    RAD_ARCH_TRY(map_copy(b, nm.f("txt.norm.weight"), nm.ckpt("transformer.txt_in.text_norm.weight")));
    m.w_txn = decl_w(b, nm.f("txt.norm.weight"), RAD_F32, { g.n_embd }, RAD_ACCESS_PER_REQUEST,
                     RAD_SHARD_NONE, grp_layer(gti));
    /* ZeroCenterRMSNorm: the effective gain is 1 + w, in f32 -- rmsnorm's `wadd`. */
    m.op_txn = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", rad_meta_getf(meta, "transformer.eps", 1e-6)),
                                       RAD_STR("dtype", "bf16"), RAD_F64("wadd", 1.0)),
                            RAD_WEIGHTS(m.w_txn)),
                  { m.b_tx }, { m.b_txa });
    m.op_tx1 = dense(b, nm.f("txt.in.weight"), nm.ckpt("transformer.txt_in.in_layer.weight"),
                     D, g.n_embd, T, m.b_txa, m.b_txb, &m.w_tx1, gti);
    m.op_txg = unary(b, "gelu_tanh", T, D, m.b_txb, m.b_txa);
    m.op_tx2 = dense(b, nm.f("txt.out.weight"), nm.ckpt("transformer.txt_in.out_layer.weight"),
                     D, D, T, m.b_txa, m.b_txt, &m.w_tx2, gti);
    if (!m.op_tx1 || !m.op_txg || !m.op_tx2 || !m.op_txn) return RAD_E_INVAL;
    return RAD_OK;
}

/* The VAE encoder (Wan 2.2's residual one on a single frame) and the latent's normalisation. */
static int declare_encoder(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Vae& v = m.enc;
    v.nm = &m.nm;
    v.b = b;
    v.scope = "venc";
    v.down = true;
    v.first_group = kGrpVaeEnc;
    /* Repaint inputs match the output. References keep their own resize area. */
    v.base_px = std::max(m.max_image_tokens, ref_tokens_max()) * 256;
    const int64_t base = rad_meta_geti(meta, "vae.base_dim", 96);
    const std::vector<double> mult = meta_list(meta, "vae.dim_mult");
    const std::vector<double> td = meta_list(meta, "vae.temperal_downsample");
    const int64_t nres = rad_meta_geti(meta, "vae.num_res_blocks", 2);
    const int64_t zd = rad_meta_geti(meta, "vae.z_dim", 64);
    const int64_t cin = rad_meta_geti(meta, "vae.in_channels", 4);
    if (mult.size() != 5 || td.size() != 4) {
        fprintf(stderr, "radiance: qwenimage21: vae.dim_mult / vae.temperal_downsample are not the "
                        "five-level residual VAE's\n");
        return RAD_E_INVAL;
    }
    std::vector<int64_t> dims{ base };
    for (double u : mult) dims.push_back(base * (int64_t)u);
    Names& nm = m.nm;
    m.enc_in = VImg{ 0, cin, 0 };
    VImg x = v.conv(m.enc_in, dims[0], 3, 1, 1, "same", "vae.encoder.conv_in");
    for (int i = 0; i < 5; ++i) {
        const VImg xc = x;
        const bool dn = i != 4;
        for (int r = 0; r < nres; ++r)
            x = v.resblock(x, dims[(size_t)i + 1], nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d", i, r));
        if (dn) {
            x = v.conv(x, dims[(size_t)i + 1], 3, 2, 1, "down",
                       nm.ckpt("vae.encoder.down_blocks.%d.downsampler.resample.1", i));
            v.shortcut(xc, x, td[(size_t)i] != 0.0 ? 2 : 1);
        } else {
            x = v.add(x, xc);
        }
    }
    x = v.resblock(x, dims[5], "vae.encoder.mid_block.resnets.0");
    x = v.attention(x, "vae.encoder.mid_block.attentions.0");
    x = v.resblock(x, dims[5], "vae.encoder.mid_block.resnets.1");
    x = v.norm(x, true, "vae.encoder.norm_out", 4);
    x = v.conv(x, 2 * zd, 3, 1, 1, "same", "vae.encoder.conv_out");
    m.enc_q = v.conv(x, 2 * zd, 1, 1, 1, "same", "vae.quant_conv");
    /* The latent is the distribution's mode -- the first z channels -- normalised by the
     * container's per-channel mean and deviation, into the persistent reference latents. */
    m.op_enc_aff = rw(b, RAD_OP(b, "chan_affine",
                                RAD_PARAMS(RAD_RANGE("M", 1, v.px_at(m.enc_q.lvl)), RAD_INT("n", zd),
                                           RAD_STR("mul", m.lat_std.c_str()),
                                           RAD_STR("add", m.lat_mean.c_str()),
                                           RAD_STR("mode", "inv"), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.enc_q.buf }, { m.b_reflat, m.b_encz });
    /* A repainted picture's latent and its mask, into every lane its images run in. */
    m.op_lstore_lat = rw(b, RAD_OP(b, "lane_store",
                                   RAD_PARAMS(RAD_RANGE("M", 1, m.max_image_tokens),
                                              RAD_INT("n", zd),
                                              RAD_INT("stride", m.max_image_tokens),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS),
                         { m.b_encz, m.b_inp }, { m.b_inp });
    m.op_lstore_mask = rw(b, RAD_OP(b, "lane_store",
                                    RAD_PARAMS(RAD_RANGE("M", 1, m.max_image_tokens),
                                               RAD_INT("n", 1),
                                               RAD_INT("stride", m.max_image_tokens),
                                               RAD_STR("dtype", "f32")),
                                    RAD_NOWEIGHTS),
                          { m.b_mask }, { m.b_mask });
    return m.op_enc_aff && m.op_lstore_lat && m.op_lstore_mask ? RAD_OK : RAD_E_INVAL;
}

static int declare_decoder(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Vae& v = m.dec;
    v.nm = &m.nm;
    v.b = b;
    v.scope = "vdec";
    v.down = false;
    v.first_group = kGrpVaeDec;
    v.base_px = m.max_image_tokens;
    const int64_t base = rad_meta_geti(meta, "vae.decoder_base_dim", 144);
    const std::vector<double> mult = meta_list(meta, "vae.dim_mult");
    const std::vector<double> td = meta_list(meta, "vae.temperal_downsample");
    const int64_t nres = rad_meta_geti(meta, "vae.num_res_blocks", 2);
    const int64_t zd = rad_meta_geti(meta, "vae.z_dim", 64);
    const int64_t cout = rad_meta_geti(meta, "vae.out_channels", 4);
    if (mult.size() != 5 || td.size() != 4) return RAD_E_INVAL;
    /* dims = base * [mult[-1]] + mult reversed; the temporal upsample flags are the downsample
     * ones reversed, and on one frame they only choose the shortcut's frame (ft). */
    std::vector<int64_t> dims{ base * (int64_t)mult.back() };
    for (int i = 4; i >= 0; --i) dims.push_back(base * (int64_t)mult[(size_t)i]);
    std::vector<double> tu(td.rbegin(), td.rend());
    Names& nm = m.nm;
    m.b_decin = decl_b(b, nm.f("vdec.lat"), RAD_BF16, { v.base_px, zd });
    m.op_dec_aff = rw(b, RAD_OP(b, "chan_affine",
                                RAD_PARAMS(RAD_RANGE("M", 1, v.base_px), RAD_INT("n", zd),
                                           RAD_STR("mul", m.lat_std.c_str()),
                                           RAD_STR("add", m.lat_mean.c_str()),
                                           RAD_STR("mode", "fma"), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_state, m.b_prev }, { m.b_decin });
    m.dec_z = VImg{ m.b_decin, zd, 0 };
    VImg x = v.conv(m.dec_z, zd, 1, 1, 1, "same", "vae.post_quant_conv");
    x = v.conv(x, dims[0], 3, 1, 1, "same", "vae.decoder.conv_in");
    x = v.resblock(x, dims[0], "vae.decoder.mid_block.resnets.0");
    x = v.attention(x, "vae.decoder.mid_block.attentions.0");
    x = v.resblock(x, dims[0], "vae.decoder.mid_block.resnets.1");
    v.band_from = (int)v.ops.size();
    v.whole.push_back(m.b_decin);
    for (const VOp& o : v.ops) v.whole.push_back(o.y.buf);
    for (int i = 0; i < 5; ++i) {
        const VImg xc = x;
        const bool upf = i != 4;
        for (int r = 0; r < nres + 1; ++r)
            x = v.resblock(x, dims[(size_t)i + 1], nm.ckpt("vae.decoder.up_blocks.%d.resnets.%d", i, r));
        if (upf) {
            x = v.conv(x, dims[(size_t)i + 1], 3, 1, 2, "same",
                       nm.ckpt("vae.decoder.up_blocks.%d.upsampler.resample.1", i));
            v.shortcut(xc, x, tu[(size_t)i] != 0.0 ? 2 : 1);
        }
    }
    x = v.norm(x, true, "vae.decoder.norm_out", 4);
    m.dec_out = v.conv(x, cout, 3, 1, 1, "same", "vae.decoder.conv_out");
    m.b_img = decl_b(b, nm.f("image_out"), RAD_U8, { m.max_image_tokens * 256 * cout },
                     RAD_BUF_PERSIST);
    m.op_rgba = rw(b, RAD_OP(b, "to_rgba8",
                             RAD_PARAMS(RAD_RANGE("M", 1, m.max_image_tokens * 256),
                                        RAD_INT("n", cout), RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.dec_out.buf }, { m.b_img });
    if (!m.op_rgba) return RAD_E_INVAL;
    v.halo = v.band_halo();
    if (v.halo <= 0) {
        fprintf(stderr, "radiance: qwenimage21: the decoder after its mid-block is not row-local\n");
        return RAD_E_INVAL;
    }
    /* The bands, gathered. A band is ceil(H / world) latent rows, and the split is taken only where
     * a band plus its two halos is smaller than the whole, so H > 2 * halo, which bounds W and
     * thus the largest band: under m.max_image_tokens / world + m.max_image_tokens / (2 * halo)
     * tokens. */
    const int world = m.gt.world;
    if (world > 1) {
        const int64_t bpx = (m.max_image_tokens / world + m.max_image_tokens / (2 * v.halo) + 1) * 256;
        m.b_rgbaf = decl_b(b, nm.f("vdec.bands"), RAD_BF16, { world * bpx, cout });
        m.op_bands = rw(b, RAD_OP(b, "all_gather",
                                   RAD_PARAMS(RAD_INT("world_size", world),
                                              RAD_RANGE("numel", 1, bpx * cout),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS),
                         { m.dec_out.buf }, { m.b_rgbaf });
        m.op_rgba_bands = rw(b, RAD_OP(b, "to_rgba8",
                                   RAD_PARAMS(RAD_RANGE("M", 1, m.max_image_tokens * 256),
                                              RAD_INT("n", cout), RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS),
                         { m.b_rgbaf }, { m.b_img });
        if (!m.op_bands || !m.op_rgba_bands) return RAD_E_INVAL;
    }
    return RAD_OK;
}

static int declare_dit(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const int64_t T = g.max_tok;
    const int64_t D = g.n_embd, hd = g.head_dim, qd = g.q_dim();
    const int64_t C = rad_meta_geti(meta, "transformer.in_channels", 64);
    const int gx = kGrpDit;

    /* ---- the timestep: [cos | sin] of t * 1000 over 256 channels, an MLP to D, then the shared
     * modulation (four D-wide rows) and norm_out's scale, each through a silu. */
    const int64_t S = g.max_seqs;
    m.b_tproj = decl_b(b, nm.f("dit.tproj"), RAD_BF16, { S, 256 });
    m.b_te1   = decl_b(b, nm.f("dit.te1"),   RAD_BF16, { S, D });
    m.b_te1s  = decl_b(b, nm.f("dit.te1s"),  RAD_BF16, { S, D });
    m.b_temb  = decl_b(b, nm.f("dit.temb"),  RAD_BF16, { S, D });
    m.b_tembs = decl_b(b, nm.f("dit.tembs"), RAD_BF16, { S, D });
    m.b_mod   = decl_b(b, nm.f("dit.mod"),   RAD_BF16, { S, 4 * D });
    m.b_modo  = decl_b(b, nm.f("dit.mod_out"), RAD_BF16, { S, D });
    m.op_temb = rw(b, RAD_OP(b, "timestep_embed",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", 256),
                                        RAD_F64("scale", 1000.0), RAD_F64("max_period", 10000.0),
                                        RAD_INT("round", 1), RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   {}, { m.b_tproj });
    m.op_t1 = dense(b, nm.f("dit.t1.weight"),
                    nm.ckpt("transformer.time_text_embed.timestep_embedder.linear_1.weight"), D, 256,
                    S, m.b_tproj, m.b_te1, &m.w_t1, gx);
    m.op_ts = unary(b, "silu", S, D, m.b_te1, m.b_te1s);
    m.op_t2 = dense(b, nm.f("dit.t2.weight"),
                    nm.ckpt("transformer.time_text_embed.timestep_embedder.linear_2.weight"), D, D,
                    S, m.b_te1s, m.b_temb, &m.w_t2, gx);
    m.op_ts2 = unary(b, "silu", S, D, m.b_temb, m.b_tembs);
    m.op_mod = dense(b, nm.f("dit.mod.weight"), nm.ckpt("transformer.modulation.1.weight"),
                     4 * D, D, S, m.b_tembs, m.b_mod, &m.w_mod, gx);
    m.op_nout = dense(b, nm.f("dit.nout.weight"), nm.ckpt("transformer.norm_out.linear.weight"),
                      D, D, S, m.b_tembs, m.b_modo, &m.w_nout, gx);
    if (!m.op_temb || !m.op_t1 || !m.op_ts || !m.op_t2 || !m.op_ts2 || !m.op_mod || !m.op_nout)
        return RAD_E_INVAL;

    /* ---- the input rows: a text segment gathered out of txt_in's output, a picture's latents
     * (or the image's own, a lane at a time) through img_in. */
    m.b_h   = decl_b(b, nm.f("dit.h"), RAD_BF16, { T, D });
    m.b_lat = decl_b(b, nm.f("dit.lat"), RAD_BF16, { T, C });
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_txt }, { m.b_h });
    m.op_randn = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", C),
                                         RAD_INT("stride", m.max_image_tokens),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_state }, { m.b_state });
    m.op_lgather = rw(b, RAD_OP(b, "lane_gather",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", C),
                                           RAD_INT("stride", m.max_image_tokens),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_state }, { m.b_lat });
    m.op_imgin = dense(b, nm.f("dit.img_in.weight"), nm.ckpt("transformer.img_in.weight"), D,
                       C, T, m.b_lat, m.b_h, &m.w_imgin, gx);
    if (!m.op_gather || !m.op_randn || !m.op_lgather || !m.op_imgin) return RAD_E_INVAL;
    /* The reference pictures' latents go through img_in where they lie. */
    rw(b, m.op_imgin, { m.b_reflat }, {});

    /* ---- the blocks */
    m.a_hm.x = decl_b(b, nm.f("dit.hm"), RAD_BF16, { T, D });
    RAD_ARCH_TRY(declare_act(b, nm, g, m, m.a_hm, "dit.hm", D, "had_quant_act_i8g"));
    m.b_qkv = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { T, 3 * qd });
    m.a_attn.x = decl_b(b, nm.f("dit.attn"), RAD_BF16, { T, qd });
    RAD_ARCH_TRY(declare_act(b, nm, g, m, m.a_attn, "dit.attn", qd, "had_quant_act_i8g"));
    m.b_ao = decl_b(b, nm.f("dit.ao"), RAD_BF16, { T, D });
    /* More than one rank: the all-reduces of b_ao's halves run on the second lane (step_blocks). */
    if (g.world > 1) RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_ao));
    m.b_gu = decl_b(b, nm.f("dit.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_ffn.x = decl_b(b, nm.f("dit.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(declare_act(b, nm, g, m, m.a_ffn, "dit.ffn", g.n_ff, "gated_had_quant_i8g"));

    /* The rotary: three sections of adjacent pairs, (frame, row, column), each on its own ladder. */
    const std::vector<double> ax = meta_list(meta, "transformer.axes_dims_rope");
    if (ax.size() != 3 || (int64_t)(ax[0] + ax[1] + ax[2]) != hd) {
        fprintf(stderr, "radiance: qwenimage21: transformer.axes_dims_rope must be three widths "
                        "summing to the head\n");
        return RAD_E_INVAL;
    }
    const char* sections = nm.f("%d %d %d", (int)ax[0] / 2, (int)ax[1] / 2, (int)ax[2] / 2);
    const double eps = rad_meta_getf(meta, "transformer.eps", 1e-6);

    auto ln_mod = [&](rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "ln_mod",
                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_F64("eps", eps),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { x, m.b_mod, m.b_modo }, { y });
    };
    auto gate_add = [&](rad_buf y, rad_buf x) {
        return rw(b, RAD_OP(b, "gate_add",
                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { y, m.b_mod, x }, { x });
    };
    /* gate_add, ln_mod and the quantiser as one op, where the linears are fp8: `delta` is the
     * branch the norm adds in first, none for the first block's. An int8 model's is the rotating
     * twin, ln_mod_had_quant_i8g, where the kernels serve it and the codes are fed; otherwise it
     * takes the three apart. */
    const bool fuse_i8 = m.i8 && m.a_hm.q8_fed && rad_op_declared(b, "ln_mod_had_quant_i8g");
    const bool fuse_norm = m.i8 ? fuse_i8 : !g.w_bf16;
    auto ln_mod_q = [&](rad_buf delta) {
        rad_op op = RAD_OP(b, m.i8 ? "ln_mod_had_quant_i8g" : "ln_mod_quant_fp8",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_F64("eps", eps),
                                      RAD_INT("group", m.a_hm.cg()),
                                      RAD_INT("gated", delta ? 1 : 0), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS);
        if (!op) return op;
        rad_buf rd[3] = { m.b_mod, m.b_h, delta };
        rad_buf wr[3] = { m.a_hm.cq(), m.a_hm.cs(), m.b_h };
        rad_op_reads(b, op, rd, delta ? 3 : 2);
        rad_op_writes(b, op, wr, 3);
        return op;
    };
    auto attn = [&](int causal, bool i8) {
        rad_op op = RAD_OP(b, "attn_prefix_self",
                           RAD_PARAMS(RAD_INT("q_len", T), RAD_INT("head_dim", hd),
                                      RAD_INT("gqa", 1), RAD_INT("block_size", RAD_KV_BLOCK),
                                      RAD_INT("causal", causal), RAD_STR("q_dtype", "bf16"),
                                      RAD_STR("kv_dtype", "bf16"),
                                      RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                      RAD_INT("max_ctx", g.max_ctx), RAD_INT("n_head", g.n_head),
                                      RAD_INT("max_seqs", g.max_seqs),
                                      RAD_STR("qk", i8 ? "i8" : "bf16")),
                           RAD_NOWEIGHTS);
        return i8 ? rw(b, op, { m.b_qkv, m.b_k8, m.b_ks }, { m.a_attn.x })
                  : rw(b, op, { m.b_qkv }, { m.a_attn.x });
    };
    /* The denoising attention's scores in int8 where the linears are fp8: keys quantised per
     * (token, head) by quant_act_i8g, the query inside the op. The score product is about half of
     * the attention and runs at twice the rate in int8. The image changes less than between two
     * exact attention implementations (at 1024^2: 36.2 dB PSNR against exact scores, while
     * diffusers' SDPA kernels differ from each other by 32.6 to 34.4 dB). The conditioning pass
     * (once per request) and a bf16 container keep bf16. */
    const bool qk8 = !g.w_bf16;
    if (qk8) {
        m.b_k8 = decl_b(b, nm.f("dit.k8"), RAD_I8, { T, qd });
        m.b_ks = decl_b(b, nm.f("dit.ks"), RAD_F32, { T, qd / hd });
    }

    m.dit.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        DitLayer& L = m.dit[(size_t)l];
        const int li = (int)l;
        const int grp = kGrpDit + 1 + li;
        const char* pre = "transformer.transformer_blocks";
        RAD_ARCH_TRY(rad_bind_layer_kv(b, li, m.kv));
        RAD_ARCH_TRY(map_copy(b, nm.f("dit.blk.%d.q_norm.weight", li),
                              nm.ckpt("%s.%d.attn.norm_q.weight", pre, li)));
        L.w_qn = decl_w(b, nm.f("dit.blk.%d.q_norm.weight", li), RAD_BF16, { hd },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(grp));
        RAD_ARCH_TRY(map_copy(b, nm.f("dit.blk.%d.k_norm.weight", li),
                              nm.ckpt("%s.%d.attn.norm_k.weight", pre, li)));
        L.w_kn = decl_w(b, nm.f("dit.blk.%d.k_norm.weight", li), RAD_BF16, { hd },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(grp));

        if (fuse_norm) {
            L.op_lnq1 = ln_mod_q(l > 0 ? m.b_ao : 0);
        } else {
            L.op_ln1 = ln_mod(m.b_h, m.a_hm.x);
            RAD_ARCH_TRY(L.q_h.declare(b, g, m.a_hm, D));
        }
        if (l == 0) L.qkv = m.qkv0;
        RAD_ARCH_TRY(L.qkv.declare(b, nm, g, nm.f("dit.blk.%d.attn_qkv", li), 3 * qd, D,
                                   RAD_SHARD_ROW, grp_layer(grp),
                                   { nm.ckpt("%s.%d.attn.to_q.weight", pre, li),
                                     nm.ckpt("%s.%d.attn.to_k.weight", pre, li),
                                     nm.ckpt("%s.%d.attn.to_v.weight", pre, li) },
                                   m.a_hm, m.b_qkv, 0, { qd, qd, qd }));
        L.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", g.n_head),
                                           RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_F64("theta", g.theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "gptj_blocks"),
                                           RAD_INT("rotary_dim", hd),
                                           RAD_STR("sections", sections)),
                                RAD_WEIGHTS(L.w_qn, L.w_kn)),
                      { m.b_qkv }, { m.b_qkv });
        /* The attention first: the KV group's block size is read off the op that consumes it. */
        L.op_attn_c = attn(1, false);
        L.op_attn_b = attn(0, false);
        if (qk8) {
            L.op_attn_q = attn(0, true);
            L.op_kq = decl_quant_act(b, "quant_act_i8g", T, qd, hd, m.b_qkv, m.b_k8, m.b_ks);
            if (!L.op_attn_q || !L.op_kq) return RAD_E_INVAL;
        }
        L.op_kv = rw(b, RAD_OP(b, "kv_store",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                          RAD_INT("n_head_kv", g.n_head_kv),
                                          RAD_INT("block_size", rad_kv_block_size(b, m.kv)),
                                          RAD_STR("kv_dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_qkv }, {});
        RAD_ARCH_TRY(L.q_attn.declare(b, g, m.a_attn, qd));
        RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("dit.blk.%d.attn_output", li), D, qd,
                                 RAD_SHARD_COL, grp_layer(grp),
                                 { nm.ckpt("%s.%d.attn.to_out.0.weight", pre, li) },
                                 m.a_attn, m.b_ao));
        L.op_ar1 = all_reduce(b, g, m.b_ao, T);
        L.op_ar1h = all_reduce(b, g, m.b_ao, T, true);
        if (fuse_norm) {
            L.op_lnq2 = ln_mod_q(m.b_ao);
        } else {
            L.op_g1 = gate_add(m.b_ao, m.b_h);
            L.op_ln2 = ln_mod(m.b_h, m.a_hm.x);
            RAD_ARCH_TRY(L.q_h2.declare(b, g, m.a_hm, D));
        }
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("dit.blk.%d.ffn_gate_up", li), 2 * g.n_ff,
                                       D, RAD_SHARD_ROW, grp_layer(grp),
                                       { nm.ckpt("%s.%d.img_mlp.gate_layer.weight", pre, li),
                                         nm.ckpt("%s.%d.img_mlp.proj.weight", pre, li) },
                                       m.a_hm, m.b_gu, 0, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(L.gq.declare(b, g, m.b_gu, m.a_ffn, g.n_ff, "silu", 0, true));
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("dit.blk.%d.ffn_down", li), D, g.n_ff,
                                    RAD_SHARD_COL, grp_layer(grp),
                                    { nm.ckpt("%s.%d.img_mlp.out.weight", pre, li) },
                                    m.a_ffn, m.b_ao));
        L.op_ar2 = all_reduce(b, g, m.b_ao, T);
        L.op_ar2h = all_reduce(b, g, m.b_ao, T, true);
        /* Folded into the next block's first norm, except after the last block. */
        if (!fuse_norm || l + 1 == g.n_layer) L.op_g2 = gate_add(m.b_ao, m.b_h);
    }

    /* ---- the output: norm_out's adaptive norm, proj_out, and the sampler's step */
    m.op_nout_ln = ln_mod(m.b_h, m.a_hm.x);
    m.b_v = decl_b(b, nm.f("dit.v"), RAD_BF16, { T, C });
    m.op_pout = dense(b, nm.f("dit.proj_out.weight"), nm.ckpt("transformer.proj_out.weight"),
                      C, D, T, m.a_hm.x, m.b_v, &m.w_pout, gx);
    m.op_euler = rw(b, RAD_OP(b, "flow_euler",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", C),
                                         RAD_INT("stride", m.max_image_tokens),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_v, m.b_inp, m.b_mask, m.b_state, m.b_prev }, { m.b_state, m.b_prev });
    return m.op_pout && m.op_euler ? RAD_OK : RAD_E_INVAL;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    m.max_image_tokens = image_tokens(ctx->gen_max_pixels);
    m.max_refs = reference_count(ctx->gen_max_refs);
    if (m.max_image_tokens == 0) {
        fprintf(stderr, "radiance: qwenimage21: --image-gen-max-pixels must fit a 32x32 image\n");
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(check_format(meta));
    RAD_ARCH_TRY(geom_from(m.gt, meta, ctx, "bf16", RAD_BF16));
    Names& nm = m.nm;

    /* ---- the DiT's geometry, off the transformer's own config */
    Geom& gd = m.gd;
    gd = m.gt;
    gd.n_layer    = rad_meta_geti(meta, "transformer.num_layers", 32);
    gd.head_dim   = rad_meta_geti(meta, "transformer.attention_head_dim", 128);
    gd.n_head_all = rad_meta_geti(meta, "transformer.num_attention_heads", 32);
    gd.n_head_kv_all = gd.n_head_all;
    gd.n_embd     = gd.head_dim * gd.n_head_all;
    gd.n_ff_all   = gd.n_embd * rad_meta_geti(meta, "transformer.mlp_ratio", 3);
    gd.eps        = (float)rad_meta_getf(meta, "transformer.eps", 1e-6);
    gd.theta      = 10000.0f;   /* diffusers' QwenEmbedRope fixes it; the config does not say */
    gd.rope_mc    = false;
    RAD_ARCH_TRY(divide_or_fail(gd.n_head_all, gd.world, "transformer.num_attention_heads",
                                &gd.n_head));
    gd.n_head_kv = gd.n_head;
    gd.kv_off = 0;
    RAD_ARCH_TRY(divide_or_fail(gd.n_ff_all, gd.world, "the DiT's feed-forward width", &gd.n_ff));
    if (m.gt.n_layer >= kGrpTxtIn || gd.n_layer >= kGrpVision - kGrpDit) {
        fprintf(stderr, "radiance: qwenimage21: %lld text-encoder and %lld DiT layers overrun the "
                        "weight-group ranges\n", (long long)m.gt.n_layer, (long long)gd.n_layer);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "transformer.context_in_dim", 4096) != m.gt.n_embd) {
        fprintf(stderr, "radiance: qwenimage21: the DiT reads %lld-wide conditioning and the text "
                        "encoder writes %lld\n",
                (long long)rad_meta_geti(meta, "transformer.context_in_dim", 0),
                (long long)m.gt.n_embd);
        return RAD_E_INVAL;
    }

    /* ---- bf16 or block fp8: what the container holds for the first DiT projection decides it
     * for every linear of both networks (rad_linear.h, Geom::w_bf16). */
    {
        LinearQ& probe = m.qkv0;
        RAD_ARCH_TRY(probe.probe(b, nm, gd, "dit.blk.0.attn_qkv",
                                 { "transformer.transformer_blocks.0.attn.to_q.weight",
                                   "transformer.transformer_blocks.0.attn.to_k.weight",
                                   "transformer.transformer_blocks.0.attn.to_v.weight" }));
        RadEncoding e{};
        const bool known = weight_enc(b, nm.f("dit.blk.0.attn_qkv.weight"), &e);
        const bool plain = known && rad_enc_is(&e, "plain");
        gd.w_bf16 = plain;
        m.gt.w_bf16 = plain;
        m.i8 = known && (linear_is_w8h(e) || linear_is_wnh(e));
        if (m.i8) m.g8 = linear_w8h_group(e);
    }

    /* ---- the latent's normalisation, rounded to bf16 by the ops that read it */
    {
        const std::vector<double> mean = meta_list(meta, "vae.latents_mean");
        const std::vector<double> sd = meta_list(meta, "vae.latents_std");
        const int64_t zd = rad_meta_geti(meta, "vae.z_dim", 64);
        if ((int64_t)mean.size() != zd || (int64_t)sd.size() != zd) {
            fprintf(stderr, "radiance: qwenimage21: vae.latents_mean / latents_std are not %lld "
                            "values\n", (long long)zd);
            return RAD_E_INVAL;
        }
        for (int64_t i = 0; i < zd; ++i) {
            char t[64];
            std::snprintf(t, sizeof t, "%s%.9g", i ? " " : "", mean[(size_t)i]);
            m.lat_mean += t;
            std::snprintf(t, sizeof t, "%s%.9g", i ? " " : "", sd[(size_t)i]);
            m.lat_std += t;
        }
    }

    /* ---- the conditioning KV group: the DiT's prefix keys and values, every block */
    {
        RadKVGroupDecl fd{};
        fd.kind = RAD_KV_FULL;
        fd.dtype = RAD_BF16;
        fd.n_head_kv = gd.n_head_kv;
        fd.head_dim = gd.head_dim;
        m.kv = rad_decl_kv_group(b, nm.f("kv_cond"), &fd);
        if (!m.kv) return RAD_E_INVAL;
    }

    /* ---- the per-image state, a lane an image, and what the passes hand each other */
    const int64_t C = rad_meta_geti(meta, "transformer.in_channels", 64);
    const int64_t lane_rows = (int64_t)kMaxLanes * m.max_image_tokens;
    m.b_state = decl_b(b, nm.f("lat_state"), RAD_BF16, { lane_rows, C }, RAD_BUF_PERSIST);
    m.b_prev  = decl_b(b, nm.f("lat_preview"), RAD_BF16, { lane_rows, C }, RAD_BUF_PERSIST);
    m.b_inp   = decl_b(b, nm.f("lat_repaint"), RAD_BF16, { lane_rows, C }, RAD_BUF_PERSIST);
    m.b_mask  = decl_b(b, nm.f("lat_mask"), RAD_F32, { lane_rows, 1 }, RAD_BUF_PERSIST);
    m.max_ref_rows = m.max_refs * ref_tokens_max();
    m.b_reflat = decl_b(b, nm.f("ref_lat"), RAD_BF16, { m.max_ref_rows, C }, RAD_BUF_PERSIST);
    m.b_encz   = decl_b(b, nm.f("enc_lat"), RAD_BF16, { m.max_image_tokens, C });
    m.C = C;

    /* ---- in issue order: the tower, the language model, the encoder, the DiT, the decoder.
     * (Vae says why the order matters.) */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, m.gt.n_embd, kGrpVision, true));
    /* ---- adapter key names. The DiT's LoRA files use the checkpoint's own names, or
     * ComfyUI's, which join the SwiGLU's gate and up projections as `img_mlp.gate_up` [gate; up]
     * (comfy/ldm/qwen_image21). */
    {
        const int64_t F = gd.n_ff_all;
        RAD_ARCH_TRY(lora_names(b, "transformer.", kLoraDiffusionSpellings,
                                { { "transformer_blocks.#.img_mlp.gate_up",
                                    "transformer_blocks.#.img_mlp.gate_layer", 0, F, 0 },
                                  { "transformer_blocks.#.img_mlp.gate_up",
                                    "transformer_blocks.#.img_mlp.proj", F, 2 * F, 0 } }));
    }
    RAD_ARCH_TRY(declare_te(b, meta, m));
    RAD_ARCH_TRY(declare_encoder(b, meta, m));
    RAD_ARCH_TRY(declare_dit(b, meta, m));
    RAD_ARCH_TRY(declare_decoder(b, meta, m));

    /* ---- what the core drives */
    if (!ctx->shape_probe) {
        static thread_local std::vector<double> sig;
        sig = meta_list(meta, "pipeline.sample_sigmas");
        static thread_local std::string tmpl_text, tmpl_edit, drop;
        tmpl_text = std::string("<|im_start|>system\n") + kSysPrompt +
                    "<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
        tmpl_edit = std::string("<|im_start|>system\n") + kSysPrompt +
                    "<|im_end|>\n<|im_start|>user\n{images}{prompt}<|im_end|>\n<|im_start|>assistant\n";
        drop = std::string("<|im_start|>system\n") + kSysPrompt + "<|im_end|>\n";

        RadGeneratorDecl d{};
        d.name = sig.empty() ? "qwen-image-2.1" : "qwen-image-2.1-turbo";
        d.layout = RAD_GEN_LAYOUT_JOINT_GRID;
        d.tmpl_text = tmpl_text.c_str();
        d.tmpl_edit = tmpl_edit.c_str();
        /* `{image}` is the processor's <|vision_start|> <|image_pad|>... <|vision_end|> run. */
        d.tmpl_image = "<image{n}>{image}";
        d.tmpl_image_sep = " ";
        d.drop_prefix = drop.c_str();
        d.tmpl_transparent = "This is an RGBA image with transparency. {prompt}. The image has alpha "
                             "channel and the background is transparent.";
        d.latent_channels = C;
        d.latent_scale = 16;
        d.size_multiple = 32;
        d.pixel_channels = rad_meta_geti(meta, "vae.out_channels", 4);
        d.slot_rows = 4;
        d.max_tokens = m.max_image_tokens;
        d.max_refs = m.max_refs;
        d.ref_area = kRefArea;
        const int64_t side = (int64_t)(std::sqrt(std::min<int64_t>(m.max_image_tokens, 16384)) / 2) * 32;
        d.default_width = side;
        d.default_height = side;
        d.default_steps = sig.empty() ? 40 : (int64_t)sig.size();
        d.n_sigmas = (int32_t)sig.size();
        d.sigmas = sig.empty() ? nullptr : sig.data();
        d.shift = rad_meta_getf(meta, "scheduler.shift", 1.0);
        d.dyn_shift = (int32_t)rad_meta_geti(meta, "scheduler.use_dynamic_shifting", 0);
        d.shift_exp = std::strcmp(rad_meta_gets(meta, "scheduler.time_shift_type", "exponential"),
                                  "exponential") == 0;
        d.base_shift = rad_meta_getf(meta, "scheduler.base_shift", 0.5);
        d.max_shift = rad_meta_getf(meta, "scheduler.max_shift", 1.15);
        d.base_seq = rad_meta_geti(meta, "scheduler.base_image_seq_len", 256);
        d.max_seq = rad_meta_geti(meta, "scheduler.max_image_seq_len", 4096);
        d.shift_terminal = rad_meta_getf(meta, "scheduler.shift_terminal", 0.0);
        d.default_cfg = 1.0;
        d.max_lanes = kMaxLanes;
        d.image_out = m.b_img;
        RAD_ARCH_TRY(rad_declare_generator(b, &d));
    }

    rad_note(b, "rank %d/%d: Qwen-Image 2.1%s -- text encoder %lld layers (%lld heads, %lld KV "
                "heads a rank), DiT %lld blocks (%lld heads a rank, ff %lld), %s linears%s",
             m.gt.rank, m.gt.world, meta_list(meta, "pipeline.sample_sigmas").empty() ? "" : " Turbo",
             (long long)m.gt.n_layer, (long long)m.gt.n_head, (long long)m.gt.n_head_kv,
             (long long)gd.n_layer, (long long)gd.n_head, (long long)gd.n_ff,
             gd.w_bf16 ? "bf16" : m.i8 ? "rotated int8" : "block-fp8",
             m.media ? ", vision tower with deepstack" : "");
    return RAD_OK;
}

/* ================================================================== step */
/* The language model over the prompt, the tower's rows in place of its vision slots and the
 * deepstack rows added after the first layers. */
static void step_te(RadCtx* c, const RadBatch* batch, const Model& m) {
    const Geom& g = m.gt;
    const int64_t T = batch->n_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_tx, T));
    const int64_t nmm = batch->n_mm_rows;
    if (m.media && nmm > 0)
        RAD_ISSUE_N(c, m.op_mm, nmm, brows(m.vit.b_out, nmm), praw(batch->mm_rows, RAD_I32, nmm),
                    brows(m.b_tx, T));
    const RadOperand pos = praw2(batch->rope_pos, RAD_I32, 3, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    for (int64_t l = 0; l < (int64_t)m.te.size(); ++l) {
        const TeLayer& L = m.te[(size_t)l];
        RAD_ISSUE(c, L.op_n1, brows(m.b_tx, T), RAD_W(L.w_n1), brows(m.a_th.x, T));
        L.q_h.step(c, T);
        L.qkv.step(c, m.a_th, m.b_tkv, T);
        RAD_ISSUE(c, L.op_qkr, brows(m.b_tkv, T), pos, RAD_W(L.w_qn), RAD_W(L.w_kn));
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
        /* Deepstack: tap l's rows join the hidden state at the vision slots after layer l. */
        if (m.media && nmm > 0 && l < m.vit.n_ds)
            RAD_ISSUE_N(c, m.op_ds[l], nmm, brows(m.vit.b_ds[l], nmm),
                        praw(batch->mm_rows, RAD_I32, nmm), brows(m.b_tx, T));
    }
    RAD_ISSUE(c, m.op_txn, brows(m.b_tx, T), RAD_W(m.w_txn), brows(m.b_txa, T));
    RAD_ISSUE(c, m.op_tx1, brows(m.b_txa, T), RAD_W(m.w_tx1), brows(m.b_txb, T));
    RAD_ISSUE(c, m.op_txg, brows(m.b_txb, T), brows(m.b_txa, T));
    RAD_ISSUE(c, m.op_tx2, brows(m.b_txa, T), RAD_W(m.w_tx2), brows(m.b_txt, T));
}

/* The timestep embedding and the shared modulation for `S` entries at the sigmas in `t`. */
static void step_temb(RadCtx* c, const Model& m, const float* t, int64_t S) {
    RAD_ISSUE_N(c, m.op_temb, S, praw(t, RAD_F32, S), brows(m.b_tproj, S));
    RAD_ISSUE_N(c, m.op_t1, S, brows(m.b_tproj, S), RAD_W(m.w_t1), brows(m.b_te1, S));
    RAD_ISSUE_N(c, m.op_ts, S, brows(m.b_te1, S), brows(m.b_te1s, S));
    RAD_ISSUE_N(c, m.op_t2, S, brows(m.b_te1s, S), RAD_W(m.w_t2), brows(m.b_temb, S));
    RAD_ISSUE_N(c, m.op_ts2, S, brows(m.b_temb, S), brows(m.b_tembs, S));
    RAD_ISSUE_N(c, m.op_mod, S, brows(m.b_tembs, S), RAD_W(m.w_mod), brows(m.b_mod, S));
    RAD_ISSUE_N(c, m.op_nout, S, brows(m.b_tembs, S), RAD_W(m.w_nout), brows(m.b_modo, S));
}

/* The DiT's blocks over `R` rows in `b_h`: `S` entries bounded by `cu`, each against its prefix
 * (`bt` / `used`) and its own rows, causal or not; the rows' keys and values stored at `slot`
 * when it is given.
 *
 * Two halves, all-reduces on the second lane. With more than one rank, a denoising pass of one
 * entry runs everything after the attention (output projection, norm, feed-forward) and the next
 * block's norm and qkv projection over the rows in two halves; each half's all-reduce goes to lane
 * 1 while lane 0 computes the other half. Every op between two attentions is row-local, so a half
 * computes exactly the bits the whole would. The attention stays whole: every query reads every
 * key. One entry only, because the norms find a row's modulation by its entry; several entries run
 * unsplit. Lane 0 waits for a half's all-reduce just before its first reader, and that wait is
 * issued before the next all-reduce goes to lane 1: a join orders a lane behind everything issued
 * on the other, so the reverse order would wait for both. */
static void step_blocks(RadCtx* c, const Model& m, int64_t R, int64_t S, const RadOperand& cu,
                        const RadOperand& pos, const RadOperand& bt, const RadOperand& used,
                        const int32_t* slot, bool causal) {
    const Geom& g = m.gd;
    const int64_t D = g.n_embd, qd = g.q_dim();
    const RadOperand s1 = bcol(m.b_mod, 0, D, S), g1 = bcol(m.b_mod, D, D, S);
    const RadOperand s2 = bcol(m.b_mod, 2 * D, D, S), g2 = bcol(m.b_mod, 3 * D, D, S);
    const bool split = g.world > 1 && S == 1 && !slot && R >= 2;
    const int64_t hr0[2] = { 0, split ? R / 2 : R }, hn[2] = { hr0[1], R - hr0[1] };
    const int nh = split ? 2 : 1;
    const int64_t nD = D / m.a_hm.cg();

    /* A half's all-reduce: on lane 1 behind everything lane 0 has issued, when split. */
    auto all_reduce = [&](rad_op op, int64_t r0, int64_t n) {
        if (!op) return;
        if (split) {
            rad_lane_join(c, 0, 1);
            rad_lane(c, 1);
        }
        RAD_ISSUE_N(c, op, n * D, brow_slice(m.b_ao, r0, n, D), RAD_NONE);
        if (split) rad_lane(c, 0);
    };
    auto wait_ar = [&]() { if (split) rad_lane_join(c, 1, 0); };
    /* The block's first norm over a half: the previous block's MLP branch gated in, where folded. */
    auto norm1 = [&](const DitLayer& L, int64_t l, int64_t r0, int64_t n) {
        if (L.op_lnq1) {
            RAD_ISSUE_N(c, L.op_lnq1, n, l > 0 ? brow_slice(m.b_ao, r0, n, D) : RAD_NONE,
                        l > 0 ? g2 : RAD_NONE, s1, cu, brow_slice(m.b_h, r0, n, D),
                        brow_slice(m.a_hm.cq(), r0, n, D), brow_slice(m.a_hm.cs(), r0, n, nD),
                        RAD_NONE);
        } else {
            RAD_ISSUE_N(c, L.op_ln1, n, brow_slice(m.b_h, r0, n, D), s1, cu,
                        brow_slice(m.a_hm.x, r0, n, D));
            L.q_h.step(c, R, r0, n);
        }
        L.qkv.step(c, m.a_hm, m.b_qkv, R, r0, n);
    };
    /* The second norm and the feed-forward over a half, after its attention branch is reduced. */
    auto mlp = [&](const DitLayer& L, int64_t r0, int64_t n) {
        if (L.op_lnq2) {
            RAD_ISSUE_N(c, L.op_lnq2, n, brow_slice(m.b_ao, r0, n, D), g1, s2, cu,
                        brow_slice(m.b_h, r0, n, D), brow_slice(m.a_hm.cq(), r0, n, D),
                        brow_slice(m.a_hm.cs(), r0, n, nD), RAD_NONE);
        } else {
            RAD_ISSUE_N(c, L.op_g1, n, brow_slice(m.b_ao, r0, n, D), g1, cu,
                        brow_slice(m.b_h, r0, n, D));
            RAD_ISSUE_N(c, L.op_ln2, n, brow_slice(m.b_h, r0, n, D), s2, cu,
                        brow_slice(m.a_hm.x, r0, n, D));
            L.q_h2.step(c, R, r0, n);
        }
        L.gate_up.step(c, m.a_hm, m.b_gu, R, r0, n);
        L.gq.step(c, m.b_gu, R, r0, n);
        L.down.step(c, m.a_ffn, m.b_ao, R, r0, n);
    };

    for (int64_t l = 0; l < (int64_t)m.dit.size(); ++l) {
        const DitLayer& L = m.dit[(size_t)l];
        /* The first half's norm reads the all-reduce the previous block waited for last; the
         * second half's waits here, so its reduction ran under the first half's norm and qkv. */
        for (int h = 0; h < nh; ++h) {
            if (h > 0 && l > 0) wait_ar();
            norm1(L, l, hr0[h], hn[h]);
        }
        RAD_ISSUE_N(c, L.op_qkr, R, brows(m.b_qkv, R), pos, RAD_W(L.w_qn), RAD_W(L.w_kn));
        if (slot)
            RAD_ISSUE_N(c, L.op_kv, R, bcol(m.b_qkv, qd, qd, R), bcol(m.b_qkv, 2 * qd, qd, R),
                        praw(slot, RAD_I32, R), kv_cache(m.kv, (int)l));
        if (!slot && L.op_attn_q) {
            RAD_ISSUE_N(c, L.op_kq, R, bcol(m.b_qkv, qd, qd, R), brows(m.b_k8, R),
                        brows(m.b_ks, R));
            RAD_ISSUE(c, L.op_attn_q, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                      bcol(m.b_qkv, 2 * qd, qd, R), kv_cache(m.kv, (int)l), bt, used, cu, RAD_NONE,
                      RAD_NONE, brows(m.b_k8, R), brows(m.b_ks, R), brows(m.a_attn.x, R),
                      RAD_NONE);
        } else {
            RAD_ISSUE(c, causal ? L.op_attn_c : L.op_attn_b, bcol(m.b_qkv, 0, qd, R),
                      bcol(m.b_qkv, qd, qd, R), bcol(m.b_qkv, 2 * qd, qd, R),
                      kv_cache(m.kv, (int)l), bt, used, cu, RAD_NONE, RAD_NONE, RAD_NONE, RAD_NONE,
                      brows(m.a_attn.x, R), RAD_NONE);
        }
        /* Each wait below is for the all-reduce lane 1 holds last, which is the one lane 0 reads
         * next: the second half's projection runs under the first half's reduction, the first
         * half's feed-forward under the second's, and so on. */
        for (int h = 0; h < nh; ++h) {
            L.q_attn.step(c, R, hr0[h], hn[h]);
            L.o.step(c, m.a_attn, m.b_ao, R, hr0[h], hn[h]);
            if (h > 0) wait_ar();
            all_reduce(split ? L.op_ar1h : L.op_ar1, hr0[h], hn[h]);
        }
        for (int h = 0; h < nh; ++h) {
            mlp(L, hr0[h], hn[h]);
            wait_ar();
            all_reduce(split ? L.op_ar2h : L.op_ar2, hr0[h], hn[h]);
        }
        if (L.op_g2) {
            wait_ar();
            RAD_ISSUE_N(c, L.op_g2, R, brows(m.b_ao, R), g2, cu, brows(m.b_h, R));
        }
    }
    wait_ar();
}

/* One picture through the VAE encoder: `px` [h, w * 4] bf16 pixels in [-1, 1] -> its normalised
 * latents at rows `row0..` of `dst` (the reference latents, or a repainted picture's). `cu` is the
 * picture's {0, latent tokens}. */
static void step_encode(RadCtx* c, const Model& m, const uint16_t* px, int64_t h, int64_t w,
                        const RadOperand& cu, rad_buf dst, int64_t row0) {
    m.enc.issue(c, h, w, cu, px);
    const int64_t n = (h / 16) * (w / 16);
    const int64_t zd = m.enc_q.ch / 2;
    RAD_ISSUE_N(c, m.op_enc_aff, n, bcol(m.enc_q.buf, 0, zd, n), brow_slice(dst, row0, n, zd));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    const RadGenBatch* G = batch->gen_b;
    if (!batch->gen || !G) {
        rad_step_fail(c, "qwenimage21 is an image generator and runs generator passes only");
        return;
    }
    const RadKVGroupBatch* kvb = kv_batch(batch, m.kv);
    const int64_t C = m.C;

    switch (batch->gen) {
    case RAD_GEN_PASS_COND: {
        if (!kvb) { rad_step_fail(c, "a conditioning pass with no KV for its sequence"); return; }
        /* 1. the vision tower, when the prompt carries pictures */
        if (m.media && batch->enc_n_patch > 0) m.vit.step(c, batch);
        /* 2. the language model and txt_in */
        step_te(c, batch, m);
        /* 3. the pictures through the VAE encoder: the references into `ref_lat`, in prompt
         * order, and a repainted picture into its images' lanes with its mask. */
        int64_t lat_row = 0;
        const int64_t n_ref = G->n_px - (G->inpaint ? 1 : 0);
        for (int64_t i = 0; i < G->n_px; ++i) {
            const int32_t off = G->px_h[3 * i], ph = G->px_h[3 * i + 1], pw = G->px_h[3 * i + 2];
            const uint16_t* px = (const uint16_t*)G->px + (int64_t)off * m.enc_in.ch;
            const int64_t n = (int64_t)(ph / 16) * (pw / 16);
            const RadOperand pcu = praw(G->px_cu + 2 * i, RAD_I32, 2);
            if (i < n_ref) {
                if (n > ref_tokens_max() || n > m.max_ref_rows - lat_row) {
                    rad_step_fail(c, "qwenimage21: reference pixels exceed the declared latent buffer");
                    return;
                }
                step_encode(c, m, px, ph, pw, pcu, m.b_reflat, lat_row);
                lat_row += n;
            } else {
                step_encode(c, m, px, ph, pw, pcu, m.b_encz, 0);
                RAD_ISSUE_N(c, m.op_lstore_lat, n, brows(m.b_encz, n),
                            praw(G->lane, RAD_I32, G->n_lane), brows(m.b_inp, kMaxLanes * m.max_image_tokens));
                RAD_ISSUE_N(c, m.op_lstore_mask, n, praw2(G->inp_mask, RAD_F32, n, 1),
                            praw(G->lane, RAD_I32, G->n_lane),
                            brows(m.b_mask, kMaxLanes * m.max_image_tokens));
            }
        }
        /* 4. the modulation at t = 0, which every conditioning row reads */
        step_temb(c, m, G->sigma, 1);
        /* 5. the prefix, a segment at a time: each sees the segments before it in the cache. */
        int64_t ref_row = 0;
        const RadOperand bt = praw2(kvb->block_table, RAD_I32, 1, kvb->block_table_pitch);
        for (int64_t s = 0; s < G->n_seg; ++s) {
            const int64_t a = G->seg_h[s], len = G->seg_h[s + 1] - a;
            if (len <= 0) continue;
            const bool pic = G->seg_bidi_h[s] != 0;
            if (pic) {
                RAD_ISSUE_N(c, m.op_imgin, len, brow_slice(m.b_reflat, ref_row, len, C),
                            RAD_W(m.w_imgin), brows(m.b_h, len));
                ref_row += len;
            } else {
                RAD_ISSUE_N(c, m.op_gather, len, brows(m.b_txt, batch->n_tok),
                            praw(G->src + a, RAD_I32, len), brows(m.b_h, len));
            }
            step_blocks(c, m, len, 1, praw(G->seg_cu + 2 * s, RAD_I32, 2),
                        praw2(G->rope + 3 * a, RAD_I32, 3, len), bt,
                        praw(G->seg_start + s, RAD_I32, 1), G->slot_mapping + a, !pic);
        }
        return;
    }
    case RAD_GEN_PASS_DENOISE: {
        if (!kvb) { rad_step_fail(c, "a denoising pass with no KV for its conditioning"); return; }
        const int64_t N = batch->n_tok, S = batch->n_seq;
        const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
        const RadOperand lanes = praw(G->lane, RAD_I32, S);
        const int64_t L = (int64_t)kMaxLanes * m.max_image_tokens;
        step_temb(c, m, G->sigma, S);
        RAD_ISSUE_N(c, m.op_randn, N, praw(G->flags, RAD_I32, S), praw(G->seed, RAD_U32, 2 * S),
                    lanes, cu, brows(m.b_state, L));
        RAD_ISSUE_N(c, m.op_lgather, N, brows(m.b_state, L), lanes, cu, brows(m.b_lat, N));
        RAD_ISSUE_N(c, m.op_imgin, N, brows(m.b_lat, N), RAD_W(m.w_imgin), brows(m.b_h, N));
        step_blocks(c, m, N, S, cu, praw2(G->rope, RAD_I32, 3, N),
                    praw2(kvb->block_table, RAD_I32, S, kvb->block_table_pitch),
                    praw(kvb->seqused, RAD_I32, S), nullptr, false);
        RAD_ISSUE_N(c, m.op_nout_ln, N, brows(m.b_h, N), brows(m.b_modo, S), cu, brows(m.a_hm.x, N));
        RAD_ISSUE_N(c, m.op_pout, N, brows(m.a_hm.x, N), RAD_W(m.w_pout), brows(m.b_v, N));
        RAD_ISSUE_N(c, m.op_euler, N, brows(m.b_v, N), praw(G->sigma, RAD_F32, S),
                    praw(G->sigma_next, RAD_F32, S), praw(G->cfg, RAD_F32, S),
                    praw(G->pair, RAD_I32, S), lanes, praw(G->flags, RAD_I32, S),
                    praw(G->seed, RAD_U32, 2 * S), cu, brows(m.b_inp, L), brows(m.b_mask, L),
                    brows(m.b_state, L), brows(m.b_prev, L));
        return;
    }
    case RAD_GEN_PASS_DECODE: {
        const int64_t H = G->dec_height / 16, W = G->dec_width / 16, n = H * W;
        const rad_buf src = G->dec_preview ? m.b_prev : m.b_state;
        RAD_ISSUE_N(c, m.op_dec_aff, n, brow_slice(src, (int64_t)G->dec_lane * m.max_image_tokens, n, C),
                    brows(m.b_decin, n));
        const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
        const int64_t npx = (int64_t)G->dec_height * G->dec_width, ch = m.dec_out.ch;
        const int world = m.gt.world, rank = m.gt.rank;
        const int64_t band = (H + world - 1) / world, halo = m.dec.halo;
        if (!m.op_bands || band + 2 * halo >= H) {
            m.dec.issue(c, H, W, cu, nullptr);
            RAD_ISSUE_N(c, m.op_rgba, npx, RAD_P_T2(rad_buf_ptr(c, m.dec_out.buf), RAD_BF16, npx, ch),
                        RAD_P_T2(rad_buf_ptr(c, m.b_img), RAD_U8, npx, ch));
            return;
        }
        /* The mid-block whole, then this rank's band of latent rows [a, a + band) with `halo` rows
         * per side; its pixels sit (a - lo) * 16 rows into the tail's output. The last band may be
         * short; whatever the gather moves past the picture's last row is never read. */
        m.dec.issue(c, H, W, cu, nullptr, 0, (size_t)m.dec.band_from);
        const int64_t a = rank * band;
        const int64_t lo = std::max<int64_t>(0, a - halo), hi = std::min<int64_t>(H, a + band + halo);
        m.dec.issue(c, hi - lo, W, cu, nullptr, (size_t)m.dec.band_from, (size_t)-1, lo);
        const int64_t bpx = band * 16 * W * 16;
        const char* out = (const char*)rad_buf_ptr(c, m.dec_out.buf) +
                          (a - lo) * 16 * W * 16 * ch * (int64_t)sizeof(uint16_t);
        RAD_ISSUE_N(c, m.op_bands, bpx * ch, RAD_P_T2(out, RAD_BF16, bpx, ch),
                    RAD_P_T2(rad_buf_ptr(c, m.b_rgbaf), RAD_BF16, world * bpx, ch));
        RAD_ISSUE_N(c, m.op_rgba_bands, npx, RAD_P_T2(rad_buf_ptr(c, m.b_rgbaf), RAD_BF16, npx, ch),
                    RAD_P_T2(rad_buf_ptr(c, m.b_img), RAD_U8, npx, ch));
        return;
    }
    default:
        rad_step_fail(c, "qwenimage21: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The denoiser can carry two copies of an image for guidance. With reduced output sizes
 * also leave room for the reference conditioning and prompt, within the original row budget. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    const int64_t tokens = image_tokens(out->gen_max_pixels);
    if (tokens == 0) return RAD_E_INVAL;
    const int64_t refs = reference_count(out->gen_max_refs);
    const int64_t conditioning_rows = std::min<int64_t>(
        2 * kMaxImageTokens, refs * ref_tokens_max() + 1024);
    out->gen_rows = std::max(2 * tokens, conditioning_rows);
    const int64_t merge = rad_meta_geti(meta, "vision_config.spatial_merge_size", 2);
    if (merge < 1 || merge > 16) return RAD_E_INVAL;
    const int64_t group = merge * merge;
    const int64_t patches = refs * (kRefArea / 256) * 11 / 10;
    out->gen_patches = (patches + group - 1) / group * group;
    return RAD_OK;
}

}  /* namespace qwenimage21 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(qwenimage21)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(qwenimage21, "qwenimage21", "", "0.1.0",
                "Qwen-Image 2.1 / 2.1-Turbo: Qwen3-VL-8B conditioning, a 7.1B single-stream "
                "diffusion transformer over a cached block-causal prefix, and a 2-D RGBA VAE; "
                "bf16 or block-fp8 linears")
#endif
