/* anima -- Anima (CircleStone Labs and Comfy Org): text to an RGB image.
 *
 * ============================== THE MODEL ==============================
 *
 * Three networks behind one prompt, as ComfyUI runs them (comfy/ldm/anima, comfy/ldm/cosmos/
 * predict2.py, comfy/text_encoders/anima.py):
 *
 *   Conditioning: Qwen3-0.6B-Base reads the prompt as it is (no template), and its last hidden
 *   state after the final norm is the source. The LLM adapter (six blocks of 1024 wide, 16 heads of
 *   64) then reads the same prompt again through T5's tokenizer: each T5 token's learned embedding
 *   is a query row that attends to the other T5 rows (self-attention) and to the Qwen3 rows (cross
 *   attention), both rotated with a 64-wide NeoX rotary at the rows' own indices, then a GELU MLP;
 *   an output linear and an RMS norm close it. Its rows, zero-padded to 512, are the DiT's context.
 *   An empty prompt is one padding token to Qwen3 and T5's </s> alone.
 *
 *   Denoiser: Cosmos-Predict2's MiniTrainDIT at 2B (28 blocks of 2048, 16 heads of 128). A token
 *   is 2 x 2 positions of the 16-channel latent (64 values, channel-major; 16 pixels a side) and
 *   the patch embedding reads them beside an all-zero padding-mask channel, so only its first 64
 *   columns are kept (RAD_MAP_COLS). Each block: a self-attention with q/k RMS norms and a three-axis rotary over
 *   (frame, row, column) at 10000 x 4^(42/40) (the frame axis is 0 for an image), a cross-attention
 *   to the 512 context rows (q/k norms, no rotary), and a GELU MLP, each behind a LayerNorm
 *   without affine modulated by (shift, scale) and added back gated, `x += gate f(norm(x)(1 + scale)
 *   + shift)`. The modulation of each of the three is Linear(256, 6144)(Linear(2048, 256)(silu(e)))
 *   plus a shared AdaLN-LoRA term, `e` the RMS-normed sinusoid of the step's sigma (a timestep in
 *   [0, 1]) and the shared term Linear(2048, 6144)(silu(Linear(2048, 2048)(sinusoid))).
 *
 *   Decoder: the Qwen-Image VAE (Wan 2.1's, its causal 3-D convolutions on one frame), 16 latent
 *   channels at 8x, read under ComfyUI's original Wan names (decoder.upsamples.N.residual.K).
 *
 * The model is a rectified flow (ComfyUI's FLOW at multiplier 1): the DiT predicts the velocity,
 * sampled by Euler steps over ComfyUI's `simple` schedule at shift 3, guided as uncond + s (cond -
 * uncond). The variant (`anima.variant` in the container, set at convert) picks the card's
 * settings: Base and Aesthetic 30 steps at guidance 4 with the card's negative prompt, Turbo 10
 * steps unguided.
 *
 * ============================== THE CHECKPOINT ==============================
 *
 * ComfyUI's single files, merged by rad-convert (--add): the DiT under `net.` (Base) or
 * `model.diffusion_model.` (Aesthetic 1.1, Turbo 1.1), the text encoder under `model.` and the VAE
 * under its Wan names. The source names are the files' own, so a LoRA's module path (`blocks.N.
 * self_attn.q_proj`) is the DiT key after its prefix. The prefix is read off the checkpoint at
 * convert, and a container records it in the name of the one weight declared under it.
 *
 * ============================== WHAT IS QUANTISED ==============================
 *
 * One plugin serves bf16 and block-fp8 containers: the text encoder's linears and the DiT blocks'
 * attention and MLP linears are LinearQs (rad_linear.h), bf16 throughout when the first DiT
 * projection is plain. The adapter, the modulation and embedding linears, the final layer and the
 * VAE stay bf16. Every linear a LoRA for Anima names is a plain capable GEMM (gemm_nt or gemm_nt_q
 * without rotation), its activation applied by a separate op.
 *
 * One card: tensor parallelism is refused.
 */
#include <arch/rad_arch.h>
#include <arch/rad_linear.h>
#include <arch/rad_vae2d.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace anima {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* The largest image: 1536 x 1536 pixels, 9216 latent tokens (the card's largest resolution).
 * 1024 x 1024 is the default. */
enum { kMaxImageTokens = 9216 };
/* The context the DiT reads: the adapter's rows zero-padded to 512, as ComfyUI pads them. A prompt
 * longer than that to T5 or Qwen3 is refused. */
enum { kCondRows = 512 };
/* Images in flight, and so the lanes of latent state. */
enum { kMaxLanes = 8 };
/* The adapter: six blocks over a 32128-row T5 embedding. */
enum { kAdLayers = 6, kT5Vocab = 32128 };
/* Qwen3's padding token: an empty prompt is this one token. */
enum { kQwenPad = 151643 };

static const char* kNegative =
    "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, "
    "chromatic aberration";

/* Wan 2.1's latent statistics (ComfyUI's latent_formats.Wan21): x std + mean in f32. */
static const double kLatMean[16] = { -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517,
                                      1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497,
                                      0.2503, -0.2921 };
static const double kLatStd[16] = { 2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                                     3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160 };

/* ================================================================== the model */
struct TeLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ  qkv{}, o{}, gate_up{}, down{};
    QuantAct   q_h{}, q_attn{}, q_h2{};
    GateQuant gq{};
    rad_op     op_n1 = 0, op_qkr = 0, op_attn = 0, op_add1 = 0, op_n2 = 0, op_add2 = 0;
};

/* An adapter block: RMS-normed self-attention over the T5 rows, cross-attention to the Qwen3 rows
 * and a GELU MLP, each added back. bf16 throughout. */
struct AdLayer {
    rad_weight w_nsa = 0, w_nca = 0, w_nmlp = 0;
    rad_weight w_sqn = 0, w_skn = 0, w_cqn = 0, w_ckn = 0;
    rad_weight w_sqkv = 0, w_so = 0, w_cq = 0, w_ckv = 0, w_co = 0;
    rad_weight w_m1 = 0, b_m1 = 0, w_m2 = 0, b_m2 = 0;
    rad_op op_nsa = 0, op_sqkv = 0, op_sqkr = 0, op_sattn = 0, op_so = 0, op_sadd = 0;
    rad_op op_nca = 0, op_cq = 0, op_ckv = 0, op_cpq = 0, op_cpkv = 0, op_cqkr = 0, op_cattn = 0;
    rad_op op_co = 0, op_cadd = 0;
    rad_op op_nmlp = 0, op_m1 = 0, op_gelu = 0, op_m2 = 0, op_madd = 0;
};

/* A DiT block's three modulations and its linears. */
struct DitLayer {
    rad_weight w_qn = 0, w_kn = 0, w_cqn = 0, w_ckn = 0;
    rad_weight w_mod1[3] = { 0, 0, 0 }, w_mod2[3] = { 0, 0, 0 };
    rad_op     op_mod1[3] = { 0, 0, 0 }, op_mod2[3] = { 0, 0, 0 }, op_modadd[3] = { 0, 0, 0 };
    LinearQ  qkv{}, o{}, cq{}, ck{}, cv{}, co{}, l1{}, l2{};
    QuantAct   q_attn{}, q_cattn{}, q_ffn{};
    rad_op     op_n1 = 0, op_qkr = 0, op_attn = 0;
    rad_op     op_n2 = 0, op_cqn = 0, op_ckn = 0, op_cattn = 0;
    rad_op     op_n3 = 0, op_gelu = 0;
};

/* Weight groups (movement units) by network. */
enum : int {
    kGrpAd     = 1000,   /* the adapter's blocks, kGrpAd + i; its model-level weights kGrpAd + 10 */
    kGrpVaeDec = 1200,
    kGrpDit    = 2000,   /* the DiT's model-level weights; its blocks kGrpDit + 1.. */
};

struct Model {
    Names nm{""};
    Geom  gt{}, ga{}, gd{};    /* the text encoder's, the adapter's and the DiT's geometry */
    int   variant = 0;         /* 0 base, 1 aesthetic, 2 turbo */
    std::string pre;           /* the DiT's checkpoint prefix */

    /* ---- the text encoder */
    rad_weight w_tok = 0, w_tnorm = 0;
    rad_op  op_embed = 0, op_tnorm = 0;
    rad_buf b_tx = 0, b_tkv = 0, b_th2 = 0, b_tgu = 0, b_src = 0;
    ActQ  a_th{}, a_tattn{}, a_tffn{};
    std::vector<TeLayer> te;

    /* ---- the adapter */
    rad_weight w_t5 = 0, w_aout = 0, b_aout = 0, w_anorm = 0;
    rad_op  op_t5 = 0, op_pos = 0, op_fill = 0, op_aout = 0, op_anorm = 0, op_ctxz = 0;
    rad_op  op_store = 0;
    rad_buf b_ax = 0, b_ah = 0, b_aqkv = 0, b_aq = 0, b_akv = 0, b_aattn = 0, b_ab = 0, b_am = 0;
    rad_buf b_apos = 0, b_ctx = 0;
    std::vector<AdLayer> ad;

    /* ---- the timestep and the modulations */
    rad_weight w_t1 = 0, w_t2 = 0, w_temb_norm = 0, w_f1 = 0, w_f2 = 0;
    rad_op  op_temb = 0, op_t1 = 0, op_tsilu = 0, op_t2 = 0, op_tnormed = 0, op_esilu = 0;
    rad_op  op_f1 = 0, op_f2 = 0, op_fadd = 0;
    rad_buf b_tproj = 0, b_t1 = 0, b_t1s = 0, b_lora = 0, b_emb = 0, b_esilu = 0, b_mh = 0;
    rad_buf b_mods = 0, b_mod1 = 0, b_fh = 0, b_fmod = 0;

    /* ---- the DiT */
    int64_t C = 64;            /* latent values a token */
    rad_weight w_xemb = 0, w_final = 0;
    rad_op  op_randn = 0, op_lgather = 0, op_xemb = 0, op_cgather = 0;
    rad_op  op_g3 = 0, op_fn = 0, op_final = 0, op_euler = 0;
    rad_buf b_lat = 0, b_h = 0, b_qkv = 0, b_cq = 0, b_ck = 0, b_cv = 0, b_ao = 0, b_l1 = 0;
    rad_buf b_fz = 0, b_v = 0;
    ActQ  a_hm{}, a_attn{}, a_ffn{}, a_ctx{};
    std::vector<DitLayer> dit;
    LinearQ qkv0{};
    QuantAct  q_ctx{};

    /* ---- the per-image state and the conditioning store (PERSIST) */
    rad_buf b_state = 0, b_prev = 0, b_store = 0;

    /* ---- the VAE */
    Vae     dec{};
    VImg    dec_out{};
    rad_buf b_dl = 0, b_decin = 0, b_cu4 = 0, b_img = 0;
    rad_op  op_dnorm = 0, op_dzero = 0, op_unpack = 0, op_cu4 = 0, op_rgb = 0;
    std::string mul, add;
};

static Model g_model[MAX_SLOTS];

/* A plain bf16 linear without a bias: gemm_nt, the activation a separate op. `src` is one
 * checkpoint name, or several stacked along dim 0. */
static rad_op dense(RadBuilder* b, const char* dn, std::initializer_list<const char*> src,
                    int64_t N, int64_t K, int64_t rows, rad_buf in, rad_buf out, rad_weight* w,
                    int access, int group, std::initializer_list<int64_t> parts = {}) {
    if (map_linear(b, dn, src) < 0) return 0;
    *w = decl_w(b, dn, RAD_BF16, { N, K }, access, RAD_SHARD_NONE, grp_layer(group), 0, parts);
    if (!*w) return 0;
    return gemm_op(b, rows, N, K, *w, in, out);
}

/* A bf16 linear with a bias. */
static rad_op dense_bias(RadBuilder* b, Names& nm, const char* dn, const char* src, int64_t N,
                         int64_t K, int64_t rows, rad_buf in, rad_buf out, rad_weight* w,
                         rad_weight* bias, int access, int group) {
    *w = wcopy(b, nm.f("%s.weight", dn), nm.ckpt("%s.weight", src), RAD_BF16, { N, K }, access,
               group);
    *bias = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), RAD_BF16, { N }, access, group);
    if (!*w || !*bias) return 0;
    return gemm_bias_op(b, rows, N, K, *w, *bias, in, out);
}

/* An RMS norm with a gain, `rows` rows of `n`. */
static rad_op rms(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight w, rad_buf x,
                  rad_buf y) {
    return rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(w)),
              { x }, { y });
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (rad_meta_geti(meta, "num_hidden_layers", 0) != 28 ||
        rad_meta_geti(meta, "hidden_size", 0) != 1024) {
        fprintf(stderr, "radiance: anima: the text encoder must be Qwen3-0.6B (28 layers of "
                        "1024); convert with --config naming the directory of Qwen3-0.6B-Base's "
                        "config.json\n");
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: anima: a 2B model served on one card; --tp %d is not "
                        "supported\n", ctx->world_size);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== declare */
/* Qwen3-0.6B over the prompt: every layer, then the final norm, into `b_src`. */
static int declare_te(RadBuilder* b, Model& m) {
    Geom& g = m.gt;
    Names& nm = m.nm;
    const int64_t T = kCondRows;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();

    RAD_ARCH_TRY(map_copy(b, nm.f("te.token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("te.token_embd.weight"), RAD_BF16, { g.n_vocab_all, g.n_embd },
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.b_tx   = decl_b(b, nm.f("te.x"), RAD_BF16, { T, g.n_embd });
    m.a_th.x = decl_b(b, nm.f("te.h"), RAD_BF16, { T, g.n_embd });
    RAD_ARCH_TRY(m.a_th.declare_qs(b, nm, g, "te.h", g.n_embd, T));
    m.b_tkv  = decl_b(b, nm.f("te.qkv"), RAD_BF16, { T, q_dim + 2 * kv_dim });
    m.a_tattn.x = decl_b(b, nm.f("te.attn"), RAD_BF16, { T, q_dim });
    RAD_ARCH_TRY(m.a_tattn.declare_qs(b, nm, g, "te.attn", q_dim, T));
    m.b_th2  = decl_b(b, nm.f("te.h2"), RAD_BF16, { T, g.n_embd });
    m.b_tgu  = decl_b(b, nm.f("te.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_tffn.x = decl_b(b, nm.f("te.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_tffn.declare_qs(b, nm, g, "te.ffn", g.n_ff, T));
    m.b_src  = decl_b(b, nm.f("te.out"), RAD_BF16, { T, g.n_embd });

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", g.n_embd),
                                         RAD_INT("n_vocab", g.n_vocab_all), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", 0)),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_tx });
    m.te.resize((size_t)g.n_layer);
    for (int l = 0; l < (int)g.n_layer; ++l) {
        TeLayer& L = m.te[(size_t)l];
        const char* lp = nm.ckpt("model.layers.%d", l);
        L.w_n1 = wcopy(b, nm.f("te.blk.%d.attn_norm.weight", l),
                       nm.ckpt("%s.input_layernorm.weight", lp), RAD_F32, { g.n_embd },
                       RAD_ACCESS_PER_TOKEN, l);
        L.w_n2 = wcopy(b, nm.f("te.blk.%d.ffn_norm.weight", l),
                       nm.ckpt("%s.post_attention_layernorm.weight", lp), RAD_F32, { g.n_embd },
                       RAD_ACCESS_PER_TOKEN, l);
        L.w_qn = wcopy(b, nm.f("te.blk.%d.q_norm.weight", l),
                       nm.ckpt("%s.self_attn.q_norm.weight", lp), RAD_BF16, { g.head_dim },
                       RAD_ACCESS_PER_TOKEN, l);
        L.w_kn = wcopy(b, nm.f("te.blk.%d.k_norm.weight", l),
                       nm.ckpt("%s.self_attn.k_norm.weight", lp), RAD_BF16, { g.head_dim },
                       RAD_ACCESS_PER_TOKEN, l);
        if (!L.w_n1 || !L.w_n2 || !L.w_qn || !L.w_kn) return RAD_E_INVAL;
        L.op_n1 = rms(b, T, g.n_embd, g.eps, L.w_n1, m.b_tx, m.a_th.x);
        RAD_ARCH_TRY(L.q_h.declare(b, g, m.a_th, g.n_embd));
        RAD_ARCH_TRY(L.qkv.declare(b, nm, g, nm.f("te.blk.%d.attn_qkv", l), q_dim + 2 * kv_dim,
                                   g.n_embd, RAD_SHARD_NONE, grp_layer(l),
                                   { nm.ckpt("%s.self_attn.q_proj.weight", lp),
                                     nm.ckpt("%s.self_attn.k_proj.weight", lp),
                                     nm.ckpt("%s.self_attn.v_proj.weight", lp) },
                                   m.a_th, m.b_tkv, T, { q_dim, kv_dim, kv_dim }));
        L.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                           RAD_INT("n_head", g.n_head),
                                           RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_F64("theta", g.theta), RAD_F64("eps", g.eps),
                                           RAD_INT("rotary_dim", g.head_dim)),
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
                                 RAD_SHARD_NONE, grp_layer(l),
                                 { nm.ckpt("%s.self_attn.o_proj.weight", lp) }, m.a_tattn, m.b_th2,
                                 T));
        L.op_add1 = add_op(b, T, g.n_embd, m.b_tx, m.b_th2, m.b_tx);
        L.op_n2 = rms(b, T, g.n_embd, g.eps, L.w_n2, m.b_tx, m.a_th.x);
        RAD_ARCH_TRY(L.q_h2.declare(b, g, m.a_th, g.n_embd));
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("te.blk.%d.ffn_gate_up", l), 2 * g.n_ff,
                                       g.n_embd, RAD_SHARD_NONE, grp_layer(l),
                                       { nm.ckpt("%s.mlp.gate_proj.weight", lp),
                                         nm.ckpt("%s.mlp.up_proj.weight", lp) },
                                       m.a_th, m.b_tgu, T, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(L.gq.declare(b, g, m.b_tgu, m.a_tffn, g.n_ff, "silu", T, true));
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("te.blk.%d.ffn_down", l), g.n_embd, g.n_ff,
                                    RAD_SHARD_NONE, grp_layer(l),
                                    { nm.ckpt("%s.mlp.down_proj.weight", lp) }, m.a_tffn, m.b_th2,
                                    T));
        L.op_add2 = add_op(b, T, g.n_embd, m.b_tx, m.b_th2, m.b_tx);
        if (!L.op_n1 || !L.op_qkr || !L.op_attn || !L.op_add1 || !L.op_n2 || !L.op_add2)
            return RAD_E_INVAL;
    }
    m.w_tnorm = wcopy(b, nm.f("te.output_norm.weight"), nm.ckpt("model.norm.weight"), RAD_F32,
                      { g.n_embd }, RAD_ACCESS_PER_TOKEN, (int)g.n_layer - 1);
    m.op_tnorm = rms(b, T, g.n_embd, g.eps, m.w_tnorm, m.b_tx, m.b_src);
    return m.op_embed && m.w_tnorm && m.op_tnorm ? RAD_OK : RAD_E_INVAL;
}

/* The LLM adapter: the T5 rows' embeddings through six blocks, the output linear and norm, into
 * the 512 context rows (zero past the prompt's) and from there into every lane's store at this
 * branch. Its cross-attention lays q (the T5 rows), k and v (the Qwen3 rows) side by side in one
 * [q | k | v] row so one qk_norm_rope rotates each at its own row index. */
static int declare_adapter(RadBuilder* b, Model& m) {
    Geom& g = m.ga;
    Names& nm = m.nm;
    const char* P = m.pre.c_str();
    const int64_t R = kCondRows, D = g.n_embd, hd = g.head_dim, F = g.n_ff_all;
    const int acc = RAD_ACCESS_PER_REQUEST;
    const double eps = 1e-6;
    const int gm = kGrpAd + 10;

    RAD_ARCH_TRY(map_copy(b, nm.f("ad.embed.weight"), nm.ckpt("%sllm_adapter.embed.weight", P)));
    m.w_t5 = decl_w(b, nm.f("ad.embed.weight"), RAD_BF16, { kT5Vocab, D }, acc, RAD_SHARD_NONE,
                    grp_layer(gm));
    m.b_ax    = decl_b(b, nm.f("ad.x"), RAD_BF16, { R, D });
    m.b_ah    = decl_b(b, nm.f("ad.h"), RAD_BF16, { R, D });
    m.b_aqkv  = decl_b(b, nm.f("ad.qkv"), RAD_BF16, { R, 3 * D });
    m.b_aq    = decl_b(b, nm.f("ad.q"), RAD_BF16, { R, D });
    m.b_akv   = decl_b(b, nm.f("ad.kv"), RAD_BF16, { R, 2 * D });
    m.b_aattn = decl_b(b, nm.f("ad.attn"), RAD_BF16, { R, D });
    m.b_ab    = decl_b(b, nm.f("ad.branch"), RAD_BF16, { R, D });
    m.b_am    = decl_b(b, nm.f("ad.mlp"), RAD_BF16, { R, F });
    m.b_apos  = decl_b(b, nm.f("ad.pos"), RAD_I32, { R });
    m.b_ctx   = decl_b(b, nm.f("ad.ctx"), RAD_BF16, { R, D });

    m.op_t5 = rw(b, RAD_OP(b, "embed_lookup",
                           RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n_embd", D),
                                      RAD_INT("n_vocab", kT5Vocab), RAD_STR("dtype", "bf16"),
                                      RAD_INT("vocab_offset", 0)),
                           RAD_WEIGHTS(m.w_t5)),
                 {}, { m.b_ax });
    m.op_pos = rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("start", 0),
                                       RAD_INT("step", 1)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_apos });
    /* The [q | k | v] rows past either side's count are read by the norm and rotary and never by
     * the attention; zeroed once so they are finite. */
    m.op_fill = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", 3 * D),
                                        RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_aqkv });
    if (!m.w_t5 || !m.op_t5 || !m.op_pos || !m.op_fill) return RAD_E_INVAL;

    auto qk_rope = [&](rad_weight qn, rad_weight kn) {
        return rw(b, RAD_OP(b, "qk_norm_rope",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                       RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head),
                                       RAD_F64("theta", 10000.0), RAD_F64("eps", eps),
                                       RAD_INT("rotary_dim", hd)),
                            RAD_WEIGHTS(qn, kn)),
                  { m.b_aqkv, m.b_apos }, { m.b_aqkv });
    };
    auto gather = [&](rad_buf src, rad_buf dst) {
        return rw(b, RAD_OP(b, "gather_rows2",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { src, m.b_apos, dst }, { dst });
    };

    m.ad.resize(kAdLayers);
    for (int i = 0; i < kAdLayers; ++i) {
        AdLayer& L = m.ad[(size_t)i];
        const int grp = kGrpAd + i;
        const char* pre = nm.ckpt("%sllm_adapter.blocks.%d", P, i);
        const char* dn = nm.f("ad.blk.%d", i);
        L.w_nsa  = wcopy(b, nm.f("%s.norm_sa.weight", dn), nm.ckpt("%s.norm_self_attn.weight", pre),
                         RAD_BF16, { D }, acc, grp);
        L.w_nca  = wcopy(b, nm.f("%s.norm_ca.weight", dn), nm.ckpt("%s.norm_cross_attn.weight", pre),
                         RAD_BF16, { D }, acc, grp);
        L.w_nmlp = wcopy(b, nm.f("%s.norm_mlp.weight", dn), nm.ckpt("%s.norm_mlp.weight", pre),
                         RAD_BF16, { D }, acc, grp);
        L.w_sqn = wcopy(b, nm.f("%s.sa_q_norm.weight", dn),
                        nm.ckpt("%s.self_attn.q_norm.weight", pre), RAD_BF16, { hd }, acc, grp);
        L.w_skn = wcopy(b, nm.f("%s.sa_k_norm.weight", dn),
                        nm.ckpt("%s.self_attn.k_norm.weight", pre), RAD_BF16, { hd }, acc, grp);
        L.w_cqn = wcopy(b, nm.f("%s.ca_q_norm.weight", dn),
                        nm.ckpt("%s.cross_attn.q_norm.weight", pre), RAD_BF16, { hd }, acc, grp);
        L.w_ckn = wcopy(b, nm.f("%s.ca_k_norm.weight", dn),
                        nm.ckpt("%s.cross_attn.k_norm.weight", pre), RAD_BF16, { hd }, acc, grp);
        if (!L.w_nsa || !L.w_nca || !L.w_nmlp || !L.w_sqn || !L.w_skn || !L.w_cqn || !L.w_ckn)
            return RAD_E_INVAL;
        /* ---- self-attention over the T5 rows */
        L.op_nsa = rms(b, R, D, eps, L.w_nsa, m.b_ax, m.b_ah);
        L.op_sqkv = dense(b, nm.f("%s.sa_qkv.weight", dn),
                          { nm.ckpt("%s.self_attn.q_proj.weight", pre),
                            nm.ckpt("%s.self_attn.k_proj.weight", pre),
                            nm.ckpt("%s.self_attn.v_proj.weight", pre) },
                          3 * D, D, R, m.b_ah, m.b_aqkv, &L.w_sqkv, acc, grp, { D, D, D });
        L.op_sqkr = qk_rope(L.w_sqn, L.w_skn);
        L.op_sattn = rw(b, RAD_OP(b, "attn_dense",
                                  RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                             RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                             RAD_STR("dtype", "bf16"),
                                             RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                  RAD_NOWEIGHTS),
                        { m.b_aqkv }, { m.b_aattn });
        L.op_so = dense(b, nm.f("%s.sa_o.weight", dn), { nm.ckpt("%s.self_attn.o_proj.weight", pre) },
                        D, D, R, m.b_aattn, m.b_ab, &L.w_so, acc, grp);
        L.op_sadd = add_op(b, R, D, m.b_ax, m.b_ab, m.b_ax);
        /* ---- cross-attention to the Qwen3 rows */
        L.op_nca = rms(b, R, D, eps, L.w_nca, m.b_ax, m.b_ah);
        L.op_cq = dense(b, nm.f("%s.ca_q.weight", dn), { nm.ckpt("%s.cross_attn.q_proj.weight", pre) },
                        D, D, R, m.b_ah, m.b_aq, &L.w_cq, acc, grp);
        L.op_ckv = dense(b, nm.f("%s.ca_kv.weight", dn),
                         { nm.ckpt("%s.cross_attn.k_proj.weight", pre),
                           nm.ckpt("%s.cross_attn.v_proj.weight", pre) },
                         2 * D, m.gt.n_embd, R, m.b_src, m.b_akv, &L.w_ckv, acc, grp, { D, D });
        L.op_cpq = gather(m.b_aq, m.b_aqkv);
        L.op_cpkv = rw(b, RAD_OP(b, "gather_rows2",
                                 RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", 2 * D),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.b_akv, m.b_apos, m.b_aqkv }, { m.b_aqkv });
        L.op_cqkr = qk_rope(L.w_cqn, L.w_ckn);
        L.op_cattn = rw(b, RAD_OP(b, "attn_cross",
                                  RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                             RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                             RAD_STR("dtype", "bf16"),
                                             RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                  RAD_NOWEIGHTS),
                        { m.b_aqkv }, { m.b_aattn });
        L.op_co = dense(b, nm.f("%s.ca_o.weight", dn), { nm.ckpt("%s.cross_attn.o_proj.weight", pre) },
                        D, D, R, m.b_aattn, m.b_ab, &L.w_co, acc, grp);
        L.op_cadd = add_op(b, R, D, m.b_ax, m.b_ab, m.b_ax);
        /* ---- the MLP */
        L.op_nmlp = rms(b, R, D, eps, L.w_nmlp, m.b_ax, m.b_ah);
        L.op_m1 = dense_bias(b, nm, nm.f("%s.mlp1", dn), nm.ckpt("%s.mlp.0", pre), F, D, R, m.b_ah,
                             m.b_am, &L.w_m1, &L.b_m1, acc, grp);
        L.op_gelu = unary(b, "gelu", R, F, m.b_am, m.b_am);
        L.op_m2 = dense_bias(b, nm, nm.f("%s.mlp2", dn), nm.ckpt("%s.mlp.2", pre), D, F, R, m.b_am,
                             m.b_ab, &L.w_m2, &L.b_m2, acc, grp);
        L.op_madd = add_op(b, R, D, m.b_ax, m.b_ab, m.b_ax);
        if (!L.op_nsa || !L.op_sqkv || !L.op_sqkr || !L.op_sattn || !L.op_so || !L.op_sadd ||
            !L.op_nca || !L.op_cq || !L.op_ckv || !L.op_cpq || !L.op_cpkv || !L.op_cqkr ||
            !L.op_cattn || !L.op_co || !L.op_cadd || !L.op_nmlp || !L.op_m1 || !L.op_gelu ||
            !L.op_m2 || !L.op_madd)
            return RAD_E_INVAL;
    }
    m.op_aout = dense_bias(b, nm, nm.f("ad.out"), nm.ckpt("%sllm_adapter.out_proj", P), D, D, R,
                           m.b_ax, m.b_ah, &m.w_aout, &m.b_aout, acc, gm);
    m.w_anorm = wcopy(b, nm.f("ad.norm.weight"), nm.ckpt("%sllm_adapter.norm.weight", P), RAD_BF16,
                      { D }, acc, gm);
    m.op_ctxz = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D),
                                        RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_ctx });
    m.op_anorm = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D),
                                         RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.w_anorm)),
                    { m.b_ah, m.b_ctx }, { m.b_ctx });
    /* Lane l's branch b at rows (2 l + b) kCondRows: the store's `stride` is two branches. */
    m.op_store = rw(b, RAD_OP(b, "lane_store",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D),
                                         RAD_INT("stride", 2 * (int64_t)kCondRows),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_ctx, m.b_store }, { m.b_store });
    return m.op_aout && m.w_anorm && m.op_ctxz && m.op_anorm && m.op_store ? RAD_OK : RAD_E_INVAL;
}

/* The final linear's rows reordered from the DiT's (row, column, channel) unpatchify order to the
 * lane state's channel-major one: state value c 4 + k is output row k 16 + c. Eight ROWS maps of
 * eight one-row pieces, joined in call order. */
static int map_final(RadBuilder* b, const char* declared, const char* src, int64_t C) {
    const int64_t ch = C / 4;
    for (int64_t t0 = 0; t0 < C; t0 += 8) {
        RadNameMap mp = map_base(declared, RAD_MAP_ROWS);
        for (int64_t t = t0; t < t0 + 8 && t < C; ++t) {
            const int64_t c = t / 4, k = t % 4;
            mp.src[mp.n_src] = src;
            mp.src_index[mp.n_src] = (int)(k * ch + c);
            mp.src_rows[mp.n_src] = 1;
            ++mp.n_src;
        }
        RAD_ARCH_TRY(rad_decl_name_map(b, &mp));
    }
    return RAD_OK;
}

/* The step's timestep embedding and every modulation: the shared AdaLN-LoRA term, the normed
 * embedding, each block's three (shift | scale | gate) rows into its columns of `mods`, and the
 * final layer's (shift | scale). All `S` rows, one an entry. */
static int declare_mod(RadBuilder* b, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const char* P = m.pre.c_str();
    const int64_t S = g.max_seqs, D = g.n_embd, L = g.n_layer, M3 = 3 * D;
    const int64_t R = 256;   /* AdaLN-LoRA's rank */
    const int acc = RAD_ACCESS_PER_TOKEN;
    const int gx = kGrpDit;
    m.b_tproj = decl_b(b, nm.f("dit.tproj"), RAD_BF16, { S, D });
    m.b_t1    = decl_b(b, nm.f("dit.t1"), RAD_BF16, { S, D });
    m.b_t1s   = decl_b(b, nm.f("dit.t1s"), RAD_BF16, { S, D });
    m.b_lora  = decl_b(b, nm.f("dit.adaln_lora"), RAD_BF16, { S, M3 });
    m.b_emb   = decl_b(b, nm.f("dit.temb"), RAD_BF16, { S, D });
    m.b_esilu = decl_b(b, nm.f("dit.temb_silu"), RAD_BF16, { S, D });
    m.b_mh    = decl_b(b, nm.f("dit.mod_h"), RAD_BF16, { S, R });
    m.b_mods  = decl_b(b, nm.f("dit.mods"), RAD_BF16, { S, L * 3 * M3 });
    m.b_fh    = decl_b(b, nm.f("dit.fmod_raw"), RAD_BF16, { S, 2 * D });
    m.b_fmod  = decl_b(b, nm.f("dit.fmod"), RAD_BF16, { S, 2 * D });
    m.b_mod1  = decl_b(b, nm.f("dit.mod_one"), RAD_BF16, { S, M3 });
    const rad_buf mod1 = m.b_mod1;

    m.op_temb = rw(b, RAD_OP(b, "timestep_embed",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", D),
                                        RAD_F64("scale", 1.0), RAD_F64("max_period", 10000.0),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   {}, { m.b_tproj });
    m.op_t1 = dense(b, nm.f("dit.t_embed1.weight"), { nm.ckpt("%st_embedder.1.linear_1.weight", P) },
                    D, D, S, m.b_tproj, m.b_t1, &m.w_t1, acc, gx);
    m.op_tsilu = unary(b, "silu", S, D, m.b_t1, m.b_t1s);
    m.op_t2 = dense(b, nm.f("dit.t_embed2.weight"), { nm.ckpt("%st_embedder.1.linear_2.weight", P) },
                    M3, D, S, m.b_t1s, m.b_lora, &m.w_t2, acc, gx);
    m.op_tnormed = rms(b, S, D, 1e-6, m.w_temb_norm, m.b_tproj, m.b_emb);
    m.op_esilu = unary(b, "silu", S, D, m.b_emb, m.b_esilu);
    if (!m.op_temb || !m.op_t1 || !m.op_tsilu || !m.op_t2 || !m.op_tnormed || !m.op_esilu)
        return RAD_E_INVAL;

    static const char* kMods[3] = { "self_attn", "cross_attn", "mlp" };
    for (int64_t l = 0; l < L; ++l) {
        DitLayer& B = m.dit[(size_t)l];
        const int grp = kGrpDit + 1 + (int)l;
        for (int k = 0; k < 3; ++k) {
            const char* src = nm.ckpt("%sblocks.%d.adaln_modulation_%s", P, (int)l, kMods[k]);
            B.op_mod1[k] = dense(b, nm.f("dit.blk.%d.mod_%s.1.weight", (int)l, kMods[k]),
                                 { nm.ckpt("%s.1.weight", src) }, R, D, S, m.b_esilu, m.b_mh,
                                 &B.w_mod1[k], acc, grp);
            B.op_mod2[k] = dense(b, nm.f("dit.blk.%d.mod_%s.2.weight", (int)l, kMods[k]),
                                 { nm.ckpt("%s.2.weight", src) }, M3, R, S, m.b_mh, mod1,
                                 &B.w_mod2[k], acc, grp);
            B.op_modadd[k] = rw(b, RAD_OP(b, "add",
                                          RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", M3),
                                                     RAD_STR("dtype", "bf16")),
                                          RAD_NOWEIGHTS),
                                { mod1, m.b_lora }, { m.b_mods });
            if (!B.op_mod1[k] || !B.op_mod2[k] || !B.op_modadd[k]) return RAD_E_INVAL;
        }
    }
    m.op_f1 = dense(b, nm.f("dit.final.mod.1.weight"),
                    { nm.ckpt("%sfinal_layer.adaln_modulation.1.weight", P) }, R, D, S, m.b_esilu,
                    m.b_mh, &m.w_f1, acc, gx);
    m.op_f2 = dense(b, nm.f("dit.final.mod.2.weight"),
                    { nm.ckpt("%sfinal_layer.adaln_modulation.2.weight", P) }, 2 * D, R, S, m.b_mh,
                    m.b_fh, &m.w_f2, acc, gx);
    m.op_fadd = add_op(b, S, 2 * D, m.b_fh, m.b_lora, m.b_fmod);
    return m.op_f1 && m.op_f2 && m.op_fadd ? RAD_OK : RAD_E_INVAL;
}

static int declare_dit(RadBuilder* b, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const char* P = m.pre.c_str();
    const int64_t T = g.max_tok;
    const int64_t D = g.n_embd, hd = g.head_dim, qd = g.q_dim(), C = m.C, L = g.n_layer;
    const int64_t Dc = m.ga.n_embd;   /* the context's width */
    const double eps = 1e-6;
    const int gx = kGrpDit;
    const int64_t NI = std::min<int64_t>(T, 2 * (int64_t)kMaxImageTokens);

    /* ---- the input rows: the images' latents a lane at a time through the patch embedding, and
     * every entry's context rows out of the store */
    m.b_lat = decl_b(b, nm.f("dit.lat"), RAD_BF16, { NI, C });
    m.b_h   = decl_b(b, nm.f("dit.h"), RAD_BF16, { T, D });
    m.op_randn = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                         RAD_INT("stride", (int64_t)kMaxImageTokens),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_state }, { m.b_state });
    m.op_lgather = rw(b, RAD_OP(b, "lane_gather",
                                RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                           RAD_INT("stride", (int64_t)kMaxImageTokens),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_state }, { m.b_lat });
    {
        const char* dn = nm.f("dit.x_embed.weight");
        RadNameMap mp = map_base(dn, RAD_MAP_COLS);
        mp.n_src = 1;
        mp.src[0] = nm.ckpt("%sx_embedder.proj.1.weight", P);
        mp.src_index[0] = 0;
        mp.src_rows[0] = C;
        RAD_ARCH_TRY(rad_decl_name_map(b, &mp));
        m.w_xemb = decl_w(b, dn, RAD_BF16, { D, C }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                          grp_layer(gx));
        if (!m.w_xemb) return RAD_E_INVAL;
        m.op_xemb = rw(b, RAD_OP(b, "gemm_nt",
                                 RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("N", D), RAD_INT("K", C),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(m.w_xemb)),
                       { m.b_lat }, { m.b_h });
    }
    m.a_ctx.x = decl_b(b, nm.f("dit.ctx"), RAD_BF16, { T, Dc });
    RAD_ARCH_TRY(m.a_ctx.declare_qs(b, nm, g, "dit.ctx", Dc));
    m.op_cgather = rw(b, RAD_OP(b, "gather_rows2",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", Dc),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_store }, { m.a_ctx.x });
    RAD_ARCH_TRY(m.q_ctx.declare(b, g, m.a_ctx, Dc));
    if (!m.op_randn || !m.op_lgather || !m.op_xemb || !m.op_cgather) return RAD_E_INVAL;

    /* ---- the blocks */
    m.a_hm.x = decl_b(b, nm.f("dit.hm"), RAD_BF16, { T, D });
    RAD_ARCH_TRY(m.a_hm.declare_qs(b, nm, g, "dit.hm", D));
    m.b_qkv = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { T, 3 * qd });
    m.a_attn.x = decl_b(b, nm.f("dit.attn"), RAD_BF16, { T, qd });
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "dit.attn", qd));
    m.b_ao = decl_b(b, nm.f("dit.ao"), RAD_BF16, { T, D });
    m.b_cq = decl_b(b, nm.f("dit.cq"), RAD_BF16, { T, qd });
    m.b_ck = decl_b(b, nm.f("dit.ck"), RAD_BF16, { T, qd });
    m.b_cv = decl_b(b, nm.f("dit.cv"), RAD_BF16, { T, qd });
    m.b_l1 = decl_b(b, nm.f("dit.l1"), RAD_BF16, { T, g.n_ff });
    m.a_ffn.x = decl_b(b, nm.f("dit.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "dit.ffn", g.n_ff));

    /* Cosmos's rotary: of the 64 frequency pairs 22 turn with the frame, 21 with the row and 21
     * with the column, NeoX pairs over the head; the row and column ladders at 10000 scaled by
     * the extrapolation ratio 4 to the power 42 / 40. The frame is 0 for an image, so its
     * section's base does not matter. */
    const int64_t dim_h = hd / 6 * 2, dim_t = hd - 2 * dim_h;
    const char* sections = nm.f("%lld %lld %lld", (long long)dim_t / 2, (long long)dim_h / 2,
                                (long long)dim_h / 2);
    const double theta = 10000.0 * std::pow(4.0, (double)dim_h / (double)(dim_h - 2));

    /* The modulated LayerNorm (no affine), gated by the branch before it when `gated`, into the
     * next linear's input: codes for fp8, the bf16 plane for bf16. */
    auto mod_norm = [&](bool gated) {
        const bool q = !g.w_bf16;
        rad_op op = q ? RAD_OP(b, "rms_mod_quant_fp8",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_F64("eps", eps),
                                          RAD_INT("group", RAD_SCALE_GROUP),
                                          RAD_INT("gated", gated ? 1 : 0), RAD_STR("dtype", "bf16"),
                                          RAD_STR("norm", "layer")),
                               RAD_NOWEIGHTS)
                      : RAD_OP(b, "rms_mod",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_F64("eps", eps),
                                          RAD_INT("gated", gated ? 1 : 0), RAD_STR("dtype", "bf16"),
                                          RAD_STR("norm", "layer")),
                               RAD_NOWEIGHTS);
        if (!op) return op;
        rad_buf rd[3] = { m.b_mods, m.b_h, m.b_ao };
        rad_buf wq[3] = { m.a_hm.q, m.a_hm.s, m.b_h };
        rad_buf wb[2] = { m.a_hm.x, m.b_h };
        rad_op_reads(b, op, rd, gated ? 3 : 2);
        if (q) rad_op_writes(b, op, wq, 3);
        else   rad_op_writes(b, op, wb, 2);
        return op;
    };
    /* A per-head RMS norm in place: [rows, heads * hd] read as [rows heads, hd]. */
    auto head_norm = [&](rad_weight w, rad_buf x) {
        return rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, T * g.n_head), RAD_INT("n", hd),
                                       RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w)),
                  { x }, { x });
    };

    for (int64_t l = 0; l < L; ++l) {
        DitLayer& B = m.dit[(size_t)l];
        const int li = (int)l;
        const int grp = kGrpDit + 1 + li;
        const char* pre = nm.ckpt("%sblocks.%d", P, li);
        const char* dn = nm.f("dit.blk.%d", li);
        B.w_qn  = wcopy(b, nm.f("%s.q_norm.weight", dn), nm.ckpt("%s.self_attn.q_norm.weight", pre),
                        RAD_BF16, { hd }, RAD_ACCESS_PER_TOKEN, grp);
        B.w_kn  = wcopy(b, nm.f("%s.k_norm.weight", dn), nm.ckpt("%s.self_attn.k_norm.weight", pre),
                        RAD_BF16, { hd }, RAD_ACCESS_PER_TOKEN, grp);
        B.w_cqn = wcopy(b, nm.f("%s.cq_norm.weight", dn),
                        nm.ckpt("%s.cross_attn.q_norm.weight", pre), RAD_BF16, { hd },
                        RAD_ACCESS_PER_TOKEN, grp);
        B.w_ckn = wcopy(b, nm.f("%s.ck_norm.weight", dn),
                        nm.ckpt("%s.cross_attn.k_norm.weight", pre), RAD_BF16, { hd },
                        RAD_ACCESS_PER_TOKEN, grp);
        if (!B.w_qn || !B.w_kn || !B.w_cqn || !B.w_ckn) return RAD_E_INVAL;

        /* ---- self-attention */
        B.op_n1 = mod_norm(l > 0);
        if (l == 0) B.qkv = m.qkv0;
        RAD_ARCH_TRY(B.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 3 * qd, D, RAD_SHARD_NONE,
                                   grp_layer(grp),
                                   { nm.ckpt("%s.self_attn.q_proj.weight", pre),
                                     nm.ckpt("%s.self_attn.k_proj.weight", pre),
                                     nm.ckpt("%s.self_attn.v_proj.weight", pre) },
                                   m.a_hm, m.b_qkv, 0, { qd, qd, qd }));
        B.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head),
                                           RAD_F64("theta", theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "axial"), RAD_INT("rotary_dim", hd),
                                           RAD_STR("sections", sections)),
                                RAD_WEIGHTS(B.w_qn, B.w_kn)),
                      { m.b_qkv }, { m.b_qkv });
        B.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                 RAD_NOWEIGHTS),
                       { m.b_qkv }, { m.a_attn.x });
        RAD_ARCH_TRY(B.q_attn.declare(b, g, m.a_attn, qd));
        RAD_ARCH_TRY(B.o.declare(b, nm, g, nm.f("%s.attn_output", dn), D, qd, RAD_SHARD_NONE,
                                 grp_layer(grp), { nm.ckpt("%s.self_attn.output_proj.weight", pre) },
                                 m.a_attn, m.b_ao));

        /* ---- cross-attention to the context */
        B.op_n2 = mod_norm(true);
        RAD_ARCH_TRY(B.cq.declare(b, nm, g, nm.f("%s.cross_q", dn), qd, D, RAD_SHARD_NONE,
                                  grp_layer(grp), { nm.ckpt("%s.cross_attn.q_proj.weight", pre) },
                                  m.a_hm, m.b_cq));
        RAD_ARCH_TRY(B.ck.declare(b, nm, g, nm.f("%s.cross_k", dn), qd, Dc, RAD_SHARD_NONE,
                                  grp_layer(grp), { nm.ckpt("%s.cross_attn.k_proj.weight", pre) },
                                  m.a_ctx, m.b_ck));
        RAD_ARCH_TRY(B.cv.declare(b, nm, g, nm.f("%s.cross_v", dn), qd, Dc, RAD_SHARD_NONE,
                                  grp_layer(grp), { nm.ckpt("%s.cross_attn.v_proj.weight", pre) },
                                  m.a_ctx, m.b_cv));
        B.op_cqn = head_norm(B.w_cqn, m.b_cq);
        B.op_ckn = head_norm(B.w_ckn, m.b_ck);
        B.op_cattn = rw(b, RAD_OP(b, "attn_cross",
                                  RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                             RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                             RAD_STR("dtype", "bf16"),
                                             RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                  RAD_NOWEIGHTS),
                        { m.b_cq, m.b_ck, m.b_cv }, { m.a_attn.x });
        RAD_ARCH_TRY(B.q_cattn.declare(b, g, m.a_attn, qd));
        RAD_ARCH_TRY(B.co.declare(b, nm, g, nm.f("%s.cross_output", dn), D, qd, RAD_SHARD_NONE,
                                  grp_layer(grp), { nm.ckpt("%s.cross_attn.output_proj.weight", pre) },
                                  m.a_attn, m.b_ao));

        /* ---- the MLP: layer1, an exact GELU, layer2 */
        B.op_n3 = mod_norm(true);
        RAD_ARCH_TRY(B.l1.declare(b, nm, g, nm.f("%s.mlp_layer1", dn), g.n_ff, D, RAD_SHARD_NONE,
                                  grp_layer(grp), { nm.ckpt("%s.mlp.layer1.weight", pre) }, m.a_hm,
                                  m.b_l1));
        B.op_gelu = unary(b, "gelu", T, g.n_ff, m.b_l1, m.a_ffn.x);
        RAD_ARCH_TRY(B.q_ffn.declare(b, g, m.a_ffn, g.n_ff));
        RAD_ARCH_TRY(B.l2.declare(b, nm, g, nm.f("%s.mlp_layer2", dn), D, g.n_ff, RAD_SHARD_NONE,
                                  grp_layer(grp), { nm.ckpt("%s.mlp.layer2.weight", pre) }, m.a_ffn,
                                  m.b_ao));
        if (!B.op_n1 || !B.op_qkr || !B.op_attn || !B.op_n2 || !B.op_cqn || !B.op_ckn ||
            !B.op_cattn || !B.op_n3 || !B.op_gelu)
            return RAD_E_INVAL;
    }

    /* ---- the output: the last MLP gated in, the final modulated norm and linear, the sampler's
     * step */
    m.b_fz = decl_b(b, nm.f("dit.fz"), RAD_BF16, { NI, D });
    m.b_v  = decl_b(b, nm.f("dit.v"), RAD_BF16, { NI, C });
    m.op_g3 = rw(b, RAD_OP(b, "gate_add",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                      RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                           RAD_NOWEIGHTS),
                 { m.b_ao, m.b_mods, m.b_h }, { m.b_h });
    m.op_fn = rw(b, RAD_OP(b, "rms_mod",
                           RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", D), RAD_F64("eps", eps),
                                      RAD_INT("gated", 0), RAD_STR("dtype", "bf16"),
                                      RAD_STR("norm", "layer")),
                           RAD_NOWEIGHTS),
                 { m.b_fmod, m.b_h }, { m.b_h, m.b_fz });
    {
        const char* dn = nm.f("dit.final.weight");
        RAD_ARCH_TRY(map_final(b, dn, nm.ckpt("%sfinal_layer.linear.weight", P), C));
        m.w_final = decl_w(b, dn, RAD_BF16, { C, D }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                           grp_layer(gx));
        if (!m.w_final) return RAD_E_INVAL;
        m.op_final = rw(b, RAD_OP(b, "gemm_nt",
                                  RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("N", C), RAD_INT("K", D),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(m.w_final)),
                        { m.b_fz }, { m.b_v });
    }
    m.op_euler = rw(b, RAD_OP(b, "flow_euler",
                              RAD_PARAMS(RAD_RANGE("M", 1, NI), RAD_INT("n", C),
                                         RAD_INT("stride", (int64_t)kMaxImageTokens),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_v, m.b_state, m.b_prev }, { m.b_state, m.b_prev });
    return m.op_g3 && m.op_fn && m.op_final && m.op_euler ? RAD_OK : RAD_E_INVAL;
}

/* Wan 2.1's residual block under its own names: residual.0 (norm), .2 (conv), .3 (norm), .6
 * (conv), plus the input or its 1x1x1 `shortcut`. */
static VImg wan_resblock(Vae& v, const VImg& x, int64_t out, const char* ckpt) {
    Names& nm = *v.nm;
    VImg h = x;
    if (x.ch != out) h = v.conv(x, out, 1, 1, 1, "same", nm.f("%s.shortcut", ckpt));
    VImg t = v.norm(x, true, nm.f("%s.residual.0", ckpt), 4);
    t = v.conv(t, out, 3, 1, 1, "same", nm.f("%s.residual.2", ckpt));
    t = v.norm(t, true, nm.f("%s.residual.3", ckpt), 4);
    t = v.conv(t, out, 3, 1, 1, "same", nm.f("%s.residual.6", ckpt));
    return v.add(t, h);
}

/* The decoder: the latent denormalised and unpacked to the VAE's 16 channels, then Wan 2.1's
 * decoder on one frame. Its residual, head, conv1 and the top-level conv2 (post_quant_conv)
 * convolutions are causal 3-D ones read at their last tap; the attention's and the upsamplers'
 * are 2-D. */
static int declare_decoder(RadBuilder* b, Model& m) {
    Vae& v = m.dec;
    Names& nm = m.nm;
    v.nm = &nm;
    v.b = b;
    v.scope = "vdec";
    v.down = false;
    v.first_group = kGrpVaeDec;
    v.base_px = (int64_t)kMaxImageTokens * 4;
    v.form = VAE_CONV_CAUSAL;
    const int64_t zd = 16, base = 96;
    const double mult[4] = { 1, 2, 4, 4 };
    /* The packed channel c 4 + k reads latent channel c: the statistics repeated four times. */
    for (int64_t k = 0; k < 4 * zd; ++k) {
        char t[64];
        std::snprintf(t, sizeof t, "%s%.9g", k ? " " : "", kLatStd[k / 4]);
        m.mul += t;
        std::snprintf(t, sizeof t, "%s%.9g", k ? " " : "", kLatMean[k / 4]);
        m.add += t;
    }
    const int64_t NT = kMaxImageTokens;
    m.b_dl    = decl_b(b, nm.f("vdec.lat"), RAD_BF16, { NT, 4 * zd });
    m.b_decin = decl_b(b, nm.f("vdec.in"), RAD_BF16, { 4 * NT, zd });
    m.b_cu4   = decl_b(b, nm.f("vdec.cu"), RAD_I32, { 2 });
    /* ComfyUI's Wan21.process_out on the f32 latent: x std + mean in f32, then bf16. */
    m.op_dnorm = rw(b, RAD_OP(b, "chan_affine",
                              RAD_PARAMS(RAD_RANGE("M", 1, NT), RAD_INT("n", 4 * zd),
                                         RAD_STR("mul", m.mul.c_str()), RAD_STR("add", m.add.c_str()),
                                         RAD_STR("mode", "fma"), RAD_STR("dtype", "bf16"),
                                         RAD_INT("round", 0)),
                              RAD_NOWEIGHTS),
                    { m.b_state, m.b_prev }, { m.b_dl });
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
                     { m.b_dl, m.b_decin }, { m.b_decin });
    m.op_cu4 = rw(b, RAD_OP(b, "cu_repeat",
                            RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", 4)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_cu4 });
    if (!m.op_dnorm || !m.op_dzero || !m.op_unpack || !m.op_cu4) return RAD_E_INVAL;

    std::vector<int64_t> dims{ base * (int64_t)mult[3] };
    for (int i = 3; i >= 0; --i) dims.push_back(base * (int64_t)mult[i]);
    VImg x{ m.b_decin, zd, 0 };
    x = v.conv(x, zd, 1, 1, 1, "same", "conv2");
    x = v.conv(x, dims[0], 3, 1, 1, "same", "decoder.conv1");
    x = wan_resblock(v, x, dims[0], "decoder.middle.0");
    v.form = VAE_CONV_2D;
    x = v.attention(x, "decoder.middle.1");
    v.form = VAE_CONV_CAUSAL;
    x = wan_resblock(v, x, dims[0], "decoder.middle.2");
    int idx = 0;
    for (int i = 0; i < 4; ++i) {
        for (int r = 0; r < 3; ++r)
            x = wan_resblock(v, x, dims[(size_t)i + 1], nm.f("decoder.upsamples.%d", idx++));
        if (i != 3) {
            v.form = VAE_CONV_2D;
            x = v.conv(x, dims[(size_t)i + 1] / 2, 3, 1, 2, "same",
                       nm.f("decoder.upsamples.%d.resample.1", idx++));
            v.form = VAE_CONV_CAUSAL;
        }
    }
    x = v.norm(x, true, "decoder.head.0", 4);
    m.dec_out = v.conv(x, 3, 3, 1, 1, "same", "decoder.head.2");
    if (!m.dec_out.buf) return RAD_E_INVAL;
    RAD_ARCH_TRY(v.attn_reads(m.b_cu4));
    const int64_t px = (int64_t)kMaxImageTokens * 256;
    m.b_img = decl_b(b, nm.f("image_out"), RAD_U8, { px * 3 }, RAD_BUF_PERSIST);
    m.op_rgb = rw(b, RAD_OP(b, "to_rgba8",
                            RAD_PARAMS(RAD_RANGE("M", 1, px), RAD_INT("n", 3),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { m.dec_out.buf }, { m.b_img });
    return m.op_rgb ? RAD_OK : RAD_E_INVAL;
}

/* Which of the release's key spellings the DiT is under: `net.` (Base) or `model.diffusion_model.`
 * (Aesthetic and Turbo 1.1). The checkpoint answers at convert; a container, by which marker name
 * it holds the timestep norm's gain under (declared under the prefix it was read with). */
static int find_prefix(RadBuilder* b, Model& m, const char** marker) {
    static const char* kPre[2] = { "net.", "model.diffusion_model." };
    static const char* kTag[2] = { "dit.t_norm.net.weight", "dit.t_norm.mdm.weight" };
    int found = -1;
    for (int i = 0; i < 2; ++i) {
        RAD_ARCH_TRY(map_copy(b, m.nm.f("%s", kTag[i]),
                              m.nm.ckpt("%st_embedding_norm.weight", kPre[i])));
        RadEncoding e{};
        if (weight_enc(b, m.nm.f("%s", kTag[i]), &e)) found = found < 0 ? i : 2;
    }
    if (found < 0 || found > 1) {
        fprintf(stderr, "radiance: anima: the DiT's keys are under neither `net.` nor "
                        "`model.diffusion_model.`, or under both\n");
        return RAD_E_UNSUPPORTED;
    }
    m.pre = kPre[found];
    *marker = kTag[found];
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta, ctx));
    RAD_ARCH_TRY(geom_from(m.gt, meta, ctx, "bf16", RAD_BF16));
    Names& nm = m.nm;
    const char* variant = rad_meta_gets(meta, "anima.variant", "base");
    m.variant = std::strcmp(variant, "turbo") == 0 ? 2 : std::strcmp(variant, "aesthetic") == 0 ? 1
              : 0;
    if (m.variant == 0 && std::strcmp(variant, "base") != 0) {
        fprintf(stderr, "radiance: anima: anima.variant is '%s'; base, aesthetic or turbo\n",
                variant);
        return RAD_E_INVAL;
    }
    const char* marker = nullptr;
    RAD_ARCH_TRY(find_prefix(b, m, &marker));

    /* ---- the adapter's and the DiT's geometry */
    Geom& ga = m.ga;
    ga = m.gt;
    ga.n_embd = 1024;
    ga.n_head = ga.n_head_all = ga.n_head_kv = ga.n_head_kv_all = 16;
    ga.head_dim = 64;
    ga.n_ff = ga.n_ff_all = 4096;
    ga.w_bf16 = true;
    Geom& gd = m.gd;
    gd = m.gt;
    gd.n_layer = 28;
    gd.n_embd = 2048;
    gd.n_head = gd.n_head_all = gd.n_head_kv = gd.n_head_kv_all = 16;
    gd.head_dim = 128;
    gd.n_ff = gd.n_ff_all = 8192;
    gd.eps = 1e-6f;

    /* ---- bf16 or block fp8: what the container holds for the first DiT projection decides it
     * for every LinearQ of the text encoder and the DiT (rad_linear.h, Geom::w_bf16). */
    {
        LinearQ& probe = m.qkv0;
        RAD_ARCH_TRY(probe.probe(b, nm, gd, nm.f("dit.blk.0.attn_qkv"),
                                 { nm.ckpt("%sblocks.0.self_attn.q_proj.weight", m.pre.c_str()),
                                   nm.ckpt("%sblocks.0.self_attn.k_proj.weight", m.pre.c_str()),
                                   nm.ckpt("%sblocks.0.self_attn.v_proj.weight", m.pre.c_str()) }));
        RadEncoding e{};
        const bool plain = weight_enc(b, nm.f("dit.blk.0.attn_qkv.weight"), &e) &&
                           rad_enc_is(&e, "plain");
        gd.w_bf16 = m.gt.w_bf16 = plain;
    }

    /* ---- the per-image state, the conditioning store, and in issue order the text encoder,
     * the adapter, the modulations, the DiT and the decoder */
    const int64_t lane_rows = (int64_t)kMaxLanes * kMaxImageTokens;
    m.b_state = decl_b(b, nm.f("lat_state"), RAD_BF16, { lane_rows, m.C }, RAD_BUF_PERSIST);
    m.b_prev  = decl_b(b, nm.f("lat_preview"), RAD_BF16, { lane_rows, m.C }, RAD_BUF_PERSIST);
    m.b_store = decl_b(b, nm.f("cond_store"), RAD_BF16,
                       { 2 * (int64_t)kMaxLanes * kCondRows, m.ga.n_embd }, RAD_BUF_PERSIST);
    m.w_temb_norm = decl_w(b, nm.f("%s", marker), RAD_BF16, { gd.n_embd }, RAD_ACCESS_PER_TOKEN,
                           RAD_SHARD_NONE, grp_layer(kGrpDit));
    if (!m.w_temb_norm) return RAD_E_INVAL;
    m.dit.resize((size_t)gd.n_layer);
    /* ---- adapter key names: LoRA files name the DiT by the checkpoint's own modules behind
     * ComfyUI's or kohya's prefix, and the text encoder as kohya's `lora_te_layers_N_...`. */
    RAD_ARCH_TRY(lora_names(b, m.pre.c_str(), kLoraDiffusionSpellings, {}));
    RAD_ARCH_TRY(lora_names(b, "model.layers.", "lora_te_layers_", {}));
    RAD_ARCH_TRY(declare_te(b, m));
    RAD_ARCH_TRY(declare_adapter(b, m));
    RAD_ARCH_TRY(declare_mod(b, m));
    RAD_ARCH_TRY(declare_dit(b, m));
    RAD_ARCH_TRY(declare_decoder(b, m));

    /* ---- what the core drives */
    if (!ctx->shape_probe) {
        static thread_local std::string name;
        name = std::string("anima-") + variant;
        RadGeneratorDecl d{};
        d.name = name.c_str();
        d.layout = RAD_GEN_LAYOUT_JOINT_STEP;
        d.tmpl_text = "{prompt}";
        d.tmpl_edit = "{prompt}";
        d.drop_prefix = "";
        d.latent_channels = m.C;
        d.latent_scale = 16;
        d.size_multiple = 16;
        d.pixel_channels = 3;
        d.slot_rows = 1;
        d.max_tokens = kMaxImageTokens;
        d.max_refs = 0;
        d.ref_area = 1;
        d.default_width = 1024;
        d.default_height = 1024;
        /* ComfyUI's FLOW sampling at shift 3 over its `simple` schedule of 1000 timesteps, Euler
         * steps; Base and Aesthetic at the card's 30 steps and guidance 4 with its negative
         * prompt, Turbo at 10 steps without guidance (the card's CFG 1). */
        d.default_steps = m.variant == 2 ? 10 : 30;
        d.n_sigmas = 0;
        d.sigmas = nullptr;
        d.shift = 3.0;
        d.dyn_shift = 0;
        d.shift_exp = 0;
        d.base_shift = d.max_shift = 0.0;
        d.base_seq = 256;
        d.max_seq = 4096;
        d.shift_terminal = 0.0;
        d.sigma_grid = 1000;
        d.cfg_original = 0;
        d.default_cfg = m.variant == 2 ? 1.0 : 4.0;
        d.default_negative = kNegative;
        d.max_lanes = kMaxLanes;
        d.image_out = m.b_img;
        /* The conditioning's rows: the prompt padded to the 512 rows the DiT reads, which the
         * COND pass leaves in each lane's store and every DENOISE entry names (mode 0). */
        d.cond_rows = kCondRows;
        d.rope_hw_scale = 1.0;
        d.rope_text_l = 0;
        d.pad_multiple = kCondRows;
        d.pad_token = kQwenPad;
        d.min_tokens = 1;
        d.aux_tokens = 1;
        d.aux_max_tokens = kCondRows;
        RAD_ARCH_TRY(rad_declare_generator(b, &d));
    }
    rad_note(b, "rank %d/%d: Anima (%s) -- Qwen3-0.6B, the LLM adapter, a 2B DiT of %lld blocks "
                "(DiT keys under `%s`), %s linears", m.gt.rank, m.gt.world, variant,
             (long long)gd.n_layer, m.pre.c_str(), gd.w_bf16 ? "bf16" : "block-fp8");
    return RAD_OK;
}

/* ================================================================== step */
/* Qwen3 over the prompt's own tokens (not the padding the layout adds), into `b_src`. */
static void step_te(RadCtx* c, const RadBatch* batch, const Model& m, int64_t T) {
    const Geom& g = m.gt;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const RadGenBatch* G = batch->gen_b;
    RAD_ISSUE_N(c, m.op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok),
                brows(m.b_tx, T));
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(G->valid_cu, RAD_I32, 2);
    for (const TeLayer& L : m.te) {
        RAD_ISSUE_N(c, L.op_n1, T, brows(m.b_tx, T), RAD_W(L.w_n1), brows(m.a_th.x, T));
        L.q_h.step(c, T);
        L.qkv.step(c, m.a_th, m.b_tkv, T);
        RAD_ISSUE_N(c, L.op_qkr, T, brows(m.b_tkv, T), pos, RAD_W(L.w_qn), RAD_W(L.w_kn));
        RAD_ISSUE_N(c, L.op_attn, T, bcol(m.b_tkv, 0, q_dim, T), bcol(m.b_tkv, q_dim, kv_dim, T),
                    bcol(m.b_tkv, q_dim + kv_dim, kv_dim, T), cu, brows(m.a_tattn.x, T));
        L.q_attn.step(c, T);
        L.o.step(c, m.a_tattn, m.b_th2, T);
        RAD_ISSUE_N(c, L.op_add1, T, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        RAD_ISSUE_N(c, L.op_n2, T, brows(m.b_tx, T), RAD_W(L.w_n2), brows(m.a_th.x, T));
        L.q_h2.step(c, T);
        L.gate_up.step(c, m.a_th, m.b_tgu, T);
        L.gq.step(c, m.b_tgu, T);
        L.down.step(c, m.a_tffn, m.b_th2, T);
        RAD_ISSUE_N(c, L.op_add2, T, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
    }
    RAD_ISSUE_N(c, m.op_tnorm, T, brows(m.b_tx, T), RAD_W(m.w_tnorm), brows(m.b_src, T));
}

/* The conditioning: Qwen3, the adapter over the T5 rows, the 512 context rows into every lane's
 * store at this branch. */
static void step_cond(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t Lq = batch->n_tok - G->n_pad, Lt = G->n_aux, R = kCondRows;
    const int64_t D = m.ga.n_embd;
    if (Lq <= 0 || Lt <= 0 || Lq > R || Lt > R || !G->aux_ids || !G->aux_cu) {
        rad_step_fail(c, "anima: a conditioning pass with no prompt, or no T5 tokens of it");
        return;
    }
    step_te(c, batch, m, Lq);
    const int64_t Rm = std::max(Lq, Lt);
    const RadOperand tcu = praw(G->aux_cu, RAD_I32, 2), qcu = praw(G->valid_cu, RAD_I32, 2);
    RAD_ISSUE_N(c, m.op_t5, Lt, praw(G->aux_ids, RAD_I32, Lt), RAD_W(m.w_t5), brows(m.b_ax, Lt));
    RAD_ISSUE_N(c, m.op_pos, R, brows(m.b_apos, R));
    RAD_ISSUE_N(c, m.op_fill, R, brows(m.b_aqkv, R));
    for (const AdLayer& L : m.ad) {
        RAD_ISSUE_N(c, L.op_nsa, Lt, brows(m.b_ax, Lt), RAD_W(L.w_nsa), brows(m.b_ah, Lt));
        RAD_ISSUE_N(c, L.op_sqkv, Lt, brows(m.b_ah, Lt), RAD_W(L.w_sqkv), brows(m.b_aqkv, Lt));
        RAD_ISSUE_N(c, L.op_sqkr, Lt, brows(m.b_aqkv, Lt), brows(m.b_apos, Lt), RAD_W(L.w_sqn),
                    RAD_W(L.w_skn));
        RAD_ISSUE_N(c, L.op_sattn, Lt, bcol(m.b_aqkv, 0, D, Lt), bcol(m.b_aqkv, D, D, Lt),
                    bcol(m.b_aqkv, 2 * D, D, Lt), tcu, brows(m.b_aattn, Lt));
        RAD_ISSUE_N(c, L.op_so, Lt, brows(m.b_aattn, Lt), RAD_W(L.w_so), brows(m.b_ab, Lt));
        RAD_ISSUE_N(c, L.op_sadd, Lt, brows(m.b_ax, Lt), brows(m.b_ab, Lt), brows(m.b_ax, Lt));

        RAD_ISSUE_N(c, L.op_nca, Lt, brows(m.b_ax, Lt), RAD_W(L.w_nca), brows(m.b_ah, Lt));
        RAD_ISSUE_N(c, L.op_cq, Lt, brows(m.b_ah, Lt), RAD_W(L.w_cq), brows(m.b_aq, Lt));
        RAD_ISSUE_N(c, L.op_ckv, Lq, brows(m.b_src, Lq), RAD_W(L.w_ckv), brows(m.b_akv, Lq));
        RAD_ISSUE_N(c, L.op_cpq, Lt, brows(m.b_aq, Lt), brows(m.b_aq, Lt), brows(m.b_apos, Lt),
                    bcol(m.b_aqkv, 0, D, Lt));
        RAD_ISSUE_N(c, L.op_cpkv, Lq, brows(m.b_akv, Lq), brows(m.b_akv, Lq), brows(m.b_apos, Lq),
                    bcol(m.b_aqkv, D, 2 * D, Lq));
        RAD_ISSUE_N(c, L.op_cqkr, Rm, brows(m.b_aqkv, Rm), brows(m.b_apos, Rm), RAD_W(L.w_cqn),
                    RAD_W(L.w_ckn));
        RAD_ISSUE_N(c, L.op_cattn, Lt, bcol(m.b_aqkv, 0, D, Lt), bcol(m.b_aqkv, D, D, Lq),
                    bcol(m.b_aqkv, 2 * D, D, Lq), tcu, qcu, brows(m.b_aattn, Lt));
        RAD_ISSUE_N(c, L.op_co, Lt, brows(m.b_aattn, Lt), RAD_W(L.w_co), brows(m.b_ab, Lt));
        RAD_ISSUE_N(c, L.op_cadd, Lt, brows(m.b_ax, Lt), brows(m.b_ab, Lt), brows(m.b_ax, Lt));

        RAD_ISSUE_N(c, L.op_nmlp, Lt, brows(m.b_ax, Lt), RAD_W(L.w_nmlp), brows(m.b_ah, Lt));
        RAD_ISSUE_N(c, L.op_m1, Lt, brows(m.b_ah, Lt), RAD_W(L.w_m1), RAD_W(L.b_m1), RAD_NONE,
                    brows(m.b_am, Lt));
        RAD_ISSUE_N(c, L.op_gelu, Lt, brows(m.b_am, Lt), brows(m.b_am, Lt));
        RAD_ISSUE_N(c, L.op_m2, Lt, brows(m.b_am, Lt), RAD_W(L.w_m2), RAD_W(L.b_m2), RAD_NONE,
                    brows(m.b_ab, Lt));
        RAD_ISSUE_N(c, L.op_madd, Lt, brows(m.b_ax, Lt), brows(m.b_ab, Lt), brows(m.b_ax, Lt));
    }
    RAD_ISSUE_N(c, m.op_aout, Lt, brows(m.b_ax, Lt), RAD_W(m.w_aout), RAD_W(m.b_aout), RAD_NONE,
                brows(m.b_ah, Lt));
    RAD_ISSUE_N(c, m.op_ctxz, R, brows(m.b_ctx, R));
    RAD_ISSUE_N(c, m.op_anorm, Lt, brows(m.b_ah, Lt), RAD_W(m.w_anorm), brows(m.b_ctx, Lt));
    RAD_ISSUE_N(c, m.op_store, R, brows(m.b_ctx, R), praw(G->lane, RAD_I32, G->n_lane),
                brow_slice(m.b_store, (int64_t)G->branch * kCondRows,
                           2 * (int64_t)kMaxLanes * kCondRows - G->branch * kCondRows, D));
}

/* One denoising step: the entries' modulations, the image rows through the patch embedding, every
 * entry's context out of the store, the blocks, and the image rows' velocity through the
 * sampler. */
static void step_denoise(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const Geom& g = m.gd;
    const int64_t S = batch->n_seq, NI = G->n_img, N0 = G->n_mode_h[0];
    const int64_t D = g.n_embd, qd = g.q_dim(), hd = g.head_dim, M3 = 3 * D, L = g.n_layer;
    const int64_t W = L * 3 * M3, nh = g.n_head;
    const RadOperand icu = praw(G->img_cu, RAD_I32, S + 1);
    const RadOperand ccu = praw(G->mode_ccu, RAD_I32, S + 1);
    const RadOperand lanes = praw(G->lane, RAD_I32, S);
    const int64_t LR = (int64_t)kMaxLanes * kMaxImageTokens;
    if (NI <= 0 || N0 <= 0) {
        rad_step_fail(c, "anima: a denoising pass with no image or no context rows");
        return;
    }

    /* ---- the modulations: block l's module k is columns [(3 l + k) 3D, +3D) of `mods`, as
     * (shift | scale | gate) */
    RAD_ISSUE_N(c, m.op_temb, S, praw(G->sigma, RAD_F32, S), brows(m.b_tproj, S));
    RAD_ISSUE_N(c, m.op_t1, S, brows(m.b_tproj, S), RAD_W(m.w_t1), brows(m.b_t1, S));
    RAD_ISSUE_N(c, m.op_tsilu, S, brows(m.b_t1, S), brows(m.b_t1s, S));
    RAD_ISSUE_N(c, m.op_t2, S, brows(m.b_t1s, S), RAD_W(m.w_t2), brows(m.b_lora, S));
    RAD_ISSUE_N(c, m.op_tnormed, S, brows(m.b_tproj, S), RAD_W(m.w_temb_norm), brows(m.b_emb, S));
    RAD_ISSUE_N(c, m.op_esilu, S, brows(m.b_emb, S), brows(m.b_esilu, S));
    auto mod = [&](int64_t l, int k, int part, int64_t width) {
        return bcol_at(m.b_mods, 0, W, (3 * l + k) * M3 + part * D, width, S);
    };
    for (int64_t l = 0; l < L; ++l) {
        const DitLayer& B = m.dit[(size_t)l];
        for (int k = 0; k < 3; ++k) {
            RAD_ISSUE_N(c, B.op_mod1[k], S, brows(m.b_esilu, S), RAD_W(B.w_mod1[k]),
                        brows(m.b_mh, S));
            RAD_ISSUE_N(c, B.op_mod2[k], S, brows(m.b_mh, S), RAD_W(B.w_mod2[k]),
                        brows(m.b_mod1, S));
            RAD_ISSUE_N(c, B.op_modadd[k], S, brows(m.b_mod1, S), brows(m.b_lora, S),
                        mod(l, k, 0, M3));
        }
    }
    RAD_ISSUE_N(c, m.op_f1, S, brows(m.b_esilu, S), RAD_W(m.w_f1), brows(m.b_mh, S));
    RAD_ISSUE_N(c, m.op_f2, S, brows(m.b_mh, S), RAD_W(m.w_f2), brows(m.b_fh, S));
    RAD_ISSUE_N(c, m.op_fadd, S, brows(m.b_fh, S), bcol(m.b_lora, 0, 2 * D, S),
                brows(m.b_fmod, S));

    /* ---- the rows: the images' latents (noise on a first step) through the patch embedding,
     * and every entry's context rows out of the store (mode 0, in entry order) */
    RAD_ISSUE_N(c, m.op_randn, NI, praw(G->flags, RAD_I32, S), praw(G->seed, RAD_U32, 2 * S),
                lanes, icu, brows(m.b_state, LR));
    RAD_ISSUE_N(c, m.op_lgather, NI, brows(m.b_state, LR), lanes, icu, brows(m.b_lat, NI));
    RAD_ISSUE_N(c, m.op_xemb, NI, brows(m.b_lat, NI), RAD_W(m.w_xemb), brows(m.b_h, NI));
    const int64_t SR = 2 * (int64_t)kMaxLanes * kCondRows;
    RAD_ISSUE_N(c, m.op_cgather, N0, brows(m.b_store, SR), brows(m.b_store, SR),
                praw(G->mode_src, RAD_I32, N0), brows(m.a_ctx.x, N0));
    m.q_ctx.step(c, N0);

    /* ---- the blocks; the image rows' positions are mode 2's block of the mode-major planes */
    const int64_t img0 = G->n_mode_h[0] + G->n_mode_h[1];
    const RadOperand pos = praw2(G->mode_rope_f + 4 * img0, RAD_F32, 4, NI);
    const bool q8 = !g.w_bf16;
    auto head_rows = [&](rad_buf x, int64_t rows) {
        return RAD_P_T2(rad_buf_ptr(c, x), RAD_BF16, rows * nh, hd);
    };
    for (int64_t l = 0; l < L; ++l) {
        const DitLayer& B = m.dit[(size_t)l];
        const RadOperand gy = l > 0 ? brows(m.b_ao, NI) : RAD_NONE;
        const RadOperand gg = l > 0 ? mod(l - 1, 2, 2, D) : RAD_NONE;
        if (q8)
            RAD_ISSUE_N(c, B.op_n1, NI, gy, gg, mod(l, 0, 0, 2 * D), RAD_NONE, icu,
                        brows(m.b_h, NI), brows(m.a_hm.q, NI), brows(m.a_hm.s, NI), RAD_NONE);
        else
            RAD_ISSUE_N(c, B.op_n1, NI, gy, gg, mod(l, 0, 0, 2 * D), RAD_NONE, icu,
                        brows(m.b_h, NI), brows(m.a_hm.x, NI));
        B.qkv.step(c, m.a_hm, m.b_qkv, NI);
        RAD_ISSUE_N(c, B.op_qkr, NI, brows(m.b_qkv, NI), pos, RAD_W(B.w_qn), RAD_W(B.w_kn));
        RAD_ISSUE_N(c, B.op_attn, NI, bcol(m.b_qkv, 0, qd, NI), bcol(m.b_qkv, qd, qd, NI),
                    bcol(m.b_qkv, 2 * qd, qd, NI), icu, brows(m.a_attn.x, NI));
        B.q_attn.step(c, NI);
        B.o.step(c, m.a_attn, m.b_ao, NI);

        if (q8)
            RAD_ISSUE_N(c, B.op_n2, NI, brows(m.b_ao, NI), mod(l, 0, 2, D), mod(l, 1, 0, 2 * D),
                        RAD_NONE, icu, brows(m.b_h, NI), brows(m.a_hm.q, NI), brows(m.a_hm.s, NI),
                        RAD_NONE);
        else
            RAD_ISSUE_N(c, B.op_n2, NI, brows(m.b_ao, NI), mod(l, 0, 2, D), mod(l, 1, 0, 2 * D),
                        RAD_NONE, icu, brows(m.b_h, NI), brows(m.a_hm.x, NI));
        B.cq.step(c, m.a_hm, m.b_cq, NI);
        B.ck.step(c, m.a_ctx, m.b_ck, N0);
        B.cv.step(c, m.a_ctx, m.b_cv, N0);
        RAD_ISSUE_N(c, B.op_cqn, NI * nh, head_rows(m.b_cq, NI), RAD_W(B.w_cqn),
                    head_rows(m.b_cq, NI));
        RAD_ISSUE_N(c, B.op_ckn, N0 * nh, head_rows(m.b_ck, N0), RAD_W(B.w_ckn),
                    head_rows(m.b_ck, N0));
        RAD_ISSUE_N(c, B.op_cattn, NI, brows(m.b_cq, NI), brows(m.b_ck, N0), brows(m.b_cv, N0),
                    icu, ccu, brows(m.a_attn.x, NI));
        B.q_cattn.step(c, NI);
        B.co.step(c, m.a_attn, m.b_ao, NI);

        if (q8)
            RAD_ISSUE_N(c, B.op_n3, NI, brows(m.b_ao, NI), mod(l, 1, 2, D), mod(l, 2, 0, 2 * D),
                        RAD_NONE, icu, brows(m.b_h, NI), brows(m.a_hm.q, NI), brows(m.a_hm.s, NI),
                        RAD_NONE);
        else
            RAD_ISSUE_N(c, B.op_n3, NI, brows(m.b_ao, NI), mod(l, 1, 2, D), mod(l, 2, 0, 2 * D),
                        RAD_NONE, icu, brows(m.b_h, NI), brows(m.a_hm.x, NI));
        B.l1.step(c, m.a_hm, m.b_l1, NI);
        RAD_ISSUE_N(c, B.op_gelu, NI, brows(m.b_l1, NI), brows(m.a_ffn.x, NI));
        B.q_ffn.step(c, NI);
        B.l2.step(c, m.a_ffn, m.b_ao, NI);
    }

    /* ---- the last MLP gated in, the final layer, the sampler */
    RAD_ISSUE_N(c, m.op_g3, NI, brows(m.b_ao, NI), mod(L - 1, 2, 2, D), icu, brows(m.b_h, NI));
    RAD_ISSUE_N(c, m.op_fn, NI, RAD_NONE, RAD_NONE, brows(m.b_fmod, S), RAD_NONE, icu,
                brows(m.b_h, NI), brows(m.b_fz, NI));
    RAD_ISSUE_N(c, m.op_final, NI, brows(m.b_fz, NI), RAD_W(m.w_final), brows(m.b_v, NI));
    RAD_ISSUE_N(c, m.op_euler, NI, brows(m.b_v, NI), praw(G->sigma, RAD_F32, S),
                praw(G->sigma_next, RAD_F32, S), praw(G->cfg, RAD_F32, S),
                praw(G->pair, RAD_I32, S), lanes, praw(G->flags, RAD_I32, S),
                praw(G->seed, RAD_U32, 2 * S), icu, RAD_NONE, RAD_NONE, brows(m.b_state, LR),
                brows(m.b_prev, LR));
}

/* One image's latent to RGB: denormalised, unpacked to the VAE's grid, decoded. */
static void step_decode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t H = G->dec_height / 16, W = G->dec_width / 16, n = H * W, C = m.C;
    const rad_buf src = G->dec_preview ? m.b_prev : m.b_state;
    RAD_ISSUE_N(c, m.op_dnorm, n, brow_slice(src, (int64_t)G->dec_lane * kMaxImageTokens, n, C),
                brows(m.b_dl, n));
    const int64_t zd = C / 4, H0 = 2 * H, W0 = 2 * W;
    RAD_ISSUE_N(c, m.op_dzero, H0 * W0, brows(m.b_decin, H0 * W0));
    RAD_ISSUE_N(c, m.op_unpack, H0 * W0, RAD_P_T2(rad_buf_ptr(c, m.b_dl), RAD_BF16, H, W * C),
                RAD_P_T2(rad_buf_ptr(c, m.b_decin), RAD_BF16, H0, W0 * zd));
    RAD_ISSUE_N(c, m.op_cu4, 2, praw(batch->cu_seqlens + 1, RAD_I32, 1), brows(m.b_cu4, 2));
    m.dec.issue(c, H0, W0, brows(m.b_cu4, 2), nullptr);
    const int64_t npx = (int64_t)G->dec_height * G->dec_width;
    RAD_ISSUE_N(c, m.op_rgb, npx, RAD_P_T2(rad_buf_ptr(c, m.dec_out.buf), RAD_BF16, npx, 3),
                RAD_P_T2(rad_buf_ptr(c, m.b_img), RAD_U8, npx, 3));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->gen_b) {
        rad_step_fail(c, "anima is an image generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_COND:    step_cond(c, batch, m); return;
    case RAD_GEN_PASS_DENOISE: step_denoise(c, batch, m); return;
    case RAD_GEN_PASS_DECODE:  step_decode(c, batch, m); return;
    default: rad_step_fail(c, "anima: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows a pass carries: one image at the largest size with guidance, its context both ways. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->gen_rows = 2 * ((int64_t)kMaxImageTokens + kCondRows);
    out->gen_patches = 0;
    return RAD_OK;
}

}  /* namespace anima */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(anima)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(anima, "anima", "", "0.1.0",
                "Anima (Base, Aesthetic, Turbo): Qwen3-0.6B through the LLM adapter, Cosmos-Predict2's "
                "2B diffusion transformer and the Qwen-Image VAE; bf16 or block-fp8 linears")
#endif
