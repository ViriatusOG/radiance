/* minimaxmusic3 -- MiniMax-Music3: a song from a description and lyrics, as 44.1 kHz stereo.
 *
 * ============================== THE MODEL ==============================
 *
 * Four networks behind one prompt (diffusers' MiniMaxMusic3ModularPipeline):
 *
 *   Global language model: Qwen3-8B with its embedding and head adapted to music codes. It reads
 *   the prompt (description and lyrics in the model's tags) and writes the song one frame at a
 *   time, 25 a second: each frame's first residual code (one of 16384 semantic codes, or the end
 *   of the song), sampled from guided logits, u + 1.5 (c - u), between a conditional branch and an
 *   unconditional one whose prompt keeps only its first and last two tokens. Top 50 of the
 *   conditional logits are the candidates, top 50 of the guided ones drawn.
 *
 *   Depth decoder (0.6B): four layers that write the frame's other seven codes, one at a time,
 *   from the language model's hidden state and the codes before them, each with the same
 *   guidance. The frame's eight codes, embedded and summed, are the language model's next input.
 *
 *   Condition encoder and denoiser: each frame's nine hidden states (the language model's and the
 *   depth decoder's seven -- the eighth plane is the language model's own) are mixed with learned
 *   weights into one conditioning row. A window of 200 frames is projected (a 3-tap conv to 2048
 *   channels), resampled to the vocoder's latent rate (441/128 latents a frame) and denoised by a
 *   2.4B flow-matching transformer over 128-channel latents, 30 Euler steps with guidance 1.7
 *   against an all-zero conditioning. Windows overlap by half; each binds its first 172 latents to
 *   its predecessor's.
 *
 *   Vocoder: a DAC decoder, the latent's two 64-channel halves decoded as two streams (left and
 *   right) through four transposed convolutions (8, 8, 4, 2) and dilated residual units with snake
 *   activations, 512 samples a latent.
 *
 * The passes are the core's audio generator (abi/rad_builder.h, rad_declare_audio_generator;
 * core/gen/audio_driver.h): PROMPT runs the language model over prompt chunks, FRAME samples one
 * frame of every song in flight and runs the language model over its embedding, WINDOW steps the
 * denoiser, VOCODE plays a finished window.
 *
 * ============================== STATE ==============================
 *
 * A song is a lane. Each lane keeps its two branches' last hidden states, every frame's
 * conditioning row (9000 frames at most), its window's latent and conditioning, and what the next
 * window binds to: the previous window's latents and conditioning over the overlap, and its own
 * noise there. The depth decoder's sequence is rebuilt every frame and recomputed whole at each of
 * its seven steps (causal attention keeps the positions not yet written out of every result it
 * reads), which costs nothing at a frame's handful of rows since every pass reads the same
 * weights.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * The language model, the depth decoder and the denoiser's blocks are split by heads and by
 * feed-forward width, an all-reduce after each output projection. The heads, the sampler, the
 * condition encoder, the denoiser's input and output layers and the vocoder are replicated: every
 * rank draws the same codes from the same logits and seeds, and plays the same samples.
 */
#include <arch/rad_arch.h>
#include <arch/rad_linear.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace minimaxmusic3 {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* Songs at once: lanes. */
enum { kMaxLanes = 4 };
/* The longest prompt and song (diffusers' _MAX_PROMPT_TOKENS, _MAX_AUDIO_FRAMES). */
enum { kMaxPrompt = 5000, kMaxFrames = 9000 };
/* The frame's codes: the semantic one and seven residual ones of 1024. */
enum { kCodes = 8, kSemCodes = 16384, kResCodes = 1024 };
/* Token ids: the first semantic code, the end of the song, the unconditional branch's filler. */
enum { kCodeOffset = 151675, kEndToken = 151670, kCfgToken = 151654 };
/* Windows: 200 frames a new one every 100; latents per frame 44100/24000 * 960/512 = 441/128. */
enum { kWinFrames = 200, kWinHop = 100, kLatNum = 441, kLatDen = 128 };
enum { kWinLat = kWinFrames * kLatNum / kLatDen };          /* 689 */
/* The overlap a window binds (diffusers' _OVERLAP_LATENT_LENGTH) and the crops at a join. */
enum { kOverlap = 172, kCropLeft = 86, kCropRight = 344 - 86 };
enum { kSamplesPerLatent = 512, kSampleRate = 44100 };

static const double kArCfg = 1.5;
static const int    kTopK  = 50;
static const int    kDitSteps = 30;
static const double kDitCfg = 1.7;

/* The prompt, in the model's own tags (diffusers' MiniMaxMusic3TokenizeStep). */
static const char* kTemplate =
    "<|im_start|><|caption_start|>{instructions}<|caption_end|><|lyrics_start|>{input}"
    "<|lyrics_end|><|im_end|><|audio_start|>";

/* Weight groups (movement units) by network. */
enum : int {
    kGrpLm    = 0,      /* the language model's layers: 0.. */
    kGrpDepth = 100,
    kGrpCond  = 150,
    kGrpDit   = 200,    /* the denoiser's model-level weights; its blocks 201.. */
    kGrpVoc   = 300,
};

/* ================================================================== the networks */
/* A Qwen3 layer over a paged cache: q, k and v as three projections (paged attention reads a
 * contiguous query), per-head RMS norms, NeoX rotary, an output projection and a SwiGLU. Every
 * linear is a LinearQ, which reads bf16 or block fp8 as the container holds it. The depth
 * decoder's layers are the same without the norms on q and k, the rotary or the cache. */
struct LmLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ  q{}, k{}, v{}, o{}, gate_up{}, down{};
    QuantAct   q_h{}, q_attn{}, q_h2{};
    GateQuant gq{};
    rad_op     op_n1 = 0, op_qn = 0, op_kn = 0, op_rq = 0, op_rk = 0, op_kv = 0, op_attn = 0;
    rad_op     op_attn_p = 0;   /* a prompt chunk: its own rows causally, its earlier ones cached */
    rad_op     op_ar1 = 0, op_add1 = 0, op_n2 = 0, op_ar2 = 0, op_add2 = 0;
};

/* The denoiser's block: layer norms with gain and bias, attention over fused q|k|v with a partial
 * rotary, a SwiGLU whose value half comes first (`gate_states, gate = chunk(2)`), biases on the
 * feed-forward. */
struct DitLayer {
    rad_weight w_n1 = 0, w_b1 = 0, w_n2 = 0, w_b2 = 0, w_fi = 0, w_fib = 0, w_fo = 0, w_fob = 0;
    LinearQ  qkv{}, o{};
    QuantAct   q_h{}, q_attn{};
    rad_op     op_ln1 = 0, op_rope = 0, op_attn = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_ln2 = 0, op_fi = 0, op_act = 0, op_fo = 0, op_ar2 = 0, op_add2 = 0;
};

/* One vocoder convolution: its weight, bias and the op. */
struct VConv {
    rad_weight w = 0, b = 0;
    rad_op     op = 0;
    int64_t    cin = 0, cout = 0, k = 1, dil = 1;
};
struct VRes {
    rad_weight a1 = 0, a2 = 0;
    rad_op     op_s1 = 0, op_s2 = 0;
    VConv      c1{}, c2{};
};
struct VBlock {
    rad_weight a = 0;
    rad_op     op_s = 0;
    rad_weight wt = 0, bt = 0;
    rad_op     op_t = 0;
    int64_t    cin = 0, cout = 0, stride = 1;
    VRes       res[3];
    rad_op     op_cu = 0;   /* the stage's cu_seqlens: the window's two streams at this rate */
};

struct Model {
    Names nm{""};
    Geom  gl{}, gd{}, gt{};      /* the language model's, the depth decoder's and the DiT's */
    int64_t lanes = 0;           /* lanes declared (kMaxLanes, or what max_seqs leaves) */
    int64_t S = 0;               /* a FRAME pass's rows at most: two a lane */
    rad_kvgroup kv = 0;

    /* ---- the language model */
    rad_weight w_tok = 0, w_norm = 0, w_head = 0;
    rad_op  op_embed = 0, op_norm = 0;
    rad_buf b_x = 0, b_q = 0, b_k = 0, b_v = 0, b_h2 = 0, b_gu = 0, b_last = 0;
    ActQ  a_h{}, a_attn{}, a_ffn{};
    std::vector<LmLayer> lm;

    /* ---- the frame: the lanes' last hidden states, the logits, the codes, the depth decoder */
    rad_buf b_hid = 0;            /* [2 lanes, n_embd] PERSIST: row 2 lane + branch */
    rad_buf b_status = 0;         /* [lanes] i32 PERSIST: what the FRAME pass reports */
    rad_buf b_slog = 0;           /* [S, 16385] the semantic codes' logits, the end's last */
    rad_buf b_codes = 0;          /* [8, S] i32: code-major, each code twice (both branches) */
    rad_buf b_idx = 0;            /* [8, S] i32: depth row e * 8 + p for each position p */
    rad_buf b_planes = 0;         /* [8 * S, n_embd]: the frame's eight conditioning planes */
    rad_op  op_hsel = 0, op_sem = 0, op_status = 0, op_iota[kCodes] = {};
    rad_op  op_mix = 0, op_fb = 0, op_store = 0, op_plast = 0, op_pnorm = 0;
    rad_weight w_dproj = 0, w_dpos = 0, w_dnorm = 0, w_dheads = 0, w_demb = 0;
    rad_op  op_dproj[kCodes] = {}, op_dscat = 0, op_dgather = 0;
    rad_op  op_dnorm = 0, op_dhead = 0, op_dsamp[kCodes] = {}, op_demb[kCodes] = {};
    rad_buf b_dseq = 0, b_dx = 0, b_dq = 0, b_dk = 0, b_dv = 0, b_dh2 = 0, b_dgu = 0;
    rad_buf b_dsel = 0, b_drow = 0, b_dlog = 0, b_demb = 0, b_dcu = 0;
    rad_op  op_dcu = 0;
    ActQ  a_dh{}, a_dattn{}, a_dffn{};
    std::vector<LmLayer> dep;
    rad_buf b_mix = 0;            /* [lanes * kMaxFrames, n_embd] PERSIST */
    rad_weight w_mixl = 0, w_mixs = 0;

    /* ---- the window: condition encoder, denoiser */
    rad_weight w_cconv = 0, w_cbias = 0;
    rad_op  op_ccu = 0, op_cconv = 0, op_cres = 0, op_cjoin = 0;
    rad_buf b_ccu = 0, b_cc = 0;
    rad_buf b_lat = 0, b_wcond = 0, b_prev = 0, b_pcond = 0, b_noise = 0;   /* PERSIST, lanes */
    rad_op  op_prep = 0, op_win_in = 0, op_pre = 0, op_pin = 0, op_finish = 0, op_euler = 0;
    rad_weight w_pre = 0, w_pin = 0, w_tw = 0, w_t1 = 0, w_t1b = 0, w_t2 = 0, w_t2b = 0;
    rad_weight w_pout = 0, w_post = 0;
    rad_op  op_four = 0, op_t1 = 0, op_tsilu = 0, op_t2 = 0, op_hscat = 0, op_tscat = 0;
    rad_op  op_hgather = 0, op_pout = 0, op_post = 0;
    rad_buf b_din = 0, b_dpre = 0, b_hl = 0, b_tf = 0, b_te1 = 0, b_te1s = 0, b_temb = 0;
    rad_buf b_h = 0, b_qkv = 0, b_th2 = 0, b_ff = 0, b_ffa = 0, b_ho = 0, b_po = 0;
    rad_buf b_vel = 0;
    ActQ  a_th{}, a_tattn{};
    std::vector<DitLayer> dit;
    int64_t C = 128, Dc = 2048;

    /* ---- the vocoder */
    VConv   v_in{}, v_conv{}, v_out{};
    rad_weight v_aout = 0;
    rad_op  op_vsplit = 0, op_vsout = 0, op_vcu0 = 0, op_vmix = 0;
    VBlock  vb[4];
    int64_t v_ups[4] = { 8, 8, 4, 2 };
    rad_buf b_vx = 0, b_vin = 0, b_vout = 0;
    rad_buf b_vcu[5] = {}, b_vstage[5] = {}, b_vt[5] = {}, b_va[5] = {}, b_vb[5] = {};
    rad_buf b_audio = 0;
};

static Model g_model[MAX_SLOTS];

/* ================================================================== declare helpers */
static rad_op norm_op(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight w, rad_buf x,
                      rad_buf y) {
    return rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(w)),
              { x }, { y });
}

/* A plain weight copied from the checkpoint, declared at `dtype`. */
static rad_weight wcopy(RadBuilder* b, Names& nm, const char* dn, const char* src, uint32_t dtype,
                        std::initializer_list<int64_t> shape, int group, int shard = RAD_SHARD_NONE,
                        std::initializer_list<int64_t> row_parts = {}) {
    if (map_copy(b, dn, src) < 0) return 0;
    (void)nm;
    return decl_w(b, dn, dtype, shape, RAD_ACCESS_PER_TOKEN, shard, grp_layer(group), 0, row_parts);
}

/* ================================================================== the language model */
/* One Qwen3 layer's weights and ops over `rows` rows, `pre` the checkpoint's prefix. */
static int declare_qwen3_layer(RadBuilder* b, Names& nm, const Geom& g, LmLayer& L, int li,
                               const char* dn, const char* pre, int group, int64_t rows,
                               bool qk_norm, bool paged, rad_kvgroup kv, rad_buf x, ActQ& ah,
                               rad_buf bq, rad_buf bk, rad_buf bv, ActQ& aattn, rad_buf h2,
                               rad_buf gu, ActQ& affn, const char* n1, const char* n2,
                               const char* qn, const char* kn, const char* qs, const char* ks,
                               const char* vs, const char* os, const char* gs, const char* us,
                               const char* ds, rad_buf x_in) {
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    RAD_ARCH_TRY(map_copy(b, nm.f("%s.%d.attn_norm.weight", dn, li), nm.ckpt("%s.%d.%s", pre, li, n1)));
    L.w_n1 = decl_w(b, nm.f("%s.%d.attn_norm.weight", dn, li), RAD_F32, { g.n_embd },
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(group));
    RAD_ARCH_TRY(map_copy(b, nm.f("%s.%d.ffn_norm.weight", dn, li), nm.ckpt("%s.%d.%s", pre, li, n2)));
    L.w_n2 = decl_w(b, nm.f("%s.%d.ffn_norm.weight", dn, li), RAD_F32, { g.n_embd },
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(group));
    /* Layer 0 of a stack may read its input from another buffer than its residual: the depth
     * decoder's sequence is rebuilt a row at a time and must survive the stack, so its first layer
     * reads it and writes the residual. */
    L.op_n1 = norm_op(b, rows, g.n_embd, g.eps, L.w_n1, x_in, ah.x);
    RAD_ARCH_TRY(L.q_h.declare(b, g, ah, g.n_embd));
    RAD_ARCH_TRY(L.q.declare(b, nm, g, nm.f("%s.%d.attn_q", dn, li), q_dim, g.n_embd, RAD_SHARD_ROW,
                             grp_layer(group), { nm.ckpt("%s.%d.%s", pre, li, qs) }, ah, bq, rows));
    RAD_ARCH_TRY(L.k.declare(b, nm, g, nm.f("%s.%d.attn_k", dn, li), kv_dim, g.n_embd, RAD_SHARD_ROW,
                             grp_layer(group), { nm.ckpt("%s.%d.%s", pre, li, ks) }, ah, bk, rows));
    RAD_ARCH_TRY(L.v.declare(b, nm, g, nm.f("%s.%d.attn_v", dn, li), kv_dim, g.n_embd, RAD_SHARD_ROW,
                             grp_layer(group), { nm.ckpt("%s.%d.%s", pre, li, vs) }, ah, bv, rows));
    if (qk_norm) {
        RAD_ARCH_TRY(map_copy(b, nm.f("%s.%d.q_norm.weight", dn, li), nm.ckpt("%s.%d.%s", pre, li, qn)));
        L.w_qn = decl_w(b, nm.f("%s.%d.q_norm.weight", dn, li), RAD_F32, { g.head_dim },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(group));
        RAD_ARCH_TRY(map_copy(b, nm.f("%s.%d.k_norm.weight", dn, li), nm.ckpt("%s.%d.%s", pre, li, kn)));
        L.w_kn = decl_w(b, nm.f("%s.%d.k_norm.weight", dn, li), RAD_F32, { g.head_dim },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(group));
        /* Each norm in place: a row's scale is computed before the row is written. */
        L.op_qn = norm_op(b, rows * g.n_head, g.head_dim, g.eps, L.w_qn, bq, bq);
        L.op_kn = norm_op(b, rows * g.n_head_kv, g.head_dim, g.eps, L.w_kn, bk, bk);
        L.op_rq = rw(b, decl_rope(b, g, rows, g.head_dim, g.n_head, 0, g.head_dim, "neox"), { bq }, { bq });
        L.op_rk = rw(b, decl_rope(b, g, rows, g.head_dim, g.n_head_kv, 0, g.head_dim, "neox"), { bk }, { bk });
    }
    if (paged) {
        /* A frame is one row a sequence, decode's band; a prompt chunk attends to its cached
         * prefix and its own rows (attn_prefix_self), which serves every chunk length at this
         * head geometry. */
        L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                           RAD_PARAMS(RAD_RANGE("q_len", 1, 1), RAD_INT("head_dim", g.head_dim),
                                      RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                      RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                      RAD_INT("block_size", RAD_KV_BLOCK),
                                      RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                      RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype)),
                           RAD_NOWEIGHTS),
                       { bq }, { aattn.x });
        L.op_attn_p = rw(b, RAD_OP(b, "attn_prefix_self",
                             RAD_PARAMS(RAD_INT("q_len", rows), RAD_INT("head_dim", g.head_dim),
                                        RAD_INT("gqa", g.gqa()), RAD_INT("block_size", RAD_KV_BLOCK),
                                        RAD_INT("causal", 1), RAD_STR("q_dtype", "bf16"),
                                        RAD_STR("kv_dtype", g.kv_dtype),
                                        RAD_F64("scale", 1.0 / std::sqrt((double)g.head_dim)),
                                        RAD_INT("max_ctx", g.max_ctx), RAD_INT("n_head", g.n_head),
                                        RAD_INT("max_seqs", g.max_seqs), RAD_STR("qk", "bf16")),
                             RAD_NOWEIGHTS),
                         { bq, bk, bv }, { aattn.x });
        L.op_kv = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     { bk, bv }, {});
    } else {
        L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                           RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", g.head_dim),
                                      RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                      RAD_STR("dtype", "bf16"),
                                      RAD_F64("scale", 1.0 / std::sqrt((double)g.head_dim)),
                                      RAD_INT("max_seqlen", kCodes)),
                           RAD_NOWEIGHTS),
                       { bq, bk, bv }, { aattn.x });
    }
    RAD_ARCH_TRY(L.q_attn.declare(b, g, aattn, q_dim));
    RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("%s.%d.attn_output", dn, li), g.n_embd, q_dim,
                             RAD_SHARD_COL, grp_layer(group), { nm.ckpt("%s.%d.%s", pre, li, os) },
                             aattn, h2, rows));
    L.op_ar1 = decl_all_reduce_tp(b, g, h2, rows);
    L.op_add1 = add_op(b, rows, g.n_embd, x_in, h2, x);
    L.op_n2 = norm_op(b, rows, g.n_embd, g.eps, L.w_n2, x, ah.x);
    RAD_ARCH_TRY(L.q_h2.declare(b, g, ah, g.n_embd));
    RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("%s.%d.ffn_gate_up", dn, li), 2 * g.n_ff,
                                   g.n_embd, RAD_SHARD_ROW, grp_layer(group),
                                   { nm.ckpt("%s.%d.%s", pre, li, gs), nm.ckpt("%s.%d.%s", pre, li, us) },
                                   ah, gu, rows, { g.n_ff, g.n_ff }));
    RAD_ARCH_TRY(L.gq.declare(b, g, gu, affn, g.n_ff, "silu", rows, true));
    RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("%s.%d.ffn_down", dn, li), g.n_embd, g.n_ff,
                                RAD_SHARD_COL, grp_layer(group), { nm.ckpt("%s.%d.%s", pre, li, ds) },
                                affn, h2, rows));
    L.op_ar2 = decl_all_reduce_tp(b, g, h2, rows);
    L.op_add2 = add_op(b, rows, g.n_embd, x, h2, x);
    return RAD_OK;
}

static void step_qwen3_layer(RadCtx* c, const Model& m, const Geom& g, const LmLayer& L,
                             const RadBatch* batch, int64_t T, rad_buf x, rad_buf x_in,
                             const ActQ& ah, rad_buf bq, rad_buf bk, rad_buf bv,
                             const ActQ& aattn, rad_buf h2, rad_buf gu, const ActQ& affn,
                             const RadOperand& cu) {
    const int layer = (int)(&L - (L.op_kv ? m.lm.data() : m.dep.data()));
    RAD_ISSUE(c, L.op_n1, brows(x_in, T), RAD_W(L.w_n1), brows(ah.x, T));
    L.q_h.step(c, T);
    L.q.step(c, ah, bq, T);
    L.k.step(c, ah, bk, T);
    L.v.step(c, ah, bv, T);
    if (L.op_qn) {
        const RadOperand pos = praw(batch->positions, RAD_I32, T);
        RAD_ISSUE_N(c, L.op_qn, T * g.n_head, brows(bq, T), RAD_W(L.w_qn), brows(bq, T));
        RAD_ISSUE_N(c, L.op_kn, T * g.n_head_kv, brows(bk, T), RAD_W(L.w_kn), brows(bk, T));
        RAD_ISSUE(c, L.op_rq, brows(bq, T), pos);
        RAD_ISSUE(c, L.op_rk, brows(bk, T), pos);
    }
    if (L.op_kv) {
        const RadKVGroupBatch* kvb = kv_batch(batch, m.kv);
        const RadOperand bt = praw2(kvb->block_table, RAD_I32, batch->n_seq, kvb->block_table_pitch);
        const RadOperand bcu = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);
        if (batch->gen == RAD_GEN_PASS_PROMPT) {
            /* The prefix first, read from the cache before this chunk's rows go in. */
            RAD_ISSUE(c, L.op_attn_p, brows(bq, T), brows(bk, T), brows(bv, T), kv_cache(m.kv, layer),
                      bt, praw(batch->agen_b->prefix, RAD_I32, batch->n_seq), bcu, RAD_NONE,
                      RAD_NONE, RAD_NONE, RAD_NONE, brows(aattn.x, T), RAD_NONE);
            RAD_ISSUE(c, L.op_kv, brows(bk, T), brows(bv, T), praw(kvb->slot_mapping, RAD_I32, T),
                      kv_cache(m.kv, layer));
        } else {
            RAD_ISSUE(c, L.op_kv, brows(bk, T), brows(bv, T), praw(kvb->slot_mapping, RAD_I32, T),
                      kv_cache(m.kv, layer));
            RAD_ISSUE_N(c, L.op_attn, batch->max_q_len, brows(bq, T), kv_cache(m.kv, layer), bt,
                        praw(kvb->seqused, RAD_I32, batch->n_seq), RAD_NONE, RAD_NONE, bcu,
                        brows(aattn.x, T));
        }
    } else {
        RAD_ISSUE_N(c, L.op_attn, T, brows(bq, T), brows(bk, T), brows(bv, T), cu, brows(aattn.x, T));
    }
    L.q_attn.step(c, T);
    L.o.step(c, aattn, h2, T);
    if (L.op_ar1) RAD_ISSUE_N(c, L.op_ar1, T * g.n_embd, brows(h2, T), RAD_NONE);
    RAD_ISSUE(c, L.op_add1, brows(x_in, T), brows(h2, T), brows(x, T));
    RAD_ISSUE(c, L.op_n2, brows(x, T), RAD_W(L.w_n2), brows(ah.x, T));
    L.q_h2.step(c, T);
    L.gate_up.step(c, ah, gu, T);
    L.gq.step(c, gu, T);
    L.down.step(c, affn, h2, T);
    if (L.op_ar2) RAD_ISSUE_N(c, L.op_ar2, T * g.n_embd, brows(h2, T), RAD_NONE);
    RAD_ISSUE(c, L.op_add2, brows(x, T), brows(h2, T), brows(x, T));
}

static int declare_lm(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx, Model& m) {
    Geom& g = m.gl;
    Names& nm = m.nm;
    const int64_t T = g.max_tok;
    (void)meta;
    /* bf16 or block fp8, as the container holds the first projection: every linear of the three
     * transformers follows it (rad_linear.h, Geom::w_bf16). */
    m.lm.resize((size_t)g.n_layer);
    RAD_ARCH_TRY(m.lm[0].q.probe(b, nm, g, "lm.0.attn_q", { "model.layers.0.self_attn.q_proj.weight" }));
    RadEncoding e{};
    const bool plain = weight_enc(b, nm.f("lm.0.attn_q.weight"), &e) && rad_enc_is(&e, "plain");
    g.w_bf16 = m.gd.w_bf16 = m.gt.w_bf16 = plain;

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

    m.b_x  = decl_b(b, nm.f("lm.x"), RAD_BF16, { T, g.n_embd });
    m.a_h.x = decl_b(b, nm.f("lm.h"), RAD_BF16, { T, g.n_embd });
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "lm.h", g.n_embd));
    m.b_q  = decl_b(b, nm.f("lm.q"), RAD_BF16, { T, g.q_dim() });
    m.b_k  = decl_b(b, nm.f("lm.k"), RAD_BF16, { T, g.kv_dim() });
    m.b_v  = decl_b(b, nm.f("lm.v"), RAD_BF16, { T, g.kv_dim() });
    m.a_attn.x = decl_b(b, nm.f("lm.attn"), RAD_BF16, { T, g.q_dim() });
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "lm.attn", g.q_dim()));
    m.b_h2 = decl_b(b, nm.f("lm.h2"), RAD_BF16, { T, g.n_embd });
    m.b_gu = decl_b(b, nm.f("lm.gu"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_ffn.x = decl_b(b, nm.f("lm.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "lm.ffn", g.n_ff));
    m.b_last = decl_b(b, nm.f("lm.last"), RAD_BF16, { T, g.n_embd });

    m.w_tok = wcopy(b, nm, nm.f("lm.tok.weight"), "model.embed_tokens.weight", RAD_BF16,
                    { g.n_vocab_all, g.n_embd }, -1);
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", g.n_embd),
                                         RAD_INT("n_vocab", g.n_vocab_all), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_x });

    for (int64_t l = 0; l < g.n_layer; ++l)
        RAD_ARCH_TRY(declare_qwen3_layer(
            b, nm, g, m.lm[(size_t)l], (int)l, "lm", "model.layers", kGrpLm + (int)l, T, true, true,
            m.kv, m.b_x, m.a_h, m.b_q, m.b_k, m.b_v, m.a_attn, m.b_h2, m.b_gu, m.a_ffn,
            "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
            "self_attn.k_norm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight",
            "mlp.up_proj.weight", "mlp.down_proj.weight", m.b_x));

    RAD_ARCH_TRY(map_copy(b, nm.f("lm.norm.weight"), "model.norm.weight"));
    m.w_norm = decl_w(b, nm.f("lm.norm.weight"), RAD_F32, { g.n_embd }, RAD_ACCESS_PER_TOKEN,
                      RAD_SHARD_NONE, grp_model());
    m.op_norm = norm_op(b, T, g.n_embd, g.eps, m.w_norm, m.b_x, m.b_last);

    /* The head over the rows a frame can be: the semantic codes, then the end of the song. */
    RAD_ARCH_TRY(map_rows(b, nm.f("lm.head.weight"),
                          { { "lm_head.weight", kCodeOffset, kSemCodes },
                            { "lm_head.weight", kEndToken, 1 } }));
    m.w_head = decl_w(b, nm.f("lm.head.weight"), RAD_BF16, { kSemCodes + 1, g.n_embd },
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    return m.w_tok && m.w_norm && m.w_head && m.op_embed ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the frame */
static int declare_frame(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& gd = m.gd;
    Names& nm = m.nm;
    const int64_t S = m.S, D = m.gl.n_embd, R = (int64_t)kCodes * S;
    const int64_t lanes = m.lanes;

    m.b_hid    = decl_b(b, nm.f("frame.hid"), RAD_BF16, { 2 * lanes, D }, RAD_BUF_PERSIST);
    m.b_status = decl_b(b, nm.f("frame.status"), RAD_I32, { lanes }, RAD_BUF_PERSIST);
    m.b_slog   = decl_b(b, nm.f("frame.slog"), RAD_BF16, { S, kSemCodes + 1 });
    m.b_codes  = decl_b(b, nm.f("frame.codes"), RAD_I32, { kCodes, S });
    m.b_idx    = decl_b(b, nm.f("frame.idx"), RAD_I32, { kCodes, S });
    m.b_planes = decl_b(b, nm.f("frame.planes"), RAD_BF16, { (int64_t)kCodes * S, D });
    m.b_mix    = decl_b(b, nm.f("frame.mix"), RAD_BF16, { lanes * kMaxFrames, D }, RAD_BUF_PERSIST);

    /* The lanes' hidden states into the pass's rows, plane 0 of the frame's conditioning. */
    m.op_hsel = rows_op(b, "gather_rows", S, D, { m.b_hid }, { m.b_planes });
    m.op_sem = gemm_op(b, S, kSemCodes + 1, D, m.w_head, m.b_planes, m.b_slog);
    auto sampler = [&](int64_t V, int stream, int cond_k, rad_buf lg) {
        return rw(b, RAD_OP(b, "cfg_topk_sample",
                            RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_vocab", V), RAD_INT("k", kTopK),
                                       RAD_INT("rep", 2), RAD_INT("cond_k", cond_k),
                                       RAD_F64("cfg", kArCfg), RAD_F64("temp", 1.0),
                                       RAD_INT("stream", stream), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { lg }, { m.b_codes });
    };
    m.op_dsamp[0] = sampler(kSemCodes + 1, 0, kTopK, m.b_slog);
    m.op_status = rw(b, RAD_OP(b, "end_status",
                               RAD_PARAMS(RAD_RANGE("M", 1, lanes), RAD_INT("end", kSemCodes),
                                          RAD_INT("rep", 2)),
                               RAD_NOWEIGHTS),
                     { m.b_codes }, { m.b_status });
    for (int p = 0; p < kCodes; ++p)
        m.op_iota[p] = rw(b, RAD_OP(b, "iota",
                                    RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("start", p),
                                               RAD_INT("step", kCodes)),
                                    RAD_NOWEIGHTS),
                          {}, { m.b_idx });

    /* ---- the depth decoder */
    gd = m.gl;
    gd.n_layer = rad_meta_geti(meta, "rvq_depth_decoder.num_layers", 4);
    gd.n_embd = rad_meta_geti(meta, "rvq_depth_decoder.hidden_size", 4096);
    gd.n_head_all = rad_meta_geti(meta, "rvq_depth_decoder.num_attention_heads", 16);
    gd.n_head_kv_all = gd.n_head_all;
    gd.head_dim = gd.n_embd / gd.n_head_all;
    gd.n_ff_all = rad_meta_geti(meta, "rvq_depth_decoder.intermediate_size", 6144);
    gd.eps = 1e-6f;
    gd.max_tok = R;
    RAD_ARCH_TRY(divide_or_fail(gd.n_head_all, gd.world, "the depth decoder's heads", &gd.n_head));
    gd.n_head_kv = gd.n_head;
    gd.kv_off = 0;
    RAD_ARCH_TRY(divide_or_fail(gd.n_ff_all, gd.world, "the depth decoder's feed-forward", &gd.n_ff));
    if (gd.n_embd != D || gd.head_dim != 256 ||
        rad_meta_geti(meta, "rvq_depth_decoder.num_codebooks", kCodes) != kCodes ||
        rad_meta_geti(meta, "rvq_depth_decoder.audio_vocab_size", kResCodes) != kResCodes) {
        fprintf(stderr, "radiance: minimaxmusic3: the depth decoder is not the released one (width "
                        "%lld, head %lld)\n", (long long)gd.n_embd, (long long)gd.head_dim);
        return RAD_E_UNSUPPORTED;
    }
    const char* dp = "rvq_depth_decoder";
    m.w_dproj = wcopy(b, nm, nm.f("dep.proj.weight"), nm.ckpt("%s.projection.weight", dp), RAD_BF16,
                      { D, D }, kGrpDepth);
    m.w_dpos = wcopy(b, nm, nm.f("dep.pos.weight"), nm.ckpt("%s.pos_embedding.weight", dp), RAD_BF16,
                     { 16, D }, kGrpDepth);
    RAD_ARCH_TRY(map_copy(b, nm.f("dep.norm.weight"), nm.ckpt("%s.norm.weight", dp)));
    m.w_dnorm = decl_w(b, nm.f("dep.norm.weight"), RAD_F32, { D }, RAD_ACCESS_PER_TOKEN,
                       RAD_SHARD_NONE, grp_layer(kGrpDepth));
    RAD_ARCH_TRY(map_concat(b, nm.f("dep.heads.weight"),
                            { nm.ckpt("%s.audio_heads.0.weight", dp), nm.ckpt("%s.audio_heads.1.weight", dp),
                              nm.ckpt("%s.audio_heads.2.weight", dp), nm.ckpt("%s.audio_heads.3.weight", dp),
                              nm.ckpt("%s.audio_heads.4.weight", dp), nm.ckpt("%s.audio_heads.5.weight", dp),
                              nm.ckpt("%s.audio_heads.6.weight", dp) }));
    m.w_dheads = decl_w(b, nm.f("dep.heads.weight"), RAD_BF16, { (kCodes - 1) * kResCodes, D },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(kGrpDepth));
    m.w_demb = wcopy(b, nm, nm.f("dep.emb.weight"), nm.ckpt("%s.audio_embeddings.weight", dp),
                     RAD_BF16, { (kCodes - 1) * kResCodes, D }, kGrpDepth);

    m.b_dseq = decl_b(b, nm.f("dep.seq"), RAD_BF16, { R, D });
    m.b_dx   = decl_b(b, nm.f("dep.x"), RAD_BF16, { R, D });
    m.a_dh.x = decl_b(b, nm.f("dep.h"), RAD_BF16, { R, D });
    RAD_ARCH_TRY(m.a_dh.declare_qs(b, nm, gd, "dep.h", D, R));
    m.b_dq = decl_b(b, nm.f("dep.q"), RAD_BF16, { R, gd.q_dim() });
    m.b_dk = decl_b(b, nm.f("dep.k"), RAD_BF16, { R, gd.kv_dim() });
    m.b_dv = decl_b(b, nm.f("dep.v"), RAD_BF16, { R, gd.kv_dim() });
    m.a_dattn.x = decl_b(b, nm.f("dep.attn"), RAD_BF16, { R, gd.q_dim() });
    RAD_ARCH_TRY(m.a_dattn.declare_qs(b, nm, gd, "dep.attn", gd.q_dim(), R));
    m.b_dh2 = decl_b(b, nm.f("dep.h2"), RAD_BF16, { R, D });
    m.b_dgu = decl_b(b, nm.f("dep.gu"), RAD_BF16, { R, 2 * gd.n_ff });
    m.a_dffn.x = decl_b(b, nm.f("dep.ffn"), RAD_BF16, { R, gd.n_ff });
    RAD_ARCH_TRY(m.a_dffn.declare_qs(b, nm, gd, "dep.ffn", gd.n_ff, R));
    m.b_dcu = decl_b(b, nm.f("dep.cu"), RAD_I32, { S + 1 });
    m.op_dcu = rw(b, RAD_OP(b, "cu_repeat",
                            RAD_PARAMS(RAD_RANGE("M", 1, S + 1), RAD_INT("mult", kCodes)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_dcu });
    m.b_dsel = decl_b(b, nm.f("dep.sel"), RAD_BF16, { S, D });
    m.b_drow = decl_b(b, nm.f("dep.row"), RAD_BF16, { S, D });
    m.b_dlog = decl_b(b, nm.f("dep.log"), RAD_BF16, { S, kResCodes });
    m.b_demb = decl_b(b, nm.f("dep.emb"), RAD_BF16, { S, D });

    m.dep.resize((size_t)gd.n_layer);
    for (int64_t l = 0; l < gd.n_layer; ++l)
        RAD_ARCH_TRY(declare_qwen3_layer(
            b, nm, gd, m.dep[(size_t)l], (int)l, "dep", "rvq_depth_decoder.layers",
            kGrpDepth + 1 + (int)l, R, false, false, 0, m.b_dx, m.a_dh, m.b_dq, m.b_dk, m.b_dv,
            m.a_dattn, m.b_dh2, m.b_dgu, m.a_dffn, "input_layernorm.weight",
            "post_attention_layernorm.weight", nullptr, nullptr, "attn.to_q.weight",
            "attn.to_k.weight", "attn.to_v.weight", "attn.to_out.weight", "gate_proj.weight",
            "up_proj.weight", "down_proj.weight", l == 0 ? m.b_dseq : m.b_dx));
    /* Each layer's attention also reads the frame's sequence offsets. */
    for (const LmLayer& L : m.dep) RAD_ARCH_TRY(rad_op_reads(b, L.op_attn, &m.b_dcu, 1));

    /* A depth position's input row: projection plus the position's embedding (one bias row of the
     * table), scattered to the position in every sequence. */
    for (int p = 0; p < kCodes; ++p)
        m.op_dproj[p] = rw(b, RAD_OP(b, "gemm_nt_bias",
                                     RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("N", D), RAD_INT("K", D),
                                                RAD_STR("dtype", "bf16")),
                                     RAD_WEIGHTS(m.w_dproj, m.w_dpos)),
                           { m.b_demb }, { m.b_drow });
    m.op_dscat = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", D), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_drow, m.b_idx }, { m.b_dseq });
    m.op_dgather = rows_op(b, "gather_rows", S, D, { m.b_dx, m.b_idx }, { m.b_dsel });
    m.op_dnorm = norm_op(b, S, D, gd.eps, m.w_dnorm, m.b_dsel, m.b_planes);
    m.op_dhead = gemm_op(b, S, kResCodes, D, m.w_dheads, m.b_planes, m.b_dlog);
    for (int p = 1; p < kCodes; ++p) m.op_dsamp[p] = sampler(kResCodes, p, 0, m.b_dlog);
    /* A code's embedding: position 1 reads the language model's table at the semantic code,
     * position p > 1 the residual table at code p - 1's block. */
    auto embed1 = [&](rad_weight t, int64_t off) {
        return rw(b, RAD_OP(b, "rvq_embed",
                            RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_embd", D), RAD_INT("n_codes", 1),
                                       RAD_INT("vocab", kResCodes), RAD_INT("off0", off),
                                       RAD_F64("scale", 1.0), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(t, m.w_demb)),
                  { m.b_codes }, { m.b_demb });
    };
    m.op_demb[1] = embed1(m.w_tok, kCodeOffset);
    for (int p = 2; p < kCodes; ++p) m.op_demb[p] = embed1(m.w_demb, (int64_t)(p - 2) * kResCodes);

    /* ---- the frame's conditioning row, and the next input of the language model */
    m.w_mixl = wcopy(b, nm, nm.f("cond.mix.logits"), "condition_encoder.layer_weight_logits", RAD_F32,
                     { kCodes }, kGrpCond);
    m.w_mixs = wcopy(b, nm, nm.f("cond.mix.scale"), "condition_encoder.layer_scale", RAD_F32, { 1 },
                     kGrpCond);
    m.op_mix = rw(b, RAD_OP(b, "frame_mix_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, lanes), RAD_INT("n_embd", D),
                                       RAD_INT("L", kCodes), RAD_INT("stride", kMaxFrames),
                                       RAD_INT("rep", 2), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.w_mixl, m.w_mixs)),
                  { m.b_planes, m.b_status }, { m.b_mix });
    m.op_fb = rw(b, RAD_OP(b, "rvq_embed",
                           RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_embd", D),
                                      RAD_INT("n_codes", kCodes), RAD_INT("vocab", kResCodes),
                                      RAD_INT("off0", kCodeOffset),
                                      RAD_F64("scale", 1.0 / std::sqrt((double)kCodes)),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(m.w_tok, m.w_demb)),
                 { m.b_codes }, { m.b_x });
    /* The language model's last rows: normed (a prompt's last row a branch, or a frame's row) and
     * into the lane's state. */
    m.op_plast = rows_op(b, "gather_rows", S, D, { m.b_x }, { m.b_dsel });
    m.op_pnorm = norm_op(b, S, D, m.gl.eps, m.w_norm, m.b_dsel, m.b_last);
    m.op_store = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", D), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_last }, { m.b_hid });
    return m.w_dproj && m.w_dpos && m.w_dheads && m.w_demb && m.w_mixl && m.w_mixs ? RAD_OK
                                                                                  : RAD_E_INVAL;
}

/* ================================================================== the window */
static int declare_window(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& gt = m.gt;
    Names& nm = m.nm;
    const int64_t lanes = m.lanes, D = m.gl.n_embd;
    const int64_t C = rad_meta_geti(meta, "transformer.in_channels", 128);
    const int64_t Dc = rad_meta_geti(meta, "transformer.condition_dim", 2048);
    m.C = C;
    m.Dc = Dc;
    const int64_t T = gt.max_tok;

    gt = m.gl;
    gt.n_layer = rad_meta_geti(meta, "transformer.num_layers", 36);
    gt.head_dim = rad_meta_geti(meta, "transformer.attention_head_dim", 64);
    gt.n_head_all = rad_meta_geti(meta, "transformer.num_attention_heads", 32);
    gt.n_head_kv_all = gt.n_head_all;
    gt.n_embd = gt.head_dim * gt.n_head_all;
    gt.n_ff_all = rad_meta_geti(meta, "transformer.ff_inner_dim", 8192);
    gt.eps = 1e-5f;
    gt.theta = 10000.0f;
    gt.rope_mc = false;
    gt.max_tok = m.gl.max_tok;
    RAD_ARCH_TRY(divide_or_fail(gt.n_head_all, gt.world, "the DiT's heads", &gt.n_head));
    gt.n_head_kv = gt.n_head;
    gt.kv_off = 0;
    RAD_ARCH_TRY(divide_or_fail(gt.n_ff_all, gt.world, "the DiT's feed-forward", &gt.n_ff));
    const int64_t H = gt.n_embd, rot = rad_meta_geti(meta, "transformer.rotary_dim", 32);
    const int64_t F = rad_meta_geti(meta, "transformer.fourier_embedding_dim", 256);
    const int64_t Wd = 2 * C + Dc;   /* [lat | zeros | cond] */
    if (gt.head_dim != 64 || C != 128 || Dc != 2048 || F != 256) {
        fprintf(stderr, "radiance: minimaxmusic3: the DiT is not the released one (head %lld, %lld "
                        "latent channels, %lld conditioning)\n", (long long)gt.head_dim,
                (long long)C, (long long)Dc);
        return RAD_E_UNSUPPORTED;
    }

    /* ---- the window states, a lane a song */
    m.b_lat   = decl_b(b, nm.f("win.lat"), RAD_BF16, { lanes * kWinLat, C }, RAD_BUF_PERSIST);
    m.b_wcond = decl_b(b, nm.f("win.cond"), RAD_BF16, { lanes * kWinLat, Dc }, RAD_BUF_PERSIST);
    m.b_prev  = decl_b(b, nm.f("win.prev"), RAD_BF16, { lanes * kOverlap, C }, RAD_BUF_PERSIST);
    m.b_pcond = decl_b(b, nm.f("win.pcond"), RAD_BF16, { lanes * kOverlap, Dc }, RAD_BUF_PERSIST);
    m.b_noise = decl_b(b, nm.f("win.noise"), RAD_BF16, { lanes * kOverlap, C }, RAD_BUF_PERSIST);

    /* ---- the condition encoder: a 3-tap conv over the window's frames, resampled */
    m.b_ccu = decl_b(b, nm.f("win.ccu"), RAD_I32, { 2 });
    m.b_cc  = decl_b(b, nm.f("win.cc"), RAD_BF16, { (int64_t)kWinFrames, Dc });
    m.op_ccu = rw(b, RAD_OP(b, "cu_repeat",
                            RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", 1)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_ccu });
    m.w_cconv = wcopy(b, nm, nm.f("cond.conv.weight"), "condition_encoder.proj.weight", RAD_BF16,
                      { Dc, D, 3 }, kGrpCond);
    m.w_cbias = wcopy(b, nm, nm.f("cond.conv.bias"), "condition_encoder.proj.bias", RAD_BF16, { Dc },
                      kGrpCond);
    m.op_cconv = rw(b, RAD_OP(b, "conv1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, kWinFrames), RAD_INT("cin", D),
                                         RAD_INT("cout", Dc), RAD_INT("k", 3), RAD_INT("dil", 1),
                                         RAD_STR("act", "none"), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.w_cconv, m.w_cbias)),
                    { m.b_mix, m.b_ccu }, { m.b_cc });
    m.op_cres = rows_op(b, "resample_nearest", kWinLat, Dc, { m.b_cc }, { m.b_wcond });
    m.op_cjoin = cast_op(b, (int64_t)kOverlap, Dc, "bf16", "bf16", m.b_pcond, m.b_wcond);

    /* ---- the window step's front: noise and overlap, the input rows, the projections */
    m.op_prep = rw(b, RAD_OP(b, "win_prep",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", C),
                                        RAD_INT("stride", kWinLat), RAD_INT("ov", kOverlap),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_prev, m.b_noise, m.b_lat }, { m.b_noise, m.b_lat });
    m.b_din  = decl_b(b, nm.f("dit.in"), RAD_BF16, { T, Wd });
    m.b_dpre = decl_b(b, nm.f("dit.pre"), RAD_BF16, { T, Wd });
    m.b_hl   = decl_b(b, nm.f("dit.hl"), RAD_BF16, { T, H });
    m.op_win_in = rw(b, RAD_OP(b, "win_input",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_lat", C), RAD_INT("n_zero", C),
                                          RAD_INT("n_cond", Dc), RAD_INT("stride", kWinLat),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_lat, m.b_wcond }, { m.b_din });
    /* preprocess_conv is a 1x1 conv with no bias, then the residual: x = W x + x. */
    m.w_pre = wcopy(b, nm, nm.f("dit.pre.weight"), "transformer.preprocess_conv.weight", RAD_BF16,
                    { Wd, Wd, 1 }, kGrpDit);
    m.op_pre = rw(b, RAD_OP(b, "conv1d",
                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("cin", Wd), RAD_INT("cout", Wd),
                                       RAD_INT("k", 1), RAD_INT("dil", 1), RAD_STR("act", "none"),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.w_pre)),
                  { m.b_din }, { m.b_dpre });
    m.w_pin = wcopy(b, nm, nm.f("dit.proj_in.weight"), "transformer.proj_in.weight", RAD_BF16,
                    { H, Wd }, kGrpDit);
    m.op_pin = gemm_op(b, T, H, Wd, m.w_pin, m.b_dpre, m.b_hl);

    /* ---- the timestep token: Fourier features, a two-linear MLP */
    m.b_tf   = decl_b(b, nm.f("dit.tf"), RAD_BF16, { 64, F });
    m.b_te1  = decl_b(b, nm.f("dit.te1"), RAD_BF16, { 64, H });
    m.b_te1s = decl_b(b, nm.f("dit.te1s"), RAD_BF16, { 64, H });
    m.b_temb = decl_b(b, nm.f("dit.temb"), RAD_BF16, { 64, H });
    m.w_tw = wcopy(b, nm, nm.f("dit.t.fourier"), "transformer.time_proj.weight", RAD_F32,
                   { F / 2, 1 }, kGrpDit);
    m.op_four = rw(b, RAD_OP(b, "fourier_embed",
                             RAD_PARAMS(RAD_RANGE("M", 1, 64), RAD_INT("n", F), RAD_INT("round", 1),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.w_tw)),
                   {}, { m.b_tf });
    m.w_t1 = wcopy(b, nm, nm.f("dit.t.l1.weight"), "transformer.time_embed.linear_1.weight", RAD_BF16,
                   { H, F }, kGrpDit);
    m.w_t1b = wcopy(b, nm, nm.f("dit.t.l1.bias"), "transformer.time_embed.linear_1.bias", RAD_BF16,
                    { H }, kGrpDit);
    m.w_t2 = wcopy(b, nm, nm.f("dit.t.l2.weight"), "transformer.time_embed.linear_2.weight", RAD_BF16,
                   { H, H }, kGrpDit);
    m.w_t2b = wcopy(b, nm, nm.f("dit.t.l2.bias"), "transformer.time_embed.linear_2.bias", RAD_BF16,
                    { H }, kGrpDit);
    m.op_t1 = gemm_bias_op(b, 64, H, F, m.w_t1, m.w_t1b, m.b_tf, m.b_te1);
    m.op_tsilu = rows_op(b, "silu", 64, H, { m.b_te1 }, { m.b_te1s });
    m.op_t2 = gemm_bias_op(b, 64, H, H, m.w_t2, m.w_t2b, m.b_te1s, m.b_temb);

    /* ---- the denoiser's rows: each entry's timestep token, then its latents */
    m.b_h = decl_b(b, nm.f("dit.h"), RAD_BF16, { T, H });
    m.op_hscat = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_hl }, { m.b_h });
    m.op_tscat = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, 64), RAD_INT("n", H), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_temb }, { m.b_h });

    m.a_th.x = decl_b(b, nm.f("dit.hn"), RAD_BF16, { T, H });
    RAD_ARCH_TRY(m.a_th.declare_qs(b, nm, gt, "dit.hn", H));
    m.b_qkv = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { T, 3 * gt.q_dim() });
    m.a_tattn.x = decl_b(b, nm.f("dit.attn"), RAD_BF16, { T, gt.q_dim() });
    RAD_ARCH_TRY(m.a_tattn.declare_qs(b, nm, gt, "dit.attn", gt.q_dim()));
    m.b_th2 = decl_b(b, nm.f("dit.h2"), RAD_BF16, { T, H });
    m.b_ff  = decl_b(b, nm.f("dit.ff"), RAD_BF16, { T, 2 * gt.n_ff });
    m.b_ffa = decl_b(b, nm.f("dit.ffa"), RAD_BF16, { T, gt.n_ff });

    const int64_t q_dim = gt.q_dim();
    m.dit.resize((size_t)gt.n_layer);
    for (int64_t l = 0; l < gt.n_layer; ++l) {
        DitLayer& L = m.dit[(size_t)l];
        const int li = (int)l, grp = kGrpDit + 1 + li;
        const char* pre = "transformer.transformer_blocks";
        auto ln = [&](const char* dn, const char* src, rad_weight* w) {
            *w = wcopy(b, nm, nm.f("dit.%d.%s", li, dn), nm.ckpt("%s.%d.%s", pre, li, src), RAD_F32,
                       { H }, grp);
            return *w ? RAD_OK : RAD_E_INVAL;
        };
        RAD_ARCH_TRY(ln("ln1.weight", "norm1.weight", &L.w_n1));
        RAD_ARCH_TRY(ln("ln1.bias", "norm1.bias", &L.w_b1));
        RAD_ARCH_TRY(ln("ln2.weight", "norm2.weight", &L.w_n2));
        RAD_ARCH_TRY(ln("ln2.bias", "norm2.bias", &L.w_b2));
        auto layernorm = [&](rad_weight w, rad_weight bias) {
            return rw(b, RAD_OP(b, "layernorm",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H),
                                           RAD_F64("eps", gt.eps), RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(w, bias)),
                      { m.b_h }, { m.a_th.x });
        };
        L.op_ln1 = layernorm(L.w_n1, L.w_b1);
        RAD_ARCH_TRY(L.q_h.declare(b, gt, m.a_th, H));
        RAD_ARCH_TRY(L.qkv.declare(b, nm, gt, nm.f("dit.%d.attn_qkv", li), 3 * q_dim, H,
                                   RAD_SHARD_ROW, grp_layer(grp),
                                   { nm.ckpt("%s.%d.attn.to_q.weight", pre, li),
                                     nm.ckpt("%s.%d.attn.to_k.weight", pre, li),
                                     nm.ckpt("%s.%d.attn.to_v.weight", pre, li) },
                                   m.a_th, m.b_qkv, 0, { q_dim, q_dim, q_dim }));
        L.op_rope = rw(b, decl_rope(b, gt, T, gt.head_dim, gt.n_head, gt.n_head, rot, "neox"),
                       { m.b_qkv }, { m.b_qkv });
        L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", gt.head_dim),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)gt.head_dim)),
                                            RAD_INT("max_seqlen", 1 + kWinLat)),
                                 RAD_NOWEIGHTS),
                       { m.b_qkv }, { m.a_tattn.x });
        RAD_ARCH_TRY(L.q_attn.declare(b, gt, m.a_tattn, q_dim));
        RAD_ARCH_TRY(L.o.declare(b, nm, gt, nm.f("dit.%d.attn_output", li), H, q_dim, RAD_SHARD_COL,
                                 grp_layer(grp), { nm.ckpt("%s.%d.attn.to_out.0.weight", pre, li) },
                                 m.a_tattn, m.b_th2));
        L.op_ar1 = decl_all_reduce_tp(b, gt, m.b_th2, T);
        L.op_add1 = add_op(b, T, H, m.b_h, m.b_th2, m.b_h);
        L.op_ln2 = layernorm(L.w_n2, L.w_b2);
        /* The feed-forward's halves are [value | gate], each row-sharded as its own part, and its
         * output projection column-sharded with the bias on rank 0 alone (added once, before the
         * all-reduce sums the ranks). */
        L.w_fi = wcopy(b, nm, nm.f("dit.%d.ffn_in.weight", li), nm.ckpt("%s.%d.ff_in.weight", pre, li),
                       RAD_BF16, { 2 * gt.n_ff, H }, grp, RAD_SHARD_ROW, { gt.n_ff, gt.n_ff });
        L.w_fib = wcopy(b, nm, nm.f("dit.%d.ffn_in.bias", li), nm.ckpt("%s.%d.ff_in.bias", pre, li),
                        RAD_BF16, { 2 * gt.n_ff }, grp, RAD_SHARD_ROW, { gt.n_ff, gt.n_ff });
        L.w_fo = wcopy(b, nm, nm.f("dit.%d.ffn_out.weight", li), nm.ckpt("%s.%d.ff_out.weight", pre, li),
                       RAD_BF16, { H, gt.n_ff }, grp, RAD_SHARD_COL);
        L.w_fob = wcopy(b, nm, nm.f("dit.%d.ffn_out.bias", li), nm.ckpt("%s.%d.ff_out.bias", pre, li),
                        RAD_BF16, { H }, grp);
        L.op_fi = gemm_bias_op(b, T, 2 * gt.n_ff, H, L.w_fi, L.w_fib, m.a_th.x, m.b_ff);
        L.op_act = rw(b, RAD_OP(b, "act_mul",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", gt.n_ff),
                                           RAD_STR("act", "silu"), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_ff }, { m.b_ffa });
        L.op_fo = gt.rank == 0 ? gemm_bias_op(b, T, H, gt.n_ff, L.w_fo, L.w_fob, m.b_ffa, m.b_th2)
                               : gemm_op(b, T, H, gt.n_ff, L.w_fo, m.b_ffa, m.b_th2);
        L.op_ar2 = decl_all_reduce_tp(b, gt, m.b_th2, T);
        L.op_add2 = add_op(b, T, H, m.b_h, m.b_th2, m.b_h);
        if (!L.op_ln1 || !L.op_rope || !L.op_attn || !L.op_ln2 || !L.op_fi || !L.op_act || !L.op_fo)
            return RAD_E_INVAL;
    }

    /* ---- the output: the latent rows, proj_out, postprocess_conv's residual, the Euler step */
    m.b_ho = decl_b(b, nm.f("dit.ho"), RAD_BF16, { T, H });
    m.b_po = decl_b(b, nm.f("dit.po"), RAD_BF16, { T, C });
    m.b_vel = decl_b(b, nm.f("dit.v"), RAD_BF16, { T, C });
    m.op_hgather = rows_op(b, "gather_rows", T, H, { m.b_h }, { m.b_ho });
    m.w_pout = wcopy(b, nm, nm.f("dit.proj_out.weight"), "transformer.proj_out.weight", RAD_BF16,
                     { C, H }, kGrpDit);
    m.op_pout = gemm_op(b, T, C, H, m.w_pout, m.b_ho, m.b_po);
    m.w_post = wcopy(b, nm, nm.f("dit.post.weight"), "transformer.postprocess_conv.weight", RAD_BF16,
                     { C, C, 1 }, kGrpDit);
    m.op_post = rw(b, RAD_OP(b, "conv1d",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("cin", C), RAD_INT("cout", C),
                                        RAD_INT("k", 1), RAD_INT("dil", 1), RAD_STR("act", "none"),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.w_post)),
                   { m.b_po }, { m.b_vel });
    m.op_euler = rw(b, RAD_OP(b, "flow_euler",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", C),
                                         RAD_INT("stride", kWinLat), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_vel, m.b_lat }, { m.b_lat });
    m.op_finish = rw(b, RAD_OP(b, "win_finish",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", C), RAD_INT("n_cond", Dc),
                                          RAD_INT("stride", kWinLat), RAD_INT("ov", kOverlap),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_wcond, m.b_lat, m.b_prev, m.b_pcond }, { m.b_lat, m.b_prev, m.b_pcond });
    return m.op_cconv && m.op_pre && m.op_pin && m.op_four && m.op_t1 && m.op_t2 && m.op_pout &&
                   m.op_post && m.op_euler && m.op_finish && m.op_prep && m.op_win_in
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== the vocoder */
/* A weight-normalised conv's weight (folded at convert) and bias. */
static int vconv(RadBuilder* b, Names& nm, VConv& v, const char* dn, const char* src, int64_t cin,
                 int64_t cout, int64_t k, int64_t dil, bool wnorm, int64_t rows, rad_buf x,
                 rad_buf res, rad_buf y, const char* act, rad_buf cu) {
    v.cin = cin; v.cout = cout; v.k = k; v.dil = dil;
    const char* wn = nm.f("%s.weight", dn);
    if (wnorm) RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.weight_g", src), nm.ckpt("%s.weight_v", src)));
    else RAD_ARCH_TRY(map_copy(b, wn, nm.ckpt("%s.weight", src)));
    v.w = decl_w(b, wn, RAD_BF16, { cout, cin, k }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                 grp_layer(kGrpVoc));
    v.b = wcopy(b, nm, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), RAD_BF16, { cout }, kGrpVoc);
    if (!v.w || !v.b) return RAD_E_INVAL;
    v.op = res ? rw(b, RAD_OP(b, "conv1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", cin),
                                         RAD_INT("cout", cout), RAD_INT("k", k), RAD_INT("dil", dil),
                                         RAD_STR("act", act), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(v.w, v.b)),
                    { x, cu, res }, { y })
               : rw(b, RAD_OP(b, "conv1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", cin),
                                         RAD_INT("cout", cout), RAD_INT("k", k), RAD_INT("dil", dil),
                                         RAD_STR("act", act), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(v.w, v.b)),
                    { x, cu }, { y });
    return v.op ? RAD_OK : RAD_E_INVAL;
}

static rad_weight valpha(RadBuilder* b, Names& nm, const char* dn, const char* src, int64_t n) {
    return wcopy(b, nm, nm.f("%s.alpha", dn), nm.ckpt("%s.alpha", src), RAD_F32, { 1, n, 1 }, kGrpVoc);
}

static rad_op snake_op(RadBuilder* b, int64_t rows, int64_t n, rad_weight a, rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "snake",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(a)),
              { x }, { y });
}

static rad_op cu_op(RadBuilder* b, int64_t mult, rad_buf y) {
    return rw(b, RAD_OP(b, "cu_repeat",
                        RAD_PARAMS(RAD_RANGE("M", 1, 3), RAD_INT("mult", mult)),
                        RAD_NOWEIGHTS),
              {}, { y });
}

static int declare_vocoder(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const int64_t lat = rad_meta_geti(meta, "vocoder.latent_channels", 128);
    const int64_t din = rad_meta_geti(meta, "vocoder.decoder_input_dim", 1024);
    const int64_t dh = rad_meta_geti(meta, "vocoder.decoder_hidden_dim", 1536);
    if (lat != m.C || rad_meta_geti(meta, "vocoder.sampling_rate", kSampleRate) != kSampleRate) {
        fprintf(stderr, "radiance: minimaxmusic3: the vocoder is not the released one\n");
        return RAD_E_UNSUPPORTED;
    }
    /* Rows at each stage: the window's two streams, stage 0 at the latent rate. */
    const int64_t half = lat / 2;
    int64_t rows[5];
    rows[0] = 2 * (int64_t)kWinLat;
    for (int i = 0; i < 4; ++i) rows[i + 1] = rows[i] * m.v_ups[i];
    int64_t ch[5];
    ch[0] = dh;
    for (int i = 0; i < 4; ++i) ch[i + 1] = dh >> (i + 1);

    m.b_vx = decl_b(b, nm.f("voc.x"), RAD_BF16, { rows[0], half });
    m.op_vsplit = rw(b, RAD_OP(b, "chan_split",
                               RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)kWinLat), RAD_INT("n", half),
                                          RAD_INT("parts", 2), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_lat }, { m.b_vx });
    for (int s = 0; s < 5; ++s) {
        int64_t mult = 1;
        for (int i = 0; i < s; ++i) mult *= m.v_ups[i];
        m.b_vcu[s] = decl_b(b, nm.f("voc.cu%d", s), RAD_I32, { 3 });
        if (s == 0) m.op_vcu0 = cu_op(b, 1, m.b_vcu[0]);
        else m.vb[s - 1].op_cu = cu_op(b, mult, m.b_vcu[s]);
        m.b_vstage[s] = decl_b(b, nm.f("voc.s%d", s), RAD_BF16, { rows[s], ch[s] });
        m.b_vt[s] = decl_b(b, nm.f("voc.t%d", s), RAD_BF16, { rows[s], ch[s] });
        /* The residual units' two scratch planes; stage 0 has no residual units. */
        if (s > 0) {
            m.b_va[s] = decl_b(b, nm.f("voc.a%d", s), RAD_BF16, { rows[s], ch[s] });
            m.b_vb[s] = decl_b(b, nm.f("voc.b%d", s), RAD_BF16, { rows[s], ch[s] });
        }
    }
    /* dec_in_proj (1x1, plain weight) into the input width, conv_in to the hidden width. */
    m.b_vin = decl_b(b, nm.f("voc.in"), RAD_BF16, { rows[0], din });
    RAD_ARCH_TRY(vconv(b, nm, m.v_in, "voc.in_proj", "vocoder.dec_in_proj", half, din, 1, 1, false,
                       rows[0], m.b_vx, 0, m.b_vin, "none", m.b_vcu[0]));
    RAD_ARCH_TRY(vconv(b, nm, m.v_conv, "voc.conv_in", "vocoder.conv_in", din, dh, 7, 1, true, rows[0],
                       m.b_vin, 0, m.b_vstage[0], "none", m.b_vcu[0]));
    for (int i = 0; i < 4; ++i) {
        VBlock& B = m.vb[i];
        const int64_t ci = ch[i], co = ch[i + 1], s = m.v_ups[i];
        B.cin = ci; B.cout = co; B.stride = s;
        const char* bn = nm.f("voc.b%d", i);
        const char* bs = nm.ckpt("vocoder.blocks.%d", i);
        B.a = valpha(b, nm, nm.f("%s.snake", bn), nm.ckpt("%s.snake1", bs), ci);
        B.op_s = snake_op(b, rows[i], ci, B.a, m.b_vstage[i], m.b_vt[i]);
        const char* wn = nm.f("%s.conv_t.weight", bn);
        RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.conv_t1.weight_g", bs), nm.ckpt("%s.conv_t1.weight_v", bs)));
        B.wt = decl_w(b, wn, RAD_BF16, { ci, co, 2 * s }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                      grp_layer(kGrpVoc));
        B.bt = wcopy(b, nm, nm.f("%s.conv_t.bias", bn), nm.ckpt("%s.conv_t1.bias", bs), RAD_BF16,
                     { co }, kGrpVoc);
        B.op_t = rw(b, RAD_OP(b, "conv_t1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows[i]), RAD_INT("cin", ci), RAD_INT("cout", co),
                                         RAD_INT("stride", s), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(B.wt, B.bt)),
                    { m.b_vt[i], m.b_vcu[i] }, { m.b_vstage[i + 1] });
        if (!B.wt || !B.bt || !B.op_t || !B.op_s) return RAD_E_INVAL;
        /* Three residual units (dilations 1, 3, 9): x + conv2(snake2(conv1(snake1(x)))), x the
         * stage buffer, written back in place by conv2's residual. */
        static const int64_t kDil[3] = { 1, 3, 9 };
        for (int r = 0; r < 3; ++r) {
            VRes& U = B.res[r];
            const char* un = nm.f("%s.res%d", bn, r);
            const char* us = nm.ckpt("%s.res_unit%d", bs, r + 1);
            U.a1 = valpha(b, nm, nm.f("%s.snake1", un), nm.ckpt("%s.snake1", us), co);
            U.a2 = valpha(b, nm, nm.f("%s.snake2", un), nm.ckpt("%s.snake2", us), co);
            const int64_t R = rows[i + 1];
            U.op_s1 = snake_op(b, R, co, U.a1, m.b_vstage[i + 1], m.b_vt[i + 1]);
            RAD_ARCH_TRY(vconv(b, nm, U.c1, nm.f("%s.conv1", un), nm.ckpt("%s.conv1", us), co, co, 7,
                               kDil[r], true, R, m.b_vt[i + 1], 0, m.b_va[i + 1], "none",
                               m.b_vcu[i + 1]));
            U.op_s2 = snake_op(b, R, co, U.a2, m.b_va[i + 1], m.b_vb[i + 1]);
            RAD_ARCH_TRY(vconv(b, nm, U.c2, nm.f("%s.conv2", un), nm.ckpt("%s.conv2", us), co, co, 1, 1,
                               true, R, m.b_vb[i + 1], m.b_vstage[i + 1], m.b_vstage[i + 1], "none",
                               m.b_vcu[i + 1]));
            if (!U.a1 || !U.a2 || !U.op_s1 || !U.op_s2) return RAD_E_INVAL;
        }
    }
    /* snake_out, conv_out to one channel with tanh, and the two streams interleaved. */
    m.v_aout = valpha(b, nm, nm.f("voc.snake_out"), nm.ckpt("vocoder.snake_out"), ch[4]);
    m.op_vsout = snake_op(b, rows[4], ch[4], m.v_aout, m.b_vstage[4], m.b_vt[4]);
    m.b_vout = decl_b(b, nm.f("voc.out"), RAD_BF16, { rows[4], 1 });
    RAD_ARCH_TRY(vconv(b, nm, m.v_out, "voc.conv_out", "vocoder.conv_out", ch[4], 1, 7, 1, true,
                       rows[4], m.b_vt[4], 0, m.b_vout, "tanh", m.b_vcu[4]));
    m.b_audio = decl_b(b, nm.f("voc.audio"), RAD_F32, { rows[4] / 2, 2 }, RAD_BUF_PERSIST);
    m.op_vmix = rw(b, RAD_OP(b, "interleave2",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows[4] / 2), RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_vout }, { m.b_audio });
    return m.v_aout && m.op_vsout && m.op_vmix && m.op_vsplit ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    const char* cls = rad_meta_gets(meta, "pipeline._class_name", "");
    if (std::strcmp(cls, "MiniMaxMusic3ModularPipeline") != 0) {
        fprintf(stderr, "radiance: minimaxmusic3: the container's pipeline is '%s', not "
                        "MiniMaxMusic3ModularPipeline\n", cls);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "n_vocab", 0) != 200000 &&
        rad_meta_geti(meta, "vocab_size", 0) != 200000) {
        fprintf(stderr, "radiance: minimaxmusic3: the language model's vocabulary is not the "
                        "released 200000-entry one\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    RAD_ARCH_TRY(geom_from(m.gl, meta, ctx, "bf16", RAD_BF16));
    m.lanes = std::min<int64_t>(kMaxLanes, std::max<int64_t>(1, ctx->max_seqs / 2));
    m.S = 2 * m.lanes;
    m.gt.max_tok = m.gl.max_tok;
    if (m.gl.max_tok < 2 * (1 + (int64_t)kWinLat)) {
        fprintf(stderr, "radiance: minimaxmusic3: a pass carries %lld rows; a window's step needs "
                        "%lld\n", (long long)m.gl.max_tok, (long long)(2 * (1 + kWinLat)));
        return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(declare_lm(b, meta, ctx, m));
    RAD_ARCH_TRY(declare_frame(b, meta, m));
    RAD_ARCH_TRY(declare_window(b, meta, m));
    RAD_ARCH_TRY(declare_vocoder(b, meta, m));

    if (!ctx->shape_probe) {
        RadAudioGenDecl d{};
        d.name = "minimax-music3";
        d.tmpl = kTemplate;
        d.neg_token = kCfgToken;
        d.neg_keep_head = 1;
        d.neg_keep_tail = 2;
        d.max_prompt = kMaxPrompt;
        d.max_frames = kMaxFrames;
        d.frame_rate_num = 25;
        d.frame_rate_den = 1;
        d.win_frames = kWinFrames;
        d.win_hop = kWinHop;
        d.win_lead = 1;
        d.lat_num = kLatNum;
        d.lat_den = kLatDen;
        d.latent_channels = m.C;
        d.win_crop_left = kCropLeft;
        d.win_crop_right = kCropRight;
        d.samples_per_latent = kSamplesPerLatent;
        d.sample_rate = kSampleRate;
        d.channels = 2;
        d.default_steps = kDitSteps;
        d.default_cfg = kDitCfg;
        d.max_lanes = m.lanes;
        d.status = m.b_status;
        d.audio_out = m.b_audio;
        d.first_frame = 1;
        RAD_ARCH_TRY(rad_declare_audio_generator(b, &d));
    }
    rad_note(b, "rank %d/%d: MiniMax-Music3 -- language model %lld layers (%lld heads, %lld KV "
                "a rank), depth decoder %lld layers, DiT %lld blocks (%lld heads a rank), %s "
                "linears, %lld lanes",
             m.gl.rank, m.gl.world, (long long)m.gl.n_layer, (long long)m.gl.n_head,
             (long long)m.gl.n_head_kv, (long long)m.gd.n_layer, (long long)m.gt.n_layer,
             (long long)m.gt.n_head, m.gl.w_bf16 ? "bf16" : "block-fp8", (long long)m.lanes);
    return RAD_OK;
}

/* ================================================================== step */
/* The language model over the pass's rows in `b_x`, then the final norm and the store of the
 * lanes' hidden states from the rows `last` names (a prompt's last row a branch, or every row of a
 * frame). */
static void step_lm(RadCtx* c, const RadBatch* batch, const Model& m) {
    const Geom& g = m.gl;
    const int64_t T = batch->n_tok;
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);
    for (const LmLayer& L : m.lm)
        step_qwen3_layer(c, m, g, L, batch, T, m.b_x, m.b_x, m.a_h, m.b_q, m.b_k, m.b_v, m.a_attn,
                         m.b_h2, m.b_gu, m.a_ffn, cu);
}

static void store_hidden(RadCtx* c, const Model& m, const RadAudioGenBatch* A, int64_t S,
                         const int32_t* last_rows, int64_t T) {
    RAD_ISSUE_N(c, m.op_plast, S, brows(m.b_x, T), praw(last_rows, RAD_I32, S), brows(m.b_dsel, S));
    RAD_ISSUE_N(c, m.op_pnorm, S, brows(m.b_dsel, S), RAD_W(m.w_norm), brows(m.b_last, S));
    RAD_ISSUE_N(c, m.op_store, S, brows(m.b_last, S), praw(A->slot, RAD_I32, S),
                brows(m.b_hid, 2 * m.lanes));
}

static void step_prompt(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    step_lm(c, batch, m);
    bool any_last = false;
    for (int64_t i = 0; i < A->n_lane; ++i) any_last = any_last || A->last_h[i];
    if (any_last) store_hidden(c, m, A, batch->n_seq, A->last_row, T);
}

static void step_frame(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t n = A->n_lane, S = 2 * n, D = m.gl.n_embd, R = (int64_t)kCodes * S;
    const int64_t Sx = m.S;   /* the codes' and indices' row pitch */
    const RadOperand seed = praw(A->seed, RAD_U32, 2 * n), frame = praw(A->frame, RAD_I32, n);
    const RadOperand lanes = praw(A->lane, RAD_I32, n), slot = praw(A->slot, RAD_I32, S);
    auto codes = [&](int p) { return brow_slice(m.b_codes, p, 1, Sx); };
    auto idx = [&](int p) { return brow_slice(m.b_idx, p, 1, Sx); };
    auto plane = [&](int p) { return brow_slice(m.b_planes, (int64_t)p * Sx, S, D); };
    /* The positions' row indices into the depth sequence (constant, cheap to remake). */
    for (int p = 0; p < kCodes; ++p) {
        RadOperand o = idx(p);
        o.cols = S;
        RAD_ISSUE_N(c, m.op_iota[p], S, o);
    }
    auto idx_s = [&](int p) { RadOperand o = idx(p); o.cols = S; return o; };
    auto codes_s = [&](int p) { RadOperand o = codes(p); o.cols = S; return o; };

    /* 1. the semantic code, from the lanes' hidden states (plane 0) */
    RAD_ISSUE_N(c, m.op_hsel, S, brows(m.b_hid, 2 * m.lanes), slot, plane(0));
    RAD_ISSUE_N(c, m.op_sem, S, plane(0), RAD_W(m.w_head), brows(m.b_slog, S));
    RAD_ISSUE_N(c, m.op_dsamp[0], S, brows(m.b_slog, S), seed, frame, codes_s(0), RAD_NONE,
                RAD_NONE, RAD_NONE);
    RAD_ISSUE_N(c, m.op_status, n, codes_s(0), lanes, brows(m.b_status, m.lanes));

    /* 2. the depth decoder's sequence: the hidden state at position 0, the semantic code's
     * embedding at 1, then a code a step. */
    auto pos_row = [&](int p) { RadOperand o = RAD_W(m.w_dpos); o.offset = (int64_t)p * D; o.rows = 1; return o; };
    RAD_ISSUE_N(c, m.op_dproj[0], S, plane(0), RAD_W(m.w_dproj), pos_row(0), RAD_NONE,
                brows(m.b_drow, S));
    RAD_ISSUE_N(c, m.op_dscat, S, brows(m.b_drow, S), idx_s(0), brows(m.b_dseq, R));
    RAD_ISSUE_N(c, m.op_demb[1], S, codes(0), RAD_W(m.w_tok), RAD_W(m.w_demb), brows(m.b_demb, S));
    RAD_ISSUE_N(c, m.op_dproj[1], S, brows(m.b_demb, S), RAD_W(m.w_dproj), pos_row(1), RAD_NONE,
                brows(m.b_drow, S));
    RAD_ISSUE_N(c, m.op_dscat, S, brows(m.b_drow, S), idx_s(1), brows(m.b_dseq, R));
    /* The depth sequences' bounds: 8 rows each, from the frame's one row a sequence. */
    RAD_ISSUE_N(c, m.op_dcu, S + 1, praw(batch->cu_seqlens + 1, RAD_I32, 1), brows(m.b_dcu, S + 1));
    for (int p = 1; p < kCodes; ++p) {
        /* The depth stack over every sequence's eight rows; causal attention keeps the rows past
         * position p out of the rows it reads. The sequences are 8 rows each: the batch's
         * cu_seqlens over 2n frame rows times 8, made by the window op's sibling below. */
        const RadOperand cu8 = brows(m.b_dcu, S + 1);
        for (const LmLayer& L : m.dep)
            step_qwen3_layer(c, m, m.gd, L, batch, R, m.b_dx, &L == &m.dep[0] ? m.b_dseq : m.b_dx,
                             m.a_dh, m.b_dq, m.b_dk, m.b_dv, m.a_dattn, m.b_dh2, m.b_dgu, m.a_dffn,
                             cu8);
        RAD_ISSUE_N(c, m.op_dgather, S, brows(m.b_dx, R), idx_s(p), brows(m.b_dsel, S));
        RAD_ISSUE_N(c, m.op_dnorm, S, brows(m.b_dsel, S), RAD_W(m.w_dnorm), plane(p));
        RadOperand head = RAD_W(m.w_dheads);
        head.offset = (int64_t)(p - 1) * kResCodes * D;
        head.rows = kResCodes;
        RAD_ISSUE_N(c, m.op_dhead, S, plane(p), head, brows(m.b_dlog, S));
        RAD_ISSUE_N(c, m.op_dsamp[p], S, brows(m.b_dlog, S), seed, frame, codes_s(p), RAD_NONE,
                    RAD_NONE, RAD_NONE);
        if (p + 1 < kCodes) {
            RAD_ISSUE_N(c, m.op_demb[p + 1], S, codes(p), RAD_W(m.w_demb), RAD_W(m.w_demb),
                        brows(m.b_demb, S));
            RAD_ISSUE_N(c, m.op_dproj[p + 1], S, brows(m.b_demb, S), RAD_W(m.w_dproj),
                        pos_row(p + 1), RAD_NONE, brows(m.b_drow, S));
            RAD_ISSUE_N(c, m.op_dscat, S, brows(m.b_drow, S), idx_s(p + 1), brows(m.b_dseq, R));
        }
    }

    /* 3. the frame's conditioning row, and the language model over the frame's embedding */
    RAD_ISSUE_N(c, m.op_mix, n, brows(m.b_planes, (int64_t)kCodes * Sx), RAD_W(m.w_mixl),
                RAD_W(m.w_mixs), lanes, frame, brows(m.b_status, m.lanes),
                brows(m.b_mix, m.lanes * kMaxFrames));
    RAD_ISSUE_N(c, m.op_fb, S, brows(m.b_codes, kCodes), RAD_W(m.w_tok), RAD_W(m.w_demb),
                brows(m.b_x, S));
    step_lm(c, batch, m);
    store_hidden(c, m, A, S, A->last_row, S);
}

static void step_window(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t S = batch->n_seq, R = batch->n_tok, NL = A->lat_rows;
    const int64_t Dc = m.Dc, D = m.gl.n_embd, H = m.gt.n_embd;
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const RadOperand lcu = praw(A->lat_cu, RAD_I32, S + 1);
    const RadOperand wl = praw(A->win_lane, RAD_I32, S), pair = praw(A->pair, RAD_I32, S);
    const RadOperand fl = praw(A->flags, RAD_I32, S);
    const int64_t LS = m.lanes * kWinLat, LO = m.lanes * kOverlap;

    /* 0. a window's first step: its conditioning from the frames (each window on its own) */
    for (int64_t e = 0; e < S; ++e) {
        if (!(A->flags_h[e] & RAD_AGEN_F_FIRST) || (e > 0 && A->win_lane_h[e] == A->win_lane_h[e - 1]))
            continue;
        const int64_t L = A->win_lane_h[e], f0 = A->win_frame0_h[e], nf = A->win_frames_h[e];
        const int64_t nl = A->lat_cu_h[e + 1] - A->lat_cu_h[e];
        RAD_ISSUE_N(c, m.op_ccu, 2, praw(A->win_frames + e, RAD_I32, 1), brows(m.b_ccu, 2));
        RAD_ISSUE_N(c, m.op_cconv, nf, brow_slice(m.b_mix, L * kMaxFrames + f0, nf, D),
                    RAD_W(m.w_cconv), brows(m.b_ccu, 2), RAD_NONE, brows(m.b_cc, nf),
                    RAD_W(m.w_cbias));
        RAD_ISSUE_N(c, m.op_cres, nl, brows(m.b_cc, nf), brow_slice(m.b_wcond, L * kWinLat, nl, Dc));
        if (A->flags_h[e] & RAD_AGEN_F_JOIN) {
            const int64_t ov = std::min<int64_t>(kOverlap, nl);
            RAD_ISSUE_N(c, m.op_cjoin, ov, brow_slice(m.b_pcond, L * kOverlap, ov, Dc),
                        brow_slice(m.b_wcond, L * kWinLat, ov, Dc));
        }
    }
    /* 1. noise and the overlap's binding, the input rows, preprocess_conv, proj_in */
    RAD_ISSUE_N(c, m.op_prep, NL, fl, praw(A->t, RAD_F32, S), praw(A->seed, RAD_U32, 2 * S),
                praw(A->win_index, RAD_I32, S), wl, pair, lcu, brows(m.b_prev, LO),
                brows(m.b_noise, LO), brows(m.b_lat, LS));
    RAD_ISSUE_N(c, m.op_win_in, NL, brows(m.b_lat, LS), brows(m.b_wcond, LS), lcu, wl, pair,
                brows(m.b_din, NL));
    RAD_ISSUE_N(c, m.op_pre, NL, brows(m.b_din, NL), RAD_W(m.w_pre), lcu, brows(m.b_din, NL),
                brows(m.b_dpre, NL), RAD_NONE);
    RAD_ISSUE_N(c, m.op_pin, NL, brows(m.b_dpre, NL), RAD_W(m.w_pin), brows(m.b_hl, NL));
    /* 2. the timestep tokens, and every row in its place */
    RAD_ISSUE_N(c, m.op_four, S, praw(A->t, RAD_F32, S), RAD_W(m.w_tw), brows(m.b_tf, S));
    RAD_ISSUE_N(c, m.op_t1, S, brows(m.b_tf, S), RAD_W(m.w_t1), RAD_W(m.w_t1b), RAD_NONE,
                brows(m.b_te1, S));
    RAD_ISSUE_N(c, m.op_tsilu, S, brows(m.b_te1, S), brows(m.b_te1s, S));
    RAD_ISSUE_N(c, m.op_t2, S, brows(m.b_te1s, S), RAD_W(m.w_t2), RAD_W(m.w_t2b), RAD_NONE,
                brows(m.b_temb, S));
    RAD_ISSUE_N(c, m.op_hscat, NL, brows(m.b_hl, NL), praw(A->lat_src, RAD_I32, NL), brows(m.b_h, R));
    RAD_ISSUE_N(c, m.op_tscat, S, brows(m.b_temb, S), praw(batch->cu_seqlens, RAD_I32, S),
                brows(m.b_h, R));
    /* 3. the blocks */
    const int64_t qd = m.gt.q_dim();
    const RadOperand pos = praw(batch->positions, RAD_I32, R);
    for (const DitLayer& L : m.dit) {
        RAD_ISSUE(c, L.op_ln1, brows(m.b_h, R), RAD_W(L.w_n1), RAD_W(L.w_b1), brows(m.a_th.x, R));
        L.q_h.step(c, R);
        L.qkv.step(c, m.a_th, m.b_qkv, R);
        RAD_ISSUE(c, L.op_rope, brows(m.b_qkv, R), pos);
        RAD_ISSUE_N(c, L.op_attn, R, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                    bcol(m.b_qkv, 2 * qd, qd, R), cu, brows(m.a_tattn.x, R));
        L.q_attn.step(c, R);
        L.o.step(c, m.a_tattn, m.b_th2, R);
        if (L.op_ar1) RAD_ISSUE_N(c, L.op_ar1, R * H, brows(m.b_th2, R), RAD_NONE);
        RAD_ISSUE(c, L.op_add1, brows(m.b_h, R), brows(m.b_th2, R), brows(m.b_h, R));
        RAD_ISSUE(c, L.op_ln2, brows(m.b_h, R), RAD_W(L.w_n2), RAD_W(L.w_b2), brows(m.a_th.x, R));
        RAD_ISSUE(c, L.op_fi, brows(m.a_th.x, R), RAD_W(L.w_fi), RAD_W(L.w_fib), RAD_NONE,
                  brows(m.b_ff, R));
        const int64_t f = m.gt.n_ff;
        RAD_ISSUE(c, L.op_act, bcol(m.b_ff, f, f, R), bcol(m.b_ff, 0, f, R), brows(m.b_ffa, R));
        if (m.gt.rank == 0)
            RAD_ISSUE(c, L.op_fo, brows(m.b_ffa, R), RAD_W(L.w_fo), RAD_W(L.w_fob), RAD_NONE,
                      brows(m.b_th2, R));
        else
            RAD_ISSUE(c, L.op_fo, brows(m.b_ffa, R), RAD_W(L.w_fo), brows(m.b_th2, R));
        if (L.op_ar2) RAD_ISSUE_N(c, L.op_ar2, R * H, brows(m.b_th2, R), RAD_NONE);
        RAD_ISSUE(c, L.op_add2, brows(m.b_h, R), brows(m.b_th2, R), brows(m.b_h, R));
    }
    /* 4. the latent rows out, the velocity, the step, the window's end */
    RAD_ISSUE_N(c, m.op_hgather, NL, brows(m.b_h, R), praw(A->lat_src, RAD_I32, NL), brows(m.b_ho, NL));
    RAD_ISSUE_N(c, m.op_pout, NL, brows(m.b_ho, NL), RAD_W(m.w_pout), brows(m.b_po, NL));
    RAD_ISSUE_N(c, m.op_post, NL, brows(m.b_po, NL), RAD_W(m.w_post), lcu, brows(m.b_po, NL),
                brows(m.b_vel, NL), RAD_NONE);
    RAD_ISSUE_N(c, m.op_euler, NL, brows(m.b_vel, NL), praw(A->t, RAD_F32, S),
                praw(A->t_next, RAD_F32, S), praw(A->cfg, RAD_F32, S), pair, wl, fl,
                praw(A->seed, RAD_U32, 2 * S), lcu, RAD_NONE, RAD_NONE, brows(m.b_lat, LS), RAD_NONE);
    RAD_ISSUE_N(c, m.op_finish, NL, fl, pair, wl, lcu, brows(m.b_wcond, LS), brows(m.b_lat, LS),
                brows(m.b_prev, LO), brows(m.b_pcond, LO));
}

static void step_vocode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t nl = A->voc_latents;
    int64_t rows[5];
    rows[0] = 2 * nl;
    for (int i = 0; i < 4; ++i) rows[i + 1] = rows[i] * m.v_ups[i];
    const RadOperand nlen = praw(batch->cu_seqlens + 1, RAD_I32, 1);
    RAD_ISSUE_N(c, m.op_vsplit, nl, brow_slice(m.b_lat, (int64_t)A->voc_lane * kWinLat, nl, m.C),
                brows(m.b_vx, 2 * nl));
    RAD_ISSUE_N(c, m.op_vcu0, 3, nlen, brows(m.b_vcu[0], 3));
    auto conv = [&](const VConv& v, rad_buf x, rad_buf res, rad_buf y, int64_t R, rad_buf cu) {
        RAD_ISSUE_N(c, v.op, R, brows(x, R), RAD_W(v.w), brows(cu, 3),
                    res ? brows(res, R) : RAD_NONE, brows(y, R), RAD_W(v.b));
    };
    conv(m.v_in, m.b_vx, 0, m.b_vin, rows[0], m.b_vcu[0]);
    conv(m.v_conv, m.b_vin, 0, m.b_vstage[0], rows[0], m.b_vcu[0]);
    for (int i = 0; i < 4; ++i) {
        const VBlock& B = m.vb[i];
        const int64_t r0 = rows[i], r1 = rows[i + 1];
        RAD_ISSUE_N(c, B.op_cu, 3, nlen, brows(m.b_vcu[i + 1], 3));
        RAD_ISSUE_N(c, B.op_s, r0, brows(m.b_vstage[i], r0), RAD_W(B.a), brows(m.b_vt[i], r0),
                    RAD_NONE);
        RAD_ISSUE_N(c, B.op_t, r0, brows(m.b_vt[i], r0), RAD_W(B.wt), RAD_W(B.bt),
                    brows(m.b_vcu[i], 3), brows(m.b_vstage[i + 1], r1));
        for (int r = 0; r < 3; ++r) {
            const VRes& U = B.res[r];
            RAD_ISSUE_N(c, U.op_s1, r1, brows(m.b_vstage[i + 1], r1), RAD_W(U.a1),
                        brows(m.b_vt[i + 1], r1), RAD_NONE);
            conv(U.c1, m.b_vt[i + 1], 0, m.b_va[i + 1], r1, m.b_vcu[i + 1]);
            RAD_ISSUE_N(c, U.op_s2, r1, brows(m.b_va[i + 1], r1), RAD_W(U.a2),
                        brows(m.b_vb[i + 1], r1), RAD_NONE);
            conv(U.c2, m.b_vb[i + 1], m.b_vstage[i + 1], m.b_vstage[i + 1], r1, m.b_vcu[i + 1]);
        }
    }
    RAD_ISSUE_N(c, m.op_vsout, rows[4], brows(m.b_vstage[4], rows[4]), RAD_W(m.v_aout),
                brows(m.b_vt[4], rows[4]), RAD_NONE);
    conv(m.v_out, m.b_vt[4], 0, m.b_vout, rows[4], m.b_vcu[4]);
    /* The two streams, cropped, interleaved into what the core reads back. */
    const int64_t L5 = rows[4] / 2;
    const int64_t off = (int64_t)A->voc_left * kSamplesPerLatent;
    const int64_t n = (int64_t)(A->voc_latents - A->voc_left - A->voc_right) * kSamplesPerLatent;
    RAD_ISSUE_N(c, m.op_vmix, n, brow_slice(m.b_vout, off, n, 1), brow_slice(m.b_vout, L5 + off, n, 1),
                brows(m.b_audio, n));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->agen_b) {
        rad_step_fail(c, "minimaxmusic3 is an audio generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_PROMPT: step_prompt(c, batch, m); return;
    case RAD_GEN_PASS_FRAME:  step_frame(c, batch, m); return;
    case RAD_GEN_PASS_WINDOW: step_window(c, batch, m); return;
    case RAD_GEN_PASS_VOCODE: step_vocode(c, batch, m); return;
    default:
        rad_step_fail(c, "minimaxmusic3: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows a pass carries: a window's step with guidance (two entries of a timestep row and the
 * window's latents), and the longest sequence a branch holds. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->gen_rows = 2048;
    out->gen_audio = 1;
    out->gen_ctx = (int64_t)kMaxPrompt + kMaxFrames + 1;
    return RAD_OK;
}

/* ================================================================== the request's texts */
/* diffusers' _clean_caption: <|tag inner|> becomes "tag is inner", markdown heads, bullets,
 * emphasis and rules are dropped, and runs of blank lines collapse. */
static std::string clean_caption(const std::string& in) {
    std::string t;
    for (size_t i = 0; i < in.size();) {
        if (in.compare(i, 2, "<|") == 0) {
            const size_t e = in.find("|>", i + 2);
            if (e != std::string::npos) {
                std::string inner = in.substr(i + 2, e - i - 2);
                size_t a = inner.find_first_not_of(" \t"), z = inner.find_last_not_of(" \t");
                inner = a == std::string::npos ? std::string() : inner.substr(a, z - a + 1);
                const size_t sp = inner.find_first_of(" \t\n");
                if (sp != std::string::npos) {
                    const size_t r = inner.find_first_not_of(" \t\n", sp);
                    t += inner.substr(0, sp) + " is " + (r == std::string::npos ? "" : inner.substr(r));
                } else {
                    t += inner;
                }
                i = e + 2;
                continue;
            }
        }
        t += in[i++];
    }
    std::string out;
    size_t pos = 0;
    bool first = true;
    while (pos <= t.size()) {
        size_t nl = t.find('\n', pos);
        std::string line = t.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        /* a markdown heading: up to three spaces, 1-6 #, a space */
        {
            size_t k = 0;
            while (k < line.size() && k < 3 && line[k] == ' ') ++k;
            size_t h = k;
            while (h < line.size() && line[h] == '#' && h - k < 6) ++h;
            if (h > k && h < line.size() && (line[h] == ' ' || line[h] == '\t')) {
                while (h < line.size() && (line[h] == ' ' || line[h] == '\t')) ++h;
                line = line.substr(h);
            }
        }
        /* a bullet: whitespace, one of * + -, whitespace */
        {
            size_t k = 0;
            while (k < line.size() && (line[k] == ' ' || line[k] == '\t')) ++k;
            if (k + 1 < line.size() && (line[k] == '*' || line[k] == '+' || line[k] == '-') &&
                (line[k + 1] == ' ' || line[k + 1] == '\t')) {
                size_t h = k + 1;
                while (h < line.size() && (line[h] == ' ' || line[h] == '\t')) ++h;
                line = line.substr(h);
            }
        }
        /* **bold** then *italic* */
        for (;;) {
            const size_t a = line.find("**");
            if (a == std::string::npos) break;
            const size_t z = line.find("**", a + 2);
            if (z == std::string::npos || z == a + 2) break;
            if (line.substr(a + 2, z - a - 2).find('*') != std::string::npos) break;
            line = line.substr(0, a) + line.substr(a + 2, z - a - 2) + line.substr(z + 2);
        }
        for (size_t a = 0; (a = line.find('*', a)) != std::string::npos;) {
            if ((a > 0 && line[a - 1] == '*') || (a + 1 < line.size() && line[a + 1] == '*')) { ++a; continue; }
            const size_t z = line.find('*', a + 1);
            if (z == std::string::npos || z == a + 1 || line.substr(a + 1, z - a - 1).find('\n') != std::string::npos ||
                (z + 1 < line.size() && line[z + 1] == '*'))
                break;
            line = line.substr(0, a) + line.substr(a + 1, z - a - 1) + line.substr(z + 1);
            a = z - 1;
        }
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r'))
            line.pop_back();
        /* a horizontal rule */
        {
            std::string s2;
            for (char ch : line) if (ch != ' ' && ch != '\t') s2 += ch;
            bool rule = s2.size() >= 3;
            for (char ch : s2) rule = rule && (ch == '-' || ch == '*' || ch == '_');
            if (rule && (s2.find_first_not_of(s2[0]) == std::string::npos || true)) {
                bool same = true;
                for (char ch : s2) same = same && (ch == '-' || ch == '*' || ch == '_');
                if (same) line.clear();
            }
        }
        if (!first) out += '\n';
        out += line;
        first = false;
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    /* "• " and four-space runs go; runs of newlines become one. */
    std::string r;
    for (size_t i = 0; i < out.size();) {
        if (out.compare(i, 4, "\xE2\x80\xA2 ") == 0) { i += 4; continue; }
        if (out.compare(i, 4, "    ") == 0) { i += 4; continue; }
        if (out[i] == '\n' && !r.empty() && r.back() == '\n') { ++i; continue; }
        r += out[i++];
    }
    return r;
}

/* diffusers' _normalize_lyrics: a line that starts with section tags keeps only them; "] " and
 * " [" break lines, " ^ " too; tags are lower case; the lyrics start with "[start]". */
static std::string normalize_lyrics(const std::string& in) {
    std::string t;
    size_t pos = 0;
    bool first = true;
    while (pos <= in.size()) {
        const size_t nl = in.find('\n', pos);
        std::string line = in.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        size_t k = 0;
        while (k < line.size() && (line[k] == ' ' || line[k] == '\t')) ++k;
        size_t e = k;
        bool any = false;
        while (e < line.size() && line[e] == '[') {
            const size_t z = line.find(']', e + 1);
            if (z == std::string::npos || z == e + 1) break;
            e = z + 1;
            any = true;
            while (e < line.size() && (line[e] == ' ' || line[e] == '\t')) ++e;
        }
        if (any) {
            std::string tags = line.substr(k, e - k);
            while (!tags.empty() && (tags.back() == ' ' || tags.back() == '\t')) tags.pop_back();
            line = tags;
        }
        if (!first) t += '\n';
        t += line;
        first = false;
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    auto replace = [](std::string s, const std::string& a, const std::string& b) {
        for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + b.size())) s.replace(p, a.size(), b);
        return s;
    };
    t = replace(t, "] ", "]\n");
    t = replace(t, " [", "\n[");
    t = replace(t, " ^ ", "\n");
    for (size_t a = t.find('['); a != std::string::npos; a = t.find('[', a + 1)) {
        const size_t z = t.find(']', a + 1);
        if (z == std::string::npos) break;
        for (size_t i = a + 1; i < z; ++i) t[i] = (char)std::tolower((unsigned char)t[i]);
    }
    return "[start]\n" + t;
}

static int64_t text_prepare(int field, const char* in, char* out, int64_t cap) {
    if (!in) return RAD_E_INVAL;
    std::string r;
    if (field == RAD_TEXT_INSTRUCTIONS) r = clean_caption(in);
    else if (field == RAD_TEXT_INPUT) r = normalize_lyrics(in);
    else r = in;
    if ((int64_t)r.size() < cap) {
        std::memcpy(out, r.data(), r.size());
        out[r.size()] = 0;
    }
    return (int64_t)r.size();
}

}  /* namespace minimaxmusic3 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(minimaxmusic3)
extern "C" int64_t rad_arch_text_prepare(int field, const char* in, char* out, int64_t cap) {
    return minimaxmusic3::text_prepare(field, in, out, cap);
}
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(minimaxmusic3, "minimaxmusic3", "", "0.1.0",
                "MiniMax-Music3: a Qwen3-8B music language model with a residual-code depth "
                "decoder, a 2.4B flow-matching transformer over windows of frames, and a DAC "
                "vocoder; 44.1 kHz stereo")
#endif
