/* mossimage -- MOSS-Image (OpenMOSS-Team/MOSS-Image-SFT): a prompt to one RGB picture.
 *
 * ============================== THE MODEL ==============================
 *
 * Three networks behind one prompt (diffusers' MOSSImagePipeline, pipeline_moss_image.py):
 *
 *   Conditioning: Qwen3.5-9B's language model, 32 layers of a hybrid trunk -- 24 gated delta nets
 *   and 8 full-attention layers (16 query / 4 KV heads of 256, partial rotary over the leading 64,
 *   M-RoPE), hidden 4096, ffn 12288. Only the last hidden state is read, after the trunk's final
 *   norm (transformers' `last_hidden_state` is post-norm), and nothing is pooled. The pipeline
 *   runs it once over the whole prompt, unpadded and from a zero state, so the delta net takes its
 *   chunked path with a zeroed convolution window: gdn_conv_prep / gdn_kkt_solve /
 *   gdn_chunk_scan / gdn_gated_rmsnorm.
 *
 *   Denoiser: a 42-block single-stream diffusion transformer over one joint sequence,
 *   [prompt rows | image latent rows]. dim 3072, 24 heads of 128, ffn 8192, full (non-causal)
 *   attention over the whole sequence, and per block
 *       x += o(attn(norm1(x)(1 + scale_a) + shift_a)) * gate_a
 *       x += down(situ(up_gate(norm2(x)(1 + scale_f) + shift_f))) * gate_f
 *   where situ(u, g) = 25 tanh(u / 25) * 4 tanh(g / 4) sigmoid(g): `situ_mul` at caps 4 and 25,
 *   over the fused row laid out gate first (the checkpoint stores it up first; the name maps
 *   swap the halves at convert time).
 *   with the six modulation rows from one projection of the timestep's Fourier embedding
 *   (computed in f64, fed to a bf16 MLP). A two-block `text_embedder`, whose linears all carry
 *   biases, refines the prompt before it joins the sequence, and the output stage normalises the
 *   image rows, applies a head modulation and projects to 128 latent channels.
 *
 *   Decoder: the FLUX.2 VAE (AutoencoderKLFlux2), arch/common/rad_block_flux2_vae.h.
 *
 * The passes are the core generator's (core/gen) in the JOINT_STEP layout: COND leaves the
 * conditioning rows in this plugin's store at every lane and branch, DENOISE runs the joint
 * sequence once a step for each entry (two with guidance), DECODE unpacks and decodes.
 *
 * ============================== THE SAMPLER ==============================
 *
 * The latents are bf16, unlike the reference's f32: `flow_euler` (the guidance combine and the
 * Euler update in one pass) has a bf16 form only. Measured cost of that arithmetic alone, with the
 * reference's own weights and the same noise: 34.0 dB PSNR against the f32-sampler picture. The
 * int8 linears' contribution rides on top of it.
 *
 * ============================== TENSOR NAMES ==============================
 *
 * The DiT (`transformer/`) is declared as `dit.*` and its text refiner as `txt.*`; the trunk
 * (`text_encoder/`'s `language_model.*`) as `te.*` and its unused vision tower as `v.*`; the
 * VAE's decoder as `vae.*` and its unused encoder as `venc.*`. Each declaration's map_* call
 * names the checkpoint tensors it reads. The one thing the engine cannot express is the DiT's
 * rotary positions, which the plugin writes into a host buffer itself (write_positions).
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_flux2_vae.h>
#include <arch/rad_linear.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mossimage {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* The pipeline's own text cap: `max_sequence_length=4096`, above which it truncates. */
enum { kMaxCond = 4096 };
/* The pipeline makes one picture a sample (`for index in range(batch)`), so one lane an image. */
enum { kMaxLanes = 1 };
/* MOSSImagePipeline's default `uncond_skip_layers=(22,)`, active over the whole denoise (its
 * slg_start/stop are 0/1): the negative branch's forward runs the DiT WITHOUT block 22. */
enum { kUncondSkip = 22 };
/* The delta net's tile, which `gdn_conv_prep`'s gate cumsum is cut at (rad_block_gdn.h's 64). */
enum { kGdnChunk = 64 };

/* Weight groups: the trunk's layers take 0..31, so everything model-level starts well clear. */
enum : int {
    kGrpTxt    = 1000,
    kGrpVae    = 1100,
    kGrpDit    = 2000,   /* blocks are kGrpDit + 1 + l */
    kGrpVision = 3000,   /* the checkpoint's unused vision tower */
};

/* ================================================================== the model */
struct TeLayer {
    int full = 0;
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    /* The full-attention projection is three weights, not two: q_proj carries the query AND its
     * output gate interleaved at a pitch of 2*head_dim, which only a rank-3 buffer addresses, and
     * k and v stay apart because the k norm reads head_dim of every head over a contiguous row
     * (a fused k|v row has a stride the kernels refuse). The delta net's are the fused
     * in_proj_qkv|a|b, in_proj_z and out_proj. */
    LinearQ    qg{}, k{}, v{}, o{}, gate_up{}, down{};
    rad_weight w_in = 0, w_z = 0, w_conv = 0, w_a_log = 0, w_dt_bias = 0, w_sn = 0, w_out = 0;
    rad_op op_n1 = 0, op_n2 = 0, op_add1 = 0, op_add2 = 0;
    rad_op op_qn = 0, op_kn = 0, op_rq = 0, op_rk = 0, op_attn = 0, op_sig = 0, op_gmul = 0,
           op_silu = 0, op_ar = 0;
    rad_op op_in = 0, op_z = 0, op_prep = 0, op_kkt = 0, op_scan = 0, op_sn = 0, op_out = 0;
};

/* The DiT's two-block text refiner: every linear carries a bias, so it is declared explicitly
 * rather than through a shared block (the tree's blocks are bias-free). */
struct TxtLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    rad_weight w_qkv = 0, w_qkv_b = 0, w_o = 0, w_o_b = 0, w_gu = 0, w_gu_b = 0, w_dn = 0,
               w_dn_b = 0;
    rad_op op_n1 = 0, op_qkv = 0, op_qkr = 0, op_attn = 0,
           op_o = 0, op_add1 = 0, op_n2 = 0, op_gu = 0, op_situ = 0, op_dn = 0, op_add2 = 0;
};

struct DitLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ    qkv{}, o{}, gate_up{}, down{};
    rad_op op_n1 = 0, op_n2 = 0, op_qkr = 0, op_attn = 0, op_situ = 0, op_ar1 = 0, op_ar2 = 0;
};

struct Model {
    Names nm{""};
    Geom  gt{}, gd{};
    int64_t max_tok = 0;
    int64_t cond_rows = kMaxCond;
    int64_t pass_rows = 0;
    /* The delta net's geometry, which only the trunk's declare reads and only its step needs:
     * 16 key and 32 value heads of 128, one convolution of 8192 channels, `conv_width` 4. */
    int64_t nh_k = 16, nh_v = 32, hk = 128, hv = 128, cw = 4;
    /* The trunk's full attention stays replicated at every world (it runs once a request), so its
     * counts are the whole checkpoint's: geom_from has already divided gt's heads for the FFN. */
    int64_t nh_a = 0, nhk_a = 0;
    /* The refiner is replicated too: its whole fused-qkv third width, for step_txt's slices. */
    int64_t rfn_qd = 0;

    /* ---- the trunk */
    rad_weight w_tok = 0;
    rad_weight w_tnorm = 0;
    rad_op  op_embed = 0;
    rad_buf b_tx = 0, b_th = 0, b_tqg = 0, b_tk = 0, b_tv = 0, b_tq = 0, b_ta = 0, b_tgate = 0,
            b_tgu = 0, b_tff = 0, b_tadd = 0;
    rad_buf b_tin = 0, b_tz = 0, b_tgq = 0, b_tgk = 0, b_tgv = 0, b_tg = 0, b_tbeta = 0,
            b_tkkt = 0, b_tgo = 0, b_tconv = 0;
    rad_buf b_tpos = 0;
    rad_op  op_conv_zero = 0, op_tnorm = 0;
    ActQ     a_th{}, a_tattn{}, a_tffn{};
    QuantAct q_th{}, q_tattn{}, q_tffn{};
    std::vector<TeLayer> te;

    /* ---- text_embedder */
    rad_weight w_tp = 0, w_tp_b = 0;
    rad_op  op_tp = 0;
    rad_buf b_xt = 0, b_xh = 0, b_xqg = 0, b_xa = 0, b_xgu = 0,
            b_xff = 0, b_xadd = 0;
    std::vector<TxtLayer> txt;

    /* ---- the conditioning store: [lanes x branches, cond_rows] of the DiT's width */
    rad_buf b_store = 0;
    rad_op  op_rows = 0, op_imgrows = 0, op_store_t = 0;

    /* ---- the DiT */
    rad_weight w_ti = 0, w_ti_b = 0, w_t1 = 0, w_t1_b = 0, w_t2 = 0, w_t2_b = 0, w_mod = 0,
               w_mod_b = 0, w_head = 0, w_head_b = 0, w_fn = 0, w_fl = 0, w_bands = 0;
    rad_op  op_temb = 0, op_t1 = 0, op_ts = 0, op_t2 = 0, op_mod = 0, op_head = 0, op_fin = 0,
            op_gfin = 0, op_pout = 0, op_euler = 0, op_randn = 0, op_lgather = 0, op_ti = 0;
    rad_buf b_lat = 0, b_h = 0, b_hm = 0, b_qkv = 0, b_attn = 0, b_ao = 0, b_gu = 0,
            b_ff = 0, b_mod = 0, b_head = 0, b_tpr = 0, b_te1 = 0, b_te1s = 0, b_temb = 0,
            b_state = 0, b_prev = 0, b_v = 0, b_hi = 0;
    rad_buf b_pos = 0, b_posc = 0;
    ActQ     a_hm{}, a_attn{}, a_ffn{};
    QuantAct q_hm{}, q_attn{}, q_ffn{};
    /* The DiT's linears are rotated int8 (`i8`, at scale group `g8`), and the trunk's (`i8t`,
     * `g8t`): the two are read off the container separately, so a container may keep the DiT
     * bf16 -- the precision runtime LoRA adapters need -- beside an int8 trunk. */
    bool    i8 = false, i8t = false;
    bool    fuse_norm = false;
    int64_t g8 = RAD_SCALE_GROUP, g8t = RAD_SCALE_GROUP;
    LinearQ    qkv0{}, tgu0{};
    std::vector<DitLayer> dit;

    /* ---- the decoder */
    Vae        dec{};
    Flux2VaeIn vin{};
    Flux2VaeDec vdec{};
    rad_buf    b_dl = 0, b_din = 0, b_cu4 = 0, b_img = 0;
    int64_t    C = 128;
    int64_t    zc = 32;
};

static Model g_model[MAX_SLOTS];

static rad_op unary(RadBuilder* b, const char* op, int64_t rows, int64_t n, rad_buf in,
                    rad_buf out) {
    return rw(b, RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                          RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
              { in }, { out });
}

/* SiTU over a fused [gate | up] row: y = 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25). */
static rad_op situ(RadBuilder* b, int64_t rows, int64_t n, rad_buf in, rad_buf out) {
    return rw(b, RAD_OP(b, "situ_mul", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                 RAD_STR("dtype", "bf16"), RAD_F64("gate_cap", 4.0),
                                                 RAD_F64("up_cap", 25.0)),
                        RAD_NOWEIGHTS),
              { in }, { out });
}

static rad_op add_rows(RadBuilder* b, int64_t rows, int64_t n, rad_buf a, rad_buf z, rad_buf out) {
    return rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
              { a, z }, { out });
}

static rad_op all_reduce(RadBuilder* b, const Geom& g, rad_buf x, int64_t rows, int64_t n) {
    if (g.world <= 1) return 0;
    return rw(b, RAD_OP(b, "all_reduce",
                        RAD_PARAMS(RAD_INT("world_size", g.world),
                                   RAD_RANGE("numel", n, rows * n), RAD_STR("dtype", "bf16"),
                                   RAD_INT("exact", g.wire_exact),
                                   RAD_INT("min_bytes", g.wire_min_bytes)),
                        RAD_NOWEIGHTS),
              { x }, { x });
}

/* A biased bf16 linear over a checkpoint tensor pair. */
static rad_op gemm_b(RadBuilder* b, const char* wn, const char* bn, const char* sw, const char* sb,
                     int64_t N, int64_t K, int64_t rows, rad_buf in, rad_buf out, rad_weight* w,
                     rad_weight* bias, int grp, int access = RAD_ACCESS_PER_REQUEST) {
    if (map_copy(b, wn, sw) < 0) return 0;
    if (map_copy(b, bn, sb) < 0) return 0;
    *w = decl_w(b, wn, RAD_BF16, { N, K }, access, RAD_SHARD_NONE, grp_layer(grp));
    *bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_NONE, grp_layer(grp));
    if (!*w || !*bias) return 0;
    return rw(b, RAD_OP(b, "gemm_nt_bias",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(*w, *bias)),
              { in }, { out });
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    const char* arch = rad_meta_gets(meta, "pipeline._class_name", "");
    if (std::strcmp(arch, "MOSSImagePipeline") != 0) {
        fprintf(stderr, "radiance: mossimage: the container's pipeline is '%s', not "
                        "MOSSImagePipeline\n", arch);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "transformer.latent_channels", 0) != 128 ||
        rad_meta_geti(meta, "transformer.num_layers", 0) != 42 ||
        rad_meta_geti(meta, "vae.latent_channels", 0) != 32) {
        fprintf(stderr, "radiance: mossimage: the DiT must carry 128 latent channels over 42 "
                        "blocks and the VAE 32 channels (2x2 packed)\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== the trunk */
static int declare_te(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gt;
    Names& nm = m.nm;
    const int64_t T = m.cond_rows;
    const int64_t E = g.n_embd, hd = g.head_dim;
    /* The trunk's full attention is replicated at every world: its weights, buffers and op rows
     * take the whole counts, while the FFN takes g's per-rank n_ff and all-reduces its partial
     * sums. geom_from divided g's heads for the sharded FFN, so the attention's dims come from
     * the _all fields. */
    const int64_t nh_a  = g.world > 1 ? g.n_head_all    : g.n_head;
    const int64_t nhk_a = g.world > 1 ? g.n_head_kv_all : g.n_head_kv;
    const int64_t qda = nh_a * hd, kvda = nhk_a * hd;
    m.nh_a = nh_a; m.nhk_a = nhk_a;
    const double rotf = rad_meta_getf(meta, "text_config.partial_rotary_factor", 0.25);
    const int64_t n_rot = (int64_t)std::llround((double)hd * rotf);

    const int64_t nh_k = rad_meta_geti(meta, "text_config.linear_num_key_heads", 16);
    const int64_t nh_v = rad_meta_geti(meta, "text_config.linear_num_value_heads", 32);
    const int64_t hk = rad_meta_geti(meta, "text_config.linear_key_head_dim", hd);
    const int64_t hv = rad_meta_geti(meta, "text_config.linear_value_head_dim", hd);
    const int64_t cw = rad_meta_geti(meta, "text_config.linear_conv_kernel_dim", 4);
    const int64_t kdim = nh_k * hk, vdim = nh_v * hv, cdim = 2 * kdim + vdim;
    const int64_t idim = cdim + 2 * nh_v;
    const int64_t interval = rad_meta_geti(meta, "text_config.full_attention_interval", 4);
    RAD_ARCH_TRY(check_layer_schedule("mossimage",
                                      rad_meta_gets(meta, "text_config.layer_types", nullptr),
                                      g.n_layer, interval, { "full_attention" }));
    if (cdim != 8192) {
        fprintf(stderr, "radiance: mossimage: the delta net's convolution covers %lld channels; "
                        "the checkpoint's conv1d is 8192 wide\n", (long long)cdim);
        return RAD_E_UNSUPPORTED;
    }
    m.nh_k = nh_k; m.nh_v = nh_v; m.hk = hk; m.hv = hv; m.cw = cw;

    RAD_ARCH_TRY(map_copy(b, nm.f("te.tok.weight"),
                          nm.ckpt("language_model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("te.tok.weight"), RAD_BF16, { g.n_vocab_all, E }, RAD_ACCESS_VOCAB,
                     RAD_SHARD_NONE, grp_model());

    m.b_tx   = decl_b(b, nm.f("te.x"), RAD_BF16, { T, E });
    m.b_th   = decl_b(b, nm.f("te.h"), RAD_BF16, { T, E });
    /* The query-and-gate projection is rank 3 so the interleaving is addressable: query and gate
     * are the two head_dim halves of the last dimension (rad_block_attn_gated.h's idiom). */
    m.b_tqg  = decl_b(b, nm.f("te.qg"), RAD_BF16, { T, nh_a, 2 * hd });
    m.b_tk   = decl_b(b, nm.f("te.k"), RAD_BF16, { T, kvda });
    m.b_tv   = decl_b(b, nm.f("te.v"), RAD_BF16, { T, kvda });
    m.b_tq   = decl_b(b, nm.f("te.q"), RAD_BF16, { T, qda });
    m.b_ta   = decl_b(b, nm.f("te.attn"), RAD_BF16, { T, qda });
    m.b_tgate = decl_b(b, nm.f("te.gate"), RAD_BF16, { T, qda });
    m.b_tgu  = decl_b(b, nm.f("te.gu"), RAD_BF16, { T, 2 * g.n_ff });
    m.b_tff  = decl_b(b, nm.f("te.ff"), RAD_BF16, { T, g.n_ff });
    m.b_tadd = decl_b(b, nm.f("te.add"), RAD_BF16, { T, E });
    /* The activation each quantised linear consumes is the buffer its PRODUCER writes: the norms
     * write `te.h`, the gated attention `te.attn`, silu_mul `te.ff`. They were declared as three
     * extra buffers nothing wrote, so every int8 linear of the 8 full-attention layers and of all
     * 32 MLPs quantised an unwritten plane (the DiT's three alias their producers, below). */
    m.a_th.x    = m.b_th;
    m.a_tattn.x = m.b_ta;
    m.a_tffn.x  = m.b_tff;
    RAD_ARCH_TRY(m.a_th.declare_pair(b, nm, g, "te.hm", E, T, m.i8t, m.g8t, "had_quant_act_i8g"));
    RAD_ARCH_TRY(m.a_tattn.declare_pair(b, nm, g, "te.attnout", qda, T, m.i8t, m.g8t,
                                        "had_quant_act_i8g"));
    RAD_ARCH_TRY(m.a_tffn.declare_pair(b, nm, g, "te.ffnin", g.n_ff, T, m.i8t, m.g8t,
                                       "gated_had_quant_i8g"));
    /* The standalone quantisers an int8 container reads, declared unconditionally as qwenimage21
     * declares its own: no producer here writes the pair's int8 codes (`q8_fed` is never set), so
     * the codes come from these ops and only `q8_fed`'s rotating variant would come from a
     * producer. A bf16 container declares them and issues nothing (`QuantAct::declare` is a no-op
     * without `i8`). */
    RAD_ARCH_TRY(m.q_th.declare(b, g, m.a_th, E));
    RAD_ARCH_TRY(m.q_tattn.declare(b, g, m.a_tattn, qda));
    RAD_ARCH_TRY(m.q_tffn.declare(b, g, m.a_tffn, g.n_ff));
    m.b_tin   = decl_b(b, nm.f("te.in"), RAD_BF16, { T, idim });
    m.b_tz    = decl_b(b, nm.f("te.z"), RAD_BF16, { T, vdim });
    m.b_tgq   = decl_b(b, nm.f("te.gq"), RAD_BF16, { T, nh_k, hk });
    m.b_tgk   = decl_b(b, nm.f("te.gk"), RAD_BF16, { T, nh_k, hk });
    m.b_tgv   = decl_b(b, nm.f("te.gv"), RAD_BF16, { T, nh_v, hv });
    /* The decay and beta are f32: the decay is a per-chunk cumulative sum that a bf16 running sum
     * would lose the low bits of, and both feed exponentials. */
    m.b_tg    = decl_b(b, nm.f("te.g"), RAD_F32, { T, nh_v });
    m.b_tbeta = decl_b(b, nm.f("te.beta"), RAD_F32, { T, nh_v });
    m.b_tkkt  = decl_b(b, nm.f("te.kkt"), RAD_BF16, { T, nh_v, kGdnChunk });
    m.b_tgo   = decl_b(b, nm.f("te.go"), RAD_BF16, { T, nh_v, hv });
    /* One conv-state slot a sequence, `conv_width - 1` deep: the COND pass is one unpadded forward
     * and zeroes it before it convolves. */
    m.b_tconv = decl_b(b, nm.f("te.conv"), RAD_BF16, { 1, cdim, cw - 1 }, RAD_BUF_PERSIST);
    m.b_tpos  = decl_b(b, nm.f("te.posf"), RAD_F32, { 4, T }, RAD_BUF_PERSIST, RAD_DOMAIN_HOST);
    /* The trunk's FINAL norm: `Qwen3_5TextModel.forward` applies it before returning
     * `last_hidden_state` (modeling_qwen3_5.py, `hidden_states = self.norm(hidden_states)`), so
     * the pipeline's `output.last_hidden_state` is the POST-norm state. Its gain is zero-centred
     * like the block norms' (wadd 1). */
    RAD_ARCH_TRY(map_copy(b, nm.f("te.norm.weight"), nm.ckpt("language_model.norm.weight")));
    m.w_tnorm = decl_w(b, nm.f("te.norm.weight"), RAD_F32, { E }, RAD_ACCESS_PER_TOKEN,
                       RAD_SHARD_NONE, grp_layer(kGrpTxt));
    m.op_tnorm = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                         RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16"),
                                         RAD_F64("wadd", 1.0)),
                              RAD_WEIGHTS(m.w_tnorm)),
                    { m.b_tx }, { m.b_tx });

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                         RAD_INT("n_vocab", g.n_vocab_all),
                                         RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", 0)),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_tx });
    m.op_conv_zero = rw(b, RAD_OP(b, "fill",
                                  RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", (cw - 1) * cdim),
                                             RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                                  RAD_NOWEIGHTS),
                        {}, { m.b_tconv });
    m.te.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        TeLayer& L = m.te[(size_t)l];
        const int li = (int)l;
        L.full = is_full_attention(l, interval) ? 1 : 0;
        const char* pre = "language_model.layers";
        RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.attn_norm.weight", li),
                              nm.ckpt("%s.%d.input_layernorm.weight", pre, li)));
        L.w_n1 = decl_w(b, nm.f("te.blk.%d.attn_norm.weight", li), RAD_F32, { E },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
        RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.ffn_norm.weight", li),
                              nm.ckpt("%s.%d.post_attention_layernorm.weight", pre, li)));
        L.w_n2 = decl_w(b, nm.f("te.blk.%d.ffn_norm.weight", li), RAD_F32, { E },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
        L.op_n1 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16"),
                                          RAD_F64("wadd", 1.0)),
                               RAD_WEIGHTS(L.w_n1)),
                     { m.b_tx }, { m.b_th });

        if (L.full) {
            /* q_proj emits the query and a per-head output gate interleaved at a pitch of
             * 2*head_dim (`attn_output_gate`), so it is one 2*q_dim projection into a rank-3
             * buffer and the per-head q norm de-interleaves it in the same pass
             * (rad_block_attn_gated.h's idiom). k and v are separate projections into separate
             * buffers: the k norm reads head_dim of every head and must see a contiguous row
             * (rad_block_attn_gated_q.h declares them exactly so). */
            RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.q_norm.weight", li),
                                  nm.ckpt("%s.%d.self_attn.q_norm.weight", pre, li)));
            L.w_qn = decl_w(b, nm.f("te.blk.%d.q_norm.weight", li), RAD_BF16, { hd },
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
            RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.k_norm.weight", li),
                                  nm.ckpt("%s.%d.self_attn.k_norm.weight", pre, li)));
            L.w_kn = decl_w(b, nm.f("te.blk.%d.k_norm.weight", li), RAD_BF16, { hd },
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
            RAD_ARCH_TRY(L.qg.probe(b, nm, g, nm.f("te.blk.%d.attn_qg", li),
                                    { nm.ckpt("%s.%d.self_attn.q_proj.weight", pre, li) }));
            RAD_ARCH_TRY(L.qg.declare(b, nm, g, nm.f("te.blk.%d.attn_qg", li),
                                      2 * qda, E, RAD_SHARD_NONE, grp_layer(li),
                                      { nm.ckpt("%s.%d.self_attn.q_proj.weight", pre, li) },
                                      m.a_th, m.b_tqg));
            RAD_ARCH_TRY(L.k.probe(b, nm, g, nm.f("te.blk.%d.attn_k", li),
                                   { nm.ckpt("%s.%d.self_attn.k_proj.weight", pre, li) }));
            RAD_ARCH_TRY(L.k.declare(b, nm, g, nm.f("te.blk.%d.attn_k", li), kvda, E,
                                     RAD_SHARD_NONE, grp_layer(li),
                                     { nm.ckpt("%s.%d.self_attn.k_proj.weight", pre, li) },
                                     m.a_th, m.b_tk));
            RAD_ARCH_TRY(L.v.probe(b, nm, g, nm.f("te.blk.%d.attn_v", li),
                                   { nm.ckpt("%s.%d.self_attn.v_proj.weight", pre, li) }));
            RAD_ARCH_TRY(L.v.declare(b, nm, g, nm.f("te.blk.%d.attn_v", li), kvda, E,
                                     RAD_SHARD_NONE, grp_layer(li),
                                     { nm.ckpt("%s.%d.self_attn.v_proj.weight", pre, li) },
                                     m.a_th, m.b_tv));
            /* The q view of the interleaved projection: head_dim columns of every 2*head_dim. */
            L.op_qn = rw(b, RAD_OP(b, "rmsnorm",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T * nh_a), RAD_INT("n", hd),
                                              RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16"),
                                              RAD_F64("wadd", 1.0)),
                                   RAD_WEIGHTS(L.w_qn)),
                         { m.b_tqg }, { m.b_tq });
            L.op_kn = rw(b, RAD_OP(b, "rmsnorm",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T * nhk_a),
                                              RAD_INT("n", hd), RAD_F64("eps", g.eps),
                                              RAD_STR("dtype", "bf16"), RAD_F64("wadd", 1.0)),
                                   RAD_WEIGHTS(L.w_kn)),
                         { m.b_tk }, { m.b_tk });
            /* Partial rotary over the leading `n_rot` of 256, M-RoPE's interleaved sections; with
             * one position component that is neox over those 64. */
            L.op_rq = rw(b, decl_rope(b, g, T, hd, nh_a, 0, n_rot, "imrope"), { m.b_tq },
                         { m.b_tq });
            L.op_rk = rw(b, decl_rope(b, g, T, hd, nhk_a, 0, n_rot, "imrope"), { m.b_tk },
                         { m.b_tk });
            L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                                RAD_INT("gqa", nh_a / nhk_a),
                                                RAD_INT("causal", 1),
                                                RAD_STR("dtype", "bf16"),
                                                RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                                RAD_INT("max_seqlen", T)),
                                     RAD_NOWEIGHTS),
                           { m.b_tq, m.b_tk, m.b_tv }, { m.b_ta });
            L.op_sig = rw(b, RAD_OP(b, "sigmoid",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T * nh_a), RAD_INT("n", hd),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                          { m.b_tqg }, { m.b_tgate });
            L.op_gmul = rw(b, RAD_OP(b, "mul",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", qda),
                                                RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS),
                           { m.b_ta, m.b_tgate }, { m.b_ta });
            RAD_ARCH_TRY(L.o.probe(b, nm, g, nm.f("te.blk.%d.attn_output", li),
                                   { nm.ckpt("%s.%d.self_attn.o_proj.weight", pre, li) }));
            RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("te.blk.%d.attn_output", li), E, qda,
                                     RAD_SHARD_NONE, grp_layer(li),
                                     { nm.ckpt("%s.%d.self_attn.o_proj.weight", pre, li) },
                                     m.a_tattn, m.b_tadd));
        } else {
            RAD_ARCH_TRY(map_concat(b, nm.f("te.blk.%d.ssm_in.weight", li),
                                    { nm.ckpt("%s.%d.linear_attn.in_proj_qkv.weight", pre, li),
                                      nm.ckpt("%s.%d.linear_attn.in_proj_a.weight", pre, li),
                                      nm.ckpt("%s.%d.linear_attn.in_proj_b.weight", pre, li) },
                                    0));
            L.w_in = decl_w(b, nm.f("te.blk.%d.ssm_in.weight", li), RAD_BF16, { idim, E },
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li), 0,
                            { kdim, kdim, vdim, nh_v, nh_v });
            RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.ssm_z.weight", li),
                                  nm.ckpt("%s.%d.linear_attn.in_proj_z.weight", pre, li)));
            L.w_z = decl_w(b, nm.f("te.blk.%d.ssm_z.weight", li), RAD_BF16, { vdim, E },
                           RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
            RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.ssm_conv1d.weight", li),
                                  nm.ckpt("%s.%d.linear_attn.conv1d.weight", pre, li)));
            L.w_conv = decl_w(b, nm.f("te.blk.%d.ssm_conv1d.weight", li), RAD_BF16, { cdim, cw },
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
            RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.ssm_a_log", li),
                                  nm.ckpt("%s.%d.linear_attn.A_log", pre, li)));
            L.w_a_log = decl_w(b, nm.f("te.blk.%d.ssm_a_log", li), RAD_F32, { nh_v },
                               RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
            RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.ssm_dt.bias", li),
                                  nm.ckpt("%s.%d.linear_attn.dt_bias", pre, li)));
            L.w_dt_bias = decl_w(b, nm.f("te.blk.%d.ssm_dt.bias", li), RAD_F32, { nh_v },
                                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
            RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.ssm_norm.weight", li),
                                  nm.ckpt("%s.%d.linear_attn.norm.weight", pre, li)));
            L.w_sn = decl_w(b, nm.f("te.blk.%d.ssm_norm.weight", li), RAD_F32, { hv },
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));
            RAD_ARCH_TRY(map_copy(b, nm.f("te.blk.%d.ssm_out.weight", li),
                                  nm.ckpt("%s.%d.linear_attn.out_proj.weight", pre, li)));
            L.w_out = decl_w(b, nm.f("te.blk.%d.ssm_out.weight", li), RAD_BF16, { E, vdim },
                             RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(li));

            const double q_scale = 1.0 / std::sqrt((double)hk);
            L.op_in = rw(b, RAD_OP(b, "gemm_nt",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", idim),
                                              RAD_INT("K", E), RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(L.w_in)),
                         { m.b_th }, { m.b_tin });
            L.op_z = rw(b, RAD_OP(b, "gemm_nt",
                                  RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", vdim),
                                             RAD_INT("K", E), RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(L.w_z)),
                        { m.b_th }, { m.b_tz });
            L.op_prep = rw(b, RAD_OP(b, "gdn_conv_prep",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_k", hk),
                                                RAD_INT("head_v", hv), RAD_INT("chunk", kGdnChunk),
                                                RAD_INT("conv_width", cw)),
                                     RAD_WEIGHTS(L.w_conv, L.w_a_log, L.w_dt_bias)),
                           { m.b_tin }, { m.b_tgq, m.b_tgk, m.b_tgv, m.b_tg, m.b_tbeta });
            L.op_kkt = rw(b, RAD_OP(b, "gdn_kkt_solve",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_k", hk)),
                                    RAD_NOWEIGHTS),
                          { m.b_tgk, m.b_tbeta, m.b_tg }, { m.b_tkkt });
            L.op_scan = rw(b, RAD_OP(b, "gdn_chunk_scan",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_k", hk),
                                                RAD_INT("head_v", hv), RAD_F64("scale", q_scale)),
                                     RAD_NOWEIGHTS),
                           { m.b_tgq, m.b_tgk, m.b_tgv, m.b_tkkt, m.b_tg, m.b_tbeta },
                           { m.b_tgo });
            L.op_sn = rw(b, RAD_OP(b, "gdn_gated_rmsnorm",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("channels", hv),
                                              RAD_F64("eps", g.eps), RAD_STR("act", "silu")),
                                   RAD_WEIGHTS(L.w_sn)),
                         { m.b_tgo, m.b_tz }, { m.b_tgo });
            L.op_out = rw(b, RAD_OP(b, "gemm_nt",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", E),
                                               RAD_INT("K", vdim), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(L.w_out)),
                          { m.b_tgo }, { m.b_tadd });
        }
        L.op_add1 = add_rows(b, T, E, m.b_tx, m.b_tadd, m.b_tx);

        L.op_n2 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                          RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16"),
                                          RAD_F64("wadd", 1.0)),
                               RAD_WEIGHTS(L.w_n2)),
                     { m.b_tx }, { m.b_th });
        RAD_ARCH_TRY(L.gate_up.probe(b, nm, g, nm.f("te.blk.%d.ffn_gate_up", li),
                                     { nm.ckpt("%s.%d.mlp.gate_proj.weight", pre, li),
                                       nm.ckpt("%s.%d.mlp.up_proj.weight", pre, li) }));
        /* The feed-forward width `g.n_ff` is this rank's, so at more than one rank the pair is a
         * row shard of two parts -- the container's [gate | up] concatenation is cut in half and
         * each half evenly, `row_parts` naming THIS RANK'S rows of each part (the loader walks
         * them at a stride of world: part g's rank-index slice sits at the part's base plus
         * index*rows) -- and the down projection a column shard of the same width, its partial
         * sums all-reduced into the residual. At one rank both are the whole projection. */
        const bool tp = g.world > 1;
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("te.blk.%d.ffn_gate_up", li), 2 * g.n_ff, E,
                                       tp ? RAD_SHARD_ROW : RAD_SHARD_NONE, grp_layer(li),
                                       { nm.ckpt("%s.%d.mlp.gate_proj.weight", pre, li),
                                         nm.ckpt("%s.%d.mlp.up_proj.weight", pre, li) },
                                       m.a_th, m.b_tgu, 0,
                                       tp ? std::initializer_list<int64_t>{ g.n_ff, g.n_ff }
                                          : std::initializer_list<int64_t>{}));
        L.op_silu = unary(b, "silu_mul", T, g.n_ff, m.b_tgu, m.b_tff);
        RAD_ARCH_TRY(L.down.probe(b, nm, g, nm.f("te.blk.%d.ffn_down", li),
                                  { nm.ckpt("%s.%d.mlp.down_proj.weight", pre, li) }));
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("te.blk.%d.ffn_down", li), E, g.n_ff,
                                    tp ? RAD_SHARD_COL : RAD_SHARD_NONE, grp_layer(li),
                                    { nm.ckpt("%s.%d.mlp.down_proj.weight", pre, li) },
                                    m.a_tffn, m.b_tadd));
        if (tp) L.op_ar = all_reduce(b, g, m.b_tadd, T, E);
        L.op_add2 = add_rows(b, T, E, m.b_tx, m.b_tadd, m.b_tx);
    }
    return RAD_OK;
}

/* ================================================================== text_embedder */
/* The DiT's two-block text refiner: proj 4096 -> 3072, then per block two RMS norms, a biased
 * fused qkv ([q | k | v]), a biased output projection, a biased SiTU feed-forward and a biased
 * down projection. The reference applies `_situ(ffn_up(y), ffn_gate(y))`; the pair is fused as
 * [gate | up], the order `situ_mul` reads. */
static int declare_txt(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    /* The refiner runs once a request and stays replicated at every world (like the trunk's
     * attention), so its dims are the DiT's whole ones; g's heads and n_ff are divided for the
     * sharded DiT blocks below. */
    const int64_t nh = g.world > 1 ? g.n_head_all : g.n_head;
    const int64_t nhk = g.world > 1 ? g.n_head_kv_all : g.n_head_kv;
    const int64_t T = m.cond_rows, D = g.n_embd, hd = g.head_dim, qd = nh * hd,
                ff = g.world > 1 ? g.n_ff_all : g.n_ff;
    m.rfn_qd = qd;
    const char* pre = "transformer.text_embedder";
    const double eps = rad_meta_getf(meta, "transformer.eps", 1e-6);
    const int grp = kGrpTxt;

    m.b_xt   = decl_b(b, nm.f("txt.x"), RAD_BF16, { T, D });
    m.b_xh   = decl_b(b, nm.f("txt.h"), RAD_BF16, { T, D });
    m.b_xqg  = decl_b(b, nm.f("txt.qg"), RAD_BF16, { T, 3 * qd });
    m.b_xa   = decl_b(b, nm.f("txt.attn"), RAD_BF16, { T, qd });
    m.b_xgu  = decl_b(b, nm.f("txt.gu"), RAD_BF16, { T, 2 * ff });
    m.b_xff  = decl_b(b, nm.f("txt.ff"), RAD_BF16, { T, ff });
    m.b_xadd = decl_b(b, nm.f("txt.add"), RAD_BF16, { T, D });

    m.op_tp = gemm_b(b, nm.f("txt.proj.weight"), nm.f("txt.proj.bias"),
                     nm.ckpt("%s.proj.weight", pre), nm.ckpt("%s.proj.bias", pre), D,
                     m.gt.n_embd, T, m.b_tx, m.b_xt, &m.w_tp, &m.w_tp_b, grp);

    m.txt.resize(2);
    for (int i = 0; i < 2; ++i) {
        TxtLayer& L = m.txt[(size_t)i];
        RAD_ARCH_TRY(map_copy(b, nm.f("txt.blk.%d.norm1.weight", i),
                              nm.ckpt("%s.blocks.%d.norm1.weight", pre, i)));
        L.w_n1 = decl_w(b, nm.f("txt.blk.%d.norm1.weight", i), RAD_F32, { D },
                        RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_layer(grp));
        RAD_ARCH_TRY(map_copy(b, nm.f("txt.blk.%d.norm2.weight", i),
                              nm.ckpt("%s.blocks.%d.norm2.weight", pre, i)));
        L.w_n2 = decl_w(b, nm.f("txt.blk.%d.norm2.weight", i), RAD_F32, { D },
                        RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_layer(grp));
        RAD_ARCH_TRY(map_copy(b, nm.f("txt.blk.%d.q_norm.weight", i),
                              nm.ckpt("%s.blocks.%d.q_norm.weight", pre, i)));
        L.w_qn = decl_w(b, nm.f("txt.blk.%d.q_norm.weight", i), RAD_F32, { hd },
                        RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_layer(grp));
        RAD_ARCH_TRY(map_copy(b, nm.f("txt.blk.%d.k_norm.weight", i),
                              nm.ckpt("%s.blocks.%d.k_norm.weight", pre, i)));
        L.w_kn = decl_w(b, nm.f("txt.blk.%d.k_norm.weight", i), RAD_F32, { hd },
                        RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_layer(grp));

        L.op_n1 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                          RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n1)),
                     { m.b_xt }, { m.b_xh });
        L.op_qkv = gemm_b(b, nm.f("txt.%d.qkv.weight", i), nm.f("txt.%d.qkv.bias", i),
                          nm.ckpt("%s.blocks.%d.qkv.weight", pre, i),
                          nm.ckpt("%s.blocks.%d.qkv.bias", pre, i), 3 * qd, D, T, m.b_xh,
                          m.b_xqg, &L.w_qkv, &L.w_qkv_b, grp);
        /* The qkv row is one [q | k | v] of q_head rows: the norms and the rope both read it at
         * its own pitch, so they are the DiT's `qk_norm_rope` in place over the fused row
         * (the four separate norm and rope ops it replaces cannot address a fused row's middle
         * third). Its rope is the DiT's: axial, `16 16 16 16`, full head_dim, the same theta. */
        L.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", nh),
                                           RAD_INT("n_head_kv", nhk),
                                           RAD_F64("theta", g.theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "axial"), RAD_INT("rotary_dim", hd),
                                           RAD_STR("sections", "16 16 16 16")),
                                RAD_WEIGHTS(L.w_qn, L.w_kn)),
                      { m.b_xqg }, { m.b_xqg });
        L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", nh / nhk), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                            RAD_INT("max_seqlen", T)),
                                 RAD_NOWEIGHTS),
                       { m.b_xqg }, { m.b_xa });
        L.op_o = gemm_b(b, nm.f("txt.%d.proj.weight", i), nm.f("txt.%d.proj.bias", i),
                        nm.ckpt("%s.blocks.%d.proj.weight", pre, i),
                        nm.ckpt("%s.blocks.%d.proj.bias", pre, i), D, qd, T, m.b_xa, m.b_xadd,
                        &L.w_o, &L.w_o_b, grp);
        L.op_add1 = add_rows(b, T, D, m.b_xt, m.b_xadd, m.b_xt);
        L.op_n2 = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                          RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n2)),
                     { m.b_xt }, { m.b_xh });
        /* ffn_gate and ffn_up as one [gate | up] row. */
        RAD_ARCH_TRY(map_concat(b, nm.f("txt.%d.gateup.weight", i),
                                { nm.ckpt("%s.blocks.%d.ffn_gate.weight", pre, i),
                                  nm.ckpt("%s.blocks.%d.ffn_up.weight", pre, i) }, 0));
        RAD_ARCH_TRY(map_concat(b, nm.f("txt.%d.gateup.bias", i),
                                { nm.ckpt("%s.blocks.%d.ffn_gate.bias", pre, i),
                                  nm.ckpt("%s.blocks.%d.ffn_up.bias", pre, i) }, 0));
        L.w_gu = decl_w(b, nm.f("txt.%d.gateup.weight", i), RAD_BF16, { 2 * ff, D },
                        RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_layer(grp));
        L.w_gu_b = decl_w(b, nm.f("txt.%d.gateup.bias", i), RAD_BF16, { 2 * ff },
                          RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_layer(grp));
        L.op_gu = rw(b, RAD_OP(b, "gemm_nt_bias",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", 2 * ff),
                                          RAD_INT("K", D), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_gu, L.w_gu_b)),
                     { m.b_xh }, { m.b_xgu });
        L.op_situ = situ(b, T, ff, m.b_xgu, m.b_xff);
        L.op_dn = gemm_b(b, nm.f("txt.%d.dn.weight", i), nm.f("txt.%d.dn.bias", i),
                         nm.ckpt("%s.blocks.%d.ffn_down.weight", pre, i),
                         nm.ckpt("%s.blocks.%d.ffn_down.bias", pre, i), D, ff, T, m.b_xff,
                         m.b_xadd, &L.w_dn, &L.w_dn_b, grp);
        L.op_add2 = add_rows(b, T, D, m.b_xt, m.b_xadd, m.b_xt);
        if (!L.op_n1 || !L.op_qkv || !L.op_qkr ||
            !L.op_attn || !L.op_o || !L.op_add1 || !L.op_n2 || !L.op_gu || !L.op_situ ||
            !L.op_dn || !L.op_add2)
            return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* ================================================================== the DiT */
static int declare_dit(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const int64_t R = m.pass_rows, D = g.n_embd, hd = g.head_dim, qd = g.q_dim(), C = m.C;
    const int64_t S = 2 * kMaxLanes * 2;
    const int gx = kGrpDit;
    const double eps = rad_meta_getf(meta, "transformer.eps", 1e-6);
    const int64_t fd = rad_meta_geti(meta, "transformer.freq_dim", 256);

    m.b_tpr  = decl_b(b, nm.f("dit.tproj"), RAD_BF16, { S, fd });
    m.b_te1  = decl_b(b, nm.f("dit.te1"), RAD_BF16, { S, D });
    m.b_te1s = decl_b(b, nm.f("dit.te1s"), RAD_BF16, { S, D });
    m.b_temb = decl_b(b, nm.f("dit.temb"), RAD_BF16, { S, D });
    m.b_mod  = decl_b(b, nm.f("dit.mod"), RAD_BF16, { S, 6 * D });
    m.b_head = decl_b(b, nm.f("dit.head"), RAD_BF16, { S, 2 * D });
    /* The reference's own embedding: `cat((cos, sin))` of the DIFFUSERS TIMESTEP of sigma --
     * t' = 1000 * sigma, the value `scheduler.timesteps` hands the transformer (FlowMatchEuler's
     * explicit-sigma convention: timesteps = sigmas * num_train_timesteps), and the transformer
     * computes the sinusoid of it as it stands -- over 10000^-i/128, in f64 on the host side and
     * rounded once to bf16. A scale of 1 would feed raw sigma and the modulation would be
     * noise. */
    m.op_temb = rw(b, RAD_OP(b, "timestep_embed",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", fd),
                                        RAD_F64("scale", 1000.0), RAD_F64("max_period", 10000.0),
                                        RAD_INT("round", 0), RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   {}, { m.b_tpr });
    m.op_t1 = gemm_b(b, nm.f("dit.t1.weight"), nm.f("dit.t1.bias"),
                     nm.ckpt("transformer.time_embedding.0.weight"),
                     nm.ckpt("transformer.time_embedding.0.bias"), D, fd, S, m.b_tpr, m.b_te1,
                     &m.w_t1, &m.w_t1_b, gx);
    m.op_ts = unary(b, "silu", S, D, m.b_te1, m.b_te1s);
    m.op_t2 = gemm_b(b, nm.f("dit.t2.weight"), nm.f("dit.t2.bias"),
                     nm.ckpt("transformer.time_embedding.2.weight"),
                     nm.ckpt("transformer.time_embedding.2.bias"), D, D, S, m.b_te1s, m.b_temb,
                     &m.w_t2, &m.w_t2_b, gx);
    m.op_mod = gemm_b(b, nm.f("dit.mod.weight"), nm.f("dit.mod.bias"),
                      nm.ckpt("transformer.shared_block_modulation.weight"),
                      nm.ckpt("transformer.shared_block_modulation.bias"), 6 * D, D, S,
                      m.b_temb, m.b_mod, &m.w_mod, &m.w_mod_b, gx);
    m.op_head = gemm_b(b, nm.f("dit.head.weight"), nm.f("dit.head.bias"),
                       nm.ckpt("transformer.video_head_modulation.weight"),
                       nm.ckpt("transformer.video_head_modulation.bias"), 2 * D, D, S,
                       m.b_temb, m.b_head, &m.w_head, &m.w_head_b, gx);
    /* `rope.bands`, kept so the container is self-describing: it is the derived ladder
     * (10000^-i/16, bit for bit), which the rope op computes from `theta`. */
    RAD_ARCH_TRY(map_copy(b, nm.f("dit.rope.bands"), nm.ckpt("transformer.rope.bands")));
    m.w_bands = decl_w(b, nm.f("dit.rope.bands"), RAD_F32, { 16 }, RAD_ACCESS_RARE,
                       RAD_SHARD_NONE, grp_layer(gx));
    if (!m.op_temb || !m.op_t1 || !m.op_ts || !m.op_t2 || !m.op_mod || !m.op_head || !m.w_bands)
        return RAD_E_INVAL;

    /* ---- the rows */
    m.b_h    = decl_b(b, nm.f("dit.h"), RAD_BF16, { R, D });
    m.b_hm   = decl_b(b, nm.f("dit.hm"), RAD_BF16, { R, D });
    m.b_qkv  = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { R, 3 * qd });
    m.b_attn = decl_b(b, nm.f("dit.attn"), RAD_BF16, { R, qd });
    m.b_ao   = decl_b(b, nm.f("dit.ao"), RAD_BF16, { R, D });
    m.b_gu   = decl_b(b, nm.f("dit.gu"), RAD_BF16, { R, 2 * g.n_ff });
    m.b_ff   = decl_b(b, nm.f("dit.ff"), RAD_BF16, { R, g.n_ff });
    m.b_hi   = decl_b(b, nm.f("dit.himg"), RAD_BF16, { R, D });
    m.b_v    = decl_b(b, nm.f("dit.out"), RAD_BF16, { R, C });
    m.b_lat  = decl_b(b, nm.f("dit.lat"), RAD_BF16, { R, C });
    m.a_hm.x   = m.b_hm;
    m.a_attn.x = m.b_attn;
    m.a_ffn.x  = m.b_ff;
    RAD_ARCH_TRY(m.a_hm.declare_pair(b, nm, g, "dit.hm", D, R, m.i8, m.g8, "had_quant_act_i8g"));
    RAD_ARCH_TRY(m.a_attn.declare_pair(b, nm, g, "dit.attn", qd, R, m.i8, m.g8,
                                       "had_quant_act_i8g"));
    RAD_ARCH_TRY(m.a_ffn.declare_pair(b, nm, g, "dit.ff", g.n_ff, R, m.i8, m.g8,
                                      "gated_had_quant_i8g"));
    RAD_ARCH_TRY(m.q_hm.declare(b, g, m.a_hm, D));
    RAD_ARCH_TRY(m.q_attn.declare(b, g, m.a_attn, qd));
    RAD_ARCH_TRY(m.q_ffn.declare(b, g, m.a_ffn, g.n_ff));

    m.b_store = decl_b(b, nm.f("dit.store"), RAD_BF16, { 2 * kMaxLanes * 2 * m.cond_rows, D },
                       RAD_BUF_PERSIST);
    m.op_rows = rw(b, RAD_OP(b, "gather_rows2",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_store, m.b_ao }, { m.b_h });
    /* `lane_store`: one conditioning's rows into every lane's cell of the store at this branch, at
     * `(2 * lane + branch) * cond_rows` (a `stride` of 2 * cond_rows from the branch's offset). */
    m.op_store_t = rw(b, RAD_OP(b, "lane_store",
                                RAD_PARAMS(RAD_RANGE("M", 1, m.cond_rows), RAD_INT("n", D),
                                           RAD_INT("stride", 2 * m.cond_rows),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_xt, m.b_store }, { m.b_store });

    /* ---- the input: the image's latent rows through video_embedder */
    RAD_ARCH_TRY(map_copy(b, nm.f("dit.ti.weight"), nm.ckpt("transformer.video_embedder.weight")));
    m.w_ti = decl_w(b, nm.f("dit.ti.weight"), RAD_BF16, { D, C }, RAD_ACCESS_PER_REQUEST,
                    RAD_SHARD_NONE, grp_layer(gx));
    RAD_ARCH_TRY(map_copy(b, nm.f("dit.ti.bias"), nm.ckpt("transformer.video_embedder.bias")));
    m.w_ti_b = decl_w(b, nm.f("dit.ti.bias"), RAD_BF16, { D }, RAD_ACCESS_PER_REQUEST,
                      RAD_SHARD_NONE, grp_layer(gx));
    m.op_ti = rw(b, RAD_OP(b, "gemm_nt_bias",
                           RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("N", D), RAD_INT("K", C),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(m.w_ti, m.w_ti_b)),
                 { m.b_lat }, { m.b_ao });
    /* `gather_rows2`: the joint pass's `src` row by row -- a conditioning row from the store, or
     * (a negative index) one of the image's rows, in the compact image order. */
    m.op_imgrows = rw(b, RAD_OP(b, "gather_rows",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_h }, { m.b_hi });

    /* ---- the blocks */
    m.dit.resize((size_t)g.n_layer);
    const char* pre = "transformer.layers";
    for (int64_t l = 0; l < g.n_layer; ++l) {
        DitLayer& L = m.dit[(size_t)l];
        const int li = (int)l;
        const int grp = kGrpDit + 1 + li;
        RAD_ARCH_TRY(map_copy(b, nm.f("dit.blk.%d.norm1.weight", li),
                              nm.ckpt("%s.%d.norm1.weight", pre, li)));
        /* bf16 gains: `rms_mod`'s kernel takes the gain as bf16 (`rms_mod_launch` refuses any
         * other dtype, libr4d/r4d_video.hip), and the checkpoint's f32 gains are within a bf16
         * ulp. */
        L.w_n1 = decl_w(b, nm.f("dit.blk.%d.norm1.weight", li), RAD_BF16, { D },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(grp));
        RAD_ARCH_TRY(map_copy(b, nm.f("dit.blk.%d.norm2.weight", li),
                              nm.ckpt("%s.%d.norm2.weight", pre, li)));
        L.w_n2 = decl_w(b, nm.f("dit.blk.%d.norm2.weight", li), RAD_BF16, { D },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(grp));
        RAD_ARCH_TRY(map_copy(b, nm.f("dit.blk.%d.q_norm.weight", li),
                              nm.ckpt("%s.%d.q_norm.weight", pre, li)));
        L.w_qn = decl_w(b, nm.f("dit.blk.%d.q_norm.weight", li), RAD_F32, { hd },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(grp));
        RAD_ARCH_TRY(map_copy(b, nm.f("dit.blk.%d.k_norm.weight", li),
                              nm.ckpt("%s.%d.k_norm.weight", pre, li)));
        L.w_kn = decl_w(b, nm.f("dit.blk.%d.k_norm.weight", li), RAD_F32, { hd },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(grp));
        /* `rms_mod`: the branch gated in, the RMS norm with its gain, then (1 + scale) and the
         * shift, from the six-row modulation table (`b_mod`, sliced per branch below). */
        L.op_n1 = rw(b, RAD_OP(b, "rms_mod",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D), RAD_F64("eps", eps),
                                          RAD_INT("gated", l > 0 ? 1 : 0), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n1)),
                     { m.b_ao, m.b_h, m.b_mod }, { m.b_hm });
        RAD_ARCH_TRY(L.qkv.probe(b, nm, g, nm.f("dit.blk.%d.attn_qkv", li),
                                 { nm.ckpt("%s.%d.proj.q.weight", pre, li),
                                   nm.ckpt("%s.%d.proj.k.weight", pre, li),
                                   nm.ckpt("%s.%d.proj.v.weight", pre, li) }));
        RAD_ARCH_TRY(L.qkv.declare(b, nm, g, nm.f("dit.blk.%d.attn_qkv", li), 3 * qd, D,
                                   RAD_SHARD_ROW, grp_layer(grp),
                                   { nm.ckpt("%s.%d.proj.q.weight", pre, li),
                                     nm.ckpt("%s.%d.proj.k.weight", pre, li),
                                     nm.ckpt("%s.%d.proj.v.weight", pre, li) },
                                   m.a_hm, m.b_qkv, 0, { qd, qd, qd }));
        L.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", g.n_head),
                                           RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_F64("theta", g.theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "axial"), RAD_INT("rotary_dim", hd),
                                           RAD_STR("sections", "16 16 16 16")),
                                RAD_WEIGHTS(L.w_qn, L.w_kn)),
                      { m.b_qkv }, { m.b_qkv });
        L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", g.gqa()), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                            RAD_INT("max_seqlen", R)),
                                 RAD_NOWEIGHTS),
                       { m.b_qkv }, { m.b_attn });
        RAD_ARCH_TRY(L.o.probe(b, nm, g, nm.f("dit.blk.%d.attn_output", li),
                               { nm.ckpt("%s.%d.proj.o.weight", pre, li) }));
        RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("dit.blk.%d.attn_output", li), D, qd,
                                 RAD_SHARD_COL, grp_layer(grp),
                                 { nm.ckpt("%s.%d.proj.o.weight", pre, li) }, m.a_attn, m.b_ao));
        L.op_ar1 = all_reduce(b, g, m.b_ao, R, D);
        L.op_n2 = rw(b, RAD_OP(b, "rms_mod",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D), RAD_F64("eps", eps),
                                          RAD_INT("gated", 1), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(L.w_n2)),
                     { m.b_ao, m.b_h, m.b_mod }, { m.b_hm });
        /* The checkpoint's `ffn_up_gate` is [up | gate] and `situ_mul` reads [gate | up], so the
         * map swaps the halves (as minimaxh3 does for its SwiGLU), under a name of its own
         * (`ffn_gateup`) so that `rad-convert --reuse` cannot carry an up-first entry over. The
         * int8 containers quantise the mapped rows, and a LoRA's B rows follow the swap as row
         * pieces. No probe(): it would map the weight again, and the DiT's precision is read off
         * `attn_qkv`. */
        const char* fsrc = nm.ckpt("%s.%d.proj.ffn_up_gate.weight", pre, li);
        const int64_t ffa = g.world > 1 ? g.n_ff_all : g.n_ff;
        RAD_ARCH_TRY(map_rows(b, nm.f("dit.blk.%d.ffn_gateup.weight", li),
                              { { fsrc, (int)ffa, ffa }, { fsrc, 0, ffa } }));
        L.gate_up.mapped = true;
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("dit.blk.%d.ffn_gateup", li), 2 * g.n_ff,
                                       D, RAD_SHARD_ROW, grp_layer(grp), { fsrc }, m.a_hm, m.b_gu,
                                       0, { g.n_ff, g.n_ff }));
        L.op_situ = situ(b, R, g.n_ff, m.b_gu, m.b_ff);
        RAD_ARCH_TRY(L.down.probe(b, nm, g, nm.f("dit.blk.%d.ffn_down", li),
                                  { nm.ckpt("%s.%d.proj.ffn_down.weight", pre, li) }));
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("dit.blk.%d.ffn_down", li), D, g.n_ff,
                                    RAD_SHARD_COL, grp_layer(grp),
                                    { nm.ckpt("%s.%d.proj.ffn_down.weight", pre, li) },
                                    m.a_ffn, m.b_ao));
        L.op_ar2 = all_reduce(b, g, m.b_ao, R, D);
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("dit.fn.weight"), nm.ckpt("transformer.final_norm_video.weight")));
    m.w_fn = decl_w(b, nm.f("dit.fn.weight"), RAD_BF16, { D }, RAD_ACCESS_PER_REQUEST,
                    RAD_SHARD_NONE, grp_layer(gx));
    RAD_ARCH_TRY(map_copy(b, nm.f("dit.fl.weight"), nm.ckpt("transformer.final_linear_video.weight")));
    m.w_fl = decl_w(b, nm.f("dit.fl.weight"), RAD_BF16, { C, D }, RAD_ACCESS_PER_REQUEST,
                    RAD_SHARD_NONE, grp_layer(gx));
    /* The LAST block's feed-forward branch gated into the joint rows: every block defers its FFN
     * residual to the NEXT block's `rms_mod` (`x = bf16(x + bf16(g y))`), so after block 41 that
     * branch sits in `b_ao` and nothing adds it -- the head would then read the image rows missing
     * block 41's FFN contribution. The reference adds it inside the block itself
     * (`x = x + down(situ(...)) * gate_f`), and the gate is the shared modulation's `gate_f` row
     * (krea2 issues the same trailing `gate_add`). */
    m.op_gfin = rw(b, RAD_OP(b, "gate_add",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D),
                                        RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                             RAD_NOWEIGHTS),
                 { m.b_ao, m.b_mod, m.b_h }, { m.b_h });
    m.op_fin = rw(b, RAD_OP(b, "rms_mod",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", D), RAD_F64("eps", eps),
                                       RAD_INT("gated", 0), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.w_fn)),
                  { m.b_hi, m.b_head }, { m.b_hm });
    m.op_pout = rw(b, RAD_OP(b, "gemm_nt",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("N", C), RAD_INT("K", D),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.w_fl)),
                   { m.b_hm }, { m.b_v });
    if (!m.w_fn || !m.w_fl || !m.op_fin || !m.op_gfin || !m.op_pout) return RAD_E_INVAL;

    /* ---- the sampler: the latents in bf16, as every sibling image plugin keeps them. libr4d's
     * `flow_euler` -- the opacity compositing, the guidance combine and the preview in one pass --
     * has a bf16 form only; the reference keeps f32 throughout (see THE SAMPLER above for the
     * measured cost, 34.0 dB PSNR, and for what an f32 sampler would need: the combine and the step
     * composed by hand). */
    const int64_t LR = (int64_t)kMaxLanes * m.max_tok;
    m.b_state = decl_b(b, nm.f("lat_state"), RAD_BF16, { LR, C }, RAD_BUF_PERSIST);
    m.b_prev  = decl_b(b, nm.f("lat_preview"), RAD_BF16, { LR, C }, RAD_BUF_PERSIST);
    m.b_pos   = decl_b(b, nm.f("dit.posf"), RAD_F32, { 4, R }, RAD_BUF_PERSIST, RAD_DOMAIN_HOST);
    /* The first entry's positions alone, plane-pitched by its own rows, for the skip layer's
     * sliced issue: the negative branch skips block 22, so that block runs over the positive
     * entry's rows only (the reference's default `uncond_skip_layers=(22,)`). */
    m.b_posc  = decl_b(b, nm.f("dit.posfc"), RAD_F32, { 4, R }, RAD_BUF_PERSIST, RAD_DOMAIN_HOST);
    m.op_randn = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", C),
                                         RAD_INT("stride", m.max_tok), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_state }, { m.b_state });
    m.op_lgather = rw(b, RAD_OP(b, "lane_gather",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", C),
                                           RAD_INT("stride", m.max_tok), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_state }, { m.b_lat });
    m.op_euler = rw(b, RAD_OP(b, "flow_euler",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", C),
                                         RAD_INT("stride", m.max_tok), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_v, m.b_state, m.b_prev }, { m.b_state, m.b_prev });
    if (!m.op_randn || !m.op_lgather || !m.op_euler) return RAD_E_INVAL;
    return RAD_OK;
}

/* ================================================================== the checkpoint's unused parts */
/* model_index.json names the whole Qwen3.5 model and the whole AutoencoderKLFlux2, and rad-convert
 * writes every declared weight -- so the vision tower and the VAE encoder are declared here as
 * plain weights with no op that reads them. This plugin makes pictures from text only (`max_refs`
 * 0), and a declared-but-unissued weight keeps the container self-describing while the converter's
 * "read by no declared weight" report stays empty. The cost is ~0.9 GB (the vision tower) and
 * ~35 MB (the encoder) of container; a converter rule for known-unused tensors would be the
 * cleaner fix. */
static int declare_unused(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    auto plain = [&](const char* name, const char* src, std::initializer_list<int64_t> shape,
                     int grp) -> int {
        if (map_copy(b, name, src) < 0) return RAD_E_INVAL;
        return decl_w(b, name, RAD_BF16, shape, RAD_ACCESS_RARE, RAD_SHARD_NONE, grp_layer(grp))
                   ? RAD_OK : RAD_E_INVAL;
    };
    const char* vp = "visual";
    if (rad_meta_geti(meta, "vision_config.depth", 0) > 0) {
        const int64_t depth = rad_meta_geti(meta, "vision_config.depth", 0);
        const int64_t w = rad_meta_geti(meta, "vision_config.hidden_size", 1152);
        const int64_t ff = rad_meta_geti(meta, "vision_config.intermediate_size", 4304);
        const int64_t mp = rad_meta_geti(meta, "vision_config.spatial_merge_size", 2);
        RAD_ARCH_TRY(plain(nm.f("v.patch.weight"), nm.ckpt("%s.patch_embed.proj.weight", vp),
                           { w, 3, 2, 16, 16 }, kGrpVision));
        RAD_ARCH_TRY(plain(nm.f("v.patch.bias"), nm.ckpt("%s.patch_embed.proj.bias", vp), { w },
                           kGrpVision));
        RAD_ARCH_TRY(plain(nm.f("v.pos.weight"), nm.ckpt("%s.pos_embed.weight", vp), { 2304, w },
                           kGrpVision));
        for (int64_t l = 0; l < depth; ++l) {
            const int li = (int)l;
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.n1.weight", li), nm.ckpt("%s.blocks.%d.norm1.weight", vp, li),
                               { w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.n1.bias", li), nm.ckpt("%s.blocks.%d.norm1.bias", vp, li),
                               { w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.n2.weight", li), nm.ckpt("%s.blocks.%d.norm2.weight", vp, li),
                               { w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.n2.bias", li), nm.ckpt("%s.blocks.%d.norm2.bias", vp, li),
                               { w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.qkv.weight", li), nm.ckpt("%s.blocks.%d.attn.qkv.weight", vp, li),
                               { 3 * w, w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.qkv.bias", li), nm.ckpt("%s.blocks.%d.attn.qkv.bias", vp, li),
                               { 3 * w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.o.weight", li), nm.ckpt("%s.blocks.%d.attn.proj.weight", vp, li),
                               { w, w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.o.bias", li), nm.ckpt("%s.blocks.%d.attn.proj.bias", vp, li),
                               { w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.fc1.weight", li), nm.ckpt("%s.blocks.%d.mlp.linear_fc1.weight", vp, li),
                               { ff, w }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.fc1.bias", li), nm.ckpt("%s.blocks.%d.mlp.linear_fc1.bias", vp, li),
                               { ff }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.fc2.weight", li), nm.ckpt("%s.blocks.%d.mlp.linear_fc2.weight", vp, li),
                               { w, ff }, kGrpVision));
            RAD_ARCH_TRY(plain(nm.f("v.blk.%d.fc2.bias", li), nm.ckpt("%s.blocks.%d.mlp.linear_fc2.bias", vp, li),
                               { w }, kGrpVision));
        }
        RAD_ARCH_TRY(plain(nm.f("v.m.norm.weight"), nm.ckpt("%s.merger.norm.weight", vp), { w },
                           kGrpVision));
        RAD_ARCH_TRY(plain(nm.f("v.m.norm.bias"), nm.ckpt("%s.merger.norm.bias", vp), { w },
                           kGrpVision));
        RAD_ARCH_TRY(plain(nm.f("v.m.fc1.weight"), nm.ckpt("%s.merger.linear_fc1.weight", vp),
                           { w * mp * mp, w * mp * mp }, kGrpVision));
        RAD_ARCH_TRY(plain(nm.f("v.m.fc1.bias"), nm.ckpt("%s.merger.linear_fc1.bias", vp),
                           { w * mp * mp }, kGrpVision));
        RAD_ARCH_TRY(plain(nm.f("v.m.fc2.weight"), nm.ckpt("%s.merger.linear_fc2.weight", vp),
                           { 4096, w * mp * mp }, kGrpVision));
        RAD_ARCH_TRY(plain(nm.f("v.m.fc2.bias"), nm.ckpt("%s.merger.linear_fc2.bias", vp),
                           { 4096 }, kGrpVision));
    }
    /* The VAE's encoder: AutoencoderKLFlux2 is one component and /v1/images/edits is refused. */
    {
        const int64_t zd = rad_meta_geti(meta, "vae.latent_channels", 32);
        const int64_t nres = rad_meta_geti(meta, "vae.layers_per_block", 2);
        const int64_t chs[4] = { 128, 256, 512, 512 };
        RAD_ARCH_TRY(plain(nm.f("venc.conv_in.weight"), nm.ckpt("vae.encoder.conv_in.weight"),
                           { chs[0], 3, 3, 3 }, kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.conv_in.bias"), nm.ckpt("vae.encoder.conv_in.bias"),
                           { chs[0] }, kGrpVae));
        for (int i = 0; i < 4; ++i) {
            const int64_t cin = i ? chs[i - 1] : chs[0], cout = chs[i];
            for (int64_t r = 0; r < nres; ++r) {
                const int64_t ci = r ? cout : cin;
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.n1.weight", i, (int)r),
                                   nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.norm1.weight", i, (int)r),
                                   { ci }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.n1.bias", i, (int)r),
                                   nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.norm1.bias", i, (int)r),
                                   { ci }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.c1.weight", i, (int)r),
                                   nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.conv1.weight", i, (int)r),
                                   { cout, ci, 3, 3 }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.c1.bias", i, (int)r),
                                   nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.conv1.bias", i, (int)r),
                                   { cout }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.n2.weight", i, (int)r),
                                   nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.norm2.weight", i, (int)r),
                                   { cout }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.n2.bias", i, (int)r),
                                   nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.norm2.bias", i, (int)r),
                                   { cout }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.c2.weight", i, (int)r),
                                   nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.conv2.weight", i, (int)r),
                                   { cout, cout, 3, 3 }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.c2.bias", i, (int)r),
                                   nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.conv2.bias", i, (int)r),
                                   { cout }, kGrpVae));
                if (r == 0 && cin != cout) {
                    RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.sc.weight", i, (int)r),
                                       nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.conv_shortcut.weight", i, (int)r),
                                       { cout, cin, 1, 1 }, kGrpVae));
                    RAD_ARCH_TRY(plain(nm.f("venc.d%d.r%d.sc.bias", i, (int)r),
                                       nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d.conv_shortcut.bias", i, (int)r),
                                       { cout }, kGrpVae));
                }
            }
            if (i < 3) {
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.ds.weight", i),
                                   nm.ckpt("vae.encoder.down_blocks.%d.downsamplers.0.conv.weight", i),
                                   { cout, cout, 3, 3 }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.d%d.ds.bias", i),
                                   nm.ckpt("vae.encoder.down_blocks.%d.downsamplers.0.conv.bias", i),
                                   { cout }, kGrpVae));
            }
            if (i == 3) break;
            /* The last down block (index 3) has no downsampler. */
            continue;
        }
        for (int i = 0; i < 2; ++i) {
            const char* nn[2] = { "norm1", "norm2" };
            const char* cc[2] = { "conv1", "conv2" };
            for (int q = 0; q < 2; ++q) {
                RAD_ARCH_TRY(plain(nm.f("venc.m.r%d.%s.weight", i, nn[q]),
                                   nm.ckpt("vae.encoder.mid_block.resnets.%d.%s.weight", i, nn[q]),
                                   { 512 }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.m.r%d.%s.bias", i, nn[q]),
                                   nm.ckpt("vae.encoder.mid_block.resnets.%d.%s.bias", i, nn[q]),
                                   { 512 }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.m.r%d.%s.weight", i, cc[q]),
                                   nm.ckpt("vae.encoder.mid_block.resnets.%d.%s.weight", i, cc[q]),
                                   { 512, 512, 3, 3 }, kGrpVae));
                RAD_ARCH_TRY(plain(nm.f("venc.m.r%d.%s.bias", i, cc[q]),
                                   nm.ckpt("vae.encoder.mid_block.resnets.%d.%s.bias", i, cc[q]),
                                   { 512 }, kGrpVae));
            }
        }
        RAD_ARCH_TRY(plain(nm.f("venc.m.gn.weight"),
                           nm.ckpt("vae.encoder.mid_block.attentions.0.group_norm.weight"),
                           { 512 }, kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.m.gn.bias"),
                           nm.ckpt("vae.encoder.mid_block.attentions.0.group_norm.bias"),
                           { 512 }, kGrpVae));
        for (const char* pr : { "to_q", "to_k", "to_v" }) {
            RAD_ARCH_TRY(plain(nm.f("venc.m.%s.weight", pr),
                               nm.ckpt("vae.encoder.mid_block.attentions.0.%s.weight", pr),
                               { 512, 512 }, kGrpVae));
            RAD_ARCH_TRY(plain(nm.f("venc.m.%s.bias", pr),
                               nm.ckpt("vae.encoder.mid_block.attentions.0.%s.bias", pr),
                               { 512 }, kGrpVae));
        }
        RAD_ARCH_TRY(plain(nm.f("venc.m.o.weight"),
                           nm.ckpt("vae.encoder.mid_block.attentions.0.to_out.0.weight"),
                           { 512, 512 }, kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.m.o.bias"),
                           nm.ckpt("vae.encoder.mid_block.attentions.0.to_out.0.bias"), { 512 },
                           kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.out_norm.weight"), nm.ckpt("vae.encoder.conv_norm_out.weight"),
                           { 512 }, kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.out_norm.bias"), nm.ckpt("vae.encoder.conv_norm_out.bias"),
                           { 512 }, kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.out.weight"), nm.ckpt("vae.encoder.conv_out.weight"),
                           { 2 * zd, 512, 3, 3 }, kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.out.bias"), nm.ckpt("vae.encoder.conv_out.bias"),
                           { 2 * zd }, kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.q.weight"), nm.ckpt("vae.quant_conv.weight"),
                           { 2 * zd, 2 * zd, 1, 1 }, kGrpVae));
        RAD_ARCH_TRY(plain(nm.f("venc.q.bias"), nm.ckpt("vae.quant_conv.bias"), { 2 * zd },
                           kGrpVae));
    }
    return RAD_OK;
}

/* ================================================================== the decoder */
/* FLUX.2's VAE decoder: the latent denormalised by the BatchNorm over the packed width, unpacked
 * 2x2 to 32 channels, then GroupNorm residual blocks, the mid-block attention and nearest-neighbour
 * upsampling, run whole on every rank (arch/common/rad_block_flux2_vae.h). */
static int declare_vae(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    const int64_t NT = m.max_tok;
    const int64_t zc = rad_meta_geti(meta, "vae.latent_channels", 32);
    const double bn_eps = rad_meta_getf(meta, "vae.batch_norm_eps", 1e-4);
    RAD_ARCH_TRY(flux2_vae_declare_in(b, m.nm, &m.vin, NT, m.C, zc, bn_eps, m.b_state, m.b_prev,
                                      &m.b_dl, &m.b_din, &m.b_cu4));
    m.zc = zc;
    return flux2_vae_declare_dec(b, meta, m.nm, m.dec, &m.vdec, NT * 256, NT, zc, 3, &m.b_din,
                                 m.b_cu4, kGrpVae);
}

/* ================================================================== the model's largest image */
/* A latent token covers 16 x 16 pixels. The pipeline states no image cap of its own, so the bound
 * is 128 x 128 tokens (2048 x 2048, the size the DiT was trained at) unless --image-gen-max-pixels
 * names the operator's. */
enum { kMaxImageTokens = 128 * 128 };

static int64_t image_tokens(int64_t pixels) {
    const int64_t cap = (int64_t)kMaxImageTokens;
    return pixels > 0 ? std::min<int64_t>(cap, pixels / 256) : cap;
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
    /* The trunk's geometry: text_encoder/config.json is the container's model config, so its bare
     * keys (and their `text_config.` aliases) are the Qwen3.5 language model's. */
    RAD_ARCH_TRY(geom_from(m.gt, meta, ctx, "bf16", RAD_BF16));
    Names& nm = m.nm;

    /* ---- the DiT's geometry, off the transformer's own config */
    Geom& gd = m.gd;
    gd = m.gt;
    const int64_t nh = rad_meta_geti(meta, "transformer.num_heads", 24);
    const int64_t dim = rad_meta_geti(meta, "transformer.dim", 3072);
    gd.n_layer       = rad_meta_geti(meta, "transformer.num_layers", 42);
    gd.head_dim      = nh ? dim / nh : 0;
    gd.n_head_all    = nh;
    gd.n_head_kv_all = nh;   /* one joint sequence: every head is a query and a key */
    gd.n_embd        = gd.head_dim * nh;
    gd.n_ff_all      = rad_meta_geti(meta, "transformer.ffn_dim", 8192);
    gd.eps           = (float)rad_meta_getf(meta, "transformer.eps", 1e-6);
    gd.theta         = (float)rad_meta_getf(meta, "transformer.rope_base", 10000.0);
    /* Not the core's multi-component rotary: the positions are (r, f, h, w) and the plugin writes
     * them itself (write_positions). */
    gd.rope_mc       = false;
    gd.wadd          = 0.0f;
    gd.wadd_qk       = 0.0f;
    if (gd.head_dim <= 0 || gd.n_embd <= 0 || gd.n_ff_all <= 0) {
        fprintf(stderr, "radiance: mossimage: transformer.dim %lld over %lld heads is not a "
                        "geometry\n", (long long)dim, (long long)nh);
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(divide_or_fail(gd.n_head_all, gd.world, "transformer.num_heads", &gd.n_head));
    gd.n_head_kv = gd.n_head;
    gd.kv_off = 0;
    RAD_ARCH_TRY(divide_or_fail(gd.n_ff_all, gd.world, "transformer.ffn_dim", &gd.n_ff));
    if (m.gt.n_layer >= kGrpTxt || gd.n_layer >= kGrpVision - kGrpDit) {
        fprintf(stderr, "radiance: mossimage: %lld trunk and %lld DiT layers overrun the weight "
                        "group ranges\n", (long long)m.gt.n_layer, (long long)gd.n_layer);
        return RAD_E_UNSUPPORTED;
    }
    if (m.gt.n_embd != rad_meta_geti(meta, "transformer.text_dim", 4096)) {
        fprintf(stderr, "radiance: mossimage: the DiT reads %lld-wide conditioning and the trunk "
                        "writes %lld\n",
                (long long)rad_meta_geti(meta, "transformer.text_dim", 0),
                (long long)m.gt.n_embd);
        return RAD_E_INVAL;
    }

    /* ---- what the container holds: bf16 or rotated int8. The first DiT projection
     * states it for the DiT's linears and the trunk's first feed-forward for the trunk's, so the
     * two networks need not agree (the text refiner between them is bf16 in every container). */
    {
        LinearQ& probe = m.qkv0;
        RAD_ARCH_TRY(probe.probe(b, nm, gd, "dit.blk.0.attn_qkv",
                                 { nm.ckpt("transformer.layers.0.proj.q.weight"),
                                   nm.ckpt("transformer.layers.0.proj.k.weight"),
                                   nm.ckpt("transformer.layers.0.proj.v.weight") }));
        RadEncoding e{};
        const bool known = weight_enc(b, nm.f("dit.blk.0.attn_qkv.weight"), &e);
        const bool plain = known && rad_enc_is(&e, "plain");
        gd.w_bf16 = plain;
        m.i8 = known && (linear_is_w8h(e) || linear_is_wnh(e));
        if (m.i8) m.g8 = linear_w8h_group(e);
    }
    {
        LinearQ& probe = m.tgu0;
        RAD_ARCH_TRY(probe.probe(b, nm, m.gt, "te.blk.0.ffn_gate_up",
                                 { nm.ckpt("language_model.layers.0.mlp.gate_proj.weight"),
                                   nm.ckpt("language_model.layers.0.mlp.up_proj.weight") }));
        RadEncoding e{};
        const bool known = weight_enc(b, nm.f("te.blk.0.ffn_gate_up.weight"), &e);
        m.gt.w_bf16 = known && rad_enc_is(&e, "plain");
        m.i8t = known && (linear_is_w8h(e) || linear_is_wnh(e));
        if (m.i8t) m.g8t = linear_w8h_group(e);
    }

    m.C = rad_meta_geti(meta, "transformer.latent_channels", 128);
    if (m.C != 4 * rad_meta_geti(meta, "vae.latent_channels", 32)) {
        fprintf(stderr, "radiance: mossimage: %lld latent channels and a %lld-channel VAE is not "
                        "a 2x2 pack\n", (long long)m.C,
                (long long)rad_meta_geti(meta, "vae.latent_channels", 0));
        return RAD_E_UNSUPPORTED;
    }
    m.max_tok = image_tokens(ctx->gen_max_pixels);
    /* The conditioning rides in every denoising step (JOINT_STEP), so a pass is the conditioning
     * and the image together, twice with guidance: `probe()` states that sum to the core, which is
     * what --max-model-len must cover. */
    m.cond_rows = kMaxCond;
    m.pass_rows = std::max<int64_t>(m.gt.max_tok, 2 * (m.cond_rows + m.max_tok));

    /* ---- in issue order: the two unused sub-models, the trunk, the text refiner, the DiT, the
     * decoder (whose latent buffers the DiT declares). */
    RAD_ARCH_TRY(declare_unused(b, meta, m));
    /* ---- adapter key names. LoRA files for MOSS-Image name the DiT's linears as the checkpoint
     * does (`layers.N.proj.q`, `.k`, `.v`, `.o`, `.ffn_up_gate`, `.ffn_down`), under PEFT's
     * `transformer.` or ComfyUI's `diffusion_model.`, so no module needs an alias. The trunk takes
     * no adapters (the trainers keep the text encoder frozen). */
    RAD_ARCH_TRY(lora_names(b, "transformer.", kLoraDiffusionSpellings, {}));
    RAD_ARCH_TRY(declare_te(b, meta, m));
    RAD_ARCH_TRY(declare_txt(b, meta, m));
    RAD_ARCH_TRY(declare_dit(b, meta, m));
    RAD_ARCH_TRY(declare_vae(b, meta, m));
    m.b_img = m.vdec.b_img;   /* what a DECODE pass fills, handed to the core */

    /* ---- what the core drives */
    if (!ctx->shape_probe) {
        static thread_local std::string tmpl;
        /* `enable_thinking=True`, the reference's own call: the generation prompt the chat template
         * renders ends with `<think>\n`, so the conditioning carries those extra tokens. Dropping
         * them shifts every text row's position and shortens the prompt the DiT cross-references
         * the image against. */
        tmpl = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n";
        RadGeneratorDecl d{};
        d.name = "moss-image";
        d.layout = RAD_GEN_LAYOUT_JOINT_STEP;
        d.tmpl_text = tmpl.c_str();
        d.tmpl_edit = tmpl.c_str();
        /* The template's fixed turn is `<|im_start|>user\n` and the reference feeds the whole
         * hidden state through: nothing is dropped. */
        d.drop_prefix = "";
        d.latent_channels = m.C;
        d.latent_scale = 16;
        /* The pipeline checks `height % 16`, not 32. */
        d.size_multiple = 16;
        d.pixel_channels = 3;
        d.slot_rows = 1;
        d.max_tokens = m.max_tok;
        d.max_refs = 0;
        d.ref_area = 1024 * 1024;
        d.default_width = 1024;
        d.default_height = 1024;
        d.default_steps = 50;
        /* `MOSSImageScheduler`: shift 5.0, no dynamic shifting, the unshifted linspace ending at
         * shift / (num_train_timesteps + shift - 1) = 5/1004, so the last sigma is 25/1024 =
         * 0.0244 at every step count (the core's default 1/n bound would leave 42% of the noise
         * in an 8-step picture). The reference then snaps each shifted sigma to its training
         * grid (multiples of 1/1000, at most 0.0005 away); that is not reproduced: measured
         * against a snapped schedule, it moves a picture no further than a change of rounding
         * order does. */
        d.shift = 5.0;
        d.dyn_shift = 0;
        d.shift_exp = 1;
        d.base_shift = 1.0;
        d.max_shift = 1.0;
        d.base_seq = 256;
        d.max_seq = 4096;
        d.shift_terminal = 0.0;
        d.sigma_lower = 5.0 / 1004.0;
        /* `uncond + s * (cond - uncond)`, so a scale of 1 is off (cfg_original 0). */
        d.default_cfg = 3.0;
        d.cfg_original = 0;
        d.max_lanes = kMaxLanes;
        d.image_out = m.b_img;
        d.cond_rows = m.cond_rows;
        /* Unused: the positions are the plugin's own [4, rows] buffer. */
        d.rope_hw_scale = 1.0;
        d.rope_text_l = 0;
        /* No padding: one unpadded forward a prompt, as the reference runs it ("to avoid padding
         * effects in Qwen GDN"). */
        d.pad_multiple = 0;
        d.pad_token = -1;
        /* The reference's own default: a guided request that names no negative prompt gets the
         * EMPTY one (MOSSImagePipeline's `negative_prompt=''`), not a refusal. */
        d.default_negative = "";
        RAD_ARCH_TRY(rad_declare_generator(b, &d));
    }

    rad_note(b, "rank %d/%d: MOSS-Image -- trunk %lld layers (%lld heads, %lld KV a rank), a "
                "%lld-block DiT (%lld heads a rank, ff %lld), %s DiT and %s trunk linears, %lld "
                "conditioning and %lld image rows",
             m.gt.rank, m.gt.world, (long long)m.gt.n_layer, (long long)m.gt.n_head,
             (long long)m.gt.n_head_kv, (long long)gd.n_layer, (long long)gd.n_head,
             (long long)gd.n_ff, gd.w_bf16 ? "bf16" : m.i8 ? "rotated int8" : "another format",
             m.gt.w_bf16 ? "bf16" : m.i8t ? "rotated int8" : "another format",
             (long long)m.cond_rows, (long long)m.max_tok);
    return RAD_OK;
}

/* ================================================================== step */
/* The positions the DiT's rotary reads: component-major (r, f, h, w) over four sections of 16
 * bands each. `r` is 0 on every row of every pass; `f` is the row's index among the text rows and
 * the prompt's length on image rows; `h`/`w` are the image token's place in its grid, centred on
 * `(n - 1) / 2`. No core layout produces these, so they are written into the host arena the rope
 * op reads.
 *
 * The pass's segment bounds come from `G->entry_h` and not from `batch->cu_seqlens`: the driver
 * keeps that array in device memory (`B.cu_seqlens = dptr<int32_t>(ri, R_CU)`,
 * `core/gen/driver.cpp:488`, `rad_dev_alloc(..., RAD_MEM_DEVICE)` at `:180`), so dereferencing it
 * here reads a device address on the host and faults. Every row an entry has is host-side
 * knowledge anyway: its conditioning's length and, off the same pixels, `(h/16)*(w/16)` latent
 * tokens -- `latent_scale` 16 makes the driver's `j.tokens` exactly that (`driver.cpp:222`,
 * `:284`). */
static void write_positions(RadCtx* c, const RadBatch* batch, const Model& m, int64_t rows,
                            int64_t compact) {
    const RadGenBatch* G = batch->gen_b;
    float* p = (float*)rad_buf_ptr(c, m.b_pos);
    float* q = compact > 0 ? (float*)rad_buf_ptr(c, m.b_posc) : nullptr;
    if (!p || !G) return;
    int64_t r = 0, a = 0;
    for (int64_t e = 0; e < batch->n_seq && r < rows; ++e) {
        /* `entry_h`: the conditioning's rows, the height and width in pixels, the branch. */
        const bool hk = G->entry_h != nullptr;
        const int64_t ntext = hk ? (int64_t)G->entry_h[4 * e] : 0;
        const int64_t h = hk ? (int64_t)G->entry_h[4 * e + 1] / 16 : 0;
        const int64_t w = hk ? (int64_t)G->entry_h[4 * e + 2] / 16 : 0;
        const int64_t z = a + ntext + h * w;
        for (int64_t i = a; i < z && r < rows; ++i, ++r) {
            const int64_t k = i - a;
            float ph = 0.0f, pw = 0.0f;
            if (k >= ntext && h > 0 && w > 0) {
                /* The INTEGER row and column of the latent grid, centred: the reference's
                 * `meshgrid(arange(H), arange(W))` gives row `(k - ntext) / w` and column
                 * `(k - ntext) % w` -- a float division here would hand every image row a
                 * fractional height and rotate each block's q/k by the wrong angle. */
                const int64_t iy = (k - ntext) / w, ix = (k - ntext) % w;
                ph = (float)iy - (float)(h - 1) / 2.0f;
                pw = (float)ix - (float)(w - 1) / 2.0f;
            }
            p[0 * rows + r] = 0.0f;
            p[1 * rows + r] = k < ntext ? (float)k : (float)ntext;
            p[2 * rows + r] = ph;
            p[3 * rows + r] = pw;
            /* The compact mirror, plane-pitched by its own row count: the skip layer's issue
             * covers the positive entry's rows only (the reference's uncond_skip_layers). */
            if (q && r < compact) {
                q[0 * compact + r] = 0.0f;
                q[1 * compact + r] = k < ntext ? (float)k : (float)ntext;
                q[2 * compact + r] = ph;
                q[3 * compact + r] = pw;
            }
        }
        a = z;
    }
}

/* The positions the text refiner rotates by: (0, i, 0, 0) over the prompt, its `f` the row's index
 * (the reference hands the refiner `rope[:, h*w:]`). */
static void write_text_positions(RadCtx* c, const Model& m, int64_t rows) {
    float* p = (float*)rad_buf_ptr(c, m.b_tpos);
    if (!p) return;
    for (int64_t i = 0; i < rows; ++i) {
        p[0 * rows + i] = 0.0f;
        p[1 * rows + i] = (float)i;
        p[2 * rows + i] = 0.0f;
        p[3 * rows + i] = 0.0f;
    }
}

/* The trunk over the prompt: Qwen3.5's 32 layers, 8 full attention and 24 gated delta nets, from a
 * zero state. Only the last hidden state is read -- no pooling and no final norm, as transformers'
 * `last_hidden_state` is the last layer's output before `language_model.norm`. */
static void step_te(RadCtx* c, const RadBatch* batch, const Model& m) {
    const Geom& g = m.gt;
    const int64_t T = batch->n_tok;
    const int64_t conv_dim = 2 * m.nh_k * m.hk + m.nh_v * m.hv;
    const RadOperand pos = praw2(batch->rope_pos, RAD_I32, 3, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_tx, T));
    /* The convolution window starts empty every pass. `has_init` is left absent, so a kernel
     * convolves against the slot's window, which this fill has just zeroed. */
    /* `RAD_ISSUE_N(..., 1, ...)`: the fill declares M 1..1 (one row whatever the prompt's length)
     * and takes exactly one operand, so the faults were the rows and the stray RAD_NONE. The
     * operand's spelling was never one: flux3 and anima pass rad_buf_ptr on device buffers, and
     * both address the buffer; the handle is kept as the plainer form. */
    RAD_ISSUE_N(c, m.op_conv_zero, 1, brows(m.b_tconv, 1));
    for (const TeLayer& L : m.te) {
        RAD_ISSUE(c, L.op_n1, brows(m.b_tx, T), RAD_W(L.w_n1), brows(m.b_th, T));
        m.q_th.step(c, T);
        if (L.full) {
            L.qg.step(c, m.a_th, m.b_tqg, T);
            L.k.step(c, m.a_th, m.b_tk, T);
            L.v.step(c, m.a_th, m.b_tv, T);
            /* The q view of the interleaved projection: head_dim columns of every 2*head_dim. */
            RAD_ISSUE_N(c, L.op_qn, T * m.nh_a, bcol(m.b_tqg, 0, g.head_dim, T),
                        RAD_W(L.w_qn), brows(m.b_tq, T));
            RAD_ISSUE_N(c, L.op_kn, T * m.nhk_a, brows(m.b_tk, T), RAD_W(L.w_kn),
                        brows(m.b_tk, T));
            RAD_ISSUE(c, L.op_rq, brows(m.b_tq, T), pos);
            RAD_ISSUE(c, L.op_rk, brows(m.b_tk, T), pos);
            RAD_ISSUE(c, L.op_attn, brows(m.b_tq, T), brows(m.b_tk, T),
                      brows(m.b_tv, T), cu, brows(m.b_ta, T));
            /* The quantiser AFTER the op that writes the final value: op_gmul multiplies the
             * attention output by the sigmoid gate in place, so codes taken before it would be of
             * the ungated output. */
            RAD_ISSUE_N(c, L.op_sig, T * m.nh_a, bcol(m.b_tqg, g.head_dim, g.head_dim, T),
                        brows(m.b_tgate, T));
            RAD_ISSUE(c, L.op_gmul, brows(m.b_ta, T), brows(m.b_tgate, T), brows(m.b_ta, T));
            m.q_tattn.step(c, T);
            L.o.step(c, m.a_tattn, m.b_tadd, T);
        } else {
            RAD_ISSUE(c, L.op_in, brows(m.b_th, T), RAD_W(L.w_in), brows(m.b_tin, T));
            RAD_ISSUE(c, L.op_z, brows(m.b_th, T), RAD_W(L.w_z), brows(m.b_tz, T));
            /* Sixteen operands, inputs before outputs. `a` and `b_gate` are the last 2*value-head
             * columns of the fused projection row (the separate path every kernel serves). The
             * conv state cache holds one slot for the whole pass -- the window `op_conv_zero`
             * cleared -- so every sequence names slot 0; `has_init` stays absent, the zeroed window
             * being exactly the fresh start. `cache_idx` is optional in the schema, but libr4d's
             * conv_prep dereferences it (`if (!ta || !tb || !cidx) return RAD_E_INVAL`), and an
             * issue-time refusal aborts the step: there is no silent fallback. */
            RAD_ISSUE(c, L.op_prep, brows(m.b_tin, T), RAD_W(L.w_conv), RAD_NONE,
                      RAD_W(L.w_a_log), RAD_W(L.w_dt_bias), brows(m.b_tconv, 1), cu,
                      bcol(m.b_tin, conv_dim, m.nh_v, T),
                      bcol(m.b_tin, conv_dim + m.nh_v, m.nh_v, T),
                      /* `cache_idx` names the conv state slot of each sequence: the pass holds
                       * one sequence (`cu` above carries two rows), so slot 0, and
                       * `cu_seqlens[0]` is 0 by construction. It must be the batch's own array --
                       * device memory the driver uploads from host staging (`core/gen/driver.cpp`:
                       * `B.cu_seqlens = dptr<int32_t>(ri, R_CU)`) -- while a host array of ours is
                       * not in the queue's address space, and `fill` serves only bf16 and f32, so
                       * there is no way to zero an i32 buffer on the card. */
                      praw(batch->cu_seqlens, RAD_I32, 1), RAD_NONE,
                      brows(m.b_tgq, T), brows(m.b_tgk, T), brows(m.b_tgv, T), brows(m.b_tg, T),
                      brows(m.b_tbeta, T));
            RAD_ISSUE(c, L.op_kkt, brows(m.b_tgk, T), brows(m.b_tbeta, T), brows(m.b_tg, T), cu,
                      brows(m.b_tkkt, T));
            RAD_ISSUE(c, L.op_scan, brows(m.b_tgq, T), brows(m.b_tgk, T), brows(m.b_tgv, T),
                      brows(m.b_tkkt, T), brows(m.b_tg, T), brows(m.b_tbeta, T), RAD_NONE, cu,
                      brows(m.b_tgo, T), RAD_NONE, RAD_NONE);
            RAD_ISSUE(c, L.op_sn, brows(m.b_tgo, T), brows(m.b_tz, T), RAD_W(L.w_sn),
                      brows(m.b_tgo, T));
            RAD_ISSUE(c, L.op_out, brows(m.b_tgo, T), RAD_W(L.w_out), brows(m.b_tadd, T));
        }
        RAD_ISSUE(c, L.op_add1, brows(m.b_tx, T), brows(m.b_tadd, T), brows(m.b_tx, T));
        RAD_ISSUE(c, L.op_n2, brows(m.b_tx, T), RAD_W(L.w_n2), brows(m.b_th, T));
        m.q_th.step(c, T);
        L.gate_up.step(c, m.a_th, m.b_tgu, T);
        RAD_ISSUE(c, L.op_silu, brows(m.b_tgu, T), brows(m.b_tff, T));
        /* As above: silu_mul writes `te.ff`, so its quantiser follows it. */
        m.q_tffn.step(c, T);
        L.down.step(c, m.a_tffn, m.b_tadd, T);
        /* A column-sharded down projection leaves a partial sum on each rank. */
        if (L.op_ar) RAD_ISSUE_N(c, L.op_ar, T * g.n_embd, brows(m.b_tadd, T), RAD_NONE);
        RAD_ISSUE(c, L.op_add2, brows(m.b_tx, T), brows(m.b_tadd, T), brows(m.b_tx, T));
    }
    /* `last_hidden_state`: the trunk's own final norm (see op_tnorm's declaration). */
    RAD_ISSUE(c, m.op_tnorm, brows(m.b_tx, T), RAD_W(m.w_tnorm), brows(m.b_tx, T));
}

/* The DiT's two-block text refiner: proj 4096 -> 3072, then two RMS norms, a biased qkv, a biased
 * output projection, a biased SiTU feed-forward and a biased down projection, all with biases (so
 * plain `gemm_nt_bias` rather than the tree's bias-free blocks). */
static void step_txt(RadCtx* c, const RadBatch* batch, const Model& m) {
    /* The whole fused-qkv third width: the refiner is replicated, and `g`'s q_dim() is this
     * rank's (the head counts the kernels check are the whole ones). */
    const int64_t T = batch->n_tok, qd = m.rfn_qd;
    const RadOperand pos = RAD_P_T2(rad_buf_ptr(c, m.b_tpos), RAD_F32, 4, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    RAD_ISSUE(c, m.op_tp, brows(m.b_tx, T), RAD_W(m.w_tp), RAD_W(m.w_tp_b), RAD_NONE,
              brows(m.b_xt, T));
    for (const TxtLayer& L : m.txt) {
        RAD_ISSUE(c, L.op_n1, brows(m.b_xt, T), RAD_W(L.w_n1), brows(m.b_xh, T));
        RAD_ISSUE(c, L.op_qkv, brows(m.b_xh, T), RAD_W(L.w_qkv), RAD_W(L.w_qkv_b), RAD_NONE,
                  brows(m.b_xqg, T));
        RAD_ISSUE(c, L.op_qkr, brows(m.b_xqg, T), pos, RAD_W(L.w_qn), RAD_W(L.w_kn));
        RAD_ISSUE(c, L.op_attn, bcol(m.b_xqg, 0, qd, T), bcol(m.b_xqg, qd, qd, T),
                  bcol(m.b_xqg, 2 * qd, qd, T), cu, brows(m.b_xa, T));
        RAD_ISSUE(c, L.op_o, brows(m.b_xa, T), RAD_W(L.w_o), RAD_W(L.w_o_b), RAD_NONE,
                  brows(m.b_xadd, T));
        RAD_ISSUE(c, L.op_add1, brows(m.b_xt, T), brows(m.b_xadd, T), brows(m.b_xt, T));
        RAD_ISSUE(c, L.op_n2, brows(m.b_xt, T), RAD_W(L.w_n2), brows(m.b_xh, T));
        RAD_ISSUE(c, L.op_gu, brows(m.b_xh, T), RAD_W(L.w_gu), RAD_W(L.w_gu_b), RAD_NONE,
                  brows(m.b_xgu, T));
        RAD_ISSUE(c, L.op_situ, brows(m.b_xgu, T), brows(m.b_xff, T));
        RAD_ISSUE(c, L.op_dn, brows(m.b_xff, T), RAD_W(L.w_dn), RAD_W(L.w_dn_b), RAD_NONE,
                  brows(m.b_xadd, T));
        RAD_ISSUE(c, L.op_add2, brows(m.b_xt, T), brows(m.b_xadd, T), brows(m.b_xt, T));
    }
}

/* The conditioning: the prompt through the trunk and the text refiner, into every lane's store at
 * this branch. The DIN's rows a lane keeps are `(2 * lane + branch) * cond_rows`, which a
 * `lane_store` at `stride` 2 * cond_rows and a view starting at the branch's offset writes. */
static void step_cond(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t T = batch->n_tok, D = m.gd.n_embd;
    const int64_t store_rows = 2 * kMaxLanes * m.cond_rows;
    write_text_positions(c, m, T);
    step_te(c, batch, m);
    step_txt(c, batch, m);
    const int64_t base = (int64_t)G->branch * m.cond_rows;
    RAD_ISSUE(c, m.op_store_t, brows(m.b_xt, T), praw(G->lane, RAD_I32, G->n_lane),
              brow_slice(m.b_store, base, store_rows - base, D));
}

/* The DiT's blocks over the pass's rows in `b_h`, one entry a segment bounded by `cu` and seeing
 * only its own rows (the conditioning and its image together, both ways, no mask).
 *
 * `n_pos` > 0 is the first entry's row count when a negative entry follows it: the reference's
 * DEFAULT guidance runs the negative branch WITHOUT block `kUncondSkip` (MOSSImagePipeline's
 * `uncond_skip_layers=(22,)`, active over the whole denoise), so that one block is issued over
 * the positive entry's rows only -- `cu_pos` ({0, n_pos}) and `pos_c` (the compact positions)
 * bound and rotate the slice. The skipped rows keep the previous block's `b_ao`, which is
 * exactly what the next block's gated residual must read there: `b_h` is never written. */
static void step_blocks(RadCtx* c, const Model& m, int64_t N, const RadOperand& cu,
                        const RadOperand& pos, int64_t n_pos, const RadOperand& cu_pos,
                        const RadOperand& pos_c) {
    const Geom& g = m.gd;
    const int64_t D = g.n_embd, qd = g.q_dim();
    const int64_t S = 2 * kMaxLanes * 2;
    const RadOperand s1 = bcol(m.b_mod, 0, 2 * D, S), ga = bcol(m.b_mod, 2 * D, D, S);
    const RadOperand s2 = bcol(m.b_mod, 3 * D, 2 * D, S), gf = bcol(m.b_mod, 5 * D, D, S);
    for (size_t l = 0; l < m.dit.size(); ++l) {
        const DitLayer& L = m.dit[l];
        const bool skip = n_pos > 0 && n_pos < N && l == (size_t)kUncondSkip;
        const int64_t n = skip ? n_pos : N;
        const RadOperand bcu = skip ? cu_pos : cu;
        const RadOperand bpos = skip ? pos_c : pos;
        /* norm1: the previous block's feed-forward branch gated in (plain on the first block),
         * then the RMS norm, `(1 + scale_a)` and `shift_a`. */
        RAD_ISSUE_N(c, L.op_n1, n, l ? brows(m.b_ao, n) : RAD_NONE, l ? gf : RAD_NONE, s1,
                    RAD_W(L.w_n1), bcu, brows(m.b_h, n), brows(m.b_hm, n), RAD_NONE);
        m.q_hm.step(c, n);
        L.qkv.step(c, m.a_hm, m.b_qkv, n);
        RAD_ISSUE_N(c, L.op_qkr, n, brows(m.b_qkv, n), bpos, RAD_W(L.w_qn), RAD_W(L.w_kn));
        RAD_ISSUE_N(c, L.op_attn, n, bcol(m.b_qkv, 0, qd, n), bcol(m.b_qkv, qd, qd, n),
                    bcol(m.b_qkv, 2 * qd, qd, n), bcu, brows(m.b_attn, n));
        m.q_attn.step(c, n);
        L.o.step(c, m.a_attn, m.b_ao, n);
        if (L.op_ar1) RAD_ISSUE_N(c, L.op_ar1, n * D, brows(m.b_ao, n), RAD_NONE);
        /* norm2: the attention branch gated in, then `(1 + scale_f)` and `shift_f`. */
        RAD_ISSUE_N(c, L.op_n2, n, brows(m.b_ao, n), ga, s2, RAD_W(L.w_n2), bcu, brows(m.b_h, n),
                    brows(m.b_hm, n), RAD_NONE);
        m.q_hm.step(c, n);
        L.gate_up.step(c, m.a_hm, m.b_gu, n);
        RAD_ISSUE_N(c, L.op_situ, n, brows(m.b_gu, n), brows(m.b_ff, n));
        /* `situ_mul` writes `dit.ff`, so its quantiser must follow it: `ffn_down` reads
         * the codes. */
        m.q_ffn.step(c, n);
        L.down.step(c, m.a_ffn, m.b_ao, n);
        if (L.op_ar2) RAD_ISSUE_N(c, L.op_ar2, n * D, brows(m.b_ao, n), RAD_NONE);
    }
}

/* The denoiser: the timestep's Fourier embedding (in f64 on the host side, cos first), the six-row
 * modulation, the joint sequence `[conditioning | image]` an entry, the blocks, and the head on the
 * image rows. The Euler step is `flow_euler`'s and runs in bf16 (THE SAMPLER above). */
static void step_denoise(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t N = batch->n_tok, S = batch->n_seq, NI = G->n_img;
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const RadOperand icu = praw(G->img_cu, RAD_I32, S + 1);
    const RadOperand lanes = praw(G->lane, RAD_I32, S);
    const int64_t LR = (int64_t)kMaxLanes * m.max_tok;

    /* ---- the timestep and every modulation */
    RAD_ISSUE_N(c, m.op_temb, S, praw(G->sigma, RAD_F32, S), brows(m.b_tpr, S));
    RAD_ISSUE_N(c, m.op_t1, S, brows(m.b_tpr, S), RAD_W(m.w_t1), RAD_W(m.w_t1_b), RAD_NONE,
                brows(m.b_te1, S));
    RAD_ISSUE_N(c, m.op_ts, S, brows(m.b_te1, S), brows(m.b_te1s, S));
    RAD_ISSUE_N(c, m.op_t2, S, brows(m.b_te1s, S), RAD_W(m.w_t2), RAD_W(m.w_t2_b), RAD_NONE,
                brows(m.b_temb, S));
    RAD_ISSUE_N(c, m.op_mod, S, brows(m.b_temb, S), RAD_W(m.w_mod), RAD_W(m.w_mod_b), RAD_NONE,
                brows(m.b_mod, S));
    RAD_ISSUE_N(c, m.op_head, S, brows(m.b_temb, S), RAD_W(m.w_head), RAD_W(m.w_head_b), RAD_NONE,
                brows(m.b_head, S));

    /* ---- the image's latent rows, compact, through video_embedder */
    RAD_ISSUE_N(c, m.op_randn, NI, praw(G->flags, RAD_I32, S), praw(G->seed, RAD_U32, 2 * S),
                lanes, icu, brows(m.b_state, LR));
    RAD_ISSUE_N(c, m.op_lgather, NI, brows(m.b_state, LR), lanes, icu, brows(m.b_lat, NI));
    RAD_ISSUE_N(c, m.op_ti, NI, brows(m.b_lat, NI), RAD_W(m.w_ti), RAD_W(m.w_ti_b), RAD_NONE,
                brows(m.b_ao, NI));
    /* ---- the joint rows: a conditioning row from the store, or an image row (`src` signed) */
    /* The positive entry comes first and its negative follows (driver.cpp: pair[i] is -2 on the
     * negative, i + 1 on the positive it pairs with), so a guided pass's first segment is the
     * positive one -- and the reference's default guidance runs the negative branch WITHOUT
     * block `kUncondSkip`, so that block is issued over the first segment's rows only. */
    int64_t n_pos = 0;
    if (S > 1 && G->entry_h && G->entry_h[7] == 1) {
        n_pos = (int64_t)G->entry_h[0] +
                ((int64_t)G->entry_h[1] / 16) * ((int64_t)G->entry_h[2] / 16);
        if (n_pos <= 0 || n_pos >= N) n_pos = 0;
    }
    write_positions(c, batch, m, N, n_pos);
    RAD_ISSUE(c, m.op_rows, brows(m.b_store, 2 * kMaxLanes * m.cond_rows), brows(m.b_ao, NI),
              praw(G->src, RAD_I32, N), brows(m.b_h, N));
    step_blocks(c, m, N, cu, RAD_P_T2(rad_buf_ptr(c, m.b_pos), RAD_F32, 4, N), n_pos,
                praw(batch->cu_seqlens, RAD_I32, 2),
                RAD_P_T2(rad_buf_ptr(c, m.b_posc), RAD_F32, 4, n_pos > 0 ? n_pos : N));

    /* ---- the head on the image rows, and the sampler's step */
    /* The last block's feed-forward branch, gated in before the image rows are gathered (see
     * op_gfin's declaration: the deferred residual's tail). */
    RAD_ISSUE_N(c, m.op_gfin, N, brows(m.b_ao, N),
                bcol(m.b_mod, 5 * m.gd.n_embd, m.gd.n_embd, 2 * kMaxLanes * 2), cu,
                brows(m.b_h, N));
    RAD_ISSUE_N(c, m.op_imgrows, NI, brows(m.b_h, N), praw(G->img_rows, RAD_I32, NI),
                brows(m.b_hi, NI));
    RAD_ISSUE_N(c, m.op_fin, NI, RAD_NONE, RAD_NONE, brows(m.b_head, S), RAD_W(m.w_fn), icu,
                brows(m.b_hi, NI), brows(m.b_hm, NI), RAD_NONE);
    RAD_ISSUE_N(c, m.op_pout, NI, brows(m.b_hm, NI), RAD_W(m.w_fl), brows(m.b_v, NI));
    RAD_ISSUE_N(c, m.op_euler, NI, brows(m.b_v, NI), praw(G->sigma, RAD_F32, S),
                praw(G->sigma_next, RAD_F32, S), praw(G->cfg, RAD_F32, S),
                praw(G->pair, RAD_I32, S), lanes, praw(G->flags, RAD_I32, S),
                praw(G->seed, RAD_U32, 2 * S), icu, RAD_NONE, RAD_NONE, brows(m.b_state, LR),
                brows(m.b_prev, LR));
}

/* One picture through the decoder: the BatchNorm's denormalisation, the 2x2 unpack and the FLUX.2
 * block graph, then -1..1 to 0..255. */
static void step_decode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t H = G->dec_height / 16, W = G->dec_width / 16;
    const rad_buf src = G->dec_preview ? m.b_prev : m.b_state;
    flux2_vae_issue_in(c, m.vin, m.b_dl, m.b_din, m.b_cu4, src,
                       (int64_t)G->dec_lane * m.max_tok, H, W, m.C, m.zc,
                       praw(batch->cu_seqlens + 1, RAD_I32, 1));
    m.dec.issue(c, 2 * H, 2 * W, brows(m.b_cu4, 2), nullptr);
    const int64_t npx = (int64_t)G->dec_height * G->dec_width;
    RAD_ISSUE_N(c, m.vdec.op_rgb, npx,
                RAD_P_T2(rad_buf_ptr(c, m.vdec.out.buf), RAD_BF16, npx, 3),
                RAD_P_T2(rad_buf_ptr(c, m.vdec.b_img), RAD_U8, npx, 3));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->gen_b) {
        rad_step_fail(c, "mossimage is an image generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_COND:    step_cond(c, batch, m); return;
    case RAD_GEN_PASS_DENOISE: step_denoise(c, batch, m); return;
    case RAD_GEN_PASS_DECODE:  step_decode(c, batch, m); return;
    default: rad_step_fail(c, "mossimage: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* A joint pass carries the conditioning and the image together, twice with guidance: that sum is
 * what the cache must hold, and --max-model-len is set from it. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    const int64_t tokens = image_tokens(out->gen_max_pixels);
    if (tokens == 0) return RAD_E_INVAL;
    out->gen_rows = 2 * (kMaxCond + tokens);
    out->gen_patches = 0;
    return RAD_OK;
}

}  /* namespace mossimage */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(mossimage)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(mossimage, "mossimage", "", "0.1.0",
                "MOSS-Image: Qwen3.5-9B conditioning, a 4.8B joint-stream diffusion transformer "
                "over [prompt | latent] with a SiTU feed-forward, and FLUX.2's VAE; "
                "bf16 or rotated int8 linears")
#endif
