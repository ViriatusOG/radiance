/* sdxl -- Stable Diffusion XL and its finetunes (Illustrious, NoobAI, Pony, ...): text to an RGB
 * image.
 *
 * ============================== THE MODEL ==============================
 *
 * Three networks behind one prompt, as Stability's single-file checkpoint holds them (read by
 * rad-convert's SDXL importer, core/format/checkpoint.cpp open_sgm):
 *
 *   Conditioning: two CLIP text encoders over the prompt in 77-token windows (the core cuts it,
 *   RadGeneratorDecl::chunk_tokens): CLIP ViT-L (12 layers, 768 wide, quick_gelu) and OpenCLIP
 *   ViT-bigG (32 layers, 1280 wide, gelu). Each window is encoded on its own, causally. The
 *   cross-attention context is ViT-L's and bigG's penultimate hidden states side by side, 2048
 *   wide, the windows end to end; the pooled vector is bigG's last layer through its final norm at
 *   the first window's end-of-text token, through text_projection. ViT-L reads padding as
 *   end-of-text tokens, bigG as token 0 ("!"), as each tokenizer pads.
 *
 *   Denoiser: a 2.6B UNet over 4-channel latents at 1/8 of the pixels, three levels (320, 640 and
 *   1280 channels), residual blocks (GroupNorm, SiLU, 3x3 convolutions, the time embedding added
 *   per channel) and spatial transformers at the two lower levels (2 and 10 blocks of self
 *   attention, cross attention over the context and a GEGLU feed-forward, 64-wide heads). The time
 *   embedding is the timestep's plus the pooled vector's and the image size's (original size,
 *   crop at (0, 0), target size), each through its own sinusoid. It predicts the noise (eps), or
 *   with a `v_pred` checkpoint the velocity.
 *
 *   Decoder: the SDXL VAE (AutoencoderKL, 4 latent channels at 8x, scaling 0.13025), run bf16
 *   (bf16 has f32's range, so the fp16 overflow the original VAE is known for does not occur).
 *
 * Sampling is k-diffusion's over the discrete schedule (1000 steps, scaled-linear betas
 * 0.00085..0.012): the core places the sigmas (ComfyUI's schedulers) and conditions each step at
 * the nearest trained timestep; kd_input scales the f32 latent by c_in, the UNet runs at bf16, and
 * kd_step applies guidance to the denoised estimates and takes the sampler's step (Euler, Euler
 * ancestral, DPM++ 2M) in f32.
 *
 * ============================== ONE IMAGE AT A TIME ==============================
 *
 * A convolution reads its image's size from its operands, so a denoising pass runs the UNet once
 * per entry (an image and its guidance branch are two entries) at that entry's size, over the
 * same declared ops; the sampler ops run over every entry at once. The buffers the entries pass
 * between them (the model input and output, the lanes, the stores) are kept for the program's
 * life.
 *
 * ============================== RUNTIME LORA ==============================
 *
 * Every linear and every 3x3 convolution of the UNet and every linear of both text encoders is a
 * plain GEMM over a weight declared under its checkpoint name: a 3x3 convolution is `im2col_nhwc`
 * then `gemm_nt_bias` over the weight read as [cout, cin * 9] in the checkpoint's own order, a 1x1
 * skip convolution a GEMM over [cout, cin]. Only the UNet's conv_in and out.2 (4 channels) and the
 * VAE are direct convolutions.
 *
 * One card: the model is small enough that tensor parallelism would only add all-reduces.
 */
#include <arch/rad_arch.h>
#include <arch/rad_vae2d.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace sdxl {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* The largest image in latent pixels: 1536 x 1536 (or the same area at another aspect). */
enum { kMaxImageTokens = 36864 };
/* CLIP's window, and the windows a prompt may take. */
enum { kWin = 77, kMaxWin = 8, kCondRows = kWin * kMaxWin };
enum { kMaxLanes = 8 };
enum { kLatC = 4, kCtx = 2048, kPooled = 1280, kAdd = 2816, kTemb = 1280, kGroups = 32 };
enum { kBos = 49406, kEos = 49407 };
static const double kVaeScale = 0.13025;
/* group_norm's scratch: 32 groups of (2 + 3 * 1024) floats (docs/OPS.md). */
enum { kGnStats = kGroups * (2 + 3 * 1024) };

static const char* kUnet = "model.diffusion_model";
static const char* kTe1 = "conditioner.embedders.0.transformer";
static const char* kTe2 = "conditioner.embedders.1.transformer";

/* Movement groups: each text encoder one, the UNet's embeddings and blocks 21 (kGrpUnet + 0..20),
 * the VAE one. All lie below the conditioning network's layer count, the container's n_layers,
 * since the engine reads a group past it as a drafter's. */
enum : int { kGrpTe1 = 0, kGrpTe2 = 1, kGrpUnet = 2, kGrpVae = 23 };

/* ================================================================== the text encoders */
struct TextLayer {
    rad_weight ln1w = 0, ln1b = 0, ln2w = 0, ln2b = 0;
    rad_weight qw = 0, qb = 0, kw = 0, kb = 0, vw = 0, vb = 0, ow = 0, ob = 0;
    rad_weight f1w = 0, f1b = 0, f2w = 0, f2b = 0;
    rad_op op_ln1 = 0, op_q = 0, op_k = 0, op_v = 0, op_attn = 0, op_o = 0;
    rad_op op_ln2 = 0, op_f1 = 0, op_act = 0, op_f2 = 0;
};

struct TextEnc {
    int64_t D = 0, F = 0, L = 0;     /* width, feed-forward width, layers run */
    int64_t tap = 0;                 /* the layer after which the context is taken */
    int64_t ctx_col = 0;             /* where its columns of the context start */
    bool    quick = false;           /* quick_gelu (ViT-L) or gelu (bigG) */
    rad_weight tok = 0, pos = 0;
    rad_buf x = 0, pe = 0, h = 0, q = 0, k = 0, v = 0, a = 0, f1 = 0, f2 = 0;
    rad_op  op_tok = 0, op_pad = 0, op_pos = 0, op_pe = 0, op_tap = 0;
    std::vector<TextLayer> layers;
    /* bigG's pooled vector */
    rad_weight fnw = 0, fnb = 0, proj = 0;
    rad_buf e1 = 0, e2 = 0;
    rad_op  op_eos = 0, op_fn = 0, op_proj = 0;
};

/* ================================================================== the UNet, recorded
 *
 * The UNet's ops are declared once and issued per entry at the entry's size, so each op is
 * recorded with what rebuilds its operands: a value's rows are its level's pixels (level l is
 * the latent at 1 / 2^l per side), one row, or the conditioning's rows. */
enum { kRowsOne = -1, kRowsCond = -2, kRowsTwo = -3 };

struct V {
    rad_buf buf = 0;
    int     lvl = 0;                 /* >= 0 a level; kRows* otherwise */
    int64_t pitch = 0;               /* the buffer's width */
    int64_t col = 0, width = 0;      /* a column slice; width 0 is the whole row */
    int64_t w() const { return width ? width : pitch; }
};

struct Opd {
    /* ROWS a value's rows (the entry's conditioning when the value is the store), IMG a level's
     * image [h, w * channels], XIN / EPS the entry's rows of the model input and output, IOTA the
     * table's entry for a level's pixel count, STATS group_norm's scratch. */
    enum Kind : uint8_t { NONE, ROWS, IMG, WEIGHT, XIN, EPS, IOTA, STATS };
    Kind       kind = NONE;
    V          v{};
    rad_weight w = 0;
};

struct UOp {
    rad_op           op = 0;
    std::vector<Opd> o;
    int              n = 0;          /* the ranged value: a level's pixels, or kRows* */
};

/* One entry's geometry and where its per-entry operands are. */
struct Geo {
    int64_t h = 0, w = 0;            /* latent rows and columns */
    int64_t cond = 0;                /* conditioning rows */
    int64_t store_row = 0, pool_row = 0;
    int64_t row0 = 0;                /* its first row in the pass's compact image order */
    int64_t px(int l) const { return (h >> l) * (w >> l); }
    int64_t rows(int l) const {
        return l >= 0 ? px(l) : l == kRowsOne ? 1 : l == kRowsTwo ? 2 : cond;
    }
};

struct Net {
    std::vector<UOp> ops;
    RadBuilder* b = nullptr;
    Names*      nm = nullptr;
    int64_t     P0 = kMaxImageTokens;
    int         n_buf = 0;
    int         grp = kGrpUnet;
    rad_buf     stats = 0, iota = 0, store = 0, xin = 0, eps = 0;

    int64_t cap(int lvl) const {
        return lvl >= 0 ? P0 >> (2 * lvl)
             : lvl == kRowsCond ? (int64_t)kCondRows : lvl == kRowsTwo ? 2 : 1;
    }
    V buf(int lvl, int64_t ch, uint32_t dt = RAD_BF16) {
        V v;
        v.lvl = lvl;
        v.pitch = ch;
        v.buf = decl_b(b, nm->f("unet.t%d", n_buf++), dt, { cap(lvl), ch });
        return v;
    }
    static Opd R(const V& v) { Opd o; o.kind = Opd::ROWS; o.v = v; return o; }
    static Opd I(const V& v) { Opd o; o.kind = Opd::IMG; o.v = v; return o; }
    static Opd W(rad_weight w) { Opd o; o.kind = Opd::WEIGHT; o.w = w; return o; }
    static Opd Kd(Opd::Kind k) { Opd o; o.kind = k; return o; }
    void add(rad_op op, std::vector<Opd> o, int n, std::initializer_list<rad_buf> rd,
             std::initializer_list<rad_buf> wr) {
        rw(b, op, rd, wr);
        UOp u;
        u.op = op;
        u.o = std::move(o);
        u.n = n;
        ops.push_back(std::move(u));
    }
    /* A weight under its checkpoint name, declared at `shape` (a conv kernel read flat). */
    rad_weight wt(const char* src, std::initializer_list<int64_t> shape) {
        if (map_copy(b, src, src) < 0) return 0;
        return decl_w(b, src, RAD_BF16, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                      grp_layer(grp));
    }

    /* y = x W^T (+ bias) (+ res): gemm_nt without a bias, gemm_nt_bias with one. */
    V lin(const V& x, int64_t N, const char* src, bool bias, const V* res = nullptr) {
        V y = buf(x.lvl, N);
        const int64_t K = x.w();
        rad_weight w = wt(nm->ckpt("%s.weight", src), { N, K });
        if (!bias) {
            rad_op op = RAD_OP(b, "gemm_nt",
                               RAD_PARAMS(RAD_RANGE("M", 1, cap(x.lvl)), RAD_INT("N", N),
                                          RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(w));
            add(op, { R(x), W(w), R(y) }, x.lvl, { x.buf }, { y.buf });
            return y;
        }
        rad_weight bw = wt(nm->ckpt("%s.bias", src), { N });
        rad_op op = RAD_OP(b, "gemm_nt_bias",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(x.lvl)), RAD_INT("N", N),
                                      RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w, bw));
        if (res)
            add(op, { R(x), W(w), W(bw), R(*res), R(y) }, x.lvl, { x.buf, res->buf }, { y.buf });
        else
            add(op, { R(x), W(w), W(bw), Kd(Opd::NONE), R(y) }, x.lvl, { x.buf }, { y.buf });
        return y;
    }
    /* A 3x3 convolution with a bias (+ res): im2col_nhwc, then a GEMM over the weight read as
     * [cout, cin * 9], the checkpoint's order. `stride` 2 halves the side, `up` 2 doubles it. */
    V conv3(const V& x, int64_t cout, const char* src, int stride, int up, const V* res = nullptr) {
        const int lo = x.lvl + (stride == 2 ? 1 : 0) - (up == 2 ? 1 : 0);
        const int64_t cin = x.pitch;
        V col = buf(lo, cin * 9);
        rad_op op = RAD_OP(b, "im2col_nhwc",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(lo)), RAD_INT("cin", cin),
                                      RAD_INT("k", 3), RAD_INT("stride", stride),
                                      RAD_STR("pad", "same"), RAD_INT("up", up),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS);
        add(op, { I(x), R(col) }, lo, { x.buf }, { col.buf });
        V y = buf(lo, cout);
        rad_weight w = wt(nm->ckpt("%s.weight", src), { cout, cin * 9 });
        rad_weight bw = wt(nm->ckpt("%s.bias", src), { cout });
        rad_op g = RAD_OP(b, "gemm_nt_bias",
                          RAD_PARAMS(RAD_RANGE("M", 1, cap(lo)), RAD_INT("N", cout),
                                     RAD_INT("K", cin * 9), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w, bw));
        if (res)
            add(g, { R(col), W(w), W(bw), R(*res), R(y) }, lo, { col.buf, res->buf }, { y.buf });
        else
            add(g, { R(col), W(w), W(bw), Kd(Opd::NONE), R(y) }, lo, { col.buf }, { y.buf });
        return y;
    }
    /* A direct convolution (conv_in and the output's, 4 channels on one side). */
    V conv_direct(const Opd& in, int lvl, int64_t cin, int64_t cout, const char* src,
                  const Opd* out = nullptr) {
        V y{};
        y.lvl = lvl;
        y.pitch = cout;
        if (!out) y = buf(lvl, cout);
        rad_weight w = wt(nm->ckpt("%s.weight", src), { cout, cin, 3, 3 });
        rad_weight bw = wt(nm->ckpt("%s.bias", src), { cout });
        rad_op op = RAD_OP(b, "conv2d_nhwc",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(lvl)), RAD_INT("cin", cin),
                                      RAD_INT("cout", cout), RAD_INT("k", 3), RAD_INT("stride", 1),
                                      RAD_STR("pad", "same"), RAD_INT("up", 1),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w, bw));
        rw(b, op, { in.v.buf }, { out ? out->v.buf : y.buf });
        UOp u;
        u.op = op;
        u.o = { in, W(w), W(bw), out ? *out : I(y) };
        u.n = lvl;
        ops.push_back(std::move(u));
        return y;
    }
    /* GroupNorm(32) with its affine, then SiLU or not. */
    V gn(const V& x, const char* src, double eps, bool silu) {
        V y = buf(x.lvl, x.pitch);
        rad_weight w = wt(nm->ckpt("%s.weight", src), { x.pitch });
        rad_weight bw = wt(nm->ckpt("%s.bias", src), { x.pitch });
        rad_op op = RAD_OP(b, "group_norm",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(x.lvl)), RAD_INT("n", x.pitch),
                                      RAD_INT("groups", kGroups), RAD_F64("eps", eps),
                                      RAD_STR("act", silu ? "silu" : "none"),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w, bw));
        add(op, { R(x), W(w), W(bw), Kd(Opd::NONE), Kd(Opd::STATS), R(y) }, x.lvl,
            { x.buf, stats }, { y.buf, stats });
        return y;
    }
    V ln(const V& x, const char* src) {
        V y = buf(x.lvl, x.pitch);
        rad_weight w = wt(nm->ckpt("%s.weight", src), { x.pitch });
        rad_weight bw = wt(nm->ckpt("%s.bias", src), { x.pitch });
        rad_op op = RAD_OP(b, "layernorm",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(x.lvl)), RAD_INT("n", x.pitch),
                                      RAD_F64("eps", 1e-5), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w, bw));
        add(op, { R(x), W(w), W(bw), R(y) }, x.lvl, { x.buf }, { y.buf });
        return y;
    }
    /* y = x + e, e one row broadcast along x's rows. */
    V add_row(const V& x, const V& e) {
        V y = buf(x.lvl, x.pitch);
        rad_op op = RAD_OP(b, "add",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(x.lvl)), RAD_INT("n", x.pitch),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS);
        add(op, { R(x), R(e), R(y) }, x.lvl, { x.buf, e.buf }, { y.buf });
        return y;
    }
    /* dst's columns [col, col + x.w()) = x: a skip connection's half of a concatenation. */
    void copy(const V& x, const V& dst, int64_t col) {
        V d = dst;
        d.col = col;
        d.width = x.w();
        rad_op op = RAD_OP(b, "cast",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(x.lvl)), RAD_INT("n", x.w()),
                                      RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                           RAD_NOWEIGHTS);
        add(op, { R(x), R(d) }, x.lvl, { x.buf }, { dst.buf });
    }
    /* {0, n}: the bounds of one sequence of the entry's rows at `lvl`, off the iota table. */
    V cu(int lvl) {
        V y;
        y.lvl = kRowsTwo;
        y.pitch = 1;
        y.buf = decl_b(b, nm->f("unet.cu%d", n_buf++), RAD_I32, { 2 });
        rad_op op = RAD_OP(b, "cu_repeat", RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", 1)),
                           RAD_NOWEIGHTS);
        Opd it = Kd(Opd::IOTA);
        it.v.lvl = lvl;
        add(op, { it, R(y) }, kRowsTwo, { iota }, { y.buf });
        return y;
    }
    /* Self attention over the entry's rows, 64-wide heads. */
    V attn(const V& q, const V& k, const V& v, const V& cu_q) {
        V y = buf(q.lvl, q.w());
        rad_op op = RAD_OP(b, "attn_dense",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(q.lvl)), RAD_INT("head_dim", 64),
                                      RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                      RAD_STR("dtype", "bf16"), RAD_F64("scale", 0.125)),
                           RAD_NOWEIGHTS);
        add(op, { R(q), R(k), R(v), R(cu_q), R(y) }, q.lvl, { q.buf, k.buf, v.buf, cu_q.buf },
            { y.buf });
        return y;
    }
    /* Cross attention of the entry's rows over its conditioning's. */
    V xattn(const V& q, const V& k, const V& v, const V& cu_q, const V& cu_k) {
        V y = buf(q.lvl, q.w());
        rad_op op = RAD_OP(b, "attn_cross",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(q.lvl)), RAD_INT("head_dim", 64),
                                      RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                      RAD_STR("dtype", "bf16"), RAD_F64("scale", 0.125)),
                           RAD_NOWEIGHTS);
        add(op, { R(q), R(k), R(v), R(cu_q), R(cu_k), R(y) }, q.lvl,
            { q.buf, k.buf, v.buf, cu_q.buf, cu_k.buf }, { y.buf });
        return y;
    }
    /* GEGLU: the projection's first half times gelu (erf) of its second. */
    V geglu(const V& p) {
        const int64_t n = p.pitch / 2;
        V y = buf(p.lvl, n);
        V a = p, g = p;
        a.col = 0; a.width = n;
        g.col = n; g.width = n;
        rad_op op = RAD_OP(b, "act_mul",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap(p.lvl)), RAD_INT("n", n),
                                      RAD_STR("act", "gelu"), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS);
        add(op, { R(g), R(a), R(y) }, p.lvl, { p.buf }, { y.buf });
        return y;
    }

    /* ---- blocks */
    /* in_layers (GroupNorm, SiLU, conv), the time embedding's projection added per channel,
     * out_layers (GroupNorm, SiLU, conv) plus the input or its 1x1 skip convolution. */
    V resblock(const V& x, int64_t cout, const char* pre, const V& emb_s) {
        V t = gn(x, nm->ckpt("%s.in_layers.0", pre), 1e-5, true);
        V h = conv3(t, cout, nm->ckpt("%s.in_layers.2", pre), 1, 1);
        V e = lin(emb_s, cout, nm->ckpt("%s.emb_layers.1", pre), true);
        h = add_row(h, e);
        V t2 = gn(h, nm->ckpt("%s.out_layers.0", pre), 1e-5, true);
        V skip = x;
        if (x.pitch != cout) skip = lin(x, cout, nm->ckpt("%s.skip_connection", pre), true);
        return conv3(t2, cout, nm->ckpt("%s.out_layers.3", pre), 1, 1, &skip);
    }
    /* SpatialTransformer with linear projections: GroupNorm, proj_in, `depth` blocks of self
     * attention, cross attention and GEGLU each added back, proj_out plus the input. */
    V transformer(const V& x, const char* pre, int depth, const V& ctx, const V& cu_q,
                  const V& cu_k) {
        const int64_t C = x.pitch;
        V t = gn(x, nm->ckpt("%s.norm", pre), 1e-6, false);
        V h = lin(t, C, nm->ckpt("%s.proj_in", pre), true);
        for (int d = 0; d < depth; ++d) {
            const char* bp = nm->ckpt("%s.transformer_blocks.%d", pre, d);
            V a = ln(h, nm->ckpt("%s.norm1", bp));
            V q = lin(a, C, nm->ckpt("%s.attn1.to_q", bp), false);
            V k = lin(a, C, nm->ckpt("%s.attn1.to_k", bp), false);
            V v = lin(a, C, nm->ckpt("%s.attn1.to_v", bp), false);
            V o = attn(q, k, v, cu_q);
            h = lin(o, C, nm->ckpt("%s.attn1.to_out.0", bp), true, &h);
            a = ln(h, nm->ckpt("%s.norm2", bp));
            q = lin(a, C, nm->ckpt("%s.attn2.to_q", bp), false);
            k = lin(ctx, C, nm->ckpt("%s.attn2.to_k", bp), false);
            v = lin(ctx, C, nm->ckpt("%s.attn2.to_v", bp), false);
            o = xattn(q, k, v, cu_q, cu_k);
            h = lin(o, C, nm->ckpt("%s.attn2.to_out.0", bp), true, &h);
            a = ln(h, nm->ckpt("%s.norm3", bp));
            V p = lin(a, 8 * C, nm->ckpt("%s.ff.net.0.proj", bp), true);
            V f = geglu(p);
            h = lin(f, C, nm->ckpt("%s.ff.net.2", bp), true, &h);
        }
        return lin(h, C, nm->ckpt("%s.proj_out", pre), true, &x);
    }
    /* [a | b] along the channels. */
    V concat(const V& a, const V& s) {
        V y = buf(a.lvl, a.pitch + s.pitch);
        copy(a, y, 0);
        copy(s, y, a.pitch);
        return y;
    }

    RadOperand opd(RadCtx* c, const Opd& o, const Geo& g) const;
    void issue(RadCtx* c, const Geo& g) const;
};

/* ================================================================== the model */
struct Model {
    Names nm{""};
    std::string pred = "eps";
    bool zero_snr = false;

    /* ---- the conditioning */
    TextEnc te1{}, te2{};
    rad_buf b_pos = 0, b_wcu = 0, b_zero = 0, b_ctx = 0, b_pooled = 0;
    rad_op  op_pos = 0, op_wcu = 0, op_zero = 0, op_store = 0, op_pstore = 0;

    /* ---- the per-image state and the stores (PERSIST) */
    rad_buf b_state = 0, b_hist = 0, b_prev = 0, b_store = 0, b_pstore = 0;
    rad_buf b_xin = 0, b_eps = 0, b_iota = 0, b_zf = 0;
    rad_op  op_iota = 0, op_zf = 0, op_kdin = 0, op_kdstep = 0;

    /* ---- the time and size embeddings, per entry */
    rad_weight w_t1 = 0, b_t1 = 0, w_t2 = 0, b_t2 = 0, w_l1 = 0, b_l1 = 0, w_l2 = 0, b_l2 = 0;
    rad_buf b_tproj = 0, b_te1 = 0, b_te1s = 0, b_temb = 0, b_add = 0, b_le1 = 0, b_le1s = 0;
    rad_buf b_emb = 0, b_embs = 0;
    rad_op  op_tproj = 0, op_t1 = 0, op_ts = 0, op_t2 = 0, op_pool = 0, op_szo = 0, op_szc = 0,
            op_szt = 0, op_l1 = 0, op_ls = 0, op_l2 = 0, op_es = 0;

    /* ---- the UNet */
    Net net{};

    /* ---- the decoder */
    Vae     dec{};
    VImg    dec_out{};
    rad_buf b_dlf = 0, b_decin = 0, b_img = 0;
    rad_op  op_dscale = 0, op_dcast = 0, op_rgb = 0;
};

static Model g_model[MAX_SLOTS];

/* ================================================================== the recorded UNet's issue */
RadOperand Net::opd(RadCtx* c, const Opd& o, const Geo& g) const {
    switch (o.kind) {
    case Opd::NONE:   return RAD_NONE;
    case Opd::WEIGHT: return RAD_W(o.w);
    case Opd::ROWS: {
        /* The context is the entry's conditioning where the store keeps it. */
        if (o.v.buf == store) return brow_slice(store, g.store_row, g.cond, kCtx);
        const int64_t r = g.rows(o.v.lvl);
        if (o.v.width && o.v.width != o.v.pitch) return bcol(o.v.buf, o.v.col, o.v.width, r);
        return brows(o.v.buf, r);
    }
    case Opd::IMG: {
        const int l = o.v.lvl;
        return RAD_P_T2(rad_buf_ptr(c, o.v.buf), RAD_BF16, g.h >> l, (g.w >> l) * o.v.pitch);
    }
    case Opd::XIN:
        return RAD_P_T2((const char*)rad_buf_ptr(c, xin) + g.row0 * kLatC * 2, RAD_BF16, g.h,
                        g.w * kLatC);
    case Opd::EPS:
        return RAD_P_T2((char*)rad_buf_ptr(c, eps) + g.row0 * kLatC * 2, RAD_BF16, g.h,
                        g.w * kLatC);
    case Opd::IOTA:   return brow_slice(iota, g.rows(o.v.lvl), 1, 1);
    case Opd::STATS:  return RAD_P_T2(rad_buf_ptr(c, stats), RAD_F32, 1, kGnStats);
    }
    return RAD_NONE;
}

void Net::issue(RadCtx* c, const Geo& g) const {
    RadOperand a[8];
    for (const UOp& u : ops) {
        const int n = (int)u.o.size();
        for (int i = 0; i < n; ++i) a[i] = opd(c, u.o[(size_t)i], g);
        rad_issue(c, u.op, a, n, g.rows(u.n));
    }
}

/* ================================================================== declare */
static rad_weight wcopy(RadBuilder* b, const char* src, std::initializer_list<int64_t> shape,
                        int access, int group, bool mapped = false) {
    if (!mapped && map_copy(b, src, src) < 0) return 0;
    return decl_w(b, src, RAD_BF16, shape, access, RAD_SHARD_NONE, grp_layer(group));
}

/* A linear over `rows` rows: gemm_nt_bias with a bias (and a residual `res` added in), gemm_nt
 * without. */
static rad_op dense(RadBuilder* b, int64_t rows, int64_t N, int64_t K, rad_weight w,
                    rad_weight bias, rad_buf in, rad_buf out, rad_buf res = 0) {
    rad_op h = bias ? RAD_OP(b, "gemm_nt_bias",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w, bias))
                    : RAD_OP(b, "gemm_nt",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w));
    if (res) return rw(b, h, { in, res }, { out });
    return rw(b, h, { in }, { out });
}

static rad_op tstep_embed(RadBuilder* b, int64_t rows, int64_t n, rad_buf out) {
    return rw(b, RAD_OP(b, "timestep_embed",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("scale", 1.0),
                                   RAD_F64("max_period", 10000.0), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
              {}, { out });
}

/* One CLIP text encoder over T rows: token and position embeddings, `L` pre-norm layers, the
 * context's columns copied out after layer `tap`. */
static int declare_text(RadBuilder* b, Model& m, TextEnc& E, const char* root, int grp,
                        int64_t D, int64_t F, int64_t L, int64_t tap, bool pooled) {
    Names& nm = m.nm;
    const int64_t T = kCondRows;
    const char* tm = nm.ckpt("%s.text_model", root);
    E.D = D;
    E.F = F;
    E.L = L;
    E.tap = tap;
    E.tok = wcopy(b, nm.ckpt("%s.embeddings.token_embedding.weight", tm), { 49408, D },
                  RAD_ACCESS_VOCAB, grp);
    E.pos = wcopy(b, nm.ckpt("%s.embeddings.position_embedding.weight", tm), { kWin, D },
                  RAD_ACCESS_PER_TOKEN, grp);
    if (!E.tok || !E.pos) return RAD_E_INVAL;
    const char* sc = nm.f("te%d", grp == kGrpTe1 ? 1 : 2);
    E.x  = decl_b(b, nm.f("%s.x", sc), RAD_BF16, { T, D });
    E.pe = decl_b(b, nm.f("%s.pe", sc), RAD_BF16, { T, D });
    E.h  = decl_b(b, nm.f("%s.h", sc), RAD_BF16, { T, D });
    E.q  = decl_b(b, nm.f("%s.q", sc), RAD_BF16, { T, D });
    E.k  = decl_b(b, nm.f("%s.k", sc), RAD_BF16, { T, D });
    E.v  = decl_b(b, nm.f("%s.v", sc), RAD_BF16, { T, D });
    E.a  = decl_b(b, nm.f("%s.attn", sc), RAD_BF16, { T, D });
    E.f1 = decl_b(b, nm.f("%s.fc1", sc), RAD_BF16, { T, F });
    E.f2 = decl_b(b, nm.f("%s.act", sc), RAD_BF16, { T, F });
    auto embed = [&](rad_weight w, int64_t vocab, rad_buf ids, rad_buf out) {
        rad_op op = RAD_OP(b, "embed_lookup",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", D),
                                      RAD_INT("n_vocab", vocab), RAD_STR("dtype", "bf16"),
                                      RAD_INT("vocab_offset", 0)),
                           RAD_WEIGHTS(w));
        if (ids) return rw(b, op, { ids }, { out });
        return rw(b, op, {}, { out });
    };
    E.op_tok = embed(E.tok, 49408, 0, E.x);
    /* Padding rows that read token 0 (bigG's tokenizer pads with "!"). */
    if (pooled) E.op_pad = embed(E.tok, 49408, m.b_zero, E.x);
    E.op_pos = embed(E.pos, kWin, m.b_pos, E.pe);
    E.op_pe = rw(b, RAD_OP(b, "add",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                 { E.x, E.pe }, { E.x });
    if (!E.op_tok || !E.op_pos || !E.op_pe || (pooled && !E.op_pad)) return RAD_E_INVAL;
    E.layers.resize((size_t)L);
    for (int64_t l = 0; l < L; ++l) {
        TextLayer& Y = E.layers[(size_t)l];
        const char* lp = nm.ckpt("%s.encoder.layers.%lld", tm, (long long)l);
        const int g = grp;
        auto w = [&](const char* fmt, std::initializer_list<int64_t> shape) {
            return wcopy(b, nm.ckpt(fmt, lp), shape, RAD_ACCESS_PER_TOKEN, g);
        };
        Y.ln1w = w("%s.layer_norm1.weight", { D });
        Y.ln1b = w("%s.layer_norm1.bias", { D });
        Y.qw = w("%s.self_attn.q_proj.weight", { D, D });
        Y.qb = w("%s.self_attn.q_proj.bias", { D });
        Y.kw = w("%s.self_attn.k_proj.weight", { D, D });
        Y.kb = w("%s.self_attn.k_proj.bias", { D });
        Y.vw = w("%s.self_attn.v_proj.weight", { D, D });
        Y.vb = w("%s.self_attn.v_proj.bias", { D });
        Y.ow = w("%s.self_attn.out_proj.weight", { D, D });
        Y.ob = w("%s.self_attn.out_proj.bias", { D });
        Y.ln2w = w("%s.layer_norm2.weight", { D });
        Y.ln2b = w("%s.layer_norm2.bias", { D });
        Y.f1w = w("%s.mlp.fc1.weight", { F, D });
        Y.f1b = w("%s.mlp.fc1.bias", { F });
        Y.f2w = w("%s.mlp.fc2.weight", { D, F });
        Y.f2b = w("%s.mlp.fc2.bias", { D });
        if (!Y.ln1w || !Y.ln1b || !Y.qw || !Y.qb || !Y.kw || !Y.kb || !Y.vw || !Y.vb || !Y.ow ||
            !Y.ob || !Y.ln2w || !Y.ln2b || !Y.f1w || !Y.f1b || !Y.f2w || !Y.f2b)
            return RAD_E_INVAL;
        auto lnorm = [&](rad_weight lw, rad_weight lb) {
            return rw(b, RAD_OP(b, "layernorm",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                           RAD_F64("eps", 1e-5), RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(lw, lb)),
                      { E.x }, { E.h });
        };
        Y.op_ln1 = lnorm(Y.ln1w, Y.ln1b);
        Y.op_q = dense(b, T, D, D, Y.qw, Y.qb, E.h, E.q);
        Y.op_k = dense(b, T, D, D, Y.kw, Y.kb, E.h, E.k);
        Y.op_v = dense(b, T, D, D, Y.vw, Y.vb, E.h, E.v);
        Y.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", 64),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 1),
                                            RAD_STR("dtype", "bf16"), RAD_F64("scale", 0.125),
                                            RAD_INT("max_seqlen", kWin)),
                                 RAD_NOWEIGHTS),
                       { E.q, E.k, E.v, m.b_wcu }, { E.a });
        Y.op_o = dense(b, T, D, D, Y.ow, Y.ob, E.a, E.x, E.x);
        Y.op_ln2 = lnorm(Y.ln2w, Y.ln2b);
        Y.op_f1 = dense(b, T, F, D, Y.f1w, Y.f1b, E.h, E.f1);
        Y.op_act = unary(b, E.quick ? "gelu_quick" : "gelu", T, F, E.f1, E.f2);
        Y.op_f2 = dense(b, T, D, F, Y.f2w, Y.f2b, E.f2, E.x, E.x);
        if (!Y.op_ln1 || !Y.op_q || !Y.op_k || !Y.op_v || !Y.op_attn || !Y.op_o || !Y.op_ln2 ||
            !Y.op_f1 || !Y.op_act || !Y.op_f2)
            return RAD_E_INVAL;
        if (l == tap)
            E.op_tap = cast_op(b, T, D, "bf16", "bf16", E.x, m.b_ctx);
    }
    if (!E.op_tap) return RAD_E_INVAL;
    if (!pooled) return RAD_OK;
    /* The pooled vector: the first window's end-of-text row through the final norm and the
     * projection. */
    E.fnw = wcopy(b, nm.ckpt("%s.final_layer_norm.weight", tm), { D }, RAD_ACCESS_PER_TOKEN,
                  grp);
    E.fnb = wcopy(b, nm.ckpt("%s.final_layer_norm.bias", tm), { D }, RAD_ACCESS_PER_TOKEN,
                  grp);
    E.proj = wcopy(b, nm.ckpt("%s.text_projection.weight", root), { kPooled, D },
                   RAD_ACCESS_PER_TOKEN, grp);
    if (!E.fnw || !E.fnb || !E.proj) return RAD_E_INVAL;
    E.e1 = decl_b(b, nm.f("%s.eos", sc), RAD_BF16, { 1, D });
    E.e2 = decl_b(b, nm.f("%s.eos_norm", sc), RAD_BF16, { 1, D });
    E.op_eos = rw(b, RAD_OP(b, "gather_rows",
                            RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { E.x, m.b_pos }, { E.e1 });
    E.op_fn = rw(b, RAD_OP(b, "layernorm",
                           RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D), RAD_F64("eps", 1e-5),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(E.fnw, E.fnb)),
                 { E.e1 }, { E.e2 });
    E.op_proj = dense(b, 1, kPooled, D, E.proj, 0, E.e2, m.b_pooled);
    return E.op_eos && E.op_fn && E.op_proj ? RAD_OK : RAD_E_INVAL;
}

static int declare_cond(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t T = kCondRows;
    /* Positions restart in every window; the windows' bounds; a row of token-0 ids. */
    m.b_pos  = decl_b(b, nm.f("cond.pos"), RAD_I32, { T });
    m.b_wcu  = decl_b(b, nm.f("cond.cu"), RAD_I32, { kMaxWin + 1 });
    m.b_zero = decl_b(b, nm.f("cond.zero_ids"), RAD_I32, { kWin });
    m.b_ctx  = decl_b(b, nm.f("cond.ctx"), RAD_BF16, { T, kCtx });
    m.b_pooled = decl_b(b, nm.f("cond.pooled"), RAD_BF16, { 1, kPooled });
    m.op_pos = rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("start", 0),
                                       RAD_INT("step", 1), RAD_INT("period", kWin)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_pos });
    m.op_wcu = rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, kMaxWin + 1), RAD_INT("start", 0),
                                       RAD_INT("step", kWin)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_wcu });
    m.op_zero = rw(b, RAD_OP(b, "iota",
                             RAD_PARAMS(RAD_RANGE("M", 1, kWin), RAD_INT("start", 0),
                                        RAD_INT("step", 0)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_zero });
    if (!m.op_pos || !m.op_wcu || !m.op_zero) return RAD_E_INVAL;
    m.te1.quick = true;
    m.te1.ctx_col = 0;
    RAD_ARCH_TRY(declare_text(b, m, m.te1, kTe1, kGrpTe1, 768, 3072, 11, 10, false));
    m.te2.quick = false;
    m.te2.ctx_col = 768;
    RAD_ARCH_TRY(declare_text(b, m, m.te2, kTe2, kGrpTe2, 1280, 5120, 32, 30, true));
    /* Into every lane of the request at this branch: lane l's branch r at row (2 l + r) of the
     * stores. */
    m.op_store = rw(b, RAD_OP(b, "lane_store",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", kCtx),
                                         RAD_INT("stride", 2 * (int64_t)kCondRows),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_ctx, m.b_store }, { m.b_store });
    m.op_pstore = rw(b, RAD_OP(b, "lane_store",
                               RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", kPooled),
                                          RAD_INT("stride", 2), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_pooled, m.b_pstore }, { m.b_pstore });
    return m.op_store && m.op_pstore ? RAD_OK : RAD_E_INVAL;
}

/* The time embedding (the timestep's sinusoid through time_embed) plus the added embedding (the
 * pooled vector beside the image size's three sinusoid pairs, through label_emb), and its SiLU,
 * which every residual block projects. One row an entry. */
static int declare_emb(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int g = kGrpUnet;
    auto w = [&](const char* fmt, std::initializer_list<int64_t> shape) {
        return wcopy(b, nm.ckpt(fmt, kUnet), shape, RAD_ACCESS_PER_TOKEN, g);
    };
    m.w_t1 = wcopy(b, nm.ckpt("%s.time_embed.0.weight", kUnet), { kTemb, 320 },
                   RAD_ACCESS_PER_TOKEN, g, true);
    m.b_t1 = w("%s.time_embed.0.bias", { kTemb });
    m.w_t2 = w("%s.time_embed.2.weight", { kTemb, kTemb });
    m.b_t2 = w("%s.time_embed.2.bias", { kTemb });
    m.w_l1 = w("%s.label_emb.0.0.weight", { kTemb, kAdd });
    m.b_l1 = w("%s.label_emb.0.0.bias", { kTemb });
    m.w_l2 = w("%s.label_emb.0.2.weight", { kTemb, kTemb });
    m.b_l2 = w("%s.label_emb.0.2.bias", { kTemb });
    if (!m.w_t1 || !m.b_t1 || !m.w_t2 || !m.b_t2 || !m.w_l1 || !m.b_l1 || !m.w_l2 || !m.b_l2)
        return RAD_E_INVAL;
    m.b_tproj = decl_b(b, nm.f("emb.tproj"), RAD_BF16, { 1, 320 });
    m.b_te1   = decl_b(b, nm.f("emb.t1"), RAD_BF16, { 1, kTemb });
    m.b_te1s  = decl_b(b, nm.f("emb.t1s"), RAD_BF16, { 1, kTemb });
    m.b_temb  = decl_b(b, nm.f("emb.temb"), RAD_BF16, { 1, kTemb });
    m.b_add   = decl_b(b, nm.f("emb.add"), RAD_BF16, { 1, kAdd });
    m.b_le1   = decl_b(b, nm.f("emb.l1"), RAD_BF16, { 1, kTemb });
    m.b_le1s  = decl_b(b, nm.f("emb.l1s"), RAD_BF16, { 1, kTemb });
    m.b_emb   = decl_b(b, nm.f("emb.emb"), RAD_BF16, { 1, kTemb });
    m.b_embs  = decl_b(b, nm.f("emb.emb_silu"), RAD_BF16, { 1, kTemb });
    m.op_tproj = tstep_embed(b, 1, 320, m.b_tproj);
    m.op_t1 = dense(b, 1, kTemb, 320, m.w_t1, m.b_t1, m.b_tproj, m.b_te1);
    m.op_ts = unary(b, "silu", 1, kTemb, m.b_te1, m.b_te1s);
    m.op_t2 = dense(b, 1, kTemb, kTemb, m.w_t2, m.b_t2, m.b_te1s, m.b_temb);
    m.op_pool = cast_op(b, 1, kPooled, "bf16", "bf16", m.b_pstore, m.b_add);
    /* (height, width), (0, 0), (height, width): each value's 256-wide sinusoid, laid end to end
     * after the pooled vector. */
    m.op_szo = tstep_embed(b, 2, 256, m.b_add);
    m.op_szc = rw(b, RAD_OP(b, "timestep_embed",
                            RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("n", 256),
                                       RAD_F64("scale", 1.0), RAD_F64("max_period", 10000.0),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { m.b_zf }, { m.b_add });
    m.op_szt = tstep_embed(b, 2, 256, m.b_add);
    m.op_l1 = dense(b, 1, kTemb, kAdd, m.w_l1, m.b_l1, m.b_add, m.b_le1);
    m.op_ls = unary(b, "silu", 1, kTemb, m.b_le1, m.b_le1s);
    m.op_l2 = dense(b, 1, kTemb, kTemb, m.w_l2, m.b_l2, m.b_le1s, m.b_emb, m.b_temb);
    m.op_es = unary(b, "silu", 1, kTemb, m.b_emb, m.b_embs);
    return m.op_tproj && m.op_t1 && m.op_ts && m.op_t2 && m.op_pool && m.op_szo && m.op_szc &&
                   m.op_szt && m.op_l1 && m.op_ls && m.op_l2 && m.op_es
               ? RAD_OK : RAD_E_INVAL;
}

/* The UNet, in the order a step runs it. */
static int declare_unet(RadBuilder* b, Model& m) {
    Net& n = m.net;
    Names& nm = m.nm;
    n.b = b;
    n.nm = &nm;
    n.stats = decl_b(b, nm.f("unet.gn_stats"), RAD_F32, { kGnStats });
    n.iota = m.b_iota;
    n.store = m.b_store;
    n.xin = m.b_xin;
    n.eps = m.b_eps;
    V emb_s{};
    emb_s.buf = m.b_embs;
    emb_s.lvl = kRowsOne;
    emb_s.pitch = kTemb;
    V ctx{};
    ctx.buf = m.b_store;
    ctx.lvl = kRowsCond;
    ctx.pitch = kCtx;
    const char* U = kUnet;
    /* Every op's sequence bounds, off the iota table. */
    const V cu1 = n.cu(1), cu2 = n.cu(2), cuc = n.cu(kRowsCond);
    /* ---- the input blocks; each output is kept for its skip connection */
    n.grp = kGrpUnet + 1;
    Opd xin = Net::Kd(Opd::XIN);
    xin.v.buf = m.b_xin;
    V h = n.conv_direct(xin, 0, kLatC, 320, nm.ckpt("%s.input_blocks.0.0", U));
    std::vector<V> skips{ h };
    struct In { int64_t c; int depth; bool down; };
    const In ins[] = { { 320, 0, false }, { 320, 0, false }, { 320, 0, true },
                       { 640, 2, false }, { 640, 2, false }, { 640, 0, true },
                       { 1280, 10, false }, { 1280, 10, false } };
    auto st = [&](const V& x, const char* pre, int depth) {
        const V& cq = x.lvl == 1 ? cu1 : cu2;
        return n.transformer(x, pre, depth, ctx, cq, cuc);
    };
    for (int i = 1; i <= 8; ++i) {
        const In& s = ins[i - 1];
        n.grp = kGrpUnet + 1 + i;
        if (s.down) {
            h = n.conv3(h, h.pitch, nm.ckpt("%s.input_blocks.%d.0.op", U, i), 2, 1);
        } else {
            h = n.resblock(h, s.c, nm.ckpt("%s.input_blocks.%d.0", U, i), emb_s);
            if (s.depth) h = st(h, nm.ckpt("%s.input_blocks.%d.1", U, i), s.depth);
        }
        skips.push_back(h);
    }
    /* ---- the middle block */
    n.grp = kGrpUnet + 10;
    h = n.resblock(h, 1280, nm.ckpt("%s.middle_block.0", U), emb_s);
    h = st(h, nm.ckpt("%s.middle_block.1", U), 10);
    h = n.resblock(h, 1280, nm.ckpt("%s.middle_block.2", U), emb_s);
    /* ---- the output blocks: the matching input block's output concatenated after h */
    struct Out { int64_t c; int depth; bool up; };
    const Out outs[] = { { 1280, 10, false }, { 1280, 10, false }, { 1280, 10, true },
                         { 640, 2, false },   { 640, 2, false },   { 640, 2, true },
                         { 320, 0, false },   { 320, 0, false },   { 320, 0, false } };
    for (int i = 0; i < 9; ++i) {
        const Out& s = outs[i];
        n.grp = kGrpUnet + 11 + i;
        const V skip = skips.back();
        skips.pop_back();
        V cat = n.concat(h, skip);
        h = n.resblock(cat, s.c, nm.ckpt("%s.output_blocks.%d.0", U, i), emb_s);
        int sub = 1;
        if (s.depth) h = st(h, nm.ckpt("%s.output_blocks.%d.%d", U, i, sub++), s.depth);
        if (s.up) h = n.conv3(h, h.pitch, nm.ckpt("%s.output_blocks.%d.%d.conv", U, i, sub), 1, 2);
    }
    /* ---- the output: GroupNorm, SiLU, conv to the latent's channels */
    n.grp = kGrpUnet + 20;
    V o = n.gn(h, nm.ckpt("%s.out.0", U), 1e-5, true);
    Opd eps = Net::Kd(Opd::EPS);
    eps.v.buf = m.b_eps;
    eps.v.lvl = 0;
    eps.v.pitch = kLatC;
    Opd oi = Net::I(o);
    n.conv_direct(oi, 0, 320, kLatC, nm.ckpt("%s.out.2", U), &eps);
    for (const UOp& u : n.ops) if (!u.op) return RAD_E_INVAL;
    return RAD_OK;
}

/* The decoder: the f32 latent divided by the scaling factor, to bf16, then the VAE's decoder in
 * diffusers' AutoencoderKL names (GroupNorm residual blocks, a mid-block attention, three 2x
 * nearest upsamplings). */
static int declare_decoder(RadBuilder* b, Model& m) {
    Vae& v = m.dec;
    Names& nm = m.nm;
    v.nm = &nm;
    v.b = b;
    v.scope = "vdec";
    v.down = false;
    v.first_group = kGrpVae;
    v.base_px = kMaxImageTokens;
    const int64_t NT = kMaxImageTokens;
    m.b_dlf = decl_b(b, nm.f("vdec.lat_f32"), RAD_F32, { NT, kLatC });
    m.b_decin = decl_b(b, nm.f("vdec.in"), RAD_BF16, { NT, kLatC });
    char mul[64], zero[16];
    const double s = 1.0 / kVaeScale;
    std::snprintf(mul, sizeof mul, "%.9g %.9g %.9g %.9g", s, s, s, s);
    std::snprintf(zero, sizeof zero, "0 0 0 0");
    static thread_local std::string smul, szero;
    smul = mul;
    szero = zero;
    m.op_dscale = rw(b, RAD_OP(b, "chan_affine",
                               RAD_PARAMS(RAD_RANGE("M", 1, NT), RAD_INT("n", kLatC),
                                          RAD_STR("mul", smul.c_str()),
                                          RAD_STR("add", szero.c_str()),
                                          RAD_STR("mode", "fma"), RAD_STR("dtype", "f32"),
                                          RAD_INT("round", 0)),
                               RAD_NOWEIGHTS),
                     { m.b_state, m.b_prev }, { m.b_dlf });
    m.op_dcast = cast_op(b, NT, kLatC, "f32", "bf16", m.b_dlf, m.b_decin);
    if (!m.op_dscale || !m.op_dcast) return RAD_E_INVAL;
    const int64_t ch[4] = { 512, 512, 256, 128 };
    const double eps = 1e-6;
    VImg x{ m.b_decin, kLatC, 0 };
    x = v.conv(x, kLatC, 1, 1, 1, "same", "vae.post_quant_conv");
    x = v.conv(x, ch[0], 3, 1, 1, "same", "vae.decoder.conv_in");
    x = v.resblock_gn(x, x.ch, "vae.decoder.mid_block.resnets.0", kGroups, eps);
    x = v.attention_gn(x, "vae.decoder.mid_block.attentions.0", kGroups, eps);
    x = v.resblock_gn(x, x.ch, "vae.decoder.mid_block.resnets.1", kGroups, eps);
    for (int i = 0; i < 4; ++i) {
        for (int r = 0; r < 3; ++r)
            x = v.resblock_gn(x, ch[i], nm.f("vae.decoder.up_blocks.%d.resnets.%d", i, r), kGroups,
                              eps);
        if (i < 3)
            x = v.conv(x, x.ch, 3, 1, 2, "same",
                       nm.f("vae.decoder.up_blocks.%d.upsamplers.0.conv", i));
    }
    x = v.gnorm(x, true, "vae.decoder.conv_norm_out", kGroups, eps);
    m.dec_out = v.conv(x, 3, 3, 1, 1, "same", "vae.decoder.conv_out");
    if (!m.dec_out.buf) return RAD_E_INVAL;
    for (const VOp& o : v.ops) if (!o.op) return RAD_E_INVAL;
    const int64_t px = NT * 64;
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
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: sdxl runs on one card; serve it with a tensor-parallel size "
                        "of 1\n");
        return RAD_E_UNSUPPORTED;
    }
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    m.pred = rad_meta_gets(meta, "sdxl.prediction", "eps");
    m.zero_snr = rad_meta_geti(meta, "sdxl.zero_snr", 0) != 0;
    if (m.pred != "eps" && m.pred != "v") {
        fprintf(stderr, "radiance: sdxl: prediction '%s' is neither eps nor v\n", m.pred.c_str());
        return RAD_E_UNSUPPORTED;
    }
    /* The weights are served bf16; the release's f16 is cast by the recipe. The first UNet weight
     * is mapped here and declared with the embeddings. */
    {
        const char* probe = nm.ckpt("%s.time_embed.0.weight", kUnet);
        if (map_copy(b, probe, probe) < 0) return RAD_E_INVAL;
        const uint32_t dt = weight_codes_dtype(b, probe, RAD_BF16);
        if (dt != RAD_BF16) {
            fprintf(stderr, "radiance: sdxl: the UNet is stored %s; convert with "
                            "data/recipes/sdxl-bf16.recipe, which casts the release to bf16\n",
                    rad_dtype_name(dt));
            return RAD_E_UNSUPPORTED;
        }
    }
    const int64_t max_img = std::max<int64_t>(ctx->max_tok, kMaxImageTokens);
    const int64_t lane_rows = (int64_t)kMaxLanes * kMaxImageTokens;
    m.b_state = decl_b(b, nm.f("lat_state"), RAD_F32, { lane_rows, kLatC }, RAD_BUF_PERSIST);
    m.b_hist  = decl_b(b, nm.f("lat_hist"), RAD_F32, { lane_rows, kLatC }, RAD_BUF_PERSIST);
    m.b_prev  = decl_b(b, nm.f("lat_preview"), RAD_F32, { lane_rows, kLatC }, RAD_BUF_PERSIST);
    m.b_store = decl_b(b, nm.f("cond_store"), RAD_BF16,
                       { 2 * (int64_t)kMaxLanes * kCondRows, kCtx }, RAD_BUF_PERSIST);
    m.b_pstore = decl_b(b, nm.f("pooled_store"), RAD_BF16, { 2 * (int64_t)kMaxLanes, kPooled },
                        RAD_BUF_PERSIST);
    m.b_xin = decl_b(b, nm.f("unet_in"), RAD_BF16, { max_img, kLatC }, RAD_BUF_PERSIST);
    m.b_eps = decl_b(b, nm.f("unet_out"), RAD_BF16, { max_img, kLatC }, RAD_BUF_PERSIST);
    m.b_iota = decl_b(b, nm.f("iota"), RAD_I32, { kMaxImageTokens + 1 }, RAD_BUF_PERSIST);
    m.b_zf = decl_b(b, nm.f("zeros_f32"), RAD_F32, { 1, 2 }, RAD_BUF_PERSIST);
    /* ---- adapter key names: kohya's `lora_unet_` over the UNet's own single-file paths, and
     * `lora_te1_` / `lora_te2_` over the two text encoders, whose layers are named as
     * transformers' CLIPTextModel names them in both. */
    RAD_ARCH_TRY(lora_names(b, "model.diffusion_model.", kLoraDiffusionSpellings, {}));
    RAD_ARCH_TRY(lora_names(b, "conditioner.embedders.0.transformer.", "lora_te1_", {}));
    RAD_ARCH_TRY(lora_names(b, "conditioner.embedders.1.transformer.", "lora_te2_", {}));
    RAD_ARCH_TRY(declare_cond(b, m));

    /* ---- a denoising pass: the tables, the model input, then per entry the embeddings and the
     * UNet, then the sampler */
    m.op_iota = rw(b, RAD_OP(b, "iota",
                             RAD_PARAMS(RAD_RANGE("M", 1, kMaxImageTokens + 1),
                                        RAD_INT("start", 0), RAD_INT("step", 1)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_iota });
    m.op_zf = rw(b, RAD_OP(b, "fill",
                           RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 2),
                                      RAD_STR("dtype", "f32"), RAD_F64("value", 0.0)),
                           RAD_NOWEIGHTS),
                 {}, { m.b_zf });
    m.op_kdin = rw(b, RAD_OP(b, "kd_input",
                             RAD_PARAMS(RAD_RANGE("M", 1, max_img), RAD_INT("n", kLatC),
                                        RAD_INT("stride", (int64_t)kMaxImageTokens),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_state }, { m.b_state, m.b_xin });
    if (!m.op_iota || !m.op_zf || !m.op_kdin) return RAD_E_INVAL;
    RAD_ARCH_TRY(declare_emb(b, m));
    RAD_ARCH_TRY(declare_unet(b, m));
    m.op_kdstep = rw(b, RAD_OP(b, "kd_step",
                               RAD_PARAMS(RAD_RANGE("M", 1, max_img), RAD_INT("n", kLatC),
                                          RAD_INT("stride", (int64_t)kMaxImageTokens),
                                          RAD_STR("pred", m.pred.c_str()),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_eps, m.b_state, m.b_hist, m.b_prev },
                     { m.b_state, m.b_hist, m.b_prev });
    if (!m.op_kdstep) return RAD_E_INVAL;
    RAD_ARCH_TRY(declare_decoder(b, m));

    /* ---- what the core drives */
    if (!ctx->shape_probe) {
        RadGeneratorDecl d{};
        d.name = "sdxl";
        d.layout = RAD_GEN_LAYOUT_JOINT_STEP;
        d.tmpl_text = "{prompt}";
        d.tmpl_edit = "{prompt}";
        d.drop_prefix = "";
        d.latent_channels = kLatC;
        d.latent_scale = 8;
        /* Two 2x downsamplings: the latent's sides are multiples of 4. */
        d.size_multiple = 32;
        d.pixel_channels = 3;
        d.slot_rows = 1;
        d.max_tokens = kMaxImageTokens;
        d.max_refs = 0;
        d.ref_area = 1;
        d.default_width = 1024;
        d.default_height = 1024;
        d.default_steps = 28;
        d.n_sigmas = 0;
        d.sigmas = nullptr;
        d.shift = 1.0;
        d.default_cfg = 6.0;
        d.cfg_original = 0;
        d.default_negative = "";
        d.max_lanes = kMaxLanes;
        d.image_out = m.b_img;
        d.cond_rows = kCondRows;
        d.rope_hw_scale = 1.0;
        d.pad_multiple = 0;
        d.pad_token = kEos;
        d.train_steps = 1000;
        d.beta_schedule = RAD_GEN_BETAS_SCALED_LINEAR;
        d.beta_start = 0.00085;
        d.beta_end = 0.012;
        d.zero_snr = m.zero_snr ? 1 : 0;
        d.samplers = (1u << RAD_GEN_SAMPLER_EULER) | (1u << RAD_GEN_SAMPLER_EULER_ANCESTRAL) |
                     (1u << RAD_GEN_SAMPLER_DPMPP_2M);
        d.default_sampler = RAD_GEN_SAMPLER_EULER_ANCESTRAL;
        d.default_scheduler = RAD_GEN_SCHED_NORMAL;
        d.chunk_tokens = kWin - 2;
        d.chunk_bos = kBos;
        d.chunk_eos = kEos;
        RAD_ARCH_TRY(rad_declare_generator(b, &d));
    }
    rad_note(b, "Stable Diffusion XL -- CLIP ViT-L and bigG, UNet %zu ops (%s prediction%s), "
                "VAE %zu ops, bf16",
             m.net.ops.size(), m.pred.c_str(), m.zero_snr ? ", zero terminal SNR" : "",
             m.dec.ops.size());
    return RAD_OK;
}

/* ================================================================== step */
static void step_text(RadCtx* c, const RadBatch* batch, const Model& m, const TextEnc& E,
                      int64_t T, const std::vector<int64_t>& eos) {
    const int64_t D = E.D, nw = T / kWin;
    const RadOperand cu = brows(m.b_wcu, nw + 1);
    RAD_ISSUE_N(c, E.op_tok, T, praw(batch->token_ids, RAD_I32, T), RAD_W(E.tok), brows(E.x, T));
    if (E.op_pad)
        for (int64_t w = 0; w < nw; ++w) {
            const int64_t first = w * kWin + eos[(size_t)w] + 1, n = (w + 1) * kWin - first;
            if (n > 0)
                RAD_ISSUE_N(c, E.op_pad, n, brows(m.b_zero, n), RAD_W(E.tok),
                            brow_slice(E.x, first, n, D));
        }
    RAD_ISSUE_N(c, E.op_pos, T, brows(m.b_pos, T), RAD_W(E.pos), brows(E.pe, T));
    RAD_ISSUE_N(c, E.op_pe, T, brows(E.x, T), brows(E.pe, T), brows(E.x, T));
    for (int64_t l = 0; l < E.L; ++l) {
        const TextLayer& Y = E.layers[(size_t)l];
        RAD_ISSUE_N(c, Y.op_ln1, T, brows(E.x, T), RAD_W(Y.ln1w), RAD_W(Y.ln1b), brows(E.h, T));
        RAD_ISSUE_N(c, Y.op_q, T, brows(E.h, T), RAD_W(Y.qw), RAD_W(Y.qb), RAD_NONE, brows(E.q, T));
        RAD_ISSUE_N(c, Y.op_k, T, brows(E.h, T), RAD_W(Y.kw), RAD_W(Y.kb), RAD_NONE, brows(E.k, T));
        RAD_ISSUE_N(c, Y.op_v, T, brows(E.h, T), RAD_W(Y.vw), RAD_W(Y.vb), RAD_NONE, brows(E.v, T));
        RAD_ISSUE_N(c, Y.op_attn, T, brows(E.q, T), brows(E.k, T), brows(E.v, T), cu,
                    brows(E.a, T));
        RAD_ISSUE_N(c, Y.op_o, T, brows(E.a, T), RAD_W(Y.ow), RAD_W(Y.ob), brows(E.x, T),
                    brows(E.x, T));
        RAD_ISSUE_N(c, Y.op_ln2, T, brows(E.x, T), RAD_W(Y.ln2w), RAD_W(Y.ln2b), brows(E.h, T));
        RAD_ISSUE_N(c, Y.op_f1, T, brows(E.h, T), RAD_W(Y.f1w), RAD_W(Y.f1b), RAD_NONE,
                    brows(E.f1, T));
        RAD_ISSUE_N(c, Y.op_act, T, brows(E.f1, T), brows(E.f2, T));
        RAD_ISSUE_N(c, Y.op_f2, T, brows(E.f2, T), RAD_W(Y.f2w), RAD_W(Y.f2b), brows(E.x, T),
                    brows(E.x, T));
        if (l == E.tap)
            RAD_ISSUE_N(c, E.op_tap, T, brows(E.x, T), bcol(m.b_ctx, E.ctx_col, D, T));
    }
    if (!E.op_eos) return;
    /* b_pos holds 0..76 in its first window: row k of it is the index k. */
    RAD_ISSUE_N(c, E.op_eos, 1, brows(E.x, T), brow_slice(m.b_pos, eos[0], 1, 1), brows(E.e1, 1));
    RAD_ISSUE_N(c, E.op_fn, 1, brows(E.e1, 1), RAD_W(E.fnw), RAD_W(E.fnb), brows(E.e2, 1));
    RAD_ISSUE_N(c, E.op_proj, 1, brows(E.e2, 1), RAD_W(E.proj), brows(m.b_pooled, 1));
}

/* Both text encoders over the prompt's windows, the context and pooled vector into every lane
 * of the request at this branch. */
static void step_cond(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t T = batch->n_tok;
    if (T <= 0 || T % kWin || T > kCondRows || !G->tokens_h) {
        rad_step_fail(c, "sdxl: a conditioning pass is not whole 77-token windows");
        return;
    }
    const int64_t nw = T / kWin;
    /* Each window's end-of-text marker: the first after its start marker. */
    std::vector<int64_t> eos((size_t)nw, kWin - 1);
    for (int64_t w = 0; w < nw; ++w)
        for (int64_t k = 1; k < kWin; ++k)
            if (G->tokens_h[w * kWin + k] == kEos) { eos[(size_t)w] = k; break; }
    RAD_ISSUE_N(c, m.op_pos, T, brows(m.b_pos, T));
    RAD_ISSUE_N(c, m.op_wcu, nw + 1, brows(m.b_wcu, nw + 1));
    RAD_ISSUE_N(c, m.op_zero, kWin, brows(m.b_zero, kWin));
    step_text(c, batch, m, m.te1, T, eos);
    step_text(c, batch, m, m.te2, T, eos);
    const int64_t br = G->branch;
    RAD_ISSUE_N(c, m.op_store, T, brows(m.b_ctx, T), praw(G->lane, RAD_I32, G->n_lane),
                brow_slice(m.b_store, br * kCondRows,
                           2 * (int64_t)kMaxLanes * kCondRows - br * kCondRows, kCtx));
    RAD_ISSUE_N(c, m.op_pstore, 1, brows(m.b_pooled, 1), praw(G->lane, RAD_I32, G->n_lane),
                brow_slice(m.b_pstore, br, 2 * (int64_t)kMaxLanes - br, kPooled));
}

/* The model input of every entry, then each entry's embeddings and UNet at its own size, then the
 * sampler's step over every entry. */
static void step_denoise(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t S = batch->n_seq, NI = G->n_img;
    if (!G->entry_h || !G->timestep || !G->size_px) {
        rad_step_fail(c, "sdxl: a denoising pass without its per-entry geometry");
        return;
    }
    const int64_t LR = (int64_t)kMaxLanes * kMaxImageTokens;
    const RadOperand icu = praw(G->img_cu, RAD_I32, S + 1);
    RAD_ISSUE_N(c, m.op_iota, kMaxImageTokens + 1, brows(m.b_iota, kMaxImageTokens + 1));
    RAD_ISSUE_N(c, m.op_zf, 1, brows(m.b_zf, 1));
    RAD_ISSUE_N(c, m.op_kdin, NI, praw(G->sigma, RAD_F32, S), praw(G->pair, RAD_I32, S),
                praw(G->lane, RAD_I32, S), praw(G->flags, RAD_I32, S),
                praw(G->seed, RAD_U32, 2 * S), icu, brows(m.b_state, LR), brows(m.b_xin, NI));
    int64_t row0 = 0;
    for (int64_t e = 0; e < S; ++e) {
        const int32_t* eh = G->entry_h + 4 * e;
        Geo g;
        g.cond = eh[0];
        g.h = eh[1] / 8;
        g.w = eh[2] / 8;
        g.row0 = row0;
        const int64_t lane = G->lane_dh[e], br = eh[3];
        g.store_row = (2 * lane + br) * kCondRows;
        g.pool_row = 2 * lane + br;
        if (g.h % 4 || g.w % 4 || g.px(0) > kMaxImageTokens || g.cond > kCondRows) {
            rad_step_fail(c, "sdxl: an entry's size is not one this model takes");
            return;
        }
        /* ---- the time and added embeddings */
        RAD_ISSUE_N(c, m.op_tproj, 1, praw(G->timestep + e, RAD_F32, 1), brows(m.b_tproj, 1));
        RAD_ISSUE_N(c, m.op_t1, 1, brows(m.b_tproj, 1), RAD_W(m.w_t1), RAD_W(m.b_t1), RAD_NONE,
                    brows(m.b_te1, 1));
        RAD_ISSUE_N(c, m.op_ts, 1, brows(m.b_te1, 1), brows(m.b_te1s, 1));
        RAD_ISSUE_N(c, m.op_t2, 1, brows(m.b_te1s, 1), RAD_W(m.w_t2), RAD_W(m.b_t2), RAD_NONE,
                    brows(m.b_temb, 1));
        char* add = (char*)rad_buf_ptr(c, m.b_add);
        RAD_ISSUE_N(c, m.op_pool, 1, brow_slice(m.b_pstore, g.pool_row, 1, kPooled),
                    RAD_P_T2(add, RAD_BF16, 1, kPooled));
        RAD_ISSUE_N(c, m.op_szo, 2, praw(G->size_px + 2 * e, RAD_F32, 2),
                    RAD_P_T2(add + 2 * kPooled, RAD_BF16, 2, 256));
        RAD_ISSUE_N(c, m.op_szc, 2, praw(rad_buf_ptr(c, m.b_zf), RAD_F32, 2),
                    RAD_P_T2(add + 2 * (kPooled + 512), RAD_BF16, 2, 256));
        RAD_ISSUE_N(c, m.op_szt, 2, praw(G->size_px + 2 * e, RAD_F32, 2),
                    RAD_P_T2(add + 2 * (kPooled + 1024), RAD_BF16, 2, 256));
        RAD_ISSUE_N(c, m.op_l1, 1, brows(m.b_add, 1), RAD_W(m.w_l1), RAD_W(m.b_l1), RAD_NONE,
                    brows(m.b_le1, 1));
        RAD_ISSUE_N(c, m.op_ls, 1, brows(m.b_le1, 1), brows(m.b_le1s, 1));
        RAD_ISSUE_N(c, m.op_l2, 1, brows(m.b_le1s, 1), RAD_W(m.w_l2), RAD_W(m.b_l2),
                    brows(m.b_temb, 1), brows(m.b_emb, 1));
        RAD_ISSUE_N(c, m.op_es, 1, brows(m.b_emb, 1), brows(m.b_embs, 1));
        /* ---- the UNet */
        m.net.issue(c, g);
        row0 += g.px(0);
    }
    if (row0 != NI) {
        rad_step_fail(c, "sdxl: the entries' images do not add up to the pass's image rows");
        return;
    }
    RAD_ISSUE_N(c, m.op_kdstep, NI, brows(m.b_eps, NI), praw(G->sigma, RAD_F32, S),
                praw(G->sigma_next, RAD_F32, S), praw(G->sigma_prev, RAD_F32, S),
                praw(G->cfg, RAD_F32, S), praw(G->pair, RAD_I32, S), praw(G->lane, RAD_I32, S),
                praw(G->flags, RAD_I32, S), praw(G->seed, RAD_U32, 2 * S),
                praw(G->sampler, RAD_I32, S), praw(G->step, RAD_I32, S), icu,
                brows(m.b_state, LR), brows(m.b_hist, LR), brows(m.b_prev, LR));
}

/* One image's latent to RGB. */
static void step_decode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadGenBatch* G = batch->gen_b;
    const int64_t H = G->dec_height / 8, W = G->dec_width / 8, n = H * W;
    const rad_buf src = G->dec_preview ? m.b_prev : m.b_state;
    RAD_ISSUE_N(c, m.op_dscale, n,
                brow_slice(src, (int64_t)G->dec_lane * kMaxImageTokens, n, kLatC),
                brows(m.b_dlf, n));
    RAD_ISSUE_N(c, m.op_dcast, n, brows(m.b_dlf, n), brows(m.b_decin, n));
    m.dec.issue(c, H, W, praw(batch->cu_seqlens, RAD_I32, 2), nullptr);
    const int64_t npx = (int64_t)G->dec_height * G->dec_width;
    RAD_ISSUE_N(c, m.op_rgb, npx, RAD_P_T2(rad_buf_ptr(c, m.dec_out.buf), RAD_BF16, npx, 3),
                RAD_P_T2(rad_buf_ptr(c, m.b_img), RAD_U8, npx, 3));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->gen_b) {
        rad_step_fail(c, "sdxl is an image generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_COND:    step_cond(c, batch, m); return;
    case RAD_GEN_PASS_DENOISE: step_denoise(c, batch, m); return;
    case RAD_GEN_PASS_DECODE:  step_decode(c, batch, m); return;
    default: rad_step_fail(c, "sdxl: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows a pass carries: one image at the largest size with guidance, its conditioning both
 * ways. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->gen_rows = 2 * ((int64_t)kMaxImageTokens + kCondRows);
    out->gen_patches = 0;
    return RAD_OK;
}

}  /* namespace sdxl */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(sdxl)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(sdxl, "sdxl", "", "0.1.0",
                "Stable Diffusion XL and its finetunes (Illustrious, NoobAI, Pony): CLIP ViT-L and "
                "OpenCLIP bigG text encoders, a 2.6B UNet sampled by k-diffusion's Euler, Euler "
                "ancestral or DPM++ 2M, and the SDXL VAE; bf16, from the single-file checkpoint")
#endif
