/* lfm2_bf16 -- LiquidAI's LFM2 and LFM2-VL, bf16 weights and activations, one rank.
 *
 * ============================== THE SHAPE ==============================
 *
 * A hybrid of two layer kinds, in the order the checkpoint's `layer_types` lists them (read, not
 * derived: d1-3B's schedule is irregular, attention at 2, 5, 9, 13, ...):
 *
 *   conv            in_proj -> [B | C | x], y = out_proj(C * conv(B * x)), where conv is a
 *                   depthwise causal convolution of width `conv_L_cache` with no activation
 *   full_attention  GQA with per-head RMS norms on q and k, NeoX rotary over the whole head
 *
 * each followed by a SwiGLU feed-forward (w1 the gate, w3 the up, w2 the down). Every norm is a
 * plain RMSNorm (`x_hat * w`, not Gemma's `1 + w`), and the final one is `embedding_norm`. At
 * d1-3B: 30 layers (10 attention), hidden 2048, 32 heads over 8 at head_dim 64, ffn 10752,
 * vocabulary 128000 with the lm_head tied to the embedding, conv width 3.
 *
 * LFM2-VL nests the same text model under `model.language_model` and adds a SigLIP2 tower and a
 * projector (siglip2.h), declared when the container carries their weights and the deployment
 * asked for media; the projector's rows replace the image placeholders' embeddings, which rotate
 * at their token positions like any other token. A text-only LFM2 checkpoint keeps the text model
 * under `model`. Which one this is comes from the config (a VL checkpoint has a `text_config`),
 * not the model's name.
 *
 * ============================== 8-BIT LINEARS ==============================
 *
 * Every linear is a LinearFP8 (rad_fp8.h), so the container's encoding of each weight decides the
 * GEMM: plain bf16 is `gemm_nt` over the bf16 activation, exactly as a bf16 model runs; a recipe's
 * block E4M3 (`codes=fp8_e4m3 block=128x128 scale=bf16`) is `gemm_nt_q` at fp8a8 over the
 * activation quantised per 128 columns (one quant_act_fp8 per activation, shared by the linears
 * that read it); int8 (`codes=i8 group=128 scale=bf16`) is `gemm_nt_q` at i8a8, each linear
 * quantising its input. The trunk's format is read off layer 0's down projection and every linear
 * must match it. The convolution, the norms and attention stay bf16; the lm_head is read at its
 * stored codes (bf16, or E4M3 / int8 where a recipe made it so; logits_gemm_fp8 / _i8). At d1-3B
 * the linears are 5.8 of the 6.3 GB a token reads, so 8-bit codes nearly halve decode and short
 * prefill, both bound by that read.
 *
 * A block-fp8 trunk also takes the fused producers (rad_fp8.h), which matter once the GEMMs are
 * this cheap: each norm quantises in its own pass and does the preceding block's residual add
 * (rmsnorm_quant_fp8 with a fold), and the feed-forward is MlpFP8, whose gate|up GEMM applies
 * SiLU(gate) * up and quantises in its epilogue at decode and short-prompt rows. Six launches a
 * layer become two; the last layer's feed-forward keeps its add for the final norm. The fused
 * kernels are byte-identical to the chains they replace.
 *
 * ============================== WHAT IT DOES NOT SERVE ==============================
 *
 * Tensor parallelism (a 3B model needs one card). A drafter, MoE (LFM2-8B-A1B), conv bias, or a
 * checkpoint without the rotary: each is refused by name.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_vit.h>
#include <arch/rad_block_mlp_fp8.h>
#include <arch/rad_fp8.h>

#include "siglip2.h"

#include <sstream>
#include <string>
#include <vector>

namespace lfm2_bf16 {

using namespace rad::arch;

struct Layer {
    bool    attention = false;
    rad_weight w_norm = 0;
    /* in is q (attention) or the convolution's in_proj; out is the mixer's out-projection */
    LinearFP8 in, out;
    /* conv */
    rad_weight w_conv = 0;
    rad_op op_gate_in = 0, op_conv = 0, op_gate_out = 0;
    /* `gate_quant_fp8` at act none: C * conv(B * x) and its E4M3 codes in one pass. Absent
     * (bf16, int8, or no library serving it), op_gate_out and q_mix run the pair. */
    rad_op op_gate_q = 0;
    /* attention: k, v and the two head norms */
    LinearFP8 k, v;
    rad_weight w_q_norm = 0, w_k_norm = 0;
    rad_op op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op op_store = 0, op_attn = 0;
    /* both */
    rad_op op_norm = 0, op_add = 0;
    /* the feed-forward: norm, gate|up (w1 | w3), SiLU(gate) * up, down (w2), residual add */
    rad_weight w_ffn_norm = 0;
    LinearFP8 gate_up, down;
    rad_op op_ffn_norm = 0, op_act = 0, op_ffn_add = 0;
    /* fp8a8 only: the activation quantisers, one per activation the linears read */
    QuantFP8 q_h, q_mix, q_ffn_h, q_ffn;
    /* fp8a8 fused: the mixer's norm (quantising, and adding the previous feed-forward's delta
     * from layer 1 on) and the whole feed-forward */
    NormQuantFP8 nq_mix;
    MlpFP8       mlp8;
};

struct Model {
    Names   nm{""};
    Geom    g{};
    int64_t width = 0;              /* conv_L_cache: the convolution's taps */
    rad_kvgroup kv_attn = 0, kv_conv = 0;

    rad_buf b_x = 0, b_h = 0, b_bcx = 0, b_conv_in = 0, b_conv_out = 0;
    rad_buf b_q = 0, b_k = 0, b_v = 0, b_attn = 0, b_gate_up = 0, b_ffn = 0;
    rad_buf b_hout = 0, b_logits = 0;
    /* The linears' inputs: the bf16 plane plus, for an 8-bit trunk, its codes. a_mix is the
     * mixer's output (attention or convolution), the out-projection's input. */
    ActFP8 a_h{}, a_attn{}, a_conv{}, a_ffn{};
    /* The trunk's format, read off layer 0's first linear. `fused`: the fp8 trunk's fused
     * producers (see the head of the file). */
    bool q_fp8 = false, q_i8 = false, fused = false;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_gather = 0, op_final_norm = 0, op_logits = 0;

    std::vector<Layer> layers;

    /* LFM2-VL's images: the tower and the scatter of its rows over the embeddings. */
    bool      media = false;
    Siglip2   vit{};
    MediaRows mrows{};
};

static Model g_model[MAX_RANKS];

/* A text_config key first, then the top level: a VL checkpoint nests the text model's settings
 * and a text checkpoint does not. */
static const char* text_key(const RadModelMeta* m, const char* key) {
    const std::string nested = std::string("text_config.") + key;
    const char* v = rad_meta_gets(m, nested.c_str(), nullptr);
    return v ? v : rad_meta_gets(m, key, nullptr);
}
static int64_t text_int(const RadModelMeta* m, const char* key, int64_t dflt) {
    const char* v = text_key(m, key);
    return v ? std::strtoll(v, nullptr, 10) : dflt;
}
static bool text_bool(const RadModelMeta* m, const char* key, bool dflt) {
    const char* v = text_key(m, key);
    if (!v) return dflt;
    return std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0;
}

/* `layer_types`, one word per layer, space-separated in the container. */
static int read_schedule(const RadModelMeta* m, int64_t n_layer, std::vector<Layer>* out) {
    const char* s = text_key(m, "layer_types");
    if (!s || !*s) {
        fprintf(stderr, "radiance: lfm2: the container has no layer_types; this family's schedule "
                        "is irregular and is not derived\n");
        return RAD_E_FORMAT;
    }
    std::istringstream in(s);
    std::string kind;
    while (in >> kind) {
        if (kind != "conv" && kind != "full_attention") {
            fprintf(stderr, "radiance: lfm2: layer_types[%zu] is '%s'; this plugin knows 'conv' "
                            "and 'full_attention'\n", out->size(), kind.c_str());
            return RAD_E_UNSUPPORTED;
        }
        Layer l;
        l.attention = kind == "full_attention";
        out->push_back(l);
    }
    if ((int64_t)out->size() != n_layer) {
        fprintf(stderr, "radiance: lfm2: layer_types names %zu layers and the model has %lld\n",
                out->size(), (long long)n_layer);
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: lfm2_bf16 serves one rank; run it with --tp 1\n");
        return RAD_E_UNSUPPORTED;
    }
    if (meta->n_expert > 0 || text_bool(meta, "conv_bias", false) ||
        !text_bool(meta, "block_use_swiglu", true) || !text_bool(meta, "use_pos_enc", true)) {
        fprintf(stderr, "radiance: lfm2_bf16 serves the dense SwiGLU model with rotary "
                        "attention and bias-free convolutions; this checkpoint is not that\n");
        return RAD_E_UNSUPPORTED;
    }

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;

    /* The FFN width as stated, unless the config asks for LFM2's sizing rule (1.2B and smaller
     * releases): two thirds of it, times the multiplier, rounded up to `multiple_of`. */
    RadModelMeta dims = *meta;
    if (text_bool(meta, "block_auto_adjust_ff_dim", false)) {
        int64_t ff = 2 * dims.n_ff / 3;
        if (const char* mult = text_key(meta, "block_ffn_dim_multiplier")) {
            const int64_t of = text_int(meta, "block_multiple_of", 256);
            const double f = std::strtod(mult, nullptr);
            if (!(f > 0.0) || of <= 0) return RAD_E_FORMAT;
            ff = (int64_t)(f * (double)ff);
            ff = of * ((ff + of - 1) / of);
        }
        dims.n_ff = ff;
    }
    RAD_ARCH_TRY(geom_from(m.g, &dims, ctx, "bf16", RAD_BF16));
    Geom& g = m.g;
    RAD_ARCH_TRY(yarn_from(g, meta, "", g.head_dim));
    RAD_ARCH_TRY(read_schedule(meta, g.n_layer, &m.layers));
    m.width = text_int(meta, "conv_L_cache", 0);
    if (m.width < 2 || m.width > 8) {
        fprintf(stderr, "radiance: lfm2: conv_L_cache %lld is outside the 2..8 the convolution "
                        "kernels serve\n", (long long)m.width);
        return RAD_E_UNSUPPORTED;
    }

    const char* pre = rad_meta_gets(meta, "text_config.num_hidden_layers", nullptr)
                    ? "model.language_model" : "model";

    /* ---- the two state groups: paged KV for the attention layers, and a rolling window of the
     * convolution's last width-1 inputs (plus the speculative slack the manager adds) for the
     * conv layers. The conv group's channels are n_head_kv * head_dim: one "head" of n_embd. */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_attn = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    RadKVGroupDecl cd{};
    cd.kind       = RAD_KV_CONV;
    cd.dtype      = RAD_BF16;
    cd.n_head_kv  = 1;
    cd.head_dim   = g.n_embd;
    cd.conv_width = m.width;
    m.kv_conv = rad_decl_kv_group(b, nm.f("kv_conv"), &cd);
    if (!m.kv_attn || !m.kv_conv) return RAD_E_INVAL;

    m.b_x        = decl_b(b, nm.f("x"),        RAD_BF16, {g.max_tok, g.n_embd});
    m.b_h        = decl_b(b, nm.f("h"),        RAD_BF16, {g.max_tok, g.n_embd});
    m.b_bcx      = decl_b(b, nm.f("conv_bcx"), RAD_BF16, {g.max_tok, 3 * g.n_embd});
    m.b_conv_in  = decl_b(b, nm.f("conv_in"),  RAD_BF16, {g.max_tok, g.n_embd});
    m.b_conv_out = decl_b(b, nm.f("conv_out"), RAD_BF16, {g.max_tok, g.n_embd});
    m.b_q        = decl_b(b, nm.f("attn_q"),   RAD_BF16, {g.max_tok, g.q_dim()});
    m.b_k        = decl_b(b, nm.f("attn_k"),   RAD_BF16, {g.max_tok, g.kv_dim()});
    m.b_v        = decl_b(b, nm.f("attn_v"),   RAD_BF16, {g.max_tok, g.kv_dim()});
    m.b_attn     = decl_b(b, nm.f("attn_out"), RAD_BF16, {g.max_tok, g.q_dim()});
    m.b_gate_up  = decl_b(b, nm.f("gate_up"),  RAD_BF16, {g.max_tok, 2 * g.n_ff});
    m.b_ffn      = decl_b(b, nm.f("ffn"),      RAD_BF16, {g.max_tok, g.n_ff});
    m.b_hout     = decl_b(b, nm.f("h_out"),    RAD_BF16, {g.max_logit_rows, g.n_embd});
    m.b_logits   = decl_b(b, nm.f("logits"),   RAD_F32,  {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges. The lm_head is tied but declared as its own weight: the
     * embedding may live in host memory (--embedding-placement) and the head on the card, and a
     * kernel may want a head layout the lookup does not read. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", pre)));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.embedding_norm.weight", pre)));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("%s.embed_tokens.weight", pre)));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* The head at its codes' dtype: bf16, or E4M3 / int8 where a recipe quantised it. */
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* The encoder's rows overwrite the lookup; with one rank there is no reduction after it. */
    m.media = Siglip2::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, g.n_embd, m.b_x));

    /* ---- the trunk's format, from layer 0's first linear: q, or the convolution's in_proj
     * (probe() makes its name map and reads its encoding). Plain bf16, or no encoding yet, is a
     * bf16 trunk; an int8 weight makes the activations carry int8 codes, any other 8-bit weight
     * E4M3 ones. */
    const char* ffn_fmt = "%s.layers.%d.feed_forward.%s";
    {
        Layer& L0 = m.layers[0];
        const char* base0 = nm.f(L0.attention ? "blk.%d.attn_q" : "blk.%d.conv_in", 0);
        RAD_ARCH_TRY(L0.in.probe(b, nm, g, base0,
                                 { nm.ckpt(L0.attention ? "%s.layers.%d.self_attn.%s"
                                                        : "%s.layers.%d.conv.%s",
                                           pre, 0, L0.attention ? "q_proj.weight"
                                                                : "in_proj.weight") }));
        RadEncoding enc{};
        const bool known = weight_enc(b, nm.f("%s.weight", base0), &enc);
        const bool plain = !known || (rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                                      enc.plane[0].dtype == RAD_BF16);
        g.w_bf16 = plain;
        m.q_i8   = !plain && L0.in.probed_i8;
        m.q_fp8  = !plain && !m.q_i8;
        m.fused  = m.q_fp8;
    }
    m.a_h.x = m.b_h; m.a_attn.x = m.b_attn; m.a_conv.x = m.b_conv_out; m.a_ffn.x = m.b_ffn;
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd, 0, m.q_i8));
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", g.q_dim(), 0, m.q_i8));
    RAD_ARCH_TRY(m.a_conv.declare_qs(b, nm, g, "conv_out", g.n_embd, 0, m.q_i8));
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff, 0, m.q_i8));
    /* An E4M3 activation is quantised once by the model and read by every linear on it; an int8
     * linear quantises its own input (LinearFP8::op_q), and a bf16 one reads the bf16 plane. */
    auto quant = [&](QuantFP8& q, const ActFP8& a, int64_t n) {
        return m.q_fp8 ? q.declare(b, g, a, n) : RAD_OK;
    };

    /* `op`: "rmsnorm", or "rmsnorm_rows" for the head norms, whose short-row form a prefill's
     * thousands of head_dim-wide rows take. */
    auto rmsnorm = [&](int64_t rows, int64_t n, rad_weight w, rad_buf in, rad_buf out,
                       const char* op = "rmsnorm") {
        return rw(b, RAD_OP(b, op,
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", g.eps),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w)),
                  {in}, {out});
    };
    auto add = [&](rad_buf x, rad_buf h) {
        return rw(b, RAD_OP(b, "add",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {x, h}, {x});
    };
    auto layer_w = [&](const char* name, int l, std::initializer_list<int64_t> shape,
                       uint32_t dtype) {
        return decl_w(b, nm.f(name, l), dtype, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                      grp_layer(l));
    };
    /* One linear: `base` is the declared name without `.weight`. Every linear must have the
     * trunk's format; LinearFP8 refuses a bf16 weight in an 8-bit trunk with no bf16 plane, and
     * this refuses the other mixes. */
    auto linear = [&](LinearFP8& lin, int l, const char* base_fmt, int64_t N, int64_t K,
                      std::initializer_list<const char*> src, const ActFP8& a, rad_buf y,
                      std::initializer_list<int64_t> parts) -> int {
        const char* base = nm.f(base_fmt, l);
        RAD_ARCH_TRY(lin.declare(b, nm, g, base, N, K, RAD_SHARD_NONE, grp_layer(l), src, a, y, 0,
                                 parts));
        const bool ok = g.w_bf16 ? lin.bf16 : m.q_i8 ? lin.i8 : (!lin.bf16 && !lin.i8);
        if (!ok) {
            fprintf(stderr, "radiance: lfm2: '%s.weight' is not in the trunk's format (%s); quantise "
                            "every linear of the trunk alike\n",
                    base, g.w_bf16 ? "bf16" : m.q_i8 ? "int8" : "block fp8");
            return RAD_E_DTYPE;
        }
        return RAD_OK;
    };

    for (int l = 0; l < (int)m.layers.size(); ++l) {
        Layer& L = m.layers[(size_t)l];
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),
                              nm.ckpt("%s.layers.%d.operator_norm.weight", pre, l)));
        L.w_norm = layer_w("blk.%d.attn_norm.weight", l, {g.n_embd}, RAD_F32);
        if (m.fused) {
            /* From layer 1 on the previous feed-forward's delta is still in h: the norm adds it
             * into the stream, then normalises and quantises. */
            const bool fold = l > 0;
            RAD_ARCH_TRY(L.nq_mix.declare(b, g, L.w_norm, fold ? m.b_h : m.b_x, m.a_h, g.n_embd,
                                          fold ? m.b_x : 0));
        } else {
            L.op_norm = rmsnorm(g.max_tok, g.n_embd, L.w_norm, m.b_x, m.b_h);
            RAD_ARCH_TRY(quant(L.q_h, m.a_h, g.n_embd));
        }

        if (L.attention) {
            RAD_ARCH_TRY(rad_bind_layer_kv(b, l, m.kv_attn));
            const char* a = "%s.layers.%d.self_attn.%s";
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l),
                                  nm.ckpt(a, pre, l, "q_layernorm.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l),
                                  nm.ckpt(a, pre, l, "k_layernorm.weight")));
            RAD_ARCH_TRY(linear(L.in, l, "blk.%d.attn_q", g.q_dim(), g.n_embd,
                                { nm.ckpt(a, pre, l, "q_proj.weight") }, m.a_h, m.b_q, {}));
            /* k and v are the trunk's narrowest linears: past 40 rows the narrow fp8 row's split
             * leaves each slice too few waves, so they ask for the short-prompt split rule. */
            L.k.short_split = L.v.short_split = true;
            RAD_ARCH_TRY(linear(L.k, l, "blk.%d.attn_k", g.kv_dim(), g.n_embd,
                                { nm.ckpt(a, pre, l, "k_proj.weight") }, m.a_h, m.b_k, {}));
            RAD_ARCH_TRY(linear(L.v, l, "blk.%d.attn_v", g.kv_dim(), g.n_embd,
                                { nm.ckpt(a, pre, l, "v_proj.weight") }, m.a_h, m.b_v, {}));
            L.w_q_norm = layer_w("blk.%d.attn_q_norm.weight", l, {g.head_dim}, RAD_F32);
            L.w_k_norm = layer_w("blk.%d.attn_k_norm.weight", l, {g.head_dim}, RAD_F32);

            /* The head norms run over rows of head_dim: a [T, heads * head_dim] buffer is
             * T * heads rows. */
            L.op_q_norm = rmsnorm(g.max_tok * g.n_head, g.head_dim, L.w_q_norm, m.b_q, m.b_q,
                                 "rmsnorm_rows");
            L.op_k_norm = rmsnorm(g.max_tok * g.n_head_kv, g.head_dim, L.w_k_norm, m.b_k, m.b_k,
                                 "rmsnorm_rows");
            L.op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, g.head_dim,
                                          "neox", "rope_rows"), {m.b_q}, {m.b_q});
            L.op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, g.head_dim,
                                          "neox", "rope_rows"), {m.b_k}, {m.b_k});
            /* attn_paged before kv_store: the cache's block size is read from the resolved
             * attention kernel. */
            L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                               RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                          RAD_INT("head_dim", g.head_dim), RAD_INT("gqa", g.gqa()),
                                          RAD_INT("causal", 1), RAD_INT("window", 0),
                                          RAD_INT("n_head", g.n_head),
                                          RAD_INT("block_size", RAD_KV_BLOCK),
                                          RAD_INT("max_seqs", g.max_seqs),
                                          RAD_INT("max_ctx", g.max_ctx),
                                          RAD_STR("q_dtype", g.dtype),
                                          RAD_STR("kv_dtype", g.kv_dtype)),
                               RAD_NOWEIGHTS),
                           {m.b_q}, {m.b_attn});
            L.op_store = rw(b, RAD_OP(b, "kv_store",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                           RAD_INT("head_dim", g.head_dim),
                                           RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_INT("block_size", rad_kv_block_size(b, m.kv_attn)),
                                           RAD_STR("kv_dtype", g.kv_dtype)),
                                RAD_NOWEIGHTS),
                            {m.b_k, m.b_v}, {});
            RAD_ARCH_TRY(quant(L.q_mix, m.a_attn, g.q_dim()));
            RAD_ARCH_TRY(linear(L.out, l, "blk.%d.attn_output", g.n_embd, g.q_dim(),
                                { nm.ckpt(a, pre, l, "out_proj.weight") }, m.a_attn, m.b_h, {}));
        } else {
            RAD_ARCH_TRY(rad_bind_layer_kv(b, l, m.kv_conv));
            const char* c = "%s.layers.%d.conv.%s";
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.conv.weight", l), nm.ckpt(c, pre, l, "conv.weight")));
            /* in_proj's row is [B | C | x]: conv(B * x), then C * that. */
            RAD_ARCH_TRY(linear(L.in, l, "blk.%d.conv_in", 3 * g.n_embd, g.n_embd,
                                { nm.ckpt(c, pre, l, "in_proj.weight") }, m.a_h, m.b_bcx, {}));
            /* The checkpoint ships [n_embd, 1, width]; the depthwise group axis is squeezed at
             * convert, as the gated delta net's conv1d is. */
            L.w_conv = layer_w("blk.%d.conv.weight", l, {g.n_embd, m.width}, RAD_BF16);

            L.op_gate_in = rw(b, RAD_OP(b, "mul",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                             RAD_STR("dtype", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_bcx}, {m.b_conv_in});
            L.op_conv = rw(b, RAD_OP(b, "causal_conv1d",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                          RAD_INT("width", m.width), RAD_STR("dtype", g.dtype)),
                               RAD_WEIGHTS(L.w_conv)),
                           {m.b_conv_in}, {m.b_conv_out});
            /* C * conv(B * x), then the codes the out projection reads, in one pass. The kernel
             * rounds the product to bf16 before taking the codes, so it is byte-identical to the
             * mul + quant_act_fp8 pair; the bf16 product has no reader in a block-fp8 trunk. */
            if (m.q_fp8 && rad_op_declared(b, "gate_quant_fp8"))
                L.op_gate_q = rw(b, RAD_OP(b, "gate_quant_fp8",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                               RAD_INT("n", g.n_embd),
                                               RAD_INT("group", RAD_FP8_BLOCK),
                                               RAD_STR("dtype", g.dtype),
                                               RAD_STR("act", "none")),
                                    RAD_NOWEIGHTS),
                                {m.b_bcx, m.b_conv_out}, {m.a_conv.cq(), m.a_conv.cs()});
            if (!L.op_gate_q) {
                L.op_gate_out = rw(b, RAD_OP(b, "mul",
                                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                                  RAD_INT("n", g.n_embd),
                                                  RAD_STR("dtype", g.dtype)),
                                       RAD_NOWEIGHTS),
                                   {m.b_bcx, m.b_conv_out}, {m.b_conv_out});
                RAD_ARCH_TRY(quant(L.q_mix, m.a_conv, g.n_embd));
            }
            RAD_ARCH_TRY(linear(L.out, l, "blk.%d.conv_out", g.n_embd, g.n_embd,
                                { nm.ckpt(c, pre, l, "out_proj.weight") }, m.a_conv, m.b_h, {}));
        }
        if (m.fused) {
            /* The feed-forward's norm adds the mixer's delta; its own add is left to the next
             * layer's norm, except after the last layer, whose output the final norm reads. */
            MlpFP8::Wire mw{};
            mw.x = m.b_x; mw.h = m.a_h; mw.gate_up = m.b_gate_up; mw.ffn = m.a_ffn;
            MlpFP8::Src ms{};
            ms.norm = nm.ckpt("%s.layers.%d.ffn_norm.weight", pre, l);
            ms.gate = nm.ckpt(ffn_fmt, pre, l, "w1.weight");
            ms.up   = nm.ckpt(ffn_fmt, pre, l, "w3.weight");
            ms.down = nm.ckpt(ffn_fmt, pre, l, "w2.weight");
            const bool last = l + 1 == (int)m.layers.size();
            /* The whole prompt in the fold: gemm_nt_q_gated_wide serves every M, so a prefill
             * chunk does not fall back to the pair past 64 rows. */
            RAD_ARCH_TRY(L.mlp8.declare(b, nm, g, l, mw, ms, /*fold=*/true, /*add_out=*/last,
                                        0, false, 0, /*wide_fold=*/true));
            if (L.mlp8.gate_up.bf16 || L.mlp8.gate_up.i8 || L.mlp8.down.bf16 || L.mlp8.down.i8) {
                fprintf(stderr, "radiance: lfm2: layer %d's feed-forward is not block fp8 like the "
                                "rest of the trunk; quantise every linear alike\n", l);
                return RAD_E_DTYPE;
            }
            continue;
        }
        L.op_add = add(m.b_x, m.b_h);

        /* ---- the feed-forward */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l),
                              nm.ckpt("%s.layers.%d.ffn_norm.weight", pre, l)));
        L.w_ffn_norm = layer_w("blk.%d.ffn_norm.weight", l, {g.n_embd}, RAD_F32);
        L.op_ffn_norm = rw(b, RAD_OP(b, "rmsnorm",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                           RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                           RAD_STR("dtype", g.dtype)),
                                RAD_WEIGHTS(L.w_ffn_norm)),
                            {m.b_x}, {m.b_h});
        RAD_ARCH_TRY(quant(L.q_ffn_h, m.a_h, g.n_embd));
        RAD_ARCH_TRY(linear(L.gate_up, l, "blk.%d.ffn_gate_up", 2 * g.n_ff, g.n_embd,
                            { nm.ckpt(ffn_fmt, pre, l, "w1.weight"),
                              nm.ckpt(ffn_fmt, pre, l, "w3.weight") },
                            m.a_h, m.b_gate_up, { g.n_ff, g.n_ff }));
        L.op_act = rw(b, RAD_OP(b, "silu_mul",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {m.b_gate_up}, {m.b_ffn});
        RAD_ARCH_TRY(quant(L.q_ffn, m.a_ffn, g.n_ff));
        RAD_ARCH_TRY(linear(L.down, l, "blk.%d.ffn_down", g.n_embd, g.n_ff,
                            { nm.ckpt(ffn_fmt, pre, l, "w2.weight") }, m.a_ffn, m.b_h, {}));
        L.op_ffn_add = add(m.b_x, m.b_h);
    }

    /* The final norm runs on the gathered rows only: it is per token, and those are the rows the
     * lm_head reads. */
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_x}, {m.b_hout});
    m.op_final_norm = rmsnorm(g.max_logit_rows, g.n_embd, m.w_out_norm, m.b_hout, m.b_hout);
    /* A tied head is the bf16 embedding table, whatever codes the head itself is read at: a
     * scored step (/v1/systemone) leaves the final-normed rows in h_out and the core takes the
     * candidates' logits from the table on the host (rad_declare_scores), with no head read. */
    if (text_bool(meta, "tie_word_embeddings", false))
        RAD_ARCH_TRY(rad_declare_scores(b, m.b_hout, m.w_tok));
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    /* The vision tower, last, so its activations share the arena with every buffer above. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));

    int64_t n_attn = 0;
    for (const Layer& L : m.layers) n_attn += L.attention ? 1 : 0;
    rad_note(b, "lfm2: %lld layers, %lld attention + %lld short convolution of width %lld; "
                "%lld heads over %lld at head_dim %lld, ffn %lld, vocab %lld",
             (long long)g.n_layer, (long long)n_attn, (long long)(g.n_layer - n_attn),
             (long long)m.width, (long long)g.n_head, (long long)g.n_head_kv,
             (long long)g.head_dim, (long long)g.n_ff, (long long)g.n_vocab);
    return RAD_OK;
}

/* The convolution is issued once per half of the step, like the PLE's and the gated delta net's:
 * the decode rows [0, D) with `num_accepted`, leaving the window in the layout a partly rejected
 * verify reads back from, and the prefill chunks [D, S) without it, each leaving its tail at
 * offset zero. `has_init` is `ctx_lens` (a sequence with earlier tokens has a history). The
 * prefill half's per-sequence arrays start at `+ D`; the token operands stay at their base, since
 * the kernel reads rows at the absolute offsets in `cu`. */
static void conv_step(const Model& m, const Layer& L, RadCtx* c, const RadBatch* batch, int l) {
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const RadKVGroupBatch* cv = kv_batch(batch, m.kv_conv);
    const int32_t* idx = cv ? cv->state_index : nullptr;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t P = S - D;
    if (D > 0)
        RAD_ISSUE_N(c, L.op_conv, DT,
                    brows(m.b_conv_in, DT), RAD_W(L.w_conv), kv_cache(m.kv_conv, l),
                    praw(batch->cu_seqlens, RAD_I32, D + 1), praw(idx, RAD_I32, D),
                    praw(batch->ctx_lens, RAD_I32, D), praw(batch->num_accepted, RAD_I32, D),
                    brows(m.b_conv_out, DT));
    if (P > 0)
        RAD_ISSUE_N(c, L.op_conv, T,
                    brows(m.b_conv_in, T), RAD_W(L.w_conv), kv_cache(m.kv_conv, l),
                    praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                    praw(idx ? idx + D : nullptr, RAD_I32, P),
                    praw(batch->ctx_lens + D, RAD_I32, P), RAD_NONE,
                    brows(m.b_conv_out, T));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_rank(c)];
    const Geom& g = m.g;
    if (batch->draft_pass || batch->spec_parent ||
        (!m.media && (batch->enc || batch->n_mm_rows > 0))) {
        rad_step_fail(c, "lfm2_bf16: this step asks for an encoder, a draft pass or media rows, "
                         "and the program declares none of them");
        return;
    }
    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }
    const int64_t T = batch->n_tok, n = g.n_embd;
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    m.mrows.step(c, batch, m.b_x, T);

    for (int l = 0; l < (int)m.layers.size(); ++l) {
        const Layer& L = m.layers[(size_t)l];
        if (m.fused) {
            L.nq_mix.step(c, T);
        } else {
            RAD_ISSUE(c, L.op_norm, brows(m.b_x, T), RAD_W(L.w_norm), brows(m.b_h, T));
            L.q_h.step(c, T);
        }
        if (L.attention) {
            const RadKVGroupBatch* kv = kv_batch(batch, m.kv_attn);
            L.in.step(c, m.a_h, m.b_q, T);
            L.k.step(c, m.a_h, m.b_k, T);
            L.v.step(c, m.a_h, m.b_v, T);
            RAD_ISSUE_N(c, L.op_q_norm, T * g.n_head, brows(m.b_q, T), RAD_W(L.w_q_norm),
                        brows(m.b_q, T));
            RAD_ISSUE_N(c, L.op_k_norm, T * g.n_head_kv, brows(m.b_k, T), RAD_W(L.w_k_norm),
                        brows(m.b_k, T));
            RAD_ISSUE(c, L.op_rope_q, brows(m.b_q, T), rope_pos1(batch, T));
            RAD_ISSUE(c, L.op_rope_k, brows(m.b_k, T), rope_pos1(batch, T));
            RAD_ISSUE(c, L.op_store, brows(m.b_k, T), brows(m.b_v, T),
                      praw(kv ? kv->slot_mapping : nullptr, RAD_I32, T), kv_cache(m.kv_attn, l));
            RAD_ISSUE_N(c, L.op_attn, batch->max_q_len,
                        brows(m.b_q, T), kv_cache(m.kv_attn, l),
                        praw2(kv ? kv->block_table : nullptr, RAD_I32, batch->n_seq,
                              kv ? kv->block_table_pitch : 0),
                        praw(kv ? kv->seqused : nullptr, RAD_I32, batch->n_seq),
                        RAD_NONE, RAD_NONE,
                        praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                        brows(m.b_attn, T));
            L.q_mix.step(c, T);
            L.out.step(c, m.a_attn, m.b_h, T);
        } else {
            L.in.step(c, m.a_h, m.b_bcx, T);
            RAD_ISSUE(c, L.op_gate_in, bcol(m.b_bcx, 0, n, T), bcol(m.b_bcx, 2 * n, n, T),
                      brows(m.b_conv_in, T));
            conv_step(m, L, c, batch, l);
            if (L.op_gate_q) {
                RAD_ISSUE(c, L.op_gate_q, bcol(m.b_bcx, n, n, T), brows(m.b_conv_out, T),
                          brows(m.a_conv.cq(), T), brows(m.a_conv.cs(), T), RAD_NONE);
            } else {
                RAD_ISSUE(c, L.op_gate_out, bcol(m.b_bcx, n, n, T), brows(m.b_conv_out, T),
                          brows(m.b_conv_out, T));
                L.q_mix.step(c, T);
            }
            L.out.step(c, m.a_conv, m.b_h, T);
        }
        if (m.fused) {
            L.mlp8.step(c, batch);
            continue;
        }
        RAD_ISSUE(c, L.op_add, brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));

        RAD_ISSUE(c, L.op_ffn_norm, brows(m.b_x, T), RAD_W(L.w_ffn_norm), brows(m.b_h, T));
        L.q_ffn_h.step(c, T);
        L.gate_up.step(c, m.a_h, m.b_gate_up, T);
        RAD_ISSUE(c, L.op_act, brows(m.b_gate_up, T), brows(m.b_ffn, T));
        L.q_ffn.step(c, T);
        L.down.step(c, m.a_ffn, m.b_h, T);
        RAD_ISSUE(c, L.op_ffn_add, brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));
    }

    /* The lm_head runs only on the rows the sampler asked for; a prefill chunk that finishes no
     * sequence skips it. */
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.b_x, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_final_norm, batch->n_out, brows(m.b_hout, batch->n_out),
                    RAD_W(m.w_out_norm), brows(m.b_hout, batch->n_out));
        if (!batch->score_only)
            RAD_ISSUE_N(c, m.op_logits, batch->n_out, brows(m.b_hout, batch->n_out),
                        RAD_W(m.w_lm_head), brows(m.b_logits, batch->n_out));
    }
}

}  /* namespace lfm2_bf16 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(lfm2_bf16, "lfm2", "", "0.2.0",
                "LiquidAI LFM2 / LFM2-VL, bf16, or with block-fp8 or int8 linears from a recipe: "
                "short gated convolutions and GQA attention in the checkpoint's own schedule, one "
                "rank; LFM2-VL's SigLIP2 tower for images")
#endif
