/* lfm3_bf16 -- LiquidAI's LFM3, bf16 activations over bf16 or int8 (IQ8R) linears, one rank.
 *
 * ============================== THE SHAPE ==============================
 *
 * LFM2's hybrid with a third layer kind, in the order the checkpoint's `layer_types` lists them
 * (read, not derived):
 *
 *   conv               in_proj -> [B | C | x], y = out_proj(C * conv(B * x)), where conv is a
 *                      depthwise causal convolution of width `conv_L_cache` with no activation
 *   full_attention     GQA with per-head RMS norms on q and k, a partial NeoX rotary on the "head"
 *                      ladder (the first `partial_rotary_factor * head_dim` channels rotate, at the
 *                      frequencies of the whole head; the rest do not), and a gate per query head
 *   sliding_attention  the same attention over the last `sliding_window` tokens, its rotary over
 *                      the whole head at its own theta
 *
 * each followed by a SwiGLU feed-forward (w1 the gate, w3 the up, w2 the down). The norms are a
 * sandwich: `operator_norm` before the mixer and `post_operator_norm` on its output, before the
 * residual add; `ffn_norm` and `post_ffn_norm` likewise. Every norm is a plain RMSNorm
 * (`x_hat * w`), and the final one is `embedding_norm`. A post norm, its add and the norm that
 * reads the sum are one `norm_add_norm` where a library serves it (Gemma's sandwich join; see
 * declare). At 1.7B: 24 layers (13 conv, 6 full, 5
 * sliding with a window of 256), hidden 2048, 16 heads over 4 at head_dim 128, ffn 7680,
 * vocabulary 128000 with the lm_head tied to the embedding, conv width 7; theta 1e6 over half the
 * head on the full layers, 1e4 on the sliding ones.
 *
 * ============================== THE GATE ==============================
 *
 * The attention output of each query head is multiplied by sigmoid(gate_proj(h)), one scalar a
 * (token, head), before the out projection. `mul` multiplies planes of equal width, so the gate
 * is computed at the output's width: the name map repeats each of gate_proj's n_head rows head_dim
 * times (RAD_MAP_ROWS), and `gemm_nt` + `sigmoid` + `mul` then give every channel of a head that
 * head's gate, with the reference's rounding (the gate and the product are each a bf16 tensor
 * there). It costs n_embd * (q_dim - n_head) stored weights an attention layer, 2.6% of the
 * container at 1.7B.
 *
 * ============================== 8-BIT LINEARS ==============================
 *
 * Every linear but the gate is a LinearQ (rad_linear.h), as lfm2_bf16's are, so the container's
 * encoding of each weight decides the GEMM: plain bf16 is `gemm_nt` over the bf16 activation,
 * exactly as a bf16 model runs; int8 (`codes=i8 group=128 scale=bf16`) is `gemm_nt_q` at i8a8, and
 * the rotated form (those codes under the 128-wide Hadamard, data/recipes/lfm3-1.7b-iq8r.recipe,
 * or IQ8R's scale per 256 of K) the same at i8a8h, each linear quantising its input from
 * the bf16 plane. The trunk's format is read off layer 0's first linear and every linear must
 * match it. The gate stays bf16 in every container (n_head distinct rows; its product is a hard
 * multiplier on a whole head), as do the convolution, the norms, the joins and attention. The
 * lm_head is read at its stored codes.
 *
 * ============================== WHAT IT DOES NOT SERVE ==============================
 *
 * Tensor parallelism (the model needs one card), block-fp8 or narrower linears, a trunk that mixes
 * formats, a quantised gate, a drafter, media, conv bias, a checkpoint without the sandwich norms,
 * or a rotary other than the default type: each is refused by name.
 */
#include <arch/rad_arch.h>
#include <arch/rad_linear.h>

#include <sstream>
#include <string>
#include <vector>

namespace lfm3_bf16 {

using namespace rad::arch;

enum Kind { kConv = 0, kFull = 1, kSliding = 2 };

/* One plain bf16 linear: its weight and the gemm_nt over it. The attention gate is the one linear
 * declared this way: its weight is gate_proj's rows repeated, which the shared LinearQ that
 * serves every other linear does not map. */
struct Linear {
    rad_weight w = 0;
    rad_op     op = 0;
};

/* What differs between the two attention classes. */
struct AttnClass {
    double      theta = 0.0;
    int64_t     rot = 0;        /* rotary_dim */
    int64_t     window = 0;     /* 0: full */
    rad_kvgroup kv = 0;
};

struct Layer {
    Kind kind = kConv;
    rad_weight w_norm = 0, w_post_norm = 0, w_ffn_norm = 0, w_ffn_post_norm = 0;
    rad_op op_norm = 0, op_post_norm = 0, op_add = 0;
    /* conv */
    LinearQ conv_in, conv_out;
    rad_weight w_conv = 0;
    rad_op op_gate_in = 0, op_conv = 0, op_gate_out = 0;
    /* attention */
    LinearQ q, k, v, out;
    Linear gate;
    rad_weight w_q_norm = 0, w_k_norm = 0;
    rad_op op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op op_store = 0, op_attn = 0, op_sigmoid = 0, op_gate_mul = 0;
    /* the feed-forward: norm, gate|up (w1 | w3), SiLU(gate) * up, down (w2), norm, residual add */
    LinearQ gate_up, down;
    rad_op op_ffn_norm = 0, op_act = 0, op_ffn_post_norm = 0, op_ffn_add = 0;
    /* The sandwich joins, fused: op_join is post_norm + add + ffn_norm, and op_ffn_join is
     * ffn_post_norm + ffn_add + the next layer's norm, which that layer then does not declare. */
    rad_op op_join = 0, op_ffn_join = 0;
};

struct Model {
    Names   nm{""};
    Geom    g{};
    int64_t width = 0;              /* conv_L_cache: the convolution's taps */
    AttnClass full{}, sliding{};
    rad_kvgroup kv_conv = 0;

    rad_buf b_x = 0, b_h = 0, b_arm = 0, b_bcx = 0, b_conv_in = 0, b_conv_out = 0;
    rad_buf b_q = 0, b_k = 0, b_v = 0, b_attn = 0, b_gate = 0, b_gate_up = 0, b_ffn = 0;
    rad_buf b_hout = 0, b_logits = 0;
    /* The four activations a linear reads, as LinearQ takes them: the bf16 plane is the buffer
     * above. */
    ActQ a_h{}, a_attn{}, a_conv{}, a_ffn{};
    /* An int8 trunk (i8a8, or IQ8R at i8a8h), and its scale group along K, which the activations'
     * int8 pairs share (ActQ::g8). */
    bool    q_i8 = false;
    int64_t g8 = RAD_SCALE_GROUP;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_gather = 0, op_final_norm = 0, op_logits = 0;

    std::vector<Layer> layers;

    const AttnClass& cls(const Layer& L) const { return L.kind == kSliding ? sliding : full; }
};

static Model g_model[MAX_SLOTS];

static int64_t meta_int(const RadModelMeta* m, const char* key, int64_t dflt) {
    const char* v = rad_meta_gets(m, key, nullptr);
    return v ? std::strtoll(v, nullptr, 10) : dflt;
}
static double meta_f64(const RadModelMeta* m, const char* key, double dflt) {
    const char* v = rad_meta_gets(m, key, nullptr);
    return v ? std::strtod(v, nullptr) : dflt;
}
static bool meta_bool(const RadModelMeta* m, const char* key, bool dflt) {
    const char* v = rad_meta_gets(m, key, nullptr);
    if (!v) return dflt;
    return std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0;
}

/* `layer_types`, one word per layer, space-separated in the container. */
static int read_schedule(const RadModelMeta* m, int64_t n_layer, std::vector<Layer>* out) {
    const char* s = rad_meta_gets(m, "layer_types", nullptr);
    if (!s || !*s) {
        fprintf(stderr, "radiance: lfm3: the container has no layer_types; this family's schedule "
                        "is read, not derived\n");
        return RAD_E_FORMAT;
    }
    std::istringstream in(s);
    std::string kind;
    while (in >> kind) {
        Layer l;
        if (kind == "conv") l.kind = kConv;
        else if (kind == "full_attention") l.kind = kFull;
        else if (kind == "sliding_attention") l.kind = kSliding;
        else {
            fprintf(stderr, "radiance: lfm3: layer_types[%zu] is '%s'; this plugin knows 'conv', "
                            "'full_attention' and 'sliding_attention'\n", out->size(), kind.c_str());
            return RAD_E_UNSUPPORTED;
        }
        out->push_back(l);
    }
    if ((int64_t)out->size() != n_layer) {
        fprintf(stderr, "radiance: lfm3: layer_types names %zu layers and the model has %lld\n",
                out->size(), (long long)n_layer);
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}

/* One class's rotary from `rope_parameters.<type>`: theta is required (the typed rope_theta is a
 * top-level key this family does not write), the type must be the default, and the partial factor
 * gives an even rotary width inside the head. */
static int read_rotary(const RadModelMeta* m, const char* type, int64_t head_dim, AttnClass* c) {
    const std::string pre = std::string("rope_parameters.") + type + ".";
    const char* rt = rad_meta_gets(m, (pre + "rope_type").c_str(), "default");
    if (std::strcmp(rt, "default") != 0) {
        fprintf(stderr, "radiance: lfm3: rope_parameters.%s.rope_type '%s' is not 'default'\n",
                type, rt);
        return RAD_E_UNSUPPORTED;
    }
    c->theta = meta_f64(m, (pre + "rope_theta").c_str(), 0.0);
    if (!(c->theta > 0.0)) {
        fprintf(stderr, "radiance: lfm3: rope_parameters.%s.rope_theta is required\n", type);
        return RAD_E_INVAL;
    }
    const double prf = meta_f64(m, (pre + "partial_rotary_factor").c_str(), 1.0);
    /* the reference's int(prf * head_dim) // 2 angles */
    c->rot = 2 * ((int64_t)(prf * (double)head_dim) / 2);
    if (!(prf > 0.0) || c->rot < 2 || c->rot > head_dim) {
        fprintf(stderr, "radiance: lfm3: rope_parameters.%s.partial_rotary_factor %g gives a "
                        "rotary width %lld outside 2..head_dim\n", type, prf, (long long)c->rot);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS ||
        rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: lfm3_bf16 serves one rank; run it with --tp 1\n");
        return RAD_E_UNSUPPORTED;
    }
    if (meta->n_expert > 0 || meta_bool(meta, "conv_bias", false) ||
        !meta_bool(meta, "use_sandwich_norm", true)) {
        fprintf(stderr, "radiance: lfm3_bf16 serves the dense SwiGLU model with sandwich norms, "
                        "gated attention and bias-free convolutions; this checkpoint is not that\n");
        return RAD_E_UNSUPPORTED;
    }

    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;

    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom& g = m.g;
    /* The attention rows are keyed on a whole number of queries a KV head. */
    if (g.n_head_kv <= 0 || g.n_head % g.n_head_kv != 0) {
        fprintf(stderr, "radiance: lfm3: %lld attention heads over %lld key/value heads is not a "
                        "whole number of queries a key/value head\n",
                (long long)g.n_head, (long long)g.n_head_kv);
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(read_schedule(meta, g.n_layer, &m.layers));
    m.width = meta_int(meta, "conv_L_cache", 0);
    if (m.width < 2 || m.width > 8) {
        fprintf(stderr, "radiance: lfm3: conv_L_cache %lld is outside the 2..8 the convolution "
                        "kernels serve\n", (long long)m.width);
        return RAD_E_UNSUPPORTED;
    }
    bool has_full = false, has_sliding = false, has_conv = false;
    for (const Layer& L : m.layers) {
        has_full |= L.kind == kFull;
        has_sliding |= L.kind == kSliding;
        has_conv |= L.kind == kConv;
    }
    if (has_full) RAD_ARCH_TRY(read_rotary(meta, "full_attention", g.head_dim, &m.full));
    if (has_sliding) {
        RAD_ARCH_TRY(read_rotary(meta, "sliding_attention", g.head_dim, &m.sliding));
        m.sliding.window = meta_int(meta, "sliding_window", 0);
        if (m.sliding.window < 1) {
            fprintf(stderr, "radiance: lfm3: the sliding_attention layers need a sliding_window, "
                            "and the container has %lld\n", (long long)m.sliding.window);
            return RAD_E_INVAL;
        }
    }

    const char* pre = "model";

    /* ---- the state groups: paged KV for the full layers, the same bounded by the window for the
     * sliding ones, and a rolling window of the convolution's last width-1 inputs (plus the
     * speculative slack the manager adds) for the conv layers. The conv group's channels are
     * n_head_kv * head_dim: one "head" of n_embd. */
    const uint32_t kvdt = rad_kv_cache_dtype(ctx, RAD_BF16);
    g.kv_dtype = rad_dtype_name(kvdt);
    if (has_full) {
        RadKVGroupDecl fd{};
        fd.kind      = RAD_KV_FULL;
        fd.dtype     = kvdt;
        fd.n_head_kv = g.n_head_kv;
        fd.head_dim  = g.head_dim;
        m.full.kv = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
        if (!m.full.kv) return RAD_E_INVAL;
    }
    if (has_sliding) {
        RadKVGroupDecl sd{};
        sd.kind      = RAD_KV_WINDOW;
        sd.dtype     = kvdt;
        sd.window    = m.sliding.window;
        sd.n_head_kv = g.n_head_kv;
        sd.head_dim  = g.head_dim;
        m.sliding.kv = rad_decl_kv_group(b, nm.f("kv_sliding"), &sd);
        if (!m.sliding.kv) return RAD_E_INVAL;
    }
    if (has_conv) {
        RadKVGroupDecl cd{};
        cd.kind       = RAD_KV_CONV;
        cd.dtype      = RAD_BF16;
        cd.n_head_kv  = 1;
        cd.head_dim   = g.n_embd;
        cd.conv_width = m.width;
        m.kv_conv = rad_decl_kv_group(b, nm.f("kv_conv"), &cd);
        if (!m.kv_conv) return RAD_E_INVAL;
    }

    m.b_x        = decl_b(b, nm.f("x"),        RAD_BF16, {g.max_tok, g.n_embd});
    m.b_h        = decl_b(b, nm.f("h"),        RAD_BF16, {g.max_tok, g.n_embd});
    /* a mixer's or a feed-forward's output, before its post norm: a join reads it beside x and
     * writes h, so it is not h */
    m.b_arm      = decl_b(b, nm.f("arm"),      RAD_BF16, {g.max_tok, g.n_embd});
    m.b_bcx      = decl_b(b, nm.f("conv_bcx"), RAD_BF16, {g.max_tok, 3 * g.n_embd});
    m.b_conv_in  = decl_b(b, nm.f("conv_in"),  RAD_BF16, {g.max_tok, g.n_embd});
    m.b_conv_out = decl_b(b, nm.f("conv_out"), RAD_BF16, {g.max_tok, g.n_embd});
    m.b_q        = decl_b(b, nm.f("attn_q"),   RAD_BF16, {g.max_tok, g.q_dim()});
    m.b_k        = decl_b(b, nm.f("attn_k"),   RAD_BF16, {g.max_tok, g.kv_dim()});
    m.b_v        = decl_b(b, nm.f("attn_v"),   RAD_BF16, {g.max_tok, g.kv_dim()});
    m.b_attn     = decl_b(b, nm.f("attn_out"), RAD_BF16, {g.max_tok, g.q_dim()});
    m.b_gate     = decl_b(b, nm.f("attn_gate"), RAD_BF16, {g.max_tok, g.q_dim()});
    m.b_gate_up  = decl_b(b, nm.f("gate_up"),  RAD_BF16, {g.max_tok, 2 * g.n_ff});
    m.b_ffn      = decl_b(b, nm.f("ffn"),      RAD_BF16, {g.max_tok, g.n_ff});
    m.b_hout     = decl_b(b, nm.f("h_out"),    RAD_BF16, {g.max_logit_rows, g.n_embd});
    m.b_logits   = decl_b(b, nm.f("logits"),   RAD_F32,  {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));
    /* ---- the trunk's format, read off layer 0's first linear before any activation is declared
     * (probe() makes its name map and reads its encoding). Plain bf16, or no encoding yet, is a
     * bf16 trunk: every linear is `gemm_nt` over its input's bf16 plane. An int8 weight makes the
     * activations carry int8 codes beside it. */
    {
        Layer& L0 = m.layers[0];
        const bool at = L0.kind != kConv;
        LinearQ& first = at ? L0.q : L0.conv_in;
        const char* base0 = nm.f(at ? "blk.%d.attn_q" : "blk.%d.conv_in", 0);
        RAD_ARCH_TRY(first.probe(b, nm, g, base0,
                                 { nm.ckpt(at ? "%s.layers.%d.self_attn.%s" : "%s.layers.%d.conv.%s",
                                           pre, 0, at ? "q_proj.weight" : "in_proj.weight") }));
        RadEncoding enc{};
        const bool known = weight_enc(b, nm.f("%s.weight", base0), &enc);
        const bool plain = !known || (rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                                      enc.plane[0].dtype == RAD_BF16);
        g.w_bf16 = plain;
        m.q_i8 = !plain && first.probed_i8;
        if (m.q_i8) m.g8 = first.probed_g8;
        if (!plain && !m.q_i8) {
            char en[256];
            rad_enc_format(&enc, en, sizeof en);
            fprintf(stderr, "radiance: lfm3: '%s.weight' is %s; this plugin serves a bf16 trunk or "
                            "an int8 one (codes=i8 group=128, or IQ8R)\n", base0, en);
            return RAD_E_DTYPE;
        }
    }
    if (m.q_i8 && (g.n_embd % m.g8 || g.q_dim() % m.g8 || g.n_ff % m.g8)) {
        fprintf(stderr, "radiance: lfm3: the int8 trunk keeps one scale a %lld of K, which does not "
                        "divide this model's widths (hidden %lld, attention %lld, feed-forward "
                        "%lld)\n",
                (long long)m.g8, (long long)g.n_embd, (long long)g.q_dim(), (long long)g.n_ff);
        return RAD_E_UNSUPPORTED;
    }
    m.a_h.x = m.b_h; m.a_attn.x = m.b_attn; m.a_conv.x = m.b_conv_out; m.a_ffn.x = m.b_ffn;
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd, 0, m.q_i8, false, m.g8));
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", g.q_dim(), 0, m.q_i8, false, m.g8));
    RAD_ARCH_TRY(m.a_conv.declare_qs(b, nm, g, "conv_out", g.n_embd, 0, m.q_i8, false, m.g8));
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff, 0, m.q_i8, false, m.g8));

    /* ---- the vocabulary edges. The lm_head is tied but declared as its own weight: the
     * embedding may live in host memory (--embedding-placement) and the head on the card. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", pre)));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.embedding_norm.weight", pre)));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("%s.embed_tokens.weight", pre)));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* The head at its codes' dtype: bf16, or int8 where a recipe made it so. */
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});

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
    auto mul = [&](std::initializer_list<rad_buf> rd, rad_buf out, int64_t n) {
        return rw(b, RAD_OP(b, "mul",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  rd, {out});
    };
    auto layer_w = [&](const char* name, int l, std::initializer_list<int64_t> shape,
                       uint32_t dtype) {
        return decl_w(b, nm.f(name, l), dtype, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                      grp_layer(l));
    };
    /* A norm's gain: mapped, and declared f32. */
    auto norm_w = [&](rad_weight* w, const char* name, int l, const char* src, int64_t n) -> int {
        RAD_ARCH_TRY(map_copy(b, nm.f(name, l), src));
        *w = layer_w(name, l, {n}, RAD_F32);
        return *w ? RAD_OK : RAD_E_INVAL;
    };
    /* The gate: one plain bf16 linear over an already mapped weight, in every container. */
    auto plain_linear = [&](Linear& lin, int l, const char* name, int64_t N, int64_t K, rad_buf in,
                            rad_buf out) -> int {
        RadEncoding enc{};
        if (weight_enc(b, nm.f(name, l), &enc) &&
            !(rad_enc_is(&enc, "plain") && enc.n_planes == 1 && enc.plane[0].dtype == RAD_BF16)) {
            char en[256];
            rad_enc_format(&enc, en, sizeof en);
            fprintf(stderr, "radiance: lfm3: '%s' is %s; the attention gate stays bf16 (leave "
                            "gate_proj out of the recipe)\n", nm.f(name, l), en);
            return RAD_E_DTYPE;
        }
        lin.w = decl_w(b, nm.f(name, l), RAD_BF16, {N, K}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                       grp_layer(l));
        if (!lin.w) return RAD_E_INVAL;
        lin.op = rw(b, RAD_OP(b, "gemm_nt",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(lin.w)),
                    {in}, {out});
        return lin.op ? RAD_OK : RAD_E_INVAL;
    };
    /* One linear: `base` is the declared name without `.weight`, `src` its checkpoint tensor (or
     * the tensors stacked along its rows, `parts` their row counts), `a` the activation it reads. */
    auto linear = [&](LinearQ& lin, int l, const char* base_fmt, int64_t N, int64_t K,
                      std::initializer_list<const char*> src, const ActQ& a, rad_buf y,
                      std::initializer_list<int64_t> parts) -> int {
        const char* base = nm.f(base_fmt, l);
        RAD_ARCH_TRY(lin.declare(b, nm, g, base, N, K, RAD_SHARD_NONE, grp_layer(l), src, a, y, 0,
                                 parts));
        /* Every linear has the trunk's format. LinearQ refuses a quantised weight in a bf16
         * trunk; this refuses a bf16 or narrower one in an int8 trunk. */
        if (g.w_bf16 ? !lin.bf16 : !(lin.i8 && !lin.iqr)) {
            fprintf(stderr, "radiance: lfm3: '%s.weight' is not in the trunk's format (%s); "
                            "quantise every linear of the trunk alike\n",
                    base, g.w_bf16 ? "bf16" : "int8");
            return RAD_E_DTYPE;
        }
        return RAD_OK;
    };
    /* q names its query heads; k is rotated as a q-only row of the kv heads (its own buffer).
     * The partial rotary is on the "head" ladder; over the whole head the ladders coincide and
     * the parameter is left out. */
    auto rope = [&](const AttnClass& c, int64_t heads, rad_buf buf) {
        const RadParam ps[] = { RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", c.theta), RAD_F64("scale", 1.0),
                                RAD_STR("mode", "neox"), RAD_INT("rotary_dim", c.rot),
                                RAD_STR("ladder", "head") };
        const int n = (int)(sizeof ps / sizeof ps[0]) - (c.rot < g.head_dim ? 0 : 1);
        return rw(b, rad_decl_op(b, "rope_rows", ps, n, nullptr, 0), {buf}, {buf});
    };


    /* ---- the sandwich joins. `norm_add_norm` is a post norm, the residual add and the norm that
     * reads the sum in one launch: post_norm + add + ffn_norm inside a layer, and ffn_post_norm +
     * add + the next layer's norm between two. It rounds where the chain rounds. The device
     * libraries serve it and libref and libavx do not, and declaring an op no loaded library
     * declares is an error, so it is asked for first; without it, or where it does not resolve,
     * the rmsnorm / add / rmsnorm chain runs. The last layer's feed-forward keeps the chain: the
     * final norm is on the gathered rows. */
    const bool has_join = rad_op_declared(b, "norm_add_norm") != 0;
    auto join = [&](rad_weight w_a, rad_weight w_y) {
        return rw(b, RAD_OP(b, "norm_add_norm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                 RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_a, w_y)),
                  {m.b_arm, m.b_x}, {m.b_x, m.b_h});
    };
    int n_joins = 0;

    /* Every layer's input norm gain, ahead of the layers: a feed-forward join names the next
     * layer's, and an op can only name weights already declared. */
    for (int l = 0; l < (int)m.layers.size(); ++l)
        RAD_ARCH_TRY(norm_w(&m.layers[(size_t)l].w_norm, "blk.%d.attn_norm.weight", l,
                            nm.ckpt("%s.layers.%d.%s", pre, l, "operator_norm.weight"), g.n_embd));

    for (int l = 0; l < (int)m.layers.size(); ++l) {
        Layer& L = m.layers[(size_t)l];
        const char* lay = "%s.layers.%d.%s";
        RAD_ARCH_TRY(norm_w(&L.w_post_norm, "blk.%d.attn_post_norm.weight", l,
                            nm.ckpt(lay, pre, l, "post_operator_norm.weight"), g.n_embd));
        RAD_ARCH_TRY(norm_w(&L.w_ffn_norm, "blk.%d.ffn_norm.weight", l,
                            nm.ckpt(lay, pre, l, "ffn_norm.weight"), g.n_embd));
        RAD_ARCH_TRY(norm_w(&L.w_ffn_post_norm, "blk.%d.ffn_post_norm.weight", l,
                            nm.ckpt(lay, pre, l, "post_ffn_norm.weight"), g.n_embd));
        /* written by the layer before, where its feed-forward join is fused */
        if (!(l > 0 && m.layers[(size_t)l - 1].op_ffn_join))
            L.op_norm = rmsnorm(g.max_tok, g.n_embd, L.w_norm, m.b_x, m.b_h);

        if (L.kind != kConv) {
            const AttnClass& c = m.cls(L);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, l, c.kv));
            const char* a = "%s.layers.%d.self_attn.%s";
            /* The gate at the output's width: row h of gate_proj, head_dim times, for each head
             * in turn (see the head of the file). Successive ROWS maps extend the weight. */
            {
                const std::string dst = nm.f("blk.%d.attn_gate.weight", l);
                const std::string src = nm.ckpt(a, pre, l, "gate_proj.weight");
                for (int64_t h = 0; h < g.n_head; ++h)
                    for (int64_t d = 0; d < g.head_dim; d += 8) {
                        RadNameMap mp = map_base(dst.c_str(), RAD_MAP_ROWS);
                        for (int64_t i = d; i < g.head_dim && i < d + 8; ++i) {
                            mp.src[mp.n_src] = src.c_str();
                            mp.src_index[mp.n_src] = (int)h;
                            mp.src_rows[mp.n_src] = 1;
                            ++mp.n_src;
                        }
                        RAD_ARCH_TRY(rad_decl_name_map(b, &mp));
                    }
            }
            RAD_ARCH_TRY(linear(L.q, l, "blk.%d.attn_q", g.q_dim(), g.n_embd,
                                { nm.ckpt(a, pre, l, "q_proj.weight") }, m.a_h, m.b_q, {}));
            RAD_ARCH_TRY(linear(L.k, l, "blk.%d.attn_k", g.kv_dim(), g.n_embd,
                                { nm.ckpt(a, pre, l, "k_proj.weight") }, m.a_h, m.b_k, {}));
            RAD_ARCH_TRY(linear(L.v, l, "blk.%d.attn_v", g.kv_dim(), g.n_embd,
                                { nm.ckpt(a, pre, l, "v_proj.weight") }, m.a_h, m.b_v, {}));
            RAD_ARCH_TRY(plain_linear(L.gate, l, "blk.%d.attn_gate.weight", g.q_dim(), g.n_embd,
                                      m.b_h, m.b_gate));
            RAD_ARCH_TRY(norm_w(&L.w_q_norm, "blk.%d.attn_q_norm.weight", l,
                                nm.ckpt(a, pre, l, "q_norm.weight"), g.head_dim));
            RAD_ARCH_TRY(norm_w(&L.w_k_norm, "blk.%d.attn_k_norm.weight", l,
                                nm.ckpt(a, pre, l, "k_norm.weight"), g.head_dim));

            /* The head norms run over rows of head_dim: a [T, heads * head_dim] buffer is
             * T * heads rows. */
            L.op_q_norm = rmsnorm(g.max_tok * g.n_head, g.head_dim, L.w_q_norm, m.b_q, m.b_q,
                                 "rmsnorm_rows");
            L.op_k_norm = rmsnorm(g.max_tok * g.n_head_kv, g.head_dim, L.w_k_norm, m.b_k, m.b_k,
                                 "rmsnorm_rows");
            L.op_rope_q = rope(c, g.n_head, m.b_q);
            L.op_rope_k = rope(c, g.n_head_kv, m.b_k);
            /* attn_paged before kv_store: the cache's block size is read from the resolved
             * attention kernel. */
            L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                               RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                          RAD_INT("head_dim", g.head_dim), RAD_INT("gqa", g.gqa()),
                                          RAD_INT("causal", 1), RAD_INT("window", c.window),
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
                                           RAD_INT("block_size", rad_kv_block_size(b, c.kv)),
                                           RAD_STR("kv_dtype", g.kv_dtype)),
                                RAD_NOWEIGHTS),
                            {m.b_k, m.b_v}, {});
            L.op_sigmoid = rw(b, RAD_OP(b, "sigmoid",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.q_dim()),
                                             RAD_STR("dtype", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_gate}, {m.b_gate});
            L.op_gate_mul = mul({m.b_gate, m.b_attn}, m.b_attn, g.q_dim());
            RAD_ARCH_TRY(linear(L.out, l, "blk.%d.attn_output", g.n_embd, g.q_dim(),
                                { nm.ckpt(a, pre, l, "o_proj.weight") }, m.a_attn, m.b_arm, {}));
        } else {
            RAD_ARCH_TRY(rad_bind_layer_kv(b, l, m.kv_conv));
            const char* c = "%s.layers.%d.conv.%s";
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.conv.weight", l), nm.ckpt(c, pre, l, "conv.weight")));
            /* in_proj's row is [B | C | x]: conv(B * x), then C * that. */
            RAD_ARCH_TRY(linear(L.conv_in, l, "blk.%d.conv_in", 3 * g.n_embd, g.n_embd,
                                { nm.ckpt(c, pre, l, "in_proj.weight") }, m.a_h, m.b_bcx, {}));
            /* The checkpoint ships [n_embd, 1, width]; the depthwise group axis is squeezed at
             * convert, as LFM2's is. */
            L.w_conv = layer_w("blk.%d.conv.weight", l, {g.n_embd, m.width}, RAD_BF16);
            L.op_gate_in = mul({m.b_bcx}, m.b_conv_in, g.n_embd);
            L.op_conv = rw(b, RAD_OP(b, "causal_conv1d",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                          RAD_INT("width", m.width), RAD_STR("dtype", g.dtype)),
                               RAD_WEIGHTS(L.w_conv)),
                           {m.b_conv_in}, {m.b_conv_out});
            L.op_gate_out = mul({m.b_bcx, m.b_conv_out}, m.b_conv_out, g.n_embd);
            RAD_ARCH_TRY(linear(L.conv_out, l, "blk.%d.conv_out", g.n_embd, g.n_embd,
                                { nm.ckpt(c, pre, l, "out_proj.weight") }, m.a_conv, m.b_arm, {}));
        }
        /* the sandwich: the mixer's output is normed before it joins the stream, and the
         * feed-forward reads the norm of the sum */
        if (has_join) L.op_join = join(L.w_post_norm, L.w_ffn_norm);
        if (!L.op_join) {
            L.op_post_norm = rmsnorm(g.max_tok, g.n_embd, L.w_post_norm, m.b_arm, m.b_arm);
            L.op_add = add(m.b_x, m.b_arm);
            L.op_ffn_norm = rmsnorm(g.max_tok, g.n_embd, L.w_ffn_norm, m.b_x, m.b_h);
        }

        /* ---- the feed-forward */
        const char* ffn = "%s.layers.%d.feed_forward.%s";
        RAD_ARCH_TRY(linear(L.gate_up, l, "blk.%d.ffn_gate_up", 2 * g.n_ff, g.n_embd,
                            { nm.ckpt(ffn, pre, l, "w1.weight"), nm.ckpt(ffn, pre, l, "w3.weight") },
                            m.a_h, m.b_gate_up, { g.n_ff, g.n_ff }));
        L.op_act = rw(b, RAD_OP(b, "silu_mul",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {m.b_gate_up}, {m.b_ffn});
        RAD_ARCH_TRY(linear(L.down, l, "blk.%d.ffn_down", g.n_embd, g.n_ff,
                            { nm.ckpt(ffn, pre, l, "w2.weight") }, m.a_ffn, m.b_arm, {}));
        if (has_join && l + 1 < (int)m.layers.size())
            L.op_ffn_join = join(L.w_ffn_post_norm, m.layers[(size_t)l + 1].w_norm);
        if (!L.op_ffn_join) {
            L.op_ffn_post_norm = rmsnorm(g.max_tok, g.n_embd, L.w_ffn_post_norm, m.b_arm, m.b_arm);
            L.op_ffn_add = add(m.b_x, m.b_arm);
        }
        n_joins += (L.op_join ? 1 : 0) + (L.op_ffn_join ? 1 : 0);
    }

    /* The final norm runs on the gathered rows only: it is per token, and those are the rows the
     * lm_head reads. */
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_x}, {m.b_hout});
    m.op_final_norm = rmsnorm(g.max_logit_rows, g.n_embd, m.w_out_norm, m.b_hout, m.b_hout);
    /* A tied head is the bf16 embedding table: a scored step leaves the final-normed rows in
     * h_out and the core takes the candidates' logits from the table on the host. */
    if (meta_bool(meta, "tie_word_embeddings", false))
        RAD_ARCH_TRY(rad_declare_scores(b, m.b_hout, m.w_tok));
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    int64_t n_full = 0, n_sliding = 0;
    for (const Layer& L : m.layers) {
        n_full += L.kind == kFull ? 1 : 0;
        n_sliding += L.kind == kSliding ? 1 : 0;
    }
    rad_note(b, "lfm3: %lld layers, %lld full attention + %lld sliding (window %lld) + %lld short "
                "convolution of width %lld; %lld heads over %lld at head_dim %lld, ffn %lld, "
                "vocab %lld",
             (long long)g.n_layer, (long long)n_full, (long long)n_sliding,
             (long long)m.sliding.window, (long long)(g.n_layer - n_full - n_sliding),
             (long long)m.width, (long long)g.n_head, (long long)g.n_head_kv,
             (long long)g.head_dim, (long long)g.n_ff, (long long)g.n_vocab);
    rad_note(b, "sandwich joins: %d of %lld fused (norm_add_norm)%s", n_joins,
             (long long)(2 * g.n_layer - 1),
             n_joins == 0 ? " -- not served, the rmsnorm / add / rmsnorm chain" : "");
    return RAD_OK;
}

static void lin_step(RadCtx* c, const Linear& lin, rad_buf in, rad_buf out, int64_t T) {
    RAD_ISSUE(c, lin.op, brows(in, T), RAD_W(lin.w), brows(out, T));
}

/* The convolution is issued once per half of the step, as LFM2's is: the decode rows [0, D) with
 * `num_accepted`, leaving the window in the layout a partly rejected verify reads back from, and
 * the prefill chunks [D, S) without it, each leaving its tail at offset zero. `has_init` is
 * `ctx_lens` (a sequence with earlier tokens has a history). The prefill half's per-sequence
 * arrays start at `+ D`; the token operands stay at their base, since the kernel reads rows at the
 * absolute offsets in `cu`. */
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
    const Model& m = g_model[rad_slot(c)];
    const Geom& g = m.g;
    if (batch->draft_pass || batch->spec_parent || batch->enc || batch->n_mm_rows > 0) {
        rad_step_fail(c, "lfm3_bf16: this step asks for an encoder, a draft pass or media rows, "
                         "and the program declares none of them");
        return;
    }
    const int64_t T = batch->n_tok, n = g.n_embd;
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));

    for (int l = 0; l < (int)m.layers.size(); ++l) {
        const Layer& L = m.layers[(size_t)l];
        if (!(l > 0 && m.layers[(size_t)l - 1].op_ffn_join))
            RAD_ISSUE(c, L.op_norm, brows(m.b_x, T), RAD_W(L.w_norm), brows(m.b_h, T));
        if (L.kind != kConv) {
            const rad_kvgroup grp = m.cls(L).kv;
            const RadKVGroupBatch* kv = kv_batch(batch, grp);
            L.q.step(c, m.a_h, m.b_q, T);
            L.k.step(c, m.a_h, m.b_k, T);
            L.v.step(c, m.a_h, m.b_v, T);
            lin_step(c, L.gate, m.b_h, m.b_gate, T);
            RAD_ISSUE_N(c, L.op_q_norm, T * g.n_head, brows(m.b_q, T), RAD_W(L.w_q_norm),
                        brows(m.b_q, T));
            RAD_ISSUE_N(c, L.op_k_norm, T * g.n_head_kv, brows(m.b_k, T), RAD_W(L.w_k_norm),
                        brows(m.b_k, T));
            RAD_ISSUE(c, L.op_rope_q, brows(m.b_q, T), rope_pos1(batch, T));
            RAD_ISSUE(c, L.op_rope_k, brows(m.b_k, T), rope_pos1(batch, T));
            RAD_ISSUE(c, L.op_store, brows(m.b_k, T), brows(m.b_v, T),
                      praw(kv ? kv->slot_mapping : nullptr, RAD_I32, T), kv_cache(grp, l));
            RAD_ISSUE_N(c, L.op_attn, batch->max_q_len,
                        brows(m.b_q, T), kv_cache(grp, l),
                        praw2(kv ? kv->block_table : nullptr, RAD_I32, batch->n_seq,
                              kv ? kv->block_table_pitch : 0),
                        praw(kv ? kv->seqused : nullptr, RAD_I32, batch->n_seq),
                        RAD_NONE, RAD_NONE,
                        praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                        brows(m.b_attn, T));
            RAD_ISSUE(c, L.op_sigmoid, brows(m.b_gate, T), brows(m.b_gate, T));
            RAD_ISSUE(c, L.op_gate_mul, brows(m.b_gate, T), brows(m.b_attn, T), brows(m.b_attn, T));
            L.out.step(c, m.a_attn, m.b_arm, T);
        } else {
            L.conv_in.step(c, m.a_h, m.b_bcx, T);
            RAD_ISSUE(c, L.op_gate_in, bcol(m.b_bcx, 0, n, T), bcol(m.b_bcx, 2 * n, n, T),
                      brows(m.b_conv_in, T));
            conv_step(m, L, c, batch, l);
            RAD_ISSUE(c, L.op_gate_out, bcol(m.b_bcx, n, n, T), brows(m.b_conv_out, T),
                      brows(m.b_conv_out, T));
            L.conv_out.step(c, m.a_conv, m.b_arm, T);
        }
        if (L.op_join) {
            RAD_ISSUE(c, L.op_join, brows(m.b_arm, T), RAD_W(L.w_post_norm), brows(m.b_x, T),
                      RAD_W(L.w_ffn_norm), brows(m.b_h, T));
        } else {
            RAD_ISSUE(c, L.op_post_norm, brows(m.b_arm, T), RAD_W(L.w_post_norm),
                      brows(m.b_arm, T));
            RAD_ISSUE(c, L.op_add, brows(m.b_x, T), brows(m.b_arm, T), brows(m.b_x, T));
            RAD_ISSUE(c, L.op_ffn_norm, brows(m.b_x, T), RAD_W(L.w_ffn_norm), brows(m.b_h, T));
        }
        L.gate_up.step(c, m.a_h, m.b_gate_up, T);
        RAD_ISSUE(c, L.op_act, brows(m.b_gate_up, T), brows(m.b_ffn, T));
        L.down.step(c, m.a_ffn, m.b_arm, T);
        if (L.op_ffn_join) {
            /* y is the next layer's input norm */
            RAD_ISSUE(c, L.op_ffn_join, brows(m.b_arm, T), RAD_W(L.w_ffn_post_norm),
                      brows(m.b_x, T), RAD_W(m.layers[(size_t)l + 1].w_norm), brows(m.b_h, T));
        } else {
            RAD_ISSUE(c, L.op_ffn_post_norm, brows(m.b_arm, T), RAD_W(L.w_ffn_post_norm),
                      brows(m.b_arm, T));
            RAD_ISSUE(c, L.op_ffn_add, brows(m.b_x, T), brows(m.b_arm, T), brows(m.b_x, T));
        }
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

}  /* namespace lfm3_bf16 */

#ifndef RAD_ARCH_NO_EXPORTS
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(lfm3_bf16, "lfm3", "", "0.1.0",
                "LiquidAI LFM3, bf16 or IQ8R: short gated convolutions, full and sliding-window GQA "
                "attention with a partial rotary and a gate per head, sandwich norms, in the "
                "checkpoint's own schedule, one rank")
#endif
