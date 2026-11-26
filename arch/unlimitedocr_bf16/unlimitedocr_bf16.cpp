/* unlimitedocr_bf16 -- baidu/Unlimited-OCR's decoder: a small Llama-MoE, bf16 weights and bf16
 * activations.
 *
 * Scope: the text trunk plus the DeepEncoder vision tower
 * (arch/common/rad_block_sam.h: SAM ViT-B/16 -> map -> CLIP ViT-L/14 -> the projector
 * -> the 273-row layout of a 1024 view / 110 rows of a 640 tile, declared with
 * rad_declare_encoder(RAD_MM_IMAGE), patch_dim 768 = 3*16*16, merge 16, n_embd 1280). The tower
 * serves both view geometries the processor stages -- the gundam mode's 640 tiles and global
 * view, and the base mode's pages -- as two op sets over the same weights, the 40 grid's absolute
 * position tables bicubic+antialias resamples of the checkpoint's (arch/common/rad_resample.h).
 * The tower's weights are mapped always; its ops and the encoder pass are declared only when
 * SAM's `sam_attn` op resolves, because no existing op
 * can carry the decomposed rel_pos_h/rel_pos_w attention bias (rad_block_sam.h's header states the
 * op and why). Without it this plugin is the text-only trunk and refuses images by name. The
 * decoder's KV is one RAD_KV_RING group with the checkpoint's `sliding_window_size` (128), the
 * reference's own R-SWA: prefill is fully causal, and from the end of the prefill the ring of the
 * last W generated tokens is appended to the prompt (`SlidingWindowLlamaAttention`, which keeps
 * the prompt as reference and overwrites ring slots in place). The checkpoint's windowed
 * no-repeat-ngram ban is the sampler's `sample_ngram_ban` stage.
 *
 * ============================== THE SHAPE (config.json) ==============================
 *
 *   12 layers, hidden 1280, 10 heads x 128 head_dim, 10 kv-heads (MHA, `use_mla: false` -- the
 *   DeepseekV2 class names in the repo are packaging, the graph is Llama)
 *   full rotary over head_dim, theta 10000, plain RMSNorm eps 1e-6 (NOT 1 + w)
 *   vocab 129280, lm_head TIED to embed_tokens (the checkpoint ships both tensors)
 *   layer 0 dense (first_k_dense_replace 1, inter 6848); layers 1..11 MoE:
 *     64 routed experts (moe_intermediate_size 896), top-6, scoring softmax,
 *     norm_topk_prob False, routed_scaling_factor 1.0 -- an ordinary softmax gate,
 *     2 shared experts packed as ONE dense MLP 2x896 = 1792 wide, summed ungated
 *
 * The MoE block is arch/common/rad_block_moe_q.h in its plain-bf16 expert mode (probe() reads
 * format 3 off the first expert's encoding), declared EXTERNAL-INPUT (src.norm null): its
 * post-attention norm is declared by this plugin because the shared expert reads the SAME normed
 * input and the block's gather overwrites that input with the routed delta (`h.x` is the
 * fold target). The shared arm therefore lives here: gate|up -> silu_mul -> down -> add, beside
 * the routed arm, exactly `y = y + shared_experts(x)` in modeling_deepseekv2.py:627.
 *
 * The attention is this plugin's private attn_llama.h: the gated bf16 block minus the output gate
 * and the q/k head norms, which this model does not have.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_mlp.h>
#include <arch/rad_block_moe_q.h>
#include <arch/rad_block_sam.h>

#include "attn_llama.h"

#include <vector>

namespace unlimitedocr_bf16 {

using namespace rad::arch;

struct Layer {
    int      dense = 0;        /* 1 = the first_k_dense layer, plain SwiGLU */
    AttnLlamaBF16 attn;
    MlpBF16  mlp;              /* dense layers */
    MoeQ   moe;              /* routed layers */
    /* The routed layer's external norm and the plugin-owned shared arm: the gain, the two dense
     * linears' weights, and the ops. Per layer -- the ACTIVATION buffers are shared across the
     * layers (spec §8) and the ops are declared per layer, as every block here does. */
    rad_weight w_norm = 0, w_sh_gate_up = 0, w_sh_down = 0;
    rad_op op_moe_norm = 0;    /* x -> h.x, the MoE block's external prepared input */
    rad_op op_sh_gu = 0, op_sh_act = 0, op_sh_down = 0, op_sh_ar = 0, op_sh_add = 0;
};

struct Model {
    Names  nm{""};
    Geom   g{};
    int64_t n_rot  = 0;
    int64_t first_dense = 1;   /* layers [0, first_dense) are dense MLPs */

    rad_kvgroup kv = 0;

    /* the residual stream and the vocabulary edges */
    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    /* attention */
    rad_buf b_qkv = 0, b_qs = 0, b_attn = 0;
    /* dense feed-forward (layer 0) */
    rad_buf b_gate_up = 0, b_ffn = 0;
    /* routing -- one row per token */
    rad_buf b_rlogits = 0, b_eids = 0, b_ew = 0, b_eoff = 0, b_ecnt = 0;
    /* expert planes -- one row per (token, slot), a pass tall */
    rad_buf b_sorted = 0, b_egu = 0, b_edn = 0;
    ActQ  a_eff{};
    /* the shared arm reads the block's prepared input (h) before the gather overwrites it, then
     * the dense linear's planes */
    rad_buf b_sgu = 0, b_sff = 0, b_sout = 0;

    MoeQ::Config moecfg{};
    /* The DeepEncoder vision tower (arch/common/rad_block_sam.h). Its weights are mapped always;
     * its ops and the encoder pass only when SAM's attention op resolves. */
    DeepEncoder enc;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_mm = 0, op_final_norm = 0, op_gather = 0,
           op_logits = 0;

    std::vector<Layer> layers;
};

static Model g_model[MAX_SLOTS];

/* The per-MoE-layer norm and shared arm. The MoE block declares no norm in external-input mode,
 * and the shared expert must read the same normed row the routed experts do; the block's gather
 * folds the routed delta into h.x, so the shared arm is issued while h.x still holds the normed
 * row, BEFORE the block. */
static int declare_moe_norm_and_shared(RadBuilder* b, Model& m, int l) {
    Geom&  g  = m.g;
    Names& nm = m.nm;
    Layer& lay = m.layers[(size_t)l];

    const char* wname = nm.f("blk.%d.ffn_norm.weight", l);
    RAD_ARCH_TRY(map_copy(b, wname, nm.ckpt("model.layers.%lld.post_attention_layernorm.weight",
                                            (long long)l)));
    lay.w_norm = decl_w(b, wname, RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    const int64_t n_ff_sh = m.moecfg.n_ff_shared;
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ffn_gate_up_shexp.weight", l),
                            { nm.ckpt("model.layers.%lld.mlp.shared_experts.gate_proj.weight",
                                       (long long)l),
                              nm.ckpt("model.layers.%lld.mlp.shared_experts.up_proj.weight",
                                       (long long)l) }, 0));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_down_shexp.weight", l),
                          nm.ckpt("model.layers.%lld.mlp.shared_experts.down_proj.weight",
                                   (long long)l)));
    lay.w_sh_gate_up = decl_w(b, nm.f("blk.%d.ffn_gate_up_shexp.weight", l), RAD_BF16,
                              {2 * n_ff_sh, g.n_embd},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l), 0,
                              {n_ff_sh, n_ff_sh});
    lay.w_sh_down = decl_w(b, nm.f("blk.%d.ffn_down_shexp.weight", l), RAD_BF16,
                           {g.n_embd, n_ff_sh},
                           RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(l));

    /* The routed arm's own norm: x -> h.x, the buffer the block takes as its prepared input and
     * the buffer its gather writes the routed delta back into. */
    lay.op_moe_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                        RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(lay.w_norm)),
                         {m.b_x}, {m.b_h});

    lay.op_sh_gu = rw(b, RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 2 * n_ff_sh),
                                     RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(lay.w_sh_gate_up)),
                      {m.b_h}, {m.b_sgu});
    lay.op_sh_act = rw(b, RAD_OP(b, "silu_mul",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n_ff_sh),
                                      RAD_STR("dtype", g.dtype)),
                           RAD_NOWEIGHTS),
                       {m.b_sgu}, {m.b_sff});
    lay.op_sh_down = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                       RAD_INT("K", n_ff_sh), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(lay.w_sh_down)),
                        {m.b_sff}, {m.b_sout});
    /* Column-sharded like the dense MLP's down_proj: every rank holds a partial sum, reduced
     * once per step before the residual add. */
    if (g.world > 1)
        lay.op_sh_ar = decl_all_reduce(b, g, m.b_sout);
    lay.op_sh_add = rw(b, RAD_OP(b, "add",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                      RAD_STR("dtype", g.dtype)),
                           RAD_NOWEIGHTS),
                       {m.b_x, m.b_sout}, {m.b_x});
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;

    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom& g = m.g;
    Names& nm = m.nm;

    /* Plain RMSNorm (DeepseekV2RMSNorm is x_hat * w), not the GemmaRMSNorm `1 + w`. */
    g.wadd = 0.0f;
    /* The linears are the checkpoint's own bf16: no scale planes, no quantised activations. The
     * MoeQ block in particular requires it (its external-input check reads it). */
    g.w_bf16 = true;

    /* ---- geometry RadModelMeta does not name. */
    const int64_t n_expert_all = meta->n_expert > 0
        ? meta->n_expert : rad_meta_geti(meta, "n_routed_experts", 0);
    const int64_t top_k        = meta->n_expert_used > 0
        ? meta->n_expert_used : rad_meta_geti(meta, "num_experts_per_tok", 0);
    const int64_t moe_ff       = rad_meta_geti(meta, "moe_intermediate_size", 0);
    const int64_t n_shared     = rad_meta_geti(meta, "n_shared_experts", 0);
    m.first_dense              = rad_meta_geti(meta, "first_k_dense_replace", 1);
    if (n_expert_all <= 0 || top_k <= 0 || top_k > n_expert_all || moe_ff <= 0 ||
        m.first_dense < 0 || m.first_dense >= g.n_layer) {
        fprintf(stderr, "radiance: unlimitedocr: the container does not describe the routed "
                        "decoder (n_routed_experts %lld, num_experts_per_tok %lld, "
                        "moe_intermediate_size %lld, first_k_dense_replace %lld of %lld layers)\n",
                (long long)n_expert_all, (long long)top_k, (long long)moe_ff,
                (long long)m.first_dense, (long long)g.n_layer);
        return RAD_E_INVAL;
    }
    /* The gate is an ordinary softmax top-k. A container that states a scheme this plugin does not
     * serve is refused by name rather than served as a softmax gate (spec §17). */
    if (const char* sf = rad_meta_gets(meta, "scoring_func", nullptr); sf && *sf &&
        std::strcmp(sf, "softmax") != 0) {
        fprintf(stderr, "radiance: unlimitedocr: scoring_func '%s' is not implemented; the gate "
                        "this plugin declares is the softmax top-k\n", sf);
        return RAD_E_UNSUPPORTED;
    }
    const double rsf = rad_meta_getf(meta, "routed_scaling_factor", 1.0);
    if (rsf != 1.0) {
        fprintf(stderr, "radiance: unlimitedocr: routed_scaling_factor %g is not 1.0 and this "
                        "plugin declares no router output scale\n", rsf);
        return RAD_E_UNSUPPORTED;
    }
    const int64_t topk_group = rad_meta_geti(meta, "topk_group", 1);
    const int64_t n_group    = rad_meta_geti(meta, "n_group", 1);
    if (topk_group != 1 || n_group != 1) {
        fprintf(stderr, "radiance: unlimitedocr: grouped routing (n_group %lld, topk_group %lld) "
                        "is not implemented; the reference's greedy top-k over one group is\n",
                (long long)n_group, (long long)topk_group);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "use_mla", 0)) {
        fprintf(stderr, "radiance: unlimitedocr: use_mla is set; this decoder is the MHA graph "
                        "(q/k/v projections), not the DeepSeek MLA\n");
        return RAD_E_UNSUPPORTED;
    }
    /* The full-head rotary: no rope_parameters object is present, so yarn_from answers
     * "default" and theta stays the container's (the reference default 10000). */
    m.n_rot = g.head_dim;
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));

    /* ---- the routed geometry. n_expert is NOT divided (every rank holds every expert and routes
     * identically); the intermediate width is. */
    MoeQ::Config& mc = m.moecfg;
    mc.n_expert     = n_expert_all;
    mc.n_expert_all = n_expert_all;
    mc.top_k        = top_k;
    int s;
    if ((s = divide_or_fail(moe_ff, g.world, "moe_intermediate_size", &mc.n_ff_exp)) < 0)
        return s;
    /* The shared experts are packed as ONE dense MLP n_shared x moe_ff wide (DeepseekV2Model:
     * `intermediate_size = moe_intermediate_size * n_shared_experts`), summed ungated. The
     * MoeQ shared arm is the SIGMOID-gated family's shape; this model's arm is declared by the
     * plugin (declare_moe_norm_and_shared) and the block declares none. */
    const int64_t sh_all = n_shared * moe_ff;
    mc.n_ff_shared = 0;
    int64_t n_ff_sh = 0;
    if (sh_all > 0 && (s = divide_or_fail(sh_all, g.world, "n_shared_experts width",
                                          &n_ff_sh)) < 0)
        return s;
    mc.norm_topk = rad_meta_geti(meta, "norm_topk_prob", 0) ? 1 : 0;
    const int64_t moe_rows = 2048;
    mc.rows = g.max_tok < moe_rows ? g.max_tok : moe_rows;

    /* ---- KV: one R-SWA ring group.
     *
     * The reference's `SlidingWindowLlamaAttention` keeps the prefill as reference and from the
     * prefill's end keeps the last `sliding_window_size` generated tokens in a ring it overwrites
     * in place, attending over both. RAD_KV_RING is that: paged like a window group, but the floor
     * is per sequence and armed at admission (`visible_lo = prefill length`), so a decode query
     * sees [0, prefill) U [prefill, prefill + W). Prefill itself stays fully causal, which is why
     * the attention op keeps `causal=1` and `window=0`: the visible set is the block table's job,
     * and `causal=0` would make a prefill chunk non-causal.
     *
     * A checkpoint that states no window gets the plain full cache, so the text trunk still
     * serves. Prefix caching and speculation switch off for the group automatically (a runtime
     * window's KV before the prompt end is not reusable and a draft cannot know where the ring
     * starts). */
    const int64_t sw_size = rad_meta_geti(meta, "sliding_window_size", 0);
    const int64_t win = sw_size > 0 ? sw_size : rad_meta_geti(meta, "sliding_window", 0);
    RadKVGroupDecl fd{};
    fd.kind      = win > 0 ? RAD_KV_RING : RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.window    = win > 0 ? win : 0;
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    /* ---- activation buffers. */
    m.b_x    = decl_b(b, nm.f("x"),      g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h    = decl_b(b, nm.f("h"),      g.act_dtype, {g.max_tok, g.n_embd});
    m.b_qkv  = decl_b(b, nm.f("attn_qkv"), g.act_dtype,
                      {g.max_tok, g.q_dim() + 2 * g.kv_dim()});
    /* The dense q lifted out of the fused plane for attn_paged (attn_llama.h's cast). */
    m.b_qs   = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, g.q_dim()});
    m.b_attn = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, g.q_dim()});

    m.b_gate_up = decl_b(b, nm.f("gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn     = decl_b(b, nm.f("ffn"),     g.act_dtype, {g.max_tok, g.n_ff});

    const int64_t R = g.max_tok, K = mc.rows * mc.top_k;
    m.b_rlogits = decl_b(b, nm.f("moe_logits"), g.act_dtype, {R, mc.n_expert_all});
    m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {R, mc.top_k});
    m.b_ew      = decl_b(b, nm.f("moe_w"),      g.act_dtype, {R, mc.top_k});
    m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {K});
    m.b_eoff    = decl_b(b, nm.f("moe_off"),    RAD_I32,     {mc.n_expert_all + 1});
    /* PERSIST: its only reader, rad_route_report, is not a declared op. See qwen4exp_fp8. */
    m.b_ecnt    = decl_b(b, nm.f("moe_cnt"),    RAD_I32,     {mc.n_expert}, RAD_BUF_PERSIST);
    m.b_egu     = decl_b(b, nm.f("moe_gate_up"), g.act_dtype, {K, 2 * mc.n_ff_exp});
    m.a_eff.x   = decl_b(b, nm.f("moe_ffn"),     g.act_dtype, {K, mc.n_ff_exp});
    m.b_edn     = decl_b(b, nm.f("moe_down"),    g.act_dtype, {K, g.n_embd});

    if (n_ff_sh > 0) {
        mc.n_ff_shared = n_ff_sh;        /* sizes nothing in the block (arm null there); kept for
                                            the shared-arm ops below */
        m.b_sgu  = decl_b(b, nm.f("shexp_gate_up"), g.act_dtype, {R, 2 * n_ff_sh});
        m.b_sff  = decl_b(b, nm.f("shexp_ffn"),     g.act_dtype, {R, n_ff_sh});
        m.b_sout = decl_b(b, nm.f("shexp_out"),     g.act_dtype, {R, g.n_embd});
    }

    m.b_hout   = decl_b(b, nm.f("h_out"),  g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- vocabulary edges. The lm_head is TIED: the checkpoint ships lm_head.weight, and a
     * release that ships only the embedding is served by the same map (first source present
     * wins). Both edges are row-sharded with the vocabulary (spec §9). */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.embed_tokens.weight")));

    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    if (g.world > 1)
        m.op_embed_ar = decl_all_reduce_exact(b, g, m.b_x);

    /* ---- the expert format: the checkpoint's plain bf16, read off the first routed layer's
     * first expert. A quantised container would be a sibling plugin's. */
    {
        MoeQ::Src probe_src{};
        probe_src.experts = nm.ckpt("model.layers.%lld.mlp.experts",
                                    (long long)m.first_dense);
        RAD_ARCH_TRY(MoeQ::probe(b, nm, (int)m.first_dense, probe_src, &mc));
        if (!mc.expert_bf16) {
            fprintf(stderr, "radiance: unlimitedocr_bf16 serves plain bf16 experts; layer %lld's "
                            "expert 0 is quantised. Serve that container with its quantised "
                            "plugin.\n", (long long)m.first_dense);
            return RAD_E_UNSUPPORTED;
        }
    }

    /* ---- the DeepEncoder tower: map and declare its weights, then (when `sam_attn` resolves) its
     * ops and the encoder pass. Without that op the tower stays off and images are refused. */
    RAD_ARCH_TRY(m.enc.declare(b, ctx, nm, g.n_embd, (int)g.n_layer + 64));

    /* A picture's rows overwrite its placeholders' embeddings after the (all-reduced) lookup and
     * before the first block. Declared only when the tower is, since otherwise no step can carry
     * media rows. */
    if (m.enc.on)
        m.op_mm = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {m.b_x}, {m.b_x});

    AttnLlamaBF16::Wire aw{};
    aw.x = m.b_x; aw.h = m.b_h; aw.q = m.b_qkv; aw.k = m.b_qkv; aw.v = m.b_qkv;
    aw.qs = m.b_qs;
    aw.attn = m.b_attn;

    MlpBF16::Wire mw{};
    mw.x = m.b_x; mw.h = m.b_h; mw.gate_up = m.b_gate_up; mw.ffn = m.b_ffn;

    MoeQ::Wire fw{};
    fw.x = m.b_x;
    fw.h.x = m.b_h;                    /* external input: the plugin's own rmsnorm writes it */
    fw.logits = m.b_rlogits; fw.ids = m.b_eids; fw.ew = m.b_ew;
    fw.eoff = m.b_eoff; fw.ecnt = m.b_ecnt;
    fw.sorted = m.b_sorted; fw.egu = m.b_egu;
    fw.eff.x = m.a_eff.x;
    fw.edn = m.b_edn;                  /* moe, sgu, sff, sout, sgate stay null: no block arm */

    m.layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.dense = (l < m.first_dense) ? 1 : 0;

        AttnLlamaBF16::Src as{};
        as.norm = nm.ckpt("model.layers.%lld.input_layernorm.weight", (long long)l);
        as.q    = nm.ckpt("model.layers.%lld.self_attn.q_proj.weight", (long long)l);
        as.k    = nm.ckpt("model.layers.%lld.self_attn.k_proj.weight", (long long)l);
        as.v    = nm.ckpt("model.layers.%lld.self_attn.v_proj.weight", (long long)l);
        as.o    = nm.ckpt("model.layers.%lld.self_attn.o_proj.weight", (long long)l);
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv));
        RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv, m.n_rot, aw, as));

        if (lay.dense) {
            MlpBF16::Src ms{};
            ms.norm = nm.ckpt("model.layers.%lld.post_attention_layernorm.weight", (long long)l);
            ms.gate = nm.ckpt("model.layers.%lld.mlp.gate_proj.weight", (long long)l);
            ms.up   = nm.ckpt("model.layers.%lld.mlp.up_proj.weight", (long long)l);
            ms.down = nm.ckpt("model.layers.%lld.mlp.down_proj.weight", (long long)l);
            RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms));
        } else {
            /* The block's norm, the shared arm and its linears, then the block. The block's norm
             * is declared by the plugin (external input) because its gather folds the routed
             * delta into h.x, and the shared arm reads its own copy of the normed row. */
            RAD_ARCH_TRY(declare_moe_norm_and_shared(b, m, (int)l));

            MoeQ::Src fs{};
            fs.cfg     = mc;
            fs.norm    = nullptr;   /* external: declared above */
            fs.router  = nm.ckpt("model.layers.%lld.mlp.gate.weight", (long long)l);
            fs.experts = nm.ckpt("model.layers.%lld.mlp.experts", (long long)l);
            fs.shared  = nullptr;   /* the ungated shared arm lives in the plugin */
            RAD_ARCH_TRY(lay.moe.declare(b, nm, g, (int)l, fw, fs));
        }
    }

    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                        RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.b_h});

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_h}, {m.b_hout});

    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    rad_note(b, "rank %d/%d: %lld layers, layer 0 dense (n_ff %lld), layers %lld..%lld MoE: "
                "%lld experts top-%lld (softmax, norm_topk %d), shared arm %lld wide, "
                "n_head=%lld n_head_kv=%lld head_dim=%lld vocab=[%lld,%lld) tied head",
             g.rank, g.world, (long long)g.n_layer, (long long)g.n_ff,
             (long long)m.first_dense, (long long)(g.n_layer - 1),
             (long long)mc.n_expert_all, (long long)mc.top_k, mc.norm_topk,
             (long long)mc.n_ff_shared,
             (long long)g.n_head, (long long)g.n_head_kv, (long long)g.head_dim,
             (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    if (win > 0)
        rad_note(b, "R-SWA: the KV group is a runtime ring (RAD_KV_RING), window %lld tokens from "
                    "the prefill's end (the reference's SlidingWindowLlamaAttention); prefill is "
                    "fully causal, and prefix cache and speculation are off for it",
                 (long long)win);
    else
        rad_note(b, "R-SWA: the checkpoint states no sliding_window_size, so the KV group is a "
                    "plain full cache");
    /* Only when the container states the processor: gen-convert copies the checkpoint's
     * generation_config into `generation.*`, so a checkpoint without it gets no note. */
    const int64_t ng_n = rad_meta_geti(meta, "generation.no_repeat_ngram_size", 0);
    if (ng_n > 0)
        rad_note(b, "generation_config asks for SlidingWindowNoRepeatNgramProcessor n=%lld over "
                    "the last %lld tokens -- served by the sample_ngram_ban stage, overridable "
                    "per request (ngram_ban_n / ngram_window); temp-0 parity checks run with it "
                    "off",
                 (long long)ng_n,
                 (long long)rad_meta_geti(meta, "generation.no_repeat_ngram_window_size", 0));
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];

    if (batch->enc) {
        m.enc.step(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);
    if (m.op_mm && batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_mm, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, m.g.n_embd),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(m.b_x, batch->n_tok));

    for (const Layer& l : m.layers) {
        l.attn.step(c, batch);
        if (l.dense) {
            l.mlp.step(c, batch);
        } else {
            const int64_t T = batch->n_tok;
            /* The MoE block's norm (x -> h.x), then the shared arm beside the routed arm: both
             * read the same prepared input, so the shared arm is issued while h.x still holds the
             * normed row and BEFORE the block's gather overwrites it with the routed delta. The
             * two residual adds commute: x += shared, then the block's x += routed. */
            RAD_ISSUE(c, l.op_moe_norm, brows(m.b_x, T), RAD_W(l.w_norm), brows(m.b_h, T));
            RAD_ISSUE(c, l.op_sh_gu,  brows(m.b_h, T), RAD_W(l.w_sh_gate_up),
                      brows(m.b_sgu, T));
            RAD_ISSUE(c, l.op_sh_act, brows(m.b_sgu, T), brows(m.b_sff, T));
            RAD_ISSUE(c, l.op_sh_down, brows(m.b_sff, T), RAD_W(l.w_sh_down),
                      brows(m.b_sout, T));
            if (l.op_sh_ar)
                RAD_ISSUE_N(c, l.op_sh_ar, T * m.g.n_embd, brows(m.b_sout, T), RAD_NONE);
            RAD_ISSUE(c, l.op_sh_add, brows(m.b_x, T), brows(m.b_sout, T), brows(m.b_x, T));
            l.moe.step(c, batch);
        }
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.b_h, batch->n_tok));

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

/* Chained launches pay on the plain-bf16 trunk: every kernel of its decode step but the cache's
 * two-dimensional copy and the vocabulary GEMM has a chained form (libr4d's rad_kernel_chain).
 * Unlimited-OCR on one R9700, ms a decode step unchained / chained: 2.79 / 2.49 at one sequence
 * and 4.51 / 4.19 at four. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->chain_pays = RAD_CHAIN_ALWAYS;
    return RAD_OK;
}

}  /* namespace unlimitedocr_bf16 */

#ifndef RAD_ARCH_NO_EXPORTS
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PROBE(unlimitedocr_bf16)
RAD_ARCH_PLUGIN(unlimitedocr_bf16, "unlimitedocr", "", "0.1.0",
                "baidu/Unlimited-OCR decoder: 12-layer Llama-MoE (64 experts top-6 + 2 shared), "
                "bf16 weights and activations, full rotary MHA, with the DeepEncoder "
                "(SAM ViT-B + CLIP ViT-L) vision tower")
#endif
