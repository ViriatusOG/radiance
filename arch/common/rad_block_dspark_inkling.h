/* rad_block_dspark_inkling.h -- Inkling 2's DSpark drafter (`mtp.safetensors`, config.json's
 * `mtp_config`): a three-layer dense transformer that denoises an anchor and seven mask positions in
 * one pass over K and V it derives from four trunk taps, then a recurrent (`rnn_hidden`) Markov head
 * that walks the seven positions greedily over the target's own output table.
 *
 * ============================== WHY ITS OWN BLOCK ==============================
 *
 * rad_block_dspark_bf16.h serves the Spark drafts. This one shares the drive (RAD_DRAFT_BLOCK, a
 * context pass a trunk step and a query pass a proposal) and differs in every op that touches the
 * taps or the head:
 *
 *   - no `fc`: each layer mixes the four taps with its own softmax weights (`fuse_weight` row i),
 *     norms the mix with its own `dflare_norm`, and projects it with its own `k_proj_ctx` /
 *     `v_proj_ctx`, K normed per head by `k_norm_ctx`. The trunk's rows never enter the layers.
 *   - the block is [anchor, MASK x 7] and the anchor predicts nothing (`sample_from_anchor` false):
 *     mask row j sits at the anchor's position + j and proposes the token there. The anchor row is
 *     the target's embedding through the target's `embed_norm`; a mask row is `mask_embed`.
 *   - two local layers (0 and 2) see the 1024 positions before the block and the block; layer 1
 *     sees everything. Two KV groups, so the local layers' cache is recycled below the window.
 *   - the residual stream is bf16 and every norm after the first adds the sublayer's output to it
 *     first (flashinfer's fused_add_rmsnorm in the reference; `add` then `rmsnorm` here).
 *   - the head is recurrent: a rank-256 gated state over [state, w1[previous]] and the position's
 *     [hidden, f_rnn], its tanh latent folded through `hidden_out` into the target's lm_head
 *     (`dspark_rnn_markov_greedy`). f_rnn is the taps of the last committed token, mixed with
 *     `fuse_weight`'s last row and normed with the head's own `dflare_norm`.
 *
 * ============================== THE FORWARD PASS ==============================
 *
 * The reference is vLLM's models/inkling2 (nvidia/dspark.py, ops/dspark.py), which follows the
 * release's own implementation:
 *
 *   context, per committed token and draft layer i:
 *       c   = rmsnorm(sum_t alpha[i, t] * tap_t) * dflare_norm_i         alpha = softmax(fuse_weight)
 *       k   = rope(rmsnorm_head(c @ k_proj_ctx_i) * k_norm_ctx_i);  v = c @ v_proj_ctx_i
 *   query, the block [anchor, MASK x 7] at positions p .. p + 7:
 *       x   = [embed_norm(embed(anchor)), mask_embed x 7]                 bf16
 *       per layer: h = norm(x (+= previous sublayer)); q, k, v = h @ W; q, k per-head normed;
 *                  rope; attention over the layer's context and the whole block (no causal mask);
 *                  o_proj; then norm(x += o) through the SiLU MLP
 *       out = norm(x += mlp)
 *   head, mask rows only:
 *       base = bf16(out @ lm_head^T); then dspark_rnn_markov_greedy (docs in libref/ref_quant.cpp)
 *
 * ============================== PRECISION ==============================
 *
 * Weights and activations bf16, as the reference. Two places round where the reference does not:
 * each norm after a residual add reads the bf16 sum the add stored (fused_add_rmsnorm takes the
 * variance of the unrounded sum), and `mix_rows` rounds the tap mix to bf16 before its norm (the
 * reference's mix_rmsnorm keeps it f32). The softmax weights are stored bf16, which is what the
 * reference's `fuse_alpha` buffer holds. Both differences move acceptance only; every draft is
 * verified by the target.
 *
 * ============================== NOT DONE ==============================
 *
 *   One rank: the head and the walk read the whole output table (the plugin refuses tp > 1 anyway).
 *   Greedy only: the reference's sampled walk at temperature > 0 is not implemented; the drafts
 *   are the greedy path either way and the target verifies them under its own sampler.
 */
#ifndef RAD_BLOCK_DSPARK_INKLING_H
#define RAD_BLOCK_DSPARK_INKLING_H

#include "rad_arch.h"

namespace rad {
namespace arch {

/* Which handle the builder did not hand back, so a refusal reads against a line of this file. */
#define DSI_REFUSED(...) ::rad::arch::dspark_inkling_refused({ __VA_ARGS__ })
inline int dspark_inkling_refused(std::initializer_list<const char*> what) {
    std::string list;
    for (const char* w : what) { if (!list.empty()) list += ", "; list += w; }
    std::fprintf(stderr, "Inkling DSpark drafter: the builder gave no handle for %s\n", list.c_str());
    return RAD_E_INVAL;
}

struct DsparkInkling {
    enum { MAX_LAYERS = 8, N_TAPS = 4 };

    struct Config {
        int64_t block   = 0;    /* rows a sequence: the anchor and block - 1 masks */
        int64_t rank    = 0;    /* markov_rank */
        int64_t window  = 0;    /* local layers: anchor - key < window (sliding_window_size + 1) */
        int64_t n_scan  = 0;    /* vocabulary ids the walk scans: the trained (unpadded) ones */
        int32_t mask_token = 0; /* the id the core stages in every mask row */
        double  softcap = 0.0;  /* the target's final_logit_softcapping */
        bool    local[MAX_LAYERS] = {};
        int64_t steps() const { return block - 1; }   /* drafts: one per mask row */
    };

    /* The target's weights the drafter reads, and the checkpoint name of its lm_head (folded into
     * the Markov head at convert). */
    struct Target {
        rad_weight embed = 0, embed_norm = 0, head = 0;
        const char* ck_head = nullptr;
    };

    struct Wire {
        rad_buf src = 0;    /* [max_tok, 4 E] bf16, the trunk's taps, one column slice a tap */
        /* the context pass */
        rad_buf cmix = 0, cn = 0;           /* [max_tok, E] */
        rad_buf ck = 0, ckn = 0, cv = 0;    /* [max_tok, kv_dim] */
        /* the query pass */
        rad_buf e = 0;      /* [qrows, E] the target's embedding of every block row, normed */
        rad_buf anc = 0;    /* [max_seqs, E] the anchors' rows of it */
        rad_buf x = 0;      /* [qrows, E] the residual stream, bf16 */
        rad_buf h = 0;      /* [qrows, E] the normed stream */
        rad_buf d = 0;      /* [qrows, E] a sublayer's output */
        rad_buf q = 0, k = 0, v = 0, qn = 0, kn = 0, attn = 0;
        rad_buf gate_up = 0, ff = 0;
        /* the head and the walk */
        rad_buf hs = 0;     /* [srows, E] the mask rows' output */
        rad_buf dlog = 0;   /* [srows, n_vocab] f32 base logits */
        rad_buf fr = 0;     /* [max_seqs, 4 E] the last committed token's taps */
        rad_buf frm = 0, frn = 0;           /* [max_seqs, E] mixed, normed: f_rnn */
        rad_buf scratch = 0;                /* [max_seqs, steps * 3R + 2R] f32 */
        rad_buf best_score = 0, best_token = 0;
        rad_buf tokens = 0; /* [max_seqs, steps] i32, the proposal */
    };

    Geom    g{};
    Config  cfg{};
    Target  tgt{};
    Wire    w{};
    int     first_layer = 0;
    int64_t qrows = 0, srows = 0;
    rad_kvgroup kv_local = 0, kv_full = 0;

    /* ---- weights */
    rad_weight w_mask = 0, w_norm = 0;
    rad_weight w_alpha[MAX_LAYERS + 1] = {};   /* softmax(fuse_weight) row i; the last is the head's */
    rad_weight w_mk_norm = 0, w_mk1 = 0, w_rnn = 0, w_rnn_b = 0, w_fold = 0, w_fold_b = 0;
    rad_weight w_dflare[MAX_LAYERS] = {};
    rad_weight w_kc[MAX_LAYERS] = {}, w_vc[MAX_LAYERS] = {}, w_kcn[MAX_LAYERS] = {};
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    rad_weight w_q[MAX_LAYERS] = {}, w_k[MAX_LAYERS] = {}, w_v[MAX_LAYERS] = {}, w_o[MAX_LAYERS] = {};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {};
    rad_weight w_gu[MAX_LAYERS] = {}, w_dn[MAX_LAYERS] = {};

    /* ---- the context pass */
    rad_op op_cmix[MAX_LAYERS] = {}, op_cnorm[MAX_LAYERS] = {};
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};
    rad_op op_crope = 0, op_cstore_local = 0, op_cstore_full = 0;

    /* ---- the query pass */
    rad_op op_embed = 0, op_embed_norm = 0, op_mask = 0, op_anc = 0, op_anc_put = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {}, op_outnorm = 0;
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {}, op_gu[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_qrope = 0, op_krope = 0, op_store_local = 0, op_store_full = 0;
    rad_op op_attn_local = 0, op_attn_full = 0, op_act = 0, op_add = 0;

    /* ---- the head and the walk */
    rad_op op_rows = 0, op_dlogits = 0, op_fr = 0, op_frmix = 0, op_frnorm = 0, op_walk = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup local, rad_kvgroup full, const Target& t, const char* ck,
                 const Wire& wire);
    void context(RadCtx* c, const RadBatch* batch) const;
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }
    rad_kvgroup group(int64_t l) const { return cfg.local[l] ? kv_local : kv_full; }
};

/* ------------------------------------------------------------------------------ declare */

inline int DsparkInkling::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                                  int base_layer, rad_kvgroup local, rad_kvgroup full,
                                  const Target& t, const char* ck, const Wire& wire) {
    g = geom; cfg = c; tgt = t; w = wire; first_layer = base_layer;
    kv_local = local; kv_full = full;
    if (!ck || !tgt.ck_head || !tgt.embed || !tgt.embed_norm || !tgt.head)
        return DSI_REFUSED("ck", "ck_head", "the target's embedding, embed_norm or lm_head");
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS || cfg.block < 2 || cfg.rank <= 0 ||
        cfg.n_scan <= 0 || cfg.n_scan > g.n_vocab_all || !(cfg.softcap > 0.0))
        return RAD_E_INVAL;

    const int64_t qd = q_dim(), kvd = kv_dim(), E = g.n_embd, R = cfg.rank;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    srows = g.max_seqs * cfg.steps();
    const int64_t crows = g.max_tok;

    auto gemm = [&](int64_t rows, int64_t n, int64_t k, rad_weight wt, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", n),
                                       RAD_INT("K", k), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(wt)),
                  {x}, {y});
    };
    auto rms = [&](int64_t rows, int64_t n, rad_weight wt, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(wt)),
                  {x}, {y});
    };
    auto mixer = [&](int64_t rows, rad_weight wt, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "mix_rows",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", E),
                                       RAD_INT("k", N_TAPS), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(wt)),
                  {x}, {y});
    };
    auto model_w = [&](const char* name, std::initializer_list<int64_t> shape, int access) {
        return decl_w(b, name, RAD_BF16, shape, access, RAD_SHARD_NONE, grp_model());
    };

    /* ---------------------------------------------------------------- model-scope weights */
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.mask_embed"), nm.ckpt("%s.mask_embed", ck)));
    w_mask = model_w(nm.f("dspark.mask_embed"), {1, E}, RAD_ACCESS_PER_TOKEN);
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.norm.weight"), nm.ckpt("%s.norm.weight", ck)));
    w_norm = model_w(nm.f("dspark.norm.weight"), {E}, RAD_ACCESS_PER_TOKEN);
    if (!w_mask || !w_norm) return DSI_REFUSED("w_mask", "w_norm");

    /* softmax(fuse_weight), one row a draft layer and the last for the head: computed at convert
     * (RAD_MAP_SOFTMAX) and stored bf16 by the recipe, which is the reference's `fuse_alpha`. */
    for (int64_t i = 0; i <= g.n_layer; ++i) {
        const char* name = nm.f("dspark.alpha.%d", (int)i);
        RAD_ARCH_TRY(map_softmax(b, name, nm.ckpt("%s.fuse_weight", ck), (int)i));
        w_alpha[i] = model_w(name, {N_TAPS}, RAD_ACCESS_PER_TOKEN);
        if (!w_alpha[i]) return DSI_REFUSED("w_alpha[i]");
    }

    /* The Markov head. `fold` is lm_head @ hidden_out (and its bias), folded at convert in f64 as
     * the reference folds it at load; both [vocab, ...] and read whole by every position's scan. */
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov.dflare_norm"),
                          nm.ckpt("%s.markov_head.dflare_norm.weight", ck)));
    w_mk_norm = model_w(nm.f("dspark.markov.dflare_norm"), {E}, RAD_ACCESS_PER_TOKEN);
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov.w1"),
                          nm.ckpt("%s.markov_head.markov_w1.weight", ck)));
    w_mk1 = model_w(nm.f("dspark.markov.w1"), {g.n_vocab_all, R}, RAD_ACCESS_VOCAB);
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov.rnn_proj"),
                          nm.ckpt("%s.markov_head.rnn_proj.weight", ck)));
    w_rnn = model_w(nm.f("dspark.markov.rnn_proj"), {3 * R, 2 * R + 2 * E}, RAD_ACCESS_PER_TOKEN);
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov.rnn_proj_b"),
                          nm.ckpt("%s.markov_head.rnn_proj.bias", ck)));
    w_rnn_b = model_w(nm.f("dspark.markov.rnn_proj_b"), {3 * R}, RAD_ACCESS_PER_TOKEN);
    RAD_ARCH_TRY(map_matmul(b, nm.f("dspark.markov.fold"), tgt.ck_head,
                            nm.ckpt("%s.markov_head.hidden_out.weight", ck)));
    w_fold = model_w(nm.f("dspark.markov.fold"), {g.n_vocab_all, R}, RAD_ACCESS_VOCAB);
    RAD_ARCH_TRY(map_matmul(b, nm.f("dspark.markov.fold_b"), tgt.ck_head,
                            nm.ckpt("%s.markov_head.hidden_out.bias", ck)));
    w_fold_b = model_w(nm.f("dspark.markov.fold_b"), {g.n_vocab_all}, RAD_ACCESS_VOCAB);
    if (!w_mk_norm || !w_mk1 || !w_rnn || !w_rnn_b || !w_fold || !w_fold_b)
        return DSI_REFUSED("w_mk_norm", "w_mk1", "w_rnn", "w_rnn_b", "w_fold", "w_fold_b");

    /* ---------------------------------------------------------------- per-layer weights */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const RadWeightGroup grp = grp_layer(first_layer + i);
        auto lw = [&](rad_weight* out, const char* suffix, std::initializer_list<const char*> srcs,
                      std::initializer_list<int64_t> shape,
                      std::initializer_list<int64_t> parts = {}) -> int {
            const char* name = nm.f("dspark.blk.%d.%s", i, suffix);
            if (srcs.size() > 1) RAD_ARCH_TRY(map_concat(b, name, srcs));
            else                 RAD_ARCH_TRY(map_copy(b, name, *srcs.begin()));
            *out = decl_w(b, name, RAD_BF16, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp, 0,
                          parts);
            return *out ? RAD_OK : DSI_REFUSED(name);
        };
#define SRC(path) nm.ckpt("%s.layers.%d." path, ck, i)
        RAD_ARCH_TRY(lw(&w_dflare[i], "dflare_norm", { SRC("dflare_norm.weight") }, {E}));
        RAD_ARCH_TRY(lw(&w_anorm[i], "attn_norm", { SRC("input_layernorm.weight") }, {E}));
        RAD_ARCH_TRY(lw(&w_fnorm[i], "ffn_norm", { SRC("post_attention_layernorm.weight") }, {E}));
        RAD_ARCH_TRY(lw(&w_q[i], "attn_q", { SRC("self_attn.q_proj.weight") }, {qd, E}));
        RAD_ARCH_TRY(lw(&w_k[i], "attn_k", { SRC("self_attn.k_proj.weight") }, {kvd, E}));
        RAD_ARCH_TRY(lw(&w_v[i], "attn_v", { SRC("self_attn.v_proj.weight") }, {kvd, E}));
        RAD_ARCH_TRY(lw(&w_o[i], "attn_output", { SRC("self_attn.o_proj.weight") }, {E, qd}));
        RAD_ARCH_TRY(lw(&w_qn[i], "attn_q_norm", { SRC("self_attn.q_norm.weight") }, {g.head_dim}));
        RAD_ARCH_TRY(lw(&w_kn[i], "attn_k_norm", { SRC("self_attn.k_norm.weight") }, {g.head_dim}));
        RAD_ARCH_TRY(lw(&w_kc[i], "ctx_k", { SRC("self_attn.k_proj_ctx.weight") }, {kvd, E}));
        RAD_ARCH_TRY(lw(&w_vc[i], "ctx_v", { SRC("self_attn.v_proj_ctx.weight") }, {kvd, E}));
        RAD_ARCH_TRY(lw(&w_kcn[i], "ctx_k_norm", { SRC("self_attn.k_norm_ctx.weight") },
                        {g.head_dim}));
        RAD_ARCH_TRY(lw(&w_gu[i], "ffn_gate_up",
                        { SRC("mlp.gate_proj.weight"), SRC("mlp.up_proj.weight") },
                        {2 * g.n_ff, E}, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(lw(&w_dn[i], "ffn_down", { SRC("mlp.down_proj.weight") }, {E, g.n_ff}));
#undef SRC
    }

    /* ---------------------------------------------------------------- attention
     *
     * Declared before the kv_stores, which read the groups' resolved block size back. Local layers:
     * every row of a block sees the keys less than `window` behind its anchor (the reference keeps
     * sliding_window_size positions before the block) and the whole block. Layer 1: everything.
     * Neither is causal inside the block. */
    auto attn = [&](const char* op, int64_t window) {
        return rw(b, RAD_OP(b, op,
                            RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block),
                                       RAD_INT("head_dim", g.head_dim), RAD_INT("gqa", g.gqa()),
                                       RAD_INT("causal", 0), RAD_INT("window", window),
                                       RAD_INT("n_head", g.n_head),
                                       RAD_INT("block_size", RAD_KV_BLOCK),
                                       RAD_INT("max_seqs", g.max_seqs),
                                       RAD_INT("max_ctx", g.max_ctx),
                                       RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_NOWEIGHTS),
                  {w.qn}, {w.attn});
    };
    bool any_local = false, any_full = false;
    for (int64_t l = 0; l < g.n_layer; ++l) (cfg.local[l] ? any_local : any_full) = true;
    if (any_local) {
        op_attn_local = attn("attn_paged_anchor_window", cfg.window);
        if (!op_attn_local) return DSI_REFUSED("op_attn_local");
    }
    if (any_full) {
        op_attn_full = attn("attn_paged", 0);
        if (!op_attn_full) return DSI_REFUSED("op_attn_full");
    }
    auto store = [&](int64_t rows, rad_kvgroup grp, rad_buf k, rad_buf v) {
        return rw(b, RAD_OP(b, "kv_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", g.head_dim),
                                       RAD_INT("n_head_kv", g.n_head_kv),
                                       RAD_INT("block_size", rad_kv_block_size(b, grp)),
                                       RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_NOWEIGHTS),
                  {k, v}, {});
    };

    /* ---------------------------------------------------------------- the context pass
     *
     * The taps [T, 4 E] read as [4 T, E], so `mix_rows` sums each token's four consecutive rows. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_cmix[i] = mixer(crows, w_alpha[i], w.src, w.cmix);
        op_cnorm[i] = rms(crows, E, w_dflare[i], w.cmix, w.cn);
        op_ck[i] = gemm(crows, kvd, E, w_kc[i], w.cn, w.ck);
        op_cv[i] = gemm(crows, kvd, E, w_vc[i], w.cn, w.cv);
        op_cknorm[i] = rms(crows * g.n_head_kv, g.head_dim, w_kcn[i], w.ck, w.ckn);
        if (!op_cmix[i] || !op_cnorm[i] || !op_ck[i] || !op_cv[i] || !op_cknorm[i])
            return DSI_REFUSED("op_cmix[i]", "op_cnorm[i]", "op_ck[i]", "op_cv[i]", "op_cknorm[i]");
        if (l == 0) {
            /* K-only rope: n_head = n_head_kv, n_head_kv = 0; one op serves every layer */
            op_crope = rw(b, decl_rope(b, g, crows, g.head_dim, g.n_head_kv, 0, g.head_dim, "neox"),
                          {w.ckn}, {w.ckn});
            if (any_local) op_cstore_local = store(crows, kv_local, w.ckn, w.cv);
            if (any_full)  op_cstore_full  = store(crows, kv_full, w.ckn, w.cv);
            if (!op_crope || (any_local && !op_cstore_local) || (any_full && !op_cstore_full))
                return DSI_REFUSED("op_crope", "op_cstore_local", "op_cstore_full");
        }
    }

    /* ---------------------------------------------------------------- the query pass
     *
     * The block's first embedding: every row through the target's table and embed_norm (a mask
     * row's id is a padding row, harmless and discarded), the mask rows' `mask_embed` from a
     * one-row table placed at the mask id, and the anchors' normed rows put over it through the
     * sequences' first rows (cu_seqlens). */
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", E),
                                       RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(tgt.embed)),
                  {}, {w.e});
    op_embed_norm = rms(qrows, E, tgt.embed_norm, w.e, w.e);
    op_mask = rw(b, RAD_OP(b, "embed_lookup",
                           RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", E),
                                      RAD_INT("n_vocab", 1), RAD_STR("dtype", "bf16"),
                                      RAD_INT("vocab_offset", cfg.mask_token)),
                           RAD_WEIGHTS(w_mask)),
                 {}, {w.x});
    op_anc = rw(b, RAD_OP(b, "gather_rows",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", E),
                                     RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                {w.e}, {w.anc});
    op_anc_put = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", E),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    {w.anc, w.x}, {w.x});
    if (!op_embed || !op_embed_norm || !op_mask || !op_anc || !op_anc_put)
        return DSI_REFUSED("op_embed", "op_embed_norm", "op_mask", "op_anc", "op_anc_put");

    /* Declared in issue order, so the planner's liveness (declaration order) covers every use. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        if (l == 1) {
            /* x += d in bf16, the residual the next norm reads: the reference's fused_add_rmsnorm
             * stores that bf16 sum too, and normalises the unrounded one */
            op_add = rw(b, RAD_OP(b, "add",
                                  RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS),
                        {w.x, w.d}, {w.x});
            if (!op_add) return DSI_REFUSED("op_add");
        }
        op_anorm[i] = rms(qrows, E, w_anorm[i], w.x, w.h);
        op_qp[i] = gemm(qrows, qd, E, w_q[i], w.h, w.q);
        op_kp[i] = gemm(qrows, kvd, E, w_k[i], w.h, w.k);
        op_vp[i] = gemm(qrows, kvd, E, w_v[i], w.h, w.v);
        op_qn[i] = rms(qrows * g.n_head, g.head_dim, w_qn[i], w.q, w.qn);
        op_kn[i] = rms(qrows * g.n_head_kv, g.head_dim, w_kn[i], w.k, w.kn);
        if (l == 0) {
            op_qrope = rw(b, decl_rope(b, g, qrows, g.head_dim, g.n_head, 0, g.head_dim, "neox"),
                          {w.qn}, {w.qn});
            op_krope = rw(b, decl_rope(b, g, qrows, g.head_dim, g.n_head_kv, 0, g.head_dim,
                                       "neox"),
                          {w.kn}, {w.kn});
            if (any_local) op_store_local = store(qrows, kv_local, w.kn, w.v);
            if (any_full)  op_store_full  = store(qrows, kv_full, w.kn, w.v);
            if (!op_qrope || !op_krope || (any_local && !op_store_local) ||
                (any_full && !op_store_full))
                return DSI_REFUSED("op_qrope", "op_krope", "op_store_local", "op_store_full");
        }
        op_op[i] = gemm(qrows, E, qd, w_o[i], w.attn, w.d);
        if (l == 0) {
            op_add = rw(b, RAD_OP(b, "add",
                                  RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS),
                        {w.x, w.d}, {w.x});
            if (!op_add) return DSI_REFUSED("op_add");
        }
        op_fnorm[i] = rms(qrows, E, w_fnorm[i], w.x, w.h);
        op_gu[i] = gemm(qrows, 2 * g.n_ff, E, w_gu[i], w.h, w.gate_up);
        if (l == 0) {
            op_act = rw(b, RAD_OP(b, "act_mul",
                                  RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                                             RAD_STR("act", "silu"), RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS),
                        {w.gate_up}, {w.ff});
            if (!op_act) return DSI_REFUSED("op_act");
        }
        op_dn[i] = gemm(qrows, E, g.n_ff, w_dn[i], w.ff, w.d);
        if (!op_anorm[i] || !op_qp[i] || !op_kp[i] || !op_vp[i] || !op_qn[i] || !op_kn[i] ||
            !op_op[i] || !op_fnorm[i] || !op_gu[i] || !op_dn[i])
            return DSI_REFUSED("a layer op");
    }
    op_outnorm = rms(qrows, E, w_norm, w.x, w.h);
    if (!op_outnorm) return DSI_REFUSED("op_outnorm");

    /* ---------------------------------------------------------------- the head and the walk */
    op_rows = rw(b, RAD_OP(b, "gather_rows",
                           RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", E),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                 {w.h}, {w.hs});
    op_dlogits = rw(b, RAD_OP(b, "logits_gemm",
                              RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n_vocab", g.n_vocab),
                                         RAD_INT("n_embd", E), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(tgt.head)),
                    {w.hs}, {w.dlog});
    op_fr = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", N_TAPS * E),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
               {w.src}, {w.fr});
    op_frmix = mixer(g.max_seqs, w_alpha[g.n_layer], w.fr, w.frm);
    op_frnorm = rms(g.max_seqs, E, w_mk_norm, w.frm, w.frn);
    op_walk = rw(b, RAD_OP(b, "dspark_rnn_markov_greedy",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                                      RAD_INT("steps", cfg.steps()), RAD_INT("rank", R),
                                      RAD_INT("n_embd", E), RAD_INT("n_vocab", cfg.n_scan),
                                      RAD_F64("softcap", cfg.softcap),
                                      RAD_INT("anchor_stride", cfg.block),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w_mk1, w_rnn, w_rnn_b, w_fold, w_fold_b)),
                 {w.dlog, w.hs, w.frn}, {w.scratch, w.best_score, w.best_token, w.tokens});
    if (!op_rows || !op_dlogits || !op_fr || !op_frmix || !op_frnorm || !op_walk)
        return DSI_REFUSED("op_rows", "op_dlogits", "op_fr", "op_frmix", "op_frnorm", "op_walk");
    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void DsparkInkling::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    /* the taps as [4 T, E]: token r's tap t is row 4 r + t */
    const RadOperand taps = RAD_P_T2(rad_buf_ptr(c, w.src), RAD_BF16, N_TAPS * T, g.n_embd);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const rad_kvgroup grp = group(l);
        const RadKVGroupBatch* kvb = kv_batch(batch, grp);
        RAD_ISSUE_N(c, op_cmix[i], T, taps, RAD_W(w_alpha[i]), brows(w.cmix, T));
        RAD_ISSUE_N(c, op_cnorm[i], T, brows(w.cmix, T), RAD_W(w_dflare[i]), brows(w.cn, T));
        RAD_ISSUE_N(c, op_ck[i], T, brows(w.cn, T), RAD_W(w_kc[i]), brows(w.ck, T));
        RAD_ISSUE_N(c, op_cv[i], T, brows(w.cn, T), RAD_W(w_vc[i]), brows(w.cv, T));
        RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv, brows(w.ck, T), RAD_W(w_kcn[i]),
                    brows(w.ckn, T));
        RAD_ISSUE_N(c, op_crope, T, brows(w.ckn, T), pos);
        RAD_ISSUE_N(c, cfg.local[l] ? op_cstore_local : op_cstore_full, T, brows(w.ckn, T),
                    brows(w.cv, T), praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T),
                    kv_cache(grp, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void DsparkInkling::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok, n = batch->n_seq;
    if (T <= 0 || n <= 0) return;
    if (!batch->anchor_rows) {
        rad_step_fail(c, "inkling2 dspark: the query pass carries no anchor rows "
                         "(RadBatch::anchor_rows), which the Markov head's context is read at");
        return;
    }
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand ids = praw(batch->token_ids, RAD_I32, T);
    const RadOperand firsts = praw(batch->cu_seqlens, RAD_I32, n);

    RAD_ISSUE_N(c, op_embed, T, ids, RAD_W(tgt.embed), brows(w.e, T));
    RAD_ISSUE_N(c, op_embed_norm, T, brows(w.e, T), RAD_W(tgt.embed_norm), brows(w.e, T));
    RAD_ISSUE_N(c, op_mask, T, ids, RAD_W(w_mask), brows(w.x, T));
    RAD_ISSUE_N(c, op_anc, n, brows(w.e, T), firsts, brows(w.anc, n));
    RAD_ISSUE_N(c, op_anc_put, n, brows(w.anc, n), firsts, brows(w.x, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const rad_kvgroup grp = group(l);
        const RadKVGroupBatch* kvb = kv_batch(batch, grp);
        if (l > 0) RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.d, T), brows(w.x, T));
        RAD_ISSUE_N(c, op_anorm[i], T, brows(w.x, T), RAD_W(w_anorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_qp[i], T, brows(w.h, T), RAD_W(w_q[i]), brows(w.q, T));
        RAD_ISSUE_N(c, op_kp[i], T, brows(w.h, T), RAD_W(w_k[i]), brows(w.k, T));
        RAD_ISSUE_N(c, op_vp[i], T, brows(w.h, T), RAD_W(w_v[i]), brows(w.v, T));
        RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.qn, T));
        RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv, brows(w.k, T), RAD_W(w_kn[i]), brows(w.kn, T));
        RAD_ISSUE_N(c, op_qrope, T, brows(w.qn, T), pos);
        RAD_ISSUE_N(c, op_krope, T, brows(w.kn, T), pos);
        /* the block's own K and V go past `seqused`; the next context pass overwrites them */
        RAD_ISSUE_N(c, cfg.local[l] ? op_store_local : op_store_full, T, brows(w.kn, T),
                    brows(w.v, T), praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T),
                    kv_cache(grp, first_layer + i));
        RAD_ISSUE_N(c, cfg.local[l] ? op_attn_local : op_attn_full, batch->max_q_len,
                    brows(w.qn, T), kv_cache(grp, first_layer + i),
                    praw2(kvb ? kvb->block_table : nullptr, RAD_I32, n,
                          kvb ? kvb->block_table_pitch : 0),
                    praw(kvb ? kvb->seqused : nullptr, RAD_I32, n), RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, n + 1), brows(w.attn, T));
        RAD_ISSUE_N(c, op_op[i], T, brows(w.attn, T), RAD_W(w_o[i]), brows(w.d, T));
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.d, T), brows(w.x, T));
        RAD_ISSUE_N(c, op_fnorm[i], T, brows(w.x, T), RAD_W(w_fnorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_gu[i], T, brows(w.h, T), RAD_W(w_gu[i]), brows(w.gate_up, T));
        RAD_ISSUE_N(c, op_act, T, bcol(w.gate_up, 0, g.n_ff, T),
                    bcol(w.gate_up, g.n_ff, g.n_ff, T), brows(w.ff, T));
        RAD_ISSUE_N(c, op_dn[i], T, brows(w.ff, T), RAD_W(w_dn[i]), brows(w.d, T));
    }
    RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.d, T), brows(w.x, T));
    RAD_ISSUE_N(c, op_outnorm, T, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));

    /* the mask rows, named by the scheduler through `out_ids`, sequence-major */
    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_rows, S, brows(w.h, T), praw(batch->out_ids, RAD_I32, S), brows(w.hs, S));
    RAD_ISSUE_N(c, op_dlogits, S, brows(w.hs, S), RAD_W(tgt.head), brows(w.dlog, S));
    /* f_rnn: the last committed token's taps, mixed with the head's row and normed */
    RAD_ISSUE_N(c, op_fr, n, RAD_B(w.src), praw(batch->anchor_rows, RAD_I32, n), brows(w.fr, n));
    RAD_ISSUE_N(c, op_frmix, n, RAD_P_T2(rad_buf_ptr(c, w.fr), RAD_BF16, N_TAPS * n, g.n_embd),
                RAD_W(w_alpha[g.n_layer]), brows(w.frm, n));
    RAD_ISSUE_N(c, op_frnorm, n, brows(w.frm, n), RAD_W(w_mk_norm), brows(w.frn, n));
    RAD_ISSUE_N(c, op_walk, n, brows(w.dlog, S), brows(w.hs, S), brows(w.frn, n), ids,
                RAD_W(w_mk1), RAD_W(w_rnn), RAD_W(w_rnn_b), RAD_W(w_fold), RAD_W(w_fold_b),
                brows(w.scratch, n), brows(w.best_score, n), brows(w.best_token, n),
                brows(w.tokens, n));
}

/* ================================================================== the plugin's side
 *
 * Put the drafter on an Inkling trunk: its config from the container's `draft.` keys (config.json's
 * `mtp_config`, imported by rad-convert --draft-model .../mtp.safetensors), its two KV groups, its
 * buffers and graph, and the drafter declaration the core drives. `taps` is the [max_tok, 4 E] bf16
 * buffer the trunk fills; `base_layer` the first absolute layer past the trunk's. Called last. */
inline int declare_dspark_inkling(DsparkInkling* ds, DsparkInkling::Wire* w, RadBuilder* b,
                                  Names& nm, const RadModelMeta* meta, const RadBuildCtx* ctx,
                                  const Geom& target, rad_buf taps, int base_layer,
                                  const DsparkInkling::Target& tgt, int64_t n_scan,
                                  double softcap) {
    Geom dg{};
    RAD_ARCH_TRY(drafter_geom(dg, target, meta, "Inkling DSpark"));
    if (rad_meta_geti(meta, "draft.hidden_size", dg.n_embd) != dg.n_embd) {
        fprintf(stderr, "radiance: the Inkling drafter's width is not the trunk's\n");
        return RAD_E_UNSUPPORTED;
    }
    if (dg.world > 1) {
        fprintf(stderr, "radiance: the Inkling drafter reads the whole output table; run at tp 1\n");
        return RAD_E_UNSUPPORTED;
    }
    if (dg.n_layer > DsparkInkling::MAX_LAYERS) return RAD_E_UNSUPPORTED;
    const char* head = rad_meta_gets(meta, "draft.markov_head_type", "");
    if (std::strcmp(head, "rnn_hidden")) {
        fprintf(stderr, "radiance: the Inkling drafter's Markov head is '%s'; this block runs "
                        "rnn_hidden\n", head);
        return RAD_E_UNSUPPORTED;
    }
    int tap_ids[8] = {};
    const int n_taps = parse_layer_ids(rad_meta_gets(meta, "draft.target_layer_ids", nullptr),
                                       1 << 20, tap_ids, 8);
    if (n_taps != DsparkInkling::N_TAPS) {
        fprintf(stderr, "radiance: the Inkling drafter reads %d trunk taps; this block mixes %d\n",
                n_taps, (int)DsparkInkling::N_TAPS);
        return RAD_E_UNSUPPORTED;
    }

    DsparkInkling::Config c{};
    c.block = rad_meta_geti(meta, "draft.block_size", 0);
    c.rank = rad_meta_geti(meta, "draft.markov_rank", 0);
    c.n_scan = n_scan;
    c.softcap = softcap;
    if (c.block < 2 || c.rank <= 0 || c.rank % 32 || c.rank > 512) {
        fprintf(stderr, "radiance: the Inkling drafter declares block_size %lld and markov_rank "
                        "%lld; the walk takes a block of 2..17 and a rank that is a multiple of 32 "
                        "up to 512\n", (long long)c.block, (long long)c.rank);
        return RAD_E_UNSUPPORTED;
    }
    if ((long long)ctx->max_spec != c.block - 1) {
        fprintf(stderr, "radiance: the Inkling drafter denoises %lld mask positions after its "
                        "anchor, and this serve asks for %d draft tokens; run it at %lld\n",
                (long long)(c.block - 1), ctx->max_spec, (long long)(c.block - 1));
        return RAD_E_UNSUPPORTED;
    }
    /* The mask id is the last row of the (padded) table: never a sampled token, and in range of
     * the target's embedding, whose output for a mask row is discarded. */
    c.mask_token = (int32_t)(dg.n_vocab_all - 1);
    if (n_scan >= dg.n_vocab_all) {
        fprintf(stderr, "radiance: the Inkling drafter needs one padding row past the trained "
                        "vocabulary for its mask id\n");
        return RAD_E_UNSUPPORTED;
    }
    int local_ids[DsparkInkling::MAX_LAYERS] = {};
    const int n_local = parse_layer_ids(rad_meta_gets(meta, "draft.local_layer_ids", nullptr),
                                        dg.n_layer, local_ids, DsparkInkling::MAX_LAYERS);
    for (int j = 0; j < n_local; ++j) c.local[local_ids[j]] = true;
    const int64_t swa = rad_meta_geti(meta, "draft.sliding_window_size", 0);
    if (n_local > 0 && swa < c.block) {
        fprintf(stderr, "radiance: the Inkling drafter's local layers have a %lld-position "
                        "window\n", (long long)swa);
        return RAD_E_UNSUPPORTED;
    }
    /* keys at positions >= anchor - swa: anchor - key < swa + 1 */
    c.window = n_local > 0 ? swa + 1 : 0;

    /* Two KV groups: the local layers' recycled below the window, layer(s) without one kept. */
    rad_kvgroup kv_local = 0, kv_full = 0;
    for (int pass = 0; pass < 2; ++pass) {
        const bool local = pass == 0;
        bool any = false;
        for (int64_t l = 0; l < dg.n_layer; ++l) any |= c.local[l] == local;
        if (!any) continue;
        RadKVGroupDecl d{};
        d.kind      = local ? RAD_KV_WINDOW : RAD_KV_FULL;
        d.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
        d.window    = local ? c.window : 0;
        d.n_head_kv = dg.n_head_kv;
        d.head_dim  = dg.head_dim;
        rad_kvgroup grp = rad_decl_kv_group(b, nm.f(local ? "kv_draft_local" : "kv_draft_full"), &d);
        if (!grp) return DSI_REFUSED("a drafter KV group");
        for (int64_t l = 0; l < dg.n_layer; ++l)
            if (c.local[l] == local) RAD_ARCH_TRY(rad_bind_layer_kv(b, base_layer + (int)l, grp));
        (local ? kv_local : kv_full) = grp;
        dg.kv_dtype = rad_dtype_name(d.dtype);
    }

    const int64_t E = dg.n_embd, qd = dg.n_head * dg.head_dim, kvd = dg.n_head_kv * dg.head_dim;
    const int64_t MT = dg.max_tok, MS = dg.max_seqs;
    int64_t qr = MS * c.block;
    if (qr > MT) qr = MT;
    const int64_t sr = MS * c.steps();
    DsparkInkling::Wire& dw = *w;
    dw.src  = taps;
    dw.cmix = decl_b(b, nm.f("dsi_cmix"), RAD_BF16, {MT, E});
    dw.cn   = decl_b(b, nm.f("dsi_cn"), RAD_BF16, {MT, E});
    dw.ck   = decl_b(b, nm.f("dsi_ck"), RAD_BF16, {MT, kvd});
    dw.ckn  = decl_b(b, nm.f("dsi_ckn"), RAD_BF16, {MT, kvd});
    dw.cv   = decl_b(b, nm.f("dsi_cv"), RAD_BF16, {MT, kvd});
    dw.e    = decl_b(b, nm.f("dsi_e"), RAD_BF16, {qr, E});
    dw.anc  = decl_b(b, nm.f("dsi_anc"), RAD_BF16, {MS, E});
    dw.x    = decl_b(b, nm.f("dsi_x"), RAD_BF16, {qr, E});
    dw.h    = decl_b(b, nm.f("dsi_h"), RAD_BF16, {qr, E});
    dw.d    = decl_b(b, nm.f("dsi_d"), RAD_BF16, {qr, E});
    dw.q    = decl_b(b, nm.f("dsi_q"), RAD_BF16, {qr, qd});
    dw.k    = decl_b(b, nm.f("dsi_k"), RAD_BF16, {qr, kvd});
    dw.v    = decl_b(b, nm.f("dsi_v"), RAD_BF16, {qr, kvd});
    dw.qn   = decl_b(b, nm.f("dsi_qn"), RAD_BF16, {qr, qd});
    dw.kn   = decl_b(b, nm.f("dsi_kn"), RAD_BF16, {qr, kvd});
    dw.attn = decl_b(b, nm.f("dsi_attn"), RAD_BF16, {qr, qd});
    dw.gate_up = decl_b(b, nm.f("dsi_gate_up"), RAD_BF16, {qr, 2 * dg.n_ff});
    dw.ff   = decl_b(b, nm.f("dsi_ff"), RAD_BF16, {qr, dg.n_ff});
    dw.hs   = decl_b(b, nm.f("dsi_hs"), RAD_BF16, {sr, E});
    dw.dlog = decl_b(b, nm.f("dsi_dlog"), RAD_F32, {sr, dg.n_vocab});
    dw.fr   = decl_b(b, nm.f("dsi_fr"), RAD_BF16, {MS, DsparkInkling::N_TAPS * E});
    dw.frm  = decl_b(b, nm.f("dsi_frm"), RAD_BF16, {MS, E});
    dw.frn  = decl_b(b, nm.f("dsi_frn"), RAD_BF16, {MS, E});
    dw.scratch = decl_b(b, nm.f("dsi_scratch"), RAD_F32,
                        {MS, c.steps() * 3 * c.rank + 2 * c.rank});
    const int64_t tiles = (n_scan + 63) / 64;
    dw.best_score = decl_b(b, nm.f("dsi_tile_score"), RAD_F32, {MS, tiles});
    dw.best_token = decl_b(b, nm.f("dsi_tile_token"), RAD_I32, {MS, tiles});
    /* the proposal (RadDrafterDecl::proposal): the walk's tokens, never the sampler's buffer */
    dw.tokens = decl_b(b, nm.f("dsi_tokens"), RAD_I32, {MS, c.steps()});
    for (rad_buf h : { dw.cmix, dw.cn, dw.ck, dw.ckn, dw.cv, dw.e, dw.anc, dw.x, dw.h, dw.d, dw.q,
                       dw.k, dw.v, dw.qn, dw.kn, dw.attn, dw.gate_up, dw.ff, dw.hs, dw.dlog, dw.fr,
                       dw.frm, dw.frn, dw.scratch, dw.best_score, dw.best_token, dw.tokens })
        if (!h) return DSI_REFUSED("a drafter buffer");

    RAD_ARCH_TRY(ds->declare(b, nm, dg, c, base_layer, kv_local, kv_full, tgt, "draft", dw));

    RadDrafterDecl drf{};
    drf.name           = "dspark-inkling";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = c.block;          /* = depth + 1: the anchor row predicts nothing */
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = c.steps();
    drf.window         = c.window;
    drf.mask_token     = c.mask_token;
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));

    rad_note(b, "Inkling DSpark drafter: %lld layers at absolute %d..%d (%d local, %lld-key "
                "anchor window), %lld heads / %lld kv x %lld, ffn %lld; taps %d/%d/%d/%d mixed per "
                "layer; one pass over [anchor, %lld masks] proposes %lld tokens through a rank-%lld "
                "recurrent Markov walk over %lld ids",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1, n_local,
             (long long)c.window, (long long)dg.n_head, (long long)dg.n_head_kv,
             (long long)dg.head_dim, (long long)dg.n_ff, tap_ids[0], tap_ids[1], tap_ids[2],
             tap_ids[3], (long long)c.steps(), (long long)c.steps(), (long long)c.rank,
             (long long)n_scan);
    return RAD_OK;
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_DSPARK_INKLING_H */
