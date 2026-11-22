/* rad_block_gdn.h -- one gated-delta-net (linear attention) block, bf16 operands.
 *
 * Hybrid models are the normal case (spec §7.2): Qwen3.5-27B has 48 GDN layers to 16
 * full-attention ones. This block composes with AttnGatedBF16 and MlpBF16 in one plugin.
 *
 *   rmsnorm -> in gemm ([q|k|v|a|b])
 *     prefill: z gemm -> gdn_conv_prep -> gdn_kkt_solve -> gdn_chunk_scan -> gdn_gated_rmsnorm
 *     decode:  gdn_conv_update -> z gemm -> gdn_recurrent_update   (which folds the gated norm)
 *   (z after the convolution in a pure decode step; a mixed step issues it first, as prefill)
 *   -> out gemm -> [all_reduce] -> residual add
 *
 * The two paths branch on phase, which spec §10 allows: both sequences are declared, resolved
 * and in the graph dump. The chunked path amortises over a prefill chunk, the recurrent path is
 * O(1) per token, and no single kernel suits both.
 *
 * Both paths take `num_accepted`: the recurrent state and conv window have already absorbed
 * rejected draft tokens, and rollback is a change of read offset, not a recompute (spec §10,
 * OPS.md).
 *
 * ============================== TWO CONVENTIONS ==============================
 *
 * `A_log` is a log. The decay is
 *
 *     g = -exp(A_log) . softplus(a + dt_bias)
 *
 * and the kernels (libr4d and libref) take the exponent themselves. Safetensors stores
 * `linear_attn.A_log` in that form, so the weight is declared `ssm_a_log` and copied through.
 * llama.cpp's GGUF conversion writes `ssm_a = -exp(A_log)` instead, so a container built from a
 * GGUF must undo it; otherwise the decay is -exp(-exp(A_log)), near -1 for every head. The name
 * makes the convention visible in the weight manifest.
 *
 * q and k are L2-normalised, not RMS-normalised (no division by the width, no gain), inside
 * gdn_conv_prep (prefill) and gdn_recurrent_update (decode), which already hold the head in
 * registers. There is no separate l2_norm op.
 *
 * ============================== THE FUSED IN-PROJECTION ==============================
 *
 * gdn_conv_prep's `x` is the whole row [q | k | v | a | b]: conv_dim convolved channels, then
 * one decay input and one beta input per value head. So in_proj_qkv, in_proj_a and in_proj_b are
 * one declared weight and one GEMM.
 */
#ifndef RAD_BLOCK_GDN_H
#define RAD_BLOCK_GDN_H

#include "rad_arch.h"
#include "rad_norm_bf16.h"

#include <cmath>

namespace rad {
namespace arch {

struct GdnBF16 {
    struct Config {
        int64_t n_head_k   = 0;    /* per rank -- q and k share these */
        int64_t head_k     = 0;    /* the recurrent state's key width */
        int64_t n_head_v   = 0;    /* per rank; a multiple of n_head_k, 3x it on this model */
        int64_t head_v     = 0;
        int64_t conv_width = 4;
        int64_t chunk      = 64;   /* the chunked-scan interval, and the checkpoint interval */

        int64_t k_dim()    const { return n_head_k * head_k; }
        int64_t v_dim()    const { return n_head_v * head_v; }
        int64_t conv_dim() const { return 2 * k_dim() + v_dim(); }
        int64_t in_dim()   const { return conv_dim() + 2 * n_head_v; }   /* ... + a + b */
    };

    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]           the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd]           normed input, then the out projection */
        rad_buf in   = 0;   /* [max_tok, in_dim]           [q|k|v|a|b], pre-convolution */
        rad_buf z    = 0;   /* [max_tok, v_dim]            the output gate */
        rad_buf q    = 0;   /* [max_tok, n_head_k, head_k] post-convolution, l2-normed */
        rad_buf k    = 0;   /* [max_tok, n_head_k, head_k] */
        rad_buf v    = 0;   /* [max_tok, n_head_v, head_v] */
        rad_buf gdec = 0;   /* [max_tok, n_head_v]         the resolved decay, per-chunk cumsum */
        rad_buf beta = 0;   /* [max_tok, n_head_v] */
        rad_buf kkt  = 0;   /* [max_tok, n_head_v, chunk]  the within-chunk inverse */
        rad_buf o    = 0;   /* [max_tok, n_head_v, head_v] */
    };

    /* Checkpoint-side names for one layer; the plugin owns the spelling (spec §2.4). */
    struct Src {
        const char* norm     = nullptr;   /* input_layernorm.weight */
        const char* in_qkv   = nullptr;   /* linear_attn.in_proj_qkv.weight */
        const char* in_a     = nullptr;   /* linear_attn.in_proj_a.weight */
        const char* in_b     = nullptr;   /* linear_attn.in_proj_b.weight */
        const char* in_z     = nullptr;   /* linear_attn.in_proj_z.weight */
        const char* conv1d   = nullptr;   /* linear_attn.conv1d.weight */
        const char* a_log    = nullptr;   /* linear_attn.A_log, a log (see the header) */
        const char* dt_bias  = nullptr;   /* linear_attn.dt_bias */
        const char* out_norm = nullptr;   /* linear_attn.norm.weight */
        const char* out      = nullptr;   /* linear_attn.out_proj.weight */
    };

    Geom        g{};
    Config      cfg{};
    Wire        w{};
    Src         src{};
    int         layer = 0;
    rad_kvgroup kv_state = 0, kv_conv = 0;

    rad_weight w_norm = 0, w_in = 0, w_z = 0, w_conv = 0;
    rad_weight w_a_log = 0, w_dt_bias = 0, w_out_norm = 0, w_out = 0;

    rad_op op_norm = 0, op_in = 0, op_z = 0;
    rad_op op_conv_prep = 0, op_kkt = 0, op_scan = 0, op_gnorm = 0;   /* prefill */
    rad_op op_conv_update = 0, op_recur = 0;                          /* decode */
    rad_op op_out = 0, op_ar = 0, op_add = 0;
    /* The fold (rad_norm_bf16.h): the norm may take the previous block's delta and all-reduce,
     * and this block may leave its own to the next block's norm (`ar_out`, NormBF16::takes_ar). */
    NormBF16 nq{};
    bool ar_out = false;
    /* Set by the architecture after declare, `prefetch`es (decl_prefetch, rad_arch.h); 0 where
     * none is declared:
     *   op_pf / pf_w  of the next GEMM's leading weight bytes, issued after out and before this
     *                 block's own all-reduce where it issues one, so it is beside the reduction
     *                 whether this block or the next norm runs it (prefetch_beside);
     *   op_pf_out     of the out-projection's leading bytes beside the decode recurrent update (as
     *                 GdnFP8::op_pf_out), issued right before it in a pure decode step of at most
     *                 pf_out_seqs sequences. */
    rad_op     op_pf = 0;
    rad_weight pf_w  = 0;
    rad_op     op_pf_out = 0;
    int64_t    pf_out_seqs = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                 rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s,
                 bool fold = false, bool add_out = true,
                 bool ar_in = false, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int GdnBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                            rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s,
                            bool fold, bool add_out, bool ar_in, bool ar_out_) {
    ar_out = ar_out_;
    g = geom; cfg = c; w = wire; src = s; layer = l; kv_state = state; kv_conv = conv;
    if (cfg.n_head_k <= 0 || cfg.head_k <= 0 || cfg.n_head_v <= 0 || cfg.head_v <= 0 ||
        cfg.conv_width <= 0 || cfg.chunk <= 0)
        return RAD_E_INVAL;
    /* q and k are shared across a group of value heads (16 key to 48 value heads on Qwen3.5-27B);
     * the kernels take H and Hg separately and need H to be a multiple of Hg. */
    if (cfg.n_head_v % cfg.n_head_k != 0) {
        fprintf(stderr, "radiance: %lld value heads is not a multiple of %lld key heads\n",
                (long long)cfg.n_head_v, (long long)cfg.n_head_k);
        return RAD_E_INVAL;
    }

    /* The query scale is required. `gdn_chunk_scan` and `gdn_recurrent_update` multiply the output
     * by `scale` (o = scale * e^g Q S^T + ...), which defaults to 1; FLA's chunk_gated_delta_rule,
     * vLLM and llama.cpp use head_k^-0.5.
     *
     * The following gated RMS norm does not cancel it: the scan output is around 1e-6, so
     * mean(o^2) ~ 1e-12 is dominated by eps = 1e-6 and the norm reduces to a fixed division by
     * sqrt(eps). Without the scale every delta-net layer's contribution is sqrt(head_k) too
     * large. */
    const double q_scale = 1.0 / std::sqrt((double)cfg.head_k);

    const int64_t v_dim    = cfg.v_dim();
    const int64_t conv_dim = cfg.conv_dim();
    const int64_t in_dim   = cfg.in_dim();

    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.attn_norm.weight", l),  src.norm));
    /* q, k, v, a and b in one row, as gdn_conv_prep reads it. A row-sharded CONCAT shards each
     * source before concatenating, and `in_proj_qkv` is itself three head groups, so the row split
     * below has five parts. Maps are declared before the weights, since weight encodings are looked
     * up through the map. */
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ssm_in.weight", l),
                            { src.in_qkv, src.in_a, src.in_b }, 0));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_z.weight", l),      src.in_z));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_conv1d.weight", l), src.conv1d));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_a_log", l),         src.a_log));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_dt.bias", l),       src.dt_bias));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_norm.weight", l),   src.out_norm));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_out.weight", l),    src.out));

    w_norm     = decl_w(b, nm.f("blk.%d.attn_norm.weight", l),      RAD_F32,  {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    const LinearBF16 in = decl_linear_bf16(b, g, nm.f("blk.%d.ssm_in.weight", l),
                                           {in_dim, g.n_embd}, RAD_SHARD_ROW, grp_layer(l),
                                           {cfg.k_dim(), cfg.k_dim(), cfg.v_dim(), cfg.n_head_v,
                                            cfg.n_head_v});
    const LinearBF16 z  = decl_linear_bf16(b, g, nm.f("blk.%d.ssm_z.weight", l),
                                           {v_dim, g.n_embd}, RAD_SHARD_ROW, grp_layer(l));
    w_in = in.w; w_z = z.w;
    /* The depthwise convolution shards with its channels. The checkpoint ships [conv_dim, 1, 4];
     * the depthwise group extent is squeezed at convert. Qwen3.5's conv1d has bias=False, so the
     * bias operand is passed absent. */
    w_conv     = decl_w(b, nm.f("blk.%d.ssm_conv1d.weight", l),     RAD_BF16,
                        {conv_dim, cfg.conv_width},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l), 0,
                        {cfg.k_dim(), cfg.k_dim(), cfg.v_dim()});
    w_a_log    = decl_w(b, nm.f("blk.%d.ssm_a_log", l),             RAD_F32,  {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_dt_bias  = decl_w(b, nm.f("blk.%d.ssm_dt.bias", l),           RAD_F32,  {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    /* One gain vector over head_v shared by every head, so replicated. */
    w_out_norm = decl_w(b, nm.f("blk.%d.ssm_norm.weight", l),       RAD_F32,  {cfg.head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    const LinearBF16 out = decl_linear_bf16(b, g, nm.f("blk.%d.ssm_out.weight", l),
                                            {g.n_embd, v_dim}, RAD_SHARD_COL, grp_layer(l));
    w_out = out.w;

    RAD_ARCH_TRY(nq.declare(b, g, w_norm, w.x, w.h, fold, ar_in));
    op_norm = nq.op_norm;
    op_in = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", in_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", in.dtype)),
                   RAD_WEIGHTS(w_in)), {w.h}, {w.in});
    op_z  = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", v_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", z.dtype)),
                   RAD_WEIGHTS(w_z)), {w.h}, {w.z});

    /* ---- the chunked path (prefill). `chunk` is RAD_DERIVED on gdn_kkt_solve and gdn_chunk_scan
     * (the tile length belongs to the kernel), but gdn_conv_prep takes it explicitly because its
     * gate cumsum must be cut at the scan's tile boundary. Known gap: nothing checks that the
     * supplied and derived values agree; a mismatch applies the decay over the wrong span. */
    op_conv_prep = rw(b, RAD_OP(b, "gdn_conv_prep",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                     RAD_INT("head_v", cfg.head_v), RAD_INT("chunk", cfg.chunk),
                                     RAD_INT("conv_width", cfg.conv_width)),
                          RAD_WEIGHTS(w_conv, w_a_log, w_dt_bias)),
                      {w.in}, {w.q, w.k, w.v, w.gdec, w.beta});
    op_kkt  = rw(b, RAD_OP(b, "gdn_kkt_solve",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k)),
                     RAD_NOWEIGHTS),
                 {w.k, w.beta, w.gdec}, {w.kkt});
    op_scan = rw(b, RAD_OP(b, "gdn_chunk_scan",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                RAD_INT("head_v", cfg.head_v), RAD_F64("scale", q_scale)),
                     RAD_NOWEIGHTS),
                 {w.q, w.k, w.v, w.kkt, w.gdec, w.beta}, {w.o});
    op_gnorm = rw(b, RAD_OP(b, "gdn_gated_rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("channels", cfg.head_v),
                                 RAD_F64("eps", g.eps), RAD_STR("act", "silu")),
                      RAD_WEIGHTS(w_out_norm)),
                  {w.o, w.z}, {w.o});

    /* ---- the recurrent path (decode and speculative verify), ranged on q_len because a verify
     * step has up to n_spec+1 query rows per sequence. `n_head_v` is stated explicitly: with 48
     * value heads to 16 key heads, inferring it from a flattened view gives a plausible wrong
     * answer. */
    op_conv_update = rw(b, RAD_OP(b, "gdn_conv_update",
                            RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                       RAD_INT("head_k", cfg.head_k),
                                       RAD_INT("head_v", cfg.head_v),
                                       RAD_INT("conv_width", cfg.conv_width)),
                            RAD_WEIGHTS(w_conv)),
                        {w.in}, {w.q, w.k, w.v});
    /* `state_form` states what the state cache holds, which the operands do not show. libr4d's
     * recurrent update has two contracts: per_candidate writes one full state per candidate token
     * (1 + n_spec slots); anchor keeps one committed state per sequence and replays the accepted
     * prefix from a scratch slot (two slots at any depth). `o` is the same, `state` differs. It
     * matches KVGroupPlan::state_copies (2 when speculating, else 1), and lets a reference decline
     * the form it lacks and the kernel refuse a pool sized for the other. */
    op_recur = rw(b, RAD_OP(b, "gdn_recurrent_update",
                      RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                 RAD_INT("head_v", cfg.head_v),
                                 RAD_INT("n_head_v", cfg.n_head_v),
                                 RAD_F64("scale", q_scale),
                                 RAD_F64("eps", g.eps), RAD_STR("act", "silu"),
                                 RAD_STR("state_form",
                                         g.max_spec > 0 ? "anchor" : "per_candidate")),
                      RAD_WEIGHTS(w_a_log, w_dt_bias, w_out_norm)),
                  {w.q, w.k, w.v, w.in, w.z}, {w.o});

    op_out = rw(b, RAD_OP(b, "gemm_nt",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                               RAD_INT("K", v_dim), RAD_STR("dtype", out.dtype)),
                    RAD_WEIGHTS(w_out)),
                {w.o}, {w.h});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h}, {w.h});
    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h}, {w.x});
    return RAD_OK;
}

inline void GdnBF16::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* st = kv_batch(batch, kv_state);
    const RadKVGroupBatch* cv = kv_batch(batch, kv_conv);
    const int32_t* st_idx = st ? st->state_index : nullptr;
    /* Slots per state_index row: the state, plus scratch when a kernel serving speculation asks
     * for it (RadKVGroupBatch). Passed through unread. */
    const int64_t st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;

    const int64_t T        = batch->n_tok;
    const int64_t S        = batch->n_seq;
    const int64_t conv_dim = cfg.conv_dim();
    const int64_t H        = cfg.n_head_v;

    /* Each path runs over its own run of sequences (details in rad_block_gdn_fp8.h's step()). The
     * step is sorted decode-first and every kernel derives grid and rows from cu_seqlens, which
     * holds absolute offsets: the prefill half gets `cu + D` with token operands at their base,
     * the decode half has both narrowed to DT. The gated norm is per token, so it takes a row
     * slice. */
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);

    /* In a pure decode step z goes after the convolution. in and z read only the normed input, so
     * the engine runs z beside in when it follows in, and at decode heights the two then share
     * the memory and both stretch: on two R9700s at Qwen3.8-27B's verify shape, ~145 us each
     * against ~72 and ~43 apart. The convolution reads only in's output, so z behind it runs
     * beside that small kernel instead, and the recurrent update, which reads z, waits for it.
     * Only the order of two independent launches changes, so the bytes do not: measured
     * 166 -> 142 us from in to the recurrent update, and 1.6 ms off a 47 ms step, with the same
     * greedy text. A prefill or mixed step keeps z beside in: there it is a large GEMM, and the
     * order was measured at decode only. */
    const bool z_late = D > 0 && D == S;

    nq.step(c, T);
    RAD_ISSUE(c, op_in,   brows(w.h, T), RAD_W(w_in),   brows(w.in, T));
    if (!z_late) RAD_ISSUE(c, op_z, brows(w.h, T), RAD_W(w_z), brows(w.z, T));
    /* The decode half's band. In a mixed step max_q_len is the prefill chunk's; in a pure decode
     * step it is already right, so a batch leaving max_q_len_decode zero still works there. */
    const int32_t qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode : batch->max_q_len;
    const int64_t P  = S - D;
    const int64_t PT = T - DT;
    const int64_t v_dim = cfg.v_dim();

    if (D > 0) {
        RAD_ISSUE_N(c, op_conv_update, qd,
                    brows(w.in, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT));
        if (z_late) RAD_ISSUE(c, op_z, brows(w.h, T), RAD_W(w_z), brows(w.z, T));
        /* The recurrent update runs one workgroup per (sequence, head) and leaves the card's
         * memory mostly idle, so the out-projection's leading bytes are read beside it
         * (prefetch_beside: on the second lane, behind a join from lane 0). The join follows z,
         * which the update reads, so it orders nothing the update did not wait for. */
        if (op_pf_out && z_late && D <= pf_out_seqs) prefetch_beside(c, op_pf_out, w_out, T);
        /* a and b are the last 2*n_head_v columns of the projection row, read in place. The
         * recurrent kernel folds the gated norm, so it takes z and the gain. */
        RAD_ISSUE_N(c, op_recur, qd,
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT),
                    bcol(w.in, conv_dim, H, DT), bcol(w.in, conv_dim + H, H, DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.z, DT), RAD_W(w_out_norm), brows(w.o, DT),
                    /* The folded out-projection quantiser, unused: the bf16 out projection reads
                     * `o` directly. Operands are positional and the runtime checks the schema's
                     * count of sixteen (core/runtime/issue.cpp), so pass two RAD_NONE. */
                    RAD_NONE, RAD_NONE);
    }

    if (P > 0) {
        /* Sixteen operands, inputs before outputs (docs/OPS.md). `a` and `b_gate` are optional in
         * the schema (libref falls back to the fused `x` tail), but libr4d's conv_prep requires
         * them and `cache_idx`; omitting them would silently fall back to the reference kernel.
         *
         * `has_init` is ctx_lens: zero exactly when a sequence starts fresh (new, or preempted and
         * replaying, with a recycled conv slot), so the first chunk does not convolve against the
         * slot's stale window. */
        RAD_ISSUE(c, op_conv_prep,
                  brows(w.in, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log), RAD_W(w_dt_bias),
                  kv_cache(kv_conv, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  bcol(w.in, conv_dim, H, T), bcol(w.in, conv_dim + H, H, T),
                  praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
                  praw(batch->ctx_lens + D, RAD_I32, P),
                  brows(w.q, T), brows(w.k, T), brows(w.v, T),
                  brows(w.gdec, T), brows(w.beta, T));
        RAD_ISSUE(c, op_kkt,
                  brows(w.k, T), brows(w.beta, T), brows(w.gdec, T),
                  praw(batch->cu_seqlens + D, RAD_I32, P + 1), brows(w.kkt, T));
        RAD_ISSUE(c, op_scan,
                  brows(w.q, T), brows(w.k, T), brows(w.v, T), brows(w.kkt, T),
                  brows(w.gdec, T), brows(w.beta, T),
                  kv_cache(kv_state, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  brows(w.o, T), kv_cache(kv_state, layer),
                  praw2(st_idx ? st_idx + D * st_w : nullptr, RAD_I32, P, st_w));
        /* Per token, so a row slice over the prefill rows only; the recurrent kernel already
         * gated the decode rows. */
        RAD_ISSUE_N(c, op_gnorm, PT,
                    brow_slice(w.o, DT, PT, v_dim), brow_slice(w.z, DT, PT, v_dim),
                    RAD_W(w_out_norm), brow_slice(w.o, DT, PT, v_dim));
    }
    RAD_ISSUE(c, op_out, brows(w.o, T), RAD_W(w_out), brows(w.h, T));
    /* Decode steps only: a prompt's rows never take the reads (rad_arch.h, kPrefetchRows). */
    prefetch_beside(c, op_pf, pf_w, batch->phase == RAD_PHASE_DECODE ? T : INT64_MAX);
    if (op_ar && !(ar_out && ar_wire_is_exact(g, T)))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_GDN_H */
