/* rad_block_gdn_exl3.h -- gated-delta-net block with EXL3 dense projections.
 *
 * Derived from Radiance's arch/common/rad_block_gdn.h:
 * Replaces projection gemm_nt declarations (op_in, op_z, op_out) with exl3i_dense.
 * Keeps RMSNorm, conv prep / update, kkt solve, chunk scan, gated rmsnorm, and recurrent update.
 *
 * The gate and beta projections (a / b) are plain bf16 by default, two gemm_nt launches. A container that holds
 * them as one trellis weight `ssm_ab` (Config::ab_rows > 0: a's rows, b's, then padding to a multiple of 128, as the
 * quantizer's LDLQ needs) serves them as one exl3i projection in the qkv / z input group (its activation
 * mode, its plane of the group's AQ site), and the GDN ops read a and b as column views of its output.
 */
#ifndef RAD_BLOCK_GDN_EXL3_H
#define RAD_BLOCK_GDN_EXL3_H

#include <arch/rad_arch.h>
#include "exl3_dense_op.h"
#include "exl3_fuse.h"

#include <cmath>

namespace qwen35_exl3 {

using namespace rad::arch;

struct GdnEXL3 {
    struct Config {
        int64_t n_head_k   = 0;    /* per rank -- q and k share these */
        int64_t head_k     = 0;    /* the recurrent state's key width */
        int64_t n_head_v   = 0;    /* per rank; a multiple of n_head_k, 3x it on this model */
        int64_t n_head_v_all = 0;  /* the value heads in the container: ssm_ab's a | b split, the rank offsets' base */
        int64_t head_v     = 0;
        int64_t conv_width = 4;
        int64_t chunk      = 64;   /* the chunked-scan interval, and checkpoint interval */
        int64_t a_rows = 0, b_rows = 0;   /* the gate and beta projection rows */
        int64_t ab_rows = 0;   /* > 0: a | b | padding as one trellis weight of ab_rows rows (ssm_ab); 0: plain a, b */

        int64_t k_dim()    const { return n_head_k * head_k; }
        int64_t v_dim()    const { return n_head_v * head_v; }
        int64_t conv_dim() const { return 2 * k_dim() + v_dim(); }
    };

    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]           the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd]           normed input, then out projection */
        rad_buf in   = 0;   /* [max_tok, conv_dim]         [q|k|v], pre-convolution */
        rad_buf a    = 0;   /* [max_tok, a_rows]           the gate, plain bf16 */
        rad_buf b    = 0;   /* [max_tok, b_rows]           the beta, plain bf16 */
        rad_buf ab   = 0;   /* [max_tok, ab_rows]          ssm_ab's output: a | b | padding (fused layers only) */
        rad_buf z    = 0;   /* [max_tok, v_dim]            the output gate */
        rad_buf q    = 0;   /* [max_tok, n_head_k, head_k] post-convolution, l2-normed */
        rad_buf k    = 0;   /* [max_tok, n_head_k, head_k] */
        rad_buf v    = 0;   /* [max_tok, n_head_v, head_v] */
        rad_buf gdec = 0;   /* [max_tok, n_head_v]         the resolved decay, per-chunk cumsum */
        rad_buf beta = 0;   /* [max_tok, n_head_v] */
        rad_buf kkt  = 0;   /* [max_tok, n_head_v, chunk]  the within-chunk inverse */
        rad_buf o    = 0;   /* [max_tok, n_head_v, head_v] */
        Exl3AqBuf* aq = nullptr;   /* AQ's shared row buffer (exl3_dense_op.h) */
    };

    struct Src {
        const char* norm     = nullptr;   /* input_layernorm.weight */
        const char* in_qkv   = nullptr;   /* linear_attn.in_proj_qkv.weight */
        const char* in_z     = nullptr;   /* linear_attn.in_proj_z.weight */
        const char* conv1d   = nullptr;   /* linear_attn.conv1d.weight */
        const char* a_log    = nullptr;   /* linear_attn.A_log */
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

    rad_weight w_norm = 0, w_conv = 0;
    rad_weight w_a_log = 0, w_dt_bias = 0, w_out_norm = 0;
    Exl3DenseWeight wt_in{}, wt_z{}, wt_out{};
    Exl3DenseWeight wt_ab{};   /* fused layers: a | b | padding, a trellis weight in the qkv / z group */
    Exl3Aq aq_in{};   /* AQ: the norm for qkv / z (and ssm_ab); h still written when plain a / b read it */
    rad_weight w_a = 0, w_b = 0;   /* the plain gate and beta projections */

    rad_op op_norm = 0, op_in = 0, op_a = 0, op_b = 0, op_ab = 0, op_z = 0;
    rad_op op_conv_prep = 0, op_kkt = 0, op_scan = 0, op_gnorm = 0;   /* prefill */
    rad_op op_conv_update = 0, op_recur = 0;                          /* decode */
    rad_op op_out = 0, op_ar = 0, op_add = 0;
    rad_op op_addnorm = 0;   /* exl3_fuse.h addnorm: the previous block's add and this norm, 0 if not served */
    /* exl3_fuse.h gdn: conv update + recurrent update in one launch at decode, up to cr_rows tokens a sequence */
    rad_op  op_cr = 0;
    int64_t cr_rows = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                 rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s);
    /* pend: the previous block left its residual add to this norm; defer: leave this block's to the next */
    void step(RadCtx* c, const RadBatch* batch, bool pend = false, bool defer = false) const;
};

inline int GdnEXL3::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                            rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s) {
    g = geom; cfg = c; w = wire; src = s; layer = l; kv_state = state; kv_conv = conv;
    if (cfg.n_head_k <= 0 || cfg.head_k <= 0 || cfg.n_head_v <= 0 || cfg.head_v <= 0 ||
        cfg.conv_width <= 0 || cfg.chunk <= 0)
        return RAD_E_INVAL;

    if (cfg.ab_rows > 0 && (cfg.ab_rows < cfg.a_rows + cfg.b_rows || cfg.ab_rows % 128 || !w.ab)) return RAD_E_INVAL;
    if (cfg.n_head_v % cfg.n_head_k != 0) {
        fprintf(stderr, "radiance: %lld value heads is not a multiple of %lld key heads\n",
                (long long)cfg.n_head_v, (long long)cfg.n_head_k);
        return RAD_E_INVAL;
    }

    const double q_scale = 1.0 / std::sqrt((double)cfg.head_k);

    const int64_t v_dim    = cfg.v_dim();
    const int64_t conv_dim = cfg.conv_dim();

    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.attn_norm.weight", l),  src.norm));
    /* The a/b (gate and beta) projections are plain bf16: the checkpoint and the published
     * EXL3 exports hold them unquantised, and the fused ssm_in (q|k|v|a|b) is not in 128s at
     * this model (10416 rows), so ssm_qkv is the one exl3 weight and the GDN ops take a/b as
     * their separate optional gate and beta operands. The a/b name maps are the model's (every
     * GDN layer, before the buffer plan): only ssm_qkv is mapped here. A fused layer's ssm_ab (a | b | padding)
     * is mapped there too, and is a trellis weight (the model's declare looked). */
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_qkv.weight", l), src.in_qkv));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_z.weight", l),      src.in_z));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_conv1d.weight", l), src.conv1d));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_a_log", l),         src.a_log));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_dt.bias", l),       src.dt_bias));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_norm.weight", l),   src.out_norm));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_out.weight", l),    src.out));

    w_norm     = decl_w(b, nm.f("blk.%d.attn_norm.weight", l),      RAD_F32,  {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* ssm_qkv EXL3 / BF16 weight: the three parts so rank r owns its slice of each of q, k, v. */
    int64_t qkv_parts[3] = {cfg.k_dim(), cfg.k_dim(), cfg.v_dim()};
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.ssm_qkv.weight", l), cfg.conv_dim(), g.n_embd,
                                  RAD_SHARD_ROW, grp_layer(l), 3, qkv_parts, &wt_in, g.world));
    if (cfg.ab_rows > 0) {
        /* a | b | padding as one trellis weight: its rows are jointly quantized, and a rank's a / b rows (one per
         * value head) are not a slice of the stored 16-row blocks, so it serves whole on every rank at world > 1 --
         * 128 rows of GEMM, and the ops read the rank's gate and beta as column views of the full output (step). */
        RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.ssm_ab.weight", l), cfg.ab_rows, g.n_embd,
                                      RAD_SHARD_NONE, grp_layer(l), 0, nullptr, &wt_ab, g.world));
        if (!wt_ab.is_exl3) return RAD_E_FORMAT;
    } else {
        w_a = decl_w(b, nm.f("blk.%d.ssm_a.weight", l), RAD_BF16, {cfg.a_rows, g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        w_b = decl_w(b, nm.f("blk.%d.ssm_b.weight", l), RAD_BF16, {cfg.b_rows, g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    }

    /* ssm_z EXL3 / BF16 weight */
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.ssm_z.weight", l), v_dim, g.n_embd,
                                  RAD_SHARD_ROW, grp_layer(l), 0, nullptr, &wt_z, g.world));
    if (cfg.ab_rows > 0) RAD_ARCH_TRY(exl3_act_same("ssm qkv / z / ab", l, {&wt_in, &wt_z, &wt_ab}));
    else                 RAD_ARCH_TRY(exl3_act_same("ssm qkv / z", l, {&wt_in, &wt_z}));

    w_conv     = decl_w(b, nm.f("blk.%d.ssm_conv1d.weight", l),     RAD_BF16,
                        {conv_dim, cfg.conv_width},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l), 0,
                        {cfg.k_dim(), cfg.k_dim(), cfg.v_dim()});
    w_a_log    = decl_w(b, nm.f("blk.%d.ssm_a_log", l),             RAD_F32,  {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_dt_bias  = decl_w(b, nm.f("blk.%d.ssm_dt.bias", l),           RAD_F32,  {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_out_norm = decl_w(b, nm.f("blk.%d.ssm_norm.weight", l),       RAD_F32,  {cfg.head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* ssm_out EXL3 / BF16 weight */
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.ssm_out.weight", l), g.n_embd, v_dim,
                                  RAD_SHARD_COL, grp_layer(l), 0, nullptr, &wt_out, g.world));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)), {w.x}, {w.h});
    op_addnorm = decl_addnorm(b, g, g.wadd, w_norm, w.x, w.h);

    if (cfg.ab_rows > 0) {
        /* ssm_ab is the site's third consumer: its own plane (its suh is its own after the scale refit), the same
         * launch; nothing reads h on the AQ path then, so the site does not write it */
        RAD_ARCH_TRY(aq_in.declare(b, g.max_tok, g.n_embd, "rmsnorm", g.eps, g.wadd, g.act_dtype,
                                   {&wt_in, &wt_z, &wt_ab}, w.aq, w.x, 0, w_norm, 0));
        if (exl3_fuse_on(EXL3_FUSE_ADDNORM)) aq_in.declare_res(b, w.h);   /* the previous block's add (exl3_fuse.h) */
        op_in = decl_exl3_proj(b, g.max_tok, conv_dim, g.n_embd, g.dtype, wt_in, w.h, w.in, &aq_in,
                               Exl3Aq::plane_of({&wt_in, &wt_z, &wt_ab}, &wt_in));
        op_ab = decl_exl3_proj(b, g.max_tok, cfg.ab_rows, g.n_embd, g.dtype, wt_ab, w.h, w.ab, &aq_in,
                               Exl3Aq::plane_of({&wt_in, &wt_z, &wt_ab}, &wt_ab));
        op_z  = decl_exl3_proj(b, g.max_tok, v_dim, g.n_embd, g.dtype, wt_z,  w.h, w.z, &aq_in,
                               Exl3Aq::plane_of({&wt_in, &wt_z, &wt_ab}, &wt_z));
    } else {
        RAD_ARCH_TRY(aq_in.declare(b, g.max_tok, g.n_embd, "rmsnorm", g.eps, g.wadd, g.act_dtype, {&wt_in, &wt_z}, w.aq,
                                   w.x, 0, w_norm, w.h));
        if (exl3_fuse_on(EXL3_FUSE_ADDNORM)) aq_in.declare_res(b, w.h);   /* the previous block's add (exl3_fuse.h) */
        op_in = decl_exl3_proj(b, g.max_tok, conv_dim, g.n_embd, g.dtype, wt_in, w.h, w.in, &aq_in,
                               Exl3Aq::plane_of({&wt_in, &wt_z}, &wt_in));
        /* a/b: plain bf16 projections, the GDN ops' separate gate and beta operands. */
        op_a = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", cfg.a_rows),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_a)),
                     {w.h}, {w.a});
        op_b = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", cfg.b_rows),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_b)),
                     {w.h}, {w.b});
        op_z  = decl_exl3_proj(b, g.max_tok, v_dim, g.n_embd, g.dtype, wt_z,  w.h, w.z, &aq_in,
                               Exl3Aq::plane_of({&wt_in, &wt_z}, &wt_z));
    }
    /* the gate and beta as the GDN ops read them: ssm_ab's output (column views at issue) or the plain pair */
    const bool fused = cfg.ab_rows > 0;

    /* ---- chunked path (prefill) */
    op_conv_prep = rw(b, RAD_OP(b, "gdn_conv_prep",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                     RAD_INT("head_v", cfg.head_v), RAD_INT("chunk", cfg.chunk),
                                     RAD_INT("conv_width", cfg.conv_width)),
                          RAD_WEIGHTS(w_conv, w_a_log, w_dt_bias)),
                      fused ? std::initializer_list<rad_buf>{w.in, w.ab} : std::initializer_list<rad_buf>{w.in, w.a, w.b},
                      {w.q, w.k, w.v, w.gdec, w.beta});
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

    /* ---- recurrent path (decode) */
    op_conv_update = rw(b, RAD_OP(b, "gdn_conv_update",
                            RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                       RAD_INT("head_k", cfg.head_k),
                                       RAD_INT("head_v", cfg.head_v),
                                       RAD_INT("conv_width", cfg.conv_width)),
                            RAD_WEIGHTS(w_conv)),
                        {w.in}, {w.q, w.k, w.v});

    op_recur = rw(b, RAD_OP(b, "gdn_recurrent_update",
                      RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                 RAD_INT("head_v", cfg.head_v),
                                 RAD_INT("n_head_v", cfg.n_head_v),
                                 RAD_F64("scale", q_scale),
                                 RAD_F64("eps", g.eps), RAD_STR("act", "silu"),
                                 RAD_STR("state_form",
                                         g.max_spec > 0 ? "anchor" : "per_candidate")),
                      RAD_WEIGHTS(w_a_log, w_dt_bias, w_out_norm)),
                  fused ? std::initializer_list<rad_buf>{w.q, w.k, w.v, w.ab, w.z}
                        : std::initializer_list<rad_buf>{w.q, w.k, w.v, w.a, w.b, w.z},
                  {w.o});

    /* exl3_fuse.h gdn: the pair above as one launch, the convolution in the recurrent update's prologue (q, k, v never
     * reach memory). Its window is a register preload: eight tokens a sequence at most. Optional: the pair otherwise. */
    if (exl3_fuse_on(EXL3_FUSE_GDN) && rad_op_declared(b, "gdn_conv_recurrent_update")) {
        const int64_t hi = g.max_tok < 8 ? g.max_tok : 8;
        const rad_op o = RAD_OP(b, "gdn_conv_recurrent_update",
                                RAD_PARAMS(RAD_RANGE("q_len", 1, hi), RAD_INT("head_k", cfg.head_k),
                                           RAD_INT("head_v", cfg.head_v), RAD_INT("conv_width", cfg.conv_width),
                                           RAD_INT("n_head_v", cfg.n_head_v), RAD_F64("scale", q_scale),
                                           RAD_F64("eps", g.eps), RAD_STR("act", "silu"),
                                           RAD_STR("state_form", g.max_spec > 0 ? "anchor" : "per_candidate")),
                                RAD_WEIGHTS(w_conv, w_a_log, w_dt_bias, w_out_norm));
        if (o && rad_op_resolved(b, o)) {
            op_cr = rw(b, o, fused ? std::initializer_list<rad_buf>{w.in, w.ab, w.z}
                                   : std::initializer_list<rad_buf>{w.in, w.a, w.b, w.z},
                       {w.o});
            cr_rows = hi;
        }
    }

    op_out = decl_exl3_proj(b, g.max_tok, g.n_embd, v_dim, g.dtype, wt_out, w.o, w.h);

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h}, {w.h});

    op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h}, {w.x});
    return RAD_OK;
}

inline void GdnEXL3::step(RadCtx* c, const RadBatch* batch, bool pend, bool defer) const {
    const RadKVGroupBatch* st = kv_batch(batch, kv_state);
    const RadKVGroupBatch* cv = kv_batch(batch, kv_conv);
    const int32_t* st_idx = st ? st->state_index : nullptr;
    const int64_t st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;

    const int64_t T        = batch->n_tok;
    const int64_t S        = batch->n_seq;
    const int64_t conv_dim = cfg.conv_dim();

    /* AQ: from EXL3I_AQ_MIN_ROWS rows the norm writes qkv / z's planes (and ssm_ab's, or h for plain a / b) in one
     * launch */
    const Exl3Aq* aqi = issue_in_norm(c, T, pend, op_add, op_addnorm, op_norm, aq_in, w.x, w.h, w_norm);
    issue_exl3_proj(c, op_in, T, wt_in, w.h, w.in, aqi);
    const bool fused = cfg.ab_rows > 0;
    if (fused) {
        issue_exl3_proj(c, op_ab, T, wt_ab, w.h, w.ab, aqi);
    } else {
        RAD_ISSUE(c, op_a, brows(w.h, T), RAD_W(w_a), brows(w.a, T));
        RAD_ISSUE(c, op_b, brows(w.h, T), RAD_W(w_b), brows(w.b, T));
    }
    issue_exl3_proj(c, op_z,  T, wt_z,  w.h, w.z, aqi);
    /* the gate and beta over n rows: column views of ssm_ab's output (the rank's own value heads: a's rows at
     * rank * a_rows, b's at n_head_v_all + rank * b_rows; world 1: 0 / a_rows), or the plain buffers */
    const int64_t av0 = g.rank * cfg.a_rows;
    const int64_t bv0 = cfg.n_head_v_all + g.rank * cfg.b_rows;
    auto ga = [&](int64_t n) { return fused ? bcol(w.ab, av0, cfg.a_rows, n) : brows(w.a, n); };
    auto gb = [&](int64_t n) { return fused ? bcol(w.ab, bv0, cfg.b_rows, n) : brows(w.b, n); };

    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int32_t qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode : batch->max_q_len;
    const int64_t P  = S - D;
    const int64_t PT = T - DT;
    const int64_t v_dim = cfg.v_dim();

    if (D > 0 && op_cr && qd <= cr_rows) {
        RAD_ISSUE_N(c, op_cr, qd,
                    brows(w.in, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D),
                    ga(DT), gb(DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.z, DT), RAD_W(w_out_norm), brows(w.o, DT),
                    RAD_NONE, RAD_NONE);
    } else if (D > 0) {
        RAD_ISSUE_N(c, op_conv_update, qd,
                    brows(w.in, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT));
        RAD_ISSUE_N(c, op_recur, qd,
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT),
                    ga(DT), gb(DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.z, DT), RAD_W(w_out_norm), brows(w.o, DT),
                    RAD_NONE, RAD_NONE);
    }

    if (P > 0) {
        RAD_ISSUE(c, op_conv_prep,
                  brows(w.in, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log), RAD_W(w_dt_bias),
                  kv_cache(kv_conv, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  ga(T), gb(T),
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
        RAD_ISSUE_N(c, op_gnorm, PT,
                    brow_slice(w.o, DT, PT, v_dim), brow_slice(w.z, DT, PT, v_dim),
                    RAD_W(w_out_norm), brow_slice(w.o, DT, PT, v_dim));
    }

    issue_exl3_proj(c, op_out, T, wt_out, w.o, w.h);
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (!defer) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace qwen35_exl3 */

#endif /* RAD_BLOCK_GDN_EXL3_H */
