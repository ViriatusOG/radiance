/* gemma4 -- Gemma 4 decoders (26B-A4B MoE, and the dense members of the family): a bf16
 * trunk, routed experts bf16, block fp8 or rotated int8 (see FP8 EXPERTS), and the DFlash
 * block-diffusion drafter, with the shared Gemma 4 vision tower.
 *
 * ============================== THE SHAPE (gemma-4-26B-A4B-it) ==============================
 *
 *   30 layers, hidden 2816, vocab 262144 (tied to the embedding), rms_norm_eps 1e-6
 *   layer_types: five `sliding_attention` then one `full_attention`, five times over
 *     sliding: 16 heads / 8 KV heads x 256, rope theta 1e4 over the whole head, window 1024
 *     full:    16 heads / 2 KV heads x 512, proportional rope theta 1e6 over a quarter of it,
 *              and K is V (`attention_k_eq_v`): there is no v_proj, V is the k projection before
 *              its norm and its rotary
 *   attention scale 1.0 -- the q/k norms carry the temperature; q and k RMS-normed per head with a
 *   gain, v with none
 *   Gemma's sandwich norms, plain gains (x_hat * w, not 1 + w)
 *   the feed-forward is two parallel arms, both reading the same residual:
 *     dense    pre_feedforward_layernorm -> gate|up 2112 (gelu-tanh) -> down
 *              -> post_feedforward_layernorm_1
 *     routed   router over rmsnorm(x) * scale * hidden^-0.5 -> softmax -> top 8 of 128 ->
 *              renormalised -> times per_expert_scale[expert];
 *              pre_feedforward_layernorm_2 -> 128 experts of 704 (gelu-tanh) ->
 *              post_feedforward_layernorm_2
 *     sum -> post_feedforward_layernorm -> residual add
 *   layer_scalar multiplies the residual stream at the end of every layer
 *   embedding scaled by sqrt(hidden) rounded to bf16 (53.0, not 53.066); final norm; tied lm_head;
 *   final_logit_softcapping 30
 *
 * The reference is transformers' modeling_gemma4 (Gemma4TextDecoderLayer, Gemma4TextAttention,
 * Gemma4TextRouter, _compute_proportional_rope_parameters), read at 5.19.
 *
 * ============================== EASY TO GET SUBTLY WRONG ==============================
 *
 *   Proportional rope is not a partial rotary. It rotates the first 64 frequency pairs of a
 *   512-wide head at theta^(-2j/512) (the exponent divides by the head width, and pair j is
 *   channels (j, j + 256)) and leaves the other 192 pairs at frequency zero. `rope` with
 *   rotary_dim 128 would pair (j, j + 64) at theta^(-2j/128), a different function that still
 *   produces fluent text. It is declared as a full-width NeoX rotary whose frequencies past pair
 *   63 are divided by 1e30 (the YaRN blend `rope` already has, with the ramp a step at 63.5..64),
 *   which gives exactly the reference's angles below the step and cos 1, sin ~1e-31 above it. See
 *   decl_full_rope().
 *
 *   V on a full layer is the un-normed, un-rotated k projection, RMS-normed with a unit gain, so
 *   the value norm must read `k` before the key norm writes it (the issue order below).
 *
 *   The router does not read the feed-forward's input. It reads the residual stream through a
 *   gainless norm, times its own `scale` vector, times hidden^-0.5 (one `rmsnorm` with `scale` as
 *   the gain and `mult` the constant), while the experts read pre_feedforward_layernorm_2's output
 *   and the dense arm pre_feedforward_layernorm's.
 *
 *   per_expert_scale multiplies the renormalised weight, after the top-k. It is gathered by expert
 *   id with `embed_lookup` (a 128 x 1 table) and multiplied in with `mul`.
 *
 *   The arms' post-norms are per arm, so under tensor parallel each arm's partial sum is reduced
 *   before its norm. The two partials share one [2T, n_embd] buffer (dense rows first, routed
 *   after), so one all-reduce carries both.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * Heads split by rank, o and down by column with an all-reduce, the vocabulary by row. The full
 * layers' two KV heads are replicated past two ranks: at four, ranks {0,1} hold KV head 0 and
 * {2,3} KV head 1, each reading its head's row span of k_proj (rad_arch.h, geom_from). Routed
 * experts are expert-parallel (whole experts, 128/world per rank), since a 704-wide expert is 5.5
 * fp8 blocks and has no block-aligned width split. The router is replicated, so every rank agrees
 * on the routing without a collective, and the arm's all-reduce sums the ranks' experts.
 *
 * ============================== FP8 EXPERTS ==============================
 *
 * The experts are ~90% of the weight bytes and decode reads whichever were routed, so their width
 * sets the step's bandwidth. Each projection is served as the container holds it, per layer and
 * per projection (expert_format): bf16 by `moe_gemm`, block fp8 (the recipe's `rtn
 * codes=fp8_e4m3 block=128x128 scale=bf16`) by `moe_gemm_q` at fp8a8 over an E4M3 input:
 *
 *   gate_up  [2 * 704, 2816]: 1408 rows are whole 64-row tiles and eleven 128-row scale blocks
 *            (one straddles gate and up, which only shares a scale); K is 22 blocks. fp8 at the
 *            checkpoint's own width.
 *   down     [2816, 704]: K is the expert width and moe_gemm_q reads K in whole 128-column
 *            blocks, which 704 is not. It stays bf16 -- unless the checkpoint is padded to 768
 *            first, outside the engine: zero rows of gate and up give gelu(0) * 0 = 0 in the
 *            padded columns, which meet zero columns of down, so the function is unchanged, and
 *            then both projections are fp8.
 *
 * Rotated int8 experts (`rtn codes=i8 group=128 scale=bf16 transform=fwht128`) take the same
 * tables and the same widths through `moe_gemm_q` at i8a8h, over int8 codes of the experts' input
 * that the attention join's fused route writes (`norm_add_norm_route_i8g`), or the rotating norm
 * (`rmsnorm_had_quant_i8g`) where that is not served. A model's quantised experts are all one or
 * all the other.
 *
 * The cache is bf16 or int8 with an f16 scale of K and of V per token and head
 * (--kv-cache-dtype i8, libr4d's windowed kernels' int8 rows and kv_store_i8), the drafter's
 * with it.
 *
 * ============================== NOT DECLARED ==============================
 *
 *   an fp8 attention or dense arm: under a tenth of the weight bytes, and the dense arm's 2112 is
 *   not whole 128-blocks.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_gemma4_vision.h>
#include <arch/rad_linear.h>
#include <arch/rad_gemma4_chat.h>
#include <arch/rad_gemma4_dflash.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gemma4 {

using namespace rad::arch;

/* The fused router's two bounds (rad_block_moe_q.h): the most routing slots libr4d's sort
 * serves in one workgroup (kScatterSmall), and the skinny bf16 GEMM's decode band, within which the
 * fused op's GEMM matches that GEMM's arithmetic. */
static const int64_t kMoeSortRows = 1024;
static const int64_t kMoeRouterFusedRows = 64;

/* ============================================================== the model */
/* Gemma4ForConditionalGeneration nests the text model's config under `text_config`; a
 * Gemma4ForCausalLM checkpoint does not, so config keys are read with meta_text_get*. */

/* One attention kind: the sliding and the full layers differ in head width, KV heads, rotary
 * and window, and in whether V has a projection of its own. */
struct Kind {
    int64_t hd = 0;          /* head_dim */
    int64_t nh = 0;          /* query heads, this rank */
    int64_t nkv = 0;         /* KV heads, this rank */
    int64_t nkv_all = 0;
    bool    kv_repl = false; /* KV heads replicated across ranks (world > nkv_all) */
    int64_t kv_off = 0;      /* this rank's first KV row in the unsharded k/v projection */
    double  theta = 0.0;
    int64_t window = 0;      /* attn_paged's: key j visible to query i when i - j < window */
    int64_t rot_pairs = 0;   /* frequency pairs that rotate; < hd/2 is proportional rope */
    bool    k_eq_v = false;
    rad_kvgroup kv = 0;
    int64_t q()  const { return nh * hd; }
    int64_t kv_dim() const { return nkv * hd; }
};

struct Layer {
    int full = 0;
    /* attention */
    rad_weight w_anorm = 0, w_qkv = 0, w_q = 0, w_k = 0, w_qn = 0, w_kn = 0, w_o = 0, w_apost = 0;
    rad_op op_anorm = 0, op_qkv = 0, op_qknr = 0, op_lift = 0;
    rad_op op_q = 0, op_k = 0, op_vnorm = 0, op_qn = 0, op_kn = 0, op_qrope = 0, op_krope = 0;
    rad_op op_store = 0, op_attn = 0, op_o = 0, op_ar_o = 0, op_apost = 0, op_add_a = 0;
    rad_op op_bidi = 0, op_ahead = 0;
    /* dense arm */
    rad_weight w_fnorm = 0, w_gu = 0, w_dn = 0, w_fpost1 = 0;
    rad_op op_fnorm = 0, op_gu = 0, op_act = 0, op_dn = 0;
    /* routed arm. Each projection's experts are plain bf16 or block fp8, as the container holds
     * (expert_format). A bf16 projection's handles are its experts in order, one table; an fp8
     * one's are the even local experts then the odd ones, codes in w_egu/w_edn and scale views in
     * w_egus/w_edns, so each parity is one run (see declare). */
    rad_weight w_rscale = 0, w_router = 0, w_escale = 0, w_enorm = 0, w_fpost2 = 0;
    std::vector<rad_weight> w_egu, w_edn, w_egus, w_edns;
    /* A projection's experts are quantised (block fp8 or rotated int8: the table GEMM over a code
     * pair), else plain bf16. */
    bool gu_q = false, dn_q = false;
    rad_op op_rnorm = 0, op_router = 0, op_topk = 0, op_escale = 0, op_emul = 0, op_enorm = 0;
    rad_op op_scatter = 0, op_egu = 0, op_eact = 0, op_eq = 0, op_edn = 0, op_gather = 0;
    /* libr4d's fused forms of the router, its top-k and the sort, where they resolve: the pair
     * (`router_topk_scatter`) up to `topk_scatter_rows`, and the router GEMM in front of it
     * (`router_gemm_topk_scatter`) up to `router_fused_rows`. A pass past their rows issues the
     * separate ops, which are declared either way. */
    rad_op  op_topk_scatter = 0, op_router_fused = 0;
    /* expert_scale: op_escale and op_emul in one launch, where it resolves */
    rad_op  op_expert_scale = 0;
    int64_t topk_scatter_rows = 0, router_fused_rows = 0;
    /* the join */
    rad_weight w_fpost = 0, w_scalar = 0;
    rad_op op_ar_f = 0, op_fpost1 = 0, op_fpost2 = 0, op_fsum = 0, op_fpost = 0, op_add_f = 0;
    rad_op op_scale = 0;
    /* The sandwich joins, fused: op_join_a is apost + add_a + fnorm, op_join_f the routed layer's
     * fpost1 + fpost2 + fsum + fpost + add_f + scale. Zero where not served; the chain above is
     * then declared and issued instead. */
    rad_op op_join_a = 0, op_join_f = 0;
    /* op_join_r: op_join_a writing the routed arm's two inputs as well -- the router's norm into
     * b_rn and the experts' codes into b_hq / b_hqs (norm_add_norm_route's E4M3, or
     * norm_add_norm_route_i8g's rotated int8) -- where the layer's gate_up is quantised. op_rnorm
     * and op_enorm are then declared and not issued. */
    rad_op op_join_r = 0;
    /* op_join_f also writes the next norm (the following layer's input norm, or the final norm
     * after the last layer) into `h`, with that norm's gain `w_next`; the replaced norm is then
     * neither declared nor issued. */
    bool next_norm = false;
    rad_weight w_next = 0;
    /* the drafter's tap, on a layer the drafter reads (see declare) */
    rad_op op_tap = 0;
    /* Calibration: the attention's input (q, k and v share it), its output (this rank's columns,
     * o_proj's input), the dense arm's input and its gated product (this rank's columns, down's
     * input). */
    CalibTap tap_in{}, tap_o{}, tap_gu{}, tap_dn{};
};

/* ---- Google's assistant MTP head: ported from arch/gemma4_bf16, the drafter the dense Gemma 4
 * plugins serve. It reads the trunk's own cache (no cache of its own) and proposes up to
 * kHeadDepth continuations, declared only when the container carries `draft.` Gemma4Assistant
 * weights (rad-convert --draft-model) and the deployment speculates. ---- */

struct ClassGeom {
    int64_t n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int64_t window = 0;
    double  theta  = 1.0;
    float   scale  = 1.0f;
    bool    head_ladder = false;   /* true: proportional over head_dim (global) */
    bool    has_v       = true;    /* false: k == v (global) */
    /* Fewer KV heads than ranks: each rank keeps the one at row `kv_off` of the n_head_kv_all,
     * shared with the other world / n_head_kv_all ranks of its group. */
    bool    kv_rep      = false;
    int64_t kv_off = 0, n_head_kv_all = 0;
    int64_t q_dim()  const { return n_head * head_dim; }
    int64_t kv_dim() const { return n_head_kv * head_dim; }
    int64_t gqa()    const { return n_head / n_head_kv; }


};

/* ---- one linear, plain bf16 or w4a8. The recipe quantises GEMM weights to int4 codes with an
 * f16 scale per [1 x 128]; a weight no rule matches stays bf16. The weight's encoding picks the
 * op: `gemm_nt` for plain bf16, `gemm_nt_q` at w4a8 for codes + scale. w4a8 reads an int8
 * activation and an f32 per-row scale, written by a quant_act_i8 declared here over the bf16
 * activation, in the GEMM's fragment byte order. Not quant_act_i8g (row-major, per-(row, group)
 * scale; i8a8's producer), and not w4a16, whose row expects an f16 activation and does not check
 * the operand dtype, so bf16 bits would be misread as f16. The name map is made before the weight
 * is queried (Builder::finish() needs the map entry before the declaration). */
struct G4Linear {
    rad_weight w = 0, ws = 0;   /* the codes + scale (w4a8), or the single bf16 weight */
    rad_op     op = 0, op_q = 0;   /* the quantised GEMM and, for w4a8, its activation quantiser */
    rad_buf    q8 = 0, s8 = 0;     /* the int8 pair (i8 codes + f32 row scale) the quantiser writes */
    int64_t    N = 0, K = 0;
    bool       quant = false;   /* true: w4a8 / w8a8 over a per-row int8 activation */
    /* true: i8a8 as arch/gemma3's G3Linear: int8 codes with a bf16 scale per [1 x 128] (libquant
     * `codes=i8 group=128 scale=bf16`) over quant_act_i8g, an activation scale per (row, 128).
     * Gemma 4 E4B's activations carry a few channels 40-200 times the row's median (the down
     * projection's input most), and one int8 scale a row spends its codes on them. Its junctions
     * are the `_i8g` twins (see "Residual junctions" above). */
    bool       i8g = false;
    /* true: asymmetric int4 as arch/gemma3's G3Linear: u4 codes, a u8 zero point and an f16 scale
     * per [1 x 128] (libquant `codes=u4 zero=u8 group=128 scale=f16`), gemm_nt_q at w4a8i8_asym
     * over the same per-(row, 128) int8 activation as i8g. */
    bool       a4 = false;
    /* true with a4: the same grid at five bits, u5 codes and a zero point 0..31 (libquant
     * `codes=u5 zero=u8 group=128 scale=f16`), ParoQuant int5's grid with no rotation. Read by
     * w5a8i8 and w5a16 where a4 reads w4a8i8_asym and w4a16_asym, over the same activations. */
    bool       w5 = false;
    bool       grouped() const { return i8g || a4; }
    /* reads an int8 pair a junction can write, at either grid */
    bool       pair() const { return quant || grouped(); }
    /* the pair's scales a row: one, or one per 128 columns */
    int64_t    scale_w() const { return grouped() ? K / RAD_SCALE_GROUP : 1; }
    rad_buf    in = 0;            /* the bf16 input row `a` given at declare */
    /* a4 at decode heights, as LinearQ's w4 (rad_linear.h): up to kA4Split rows the bf16 input is
     * cast to f16 (`a16`) for w4a16_asym's skinny kernel; w4a8i8_asym has only a prefill kernel,
     * which at one row does 16 times the matrix work and reads its weight a quarter as fast. */
    static constexpr int64_t kA4Split = 64;
    int64_t    a4_split = 0;      /* rows up to which op_dec serves; 0: no decode band */
    rad_buf    a16 = 0;
    rad_op     op_a16 = 0, op_dec = 0;
    /* Reads another linear's int8 pair. q, k and v quantise the same norm output on the same
     * per-row grid, so with `in_from` set at declare this linear reads that linear's pair and
     * declares no quantiser of its own: one dispatch instead of three. */
    bool       shared_in = false;

    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 std::initializer_list<const char*> src,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 rad_buf a, rad_buf y,
                 std::initializer_list<int64_t> row_parts = {},
                 const G4Linear* in_from = nullptr);
    void step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const;
    /* the GEMM alone, for a caller that wrote the int8 pair itself (a fused norm + quantiser); an
     * a4 linear's decode band reads the bf16 input row instead, which that caller then wrote */
    void gemm(RadCtx* c, rad_buf y, int64_t T) const;
    /* The same two over rows [r0, r0 + n) of the step: a slice of the prefill overlap. The
     * quantiser and the GEMM are a row at a time above their bands, so a slice's rows are the
     * whole step's (see "the prefill overlap" below). */
    void step_rows(RadCtx* c, rad_buf a, rad_buf y, int64_t r0, int64_t n) const;
    void gemm_rows(RadCtx* c, rad_buf y, int64_t r0, int64_t n) const;
    /* This rank's rows of each row part where they are not an even share (a replicated KV head),
     * on the weight and on its scale plane: every grid here keeps one scale row a weight row. */
    int  span_parts(RadBuilder* b, const int64_t* lo, const int64_t* hi, const int64_t* total,
                    int n) const {
        for (rad_weight x : { w, ws }) {
            if (!x) continue;
            RAD_ARCH_TRY(n == 1 ? rad_weight_shard_span(b, x, lo[0], hi[0])
                                : rad_weight_shard_span_parts(b, x, lo, hi, total, n));
        }
        return RAD_OK;
    }
};

/* A weight on the w4a8 / w8a8 grid: i4 or i8 codes with an f16 scale per [1 x 128], the encoding
 * gemm_nt_q reads against quant_act_i8's pair. */
static bool g4_grid_w4(const RadEncoding& enc) {
    const RadEncPlane* sc = rad_enc_plane(&enc, "scale");
    const uint32_t cdt = enc.n_planes > 0 ? enc.plane[0].dtype : 0;
    return rad_enc_is(&enc, "affine") && enc.n_planes == 2 && !enc.transform[0] &&
           (cdt == RAD_I4 || cdt == RAD_I8) && enc.plane[0].block[1] == 1 &&
           sc && sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == 128;
}

/* An empty `src` means the caller made the weight's name map (a probe asked it first). */
inline int G4Linear::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                            std::initializer_list<const char*> src,
                            int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                            rad_buf a, rad_buf y,
                            std::initializer_list<int64_t> row_parts,
                            const G4Linear* in_from) {
    N = n_out; K = n_in; in = a;
    if (N <= 0 || K <= 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    if (src.size() == 1)     RAD_ARCH_TRY(map_copy(b, wn, *src.begin()));
    else if (src.size() > 1) RAD_ARCH_TRY(map_concat(b, wn, src, 0));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    /* A builder with no model behind it (a test, a synthetic tool) answers RAD_E_NOTFOUND, and
     * the weight is read as plain bf16. */
    const bool plain = !known || (rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                                  enc.plane[0].dtype == RAD_BF16);
    int64_t parts[RAD_MAX_ROW_PARTS] = {0};
    int np = 0;
    for (int64_t p : row_parts) if (np < RAD_MAX_ROW_PARTS) parts[np++] = p;

    if (plain) {
        w = decl_w(b, wn, RAD_BF16, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, 0, row_parts);
        if (!w) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "gemm_nt",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w)),
                { a }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* A w4 weight: i4 codes and an f16 scale per [1 x 128]. gemm_nt_q at w4a8 reads an int8
     * activation and an f32 per-row scale, so a quantiser over the bf16 plane `a` writes the int8
     * pair (q8, s8). The codes + scale layout is the same as w4a16's, so containers serve
     * unchanged. An 8-bit weight on the same grid (i8 codes, f16 scale per [1 x 128]): gemm_nt_q
     * at w8a8 reads the same int8 activation and per-row scale as w4a8, so the quantiser is
     * shared. A recipe uses it for linears int4 hurts (attention, down_proj). */
    const uint32_t cdt = known && enc.n_planes > 0 ? enc.plane[0].dtype : 0;
    i8g = known && linear_is_w8(enc);
    {
        const RadEncPlane* sp = rad_enc_plane(&enc, "scale");
        const RadEncPlane* zp = rad_enc_plane(&enc, "zero");
        a4 = known && rad_enc_is(&enc, "affine") && enc.n_planes == 3 && !enc.transform[0] &&
             (cdt == RAD_U4 || cdt == RAD_U5) && enc.plane[0].block[1] == 1 && sp &&
             sp->dtype == RAD_F16 && sp->block[0] == 1 && sp->block[1] == RAD_SCALE_GROUP && zp &&
             zp->dtype == RAD_U8 && zp->block[0] == 1 && zp->block[1] == RAD_SCALE_GROUP;
        w5 = a4 && cdt == RAD_U5;
    }
    if (i8g || a4) {
        if (N % 16 || K % RAD_SCALE_GROUP) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; it takes rows in 16s and K in "
                            "whole %d-column activation scale groups\n", base, (long long)N,
                    (long long)K, RAD_SCALE_GROUP);
            return RAD_E_INVAL;
        }
        if (a4) {
            w  = decl_view(b, wn, nullptr, "codes", w5 ? RAD_U5 : RAD_U4, { N, K }, RAD_ACCESS_PER_TOKEN,
                           shard, grp, parts, np);
            ws = decl_view(b, nm.f("%s.scale", base), wn, "scale,zero", RAD_F16, { N, K },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        } else {
            w  = decl_view(b, wn, nullptr, "codes", RAD_I8, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            ws = decl_view(b, nm.f("%s.scale", base), wn, "scale", RAD_BF16, { N, K / RAD_SCALE_GROUP },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        }
        if (!w || !ws) return RAD_E_INVAL;
        /* not an a4 linear's pair: below its decode band it writes none */
        if (in_from && in_from->grouped() && !in_from->a4 && in_from->q8 && in_from->K == K) {
            q8 = in_from->q8;
            s8 = in_from->s8;
            shared_in = true;
        } else {
            q8 = decl_b(b, nm.f("%s.q8", base), RAD_I8,  { g.max_tok, K });
            s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, K / RAD_SCALE_GROUP });
            if (!q8 || !s8) return RAD_E_INVAL;
            op_q = rw(b, RAD_OP(b, "quant_act_i8g",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", RAD_SCALE_GROUP), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { q8, s8 });
            if (!op_q) return RAD_E_INVAL;
        }
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_INT("group", RAD_SCALE_GROUP),
                               RAD_STR("dtype", w5 ? "w5a8i8" : a4 ? "w4a8i8_asym" : "i8a8")),
                    RAD_WEIGHTS(w, ws)),
                { q8, s8 }, { y });
        if (!op) return RAD_E_INVAL;
        if (a4) {
            a4_split = std::min<int64_t>(kA4Split, g.max_tok);
            a16 = decl_b(b, nm.f("%s.a16", base), RAD_F16, { a4_split, K });
            if (!a16) return RAD_E_INVAL;
            op_a16 = rw(b, RAD_OP(b, "cast",
                           RAD_PARAMS(RAD_RANGE("M", 1, a4_split), RAD_INT("n", K),
                                      RAD_STR("from", g.dtype), RAD_STR("to", "f16")),
                           RAD_NOWEIGHTS),
                        { a }, { a16 });
            op_dec = rw(b, RAD_OP(b, "gemm_nt_q",
                            RAD_PARAMS(RAD_RANGE("M", 1, a4_split), RAD_INT("N", N), RAD_INT("K", K),
                                       RAD_INT("group", RAD_SCALE_GROUP),
                                       RAD_STR("dtype", w5 ? "w5a16" : "w4a16_asym")),
                            RAD_WEIGHTS(w, ws)),
                        { a16 }, { y });
            if (!op_a16 || !op_dec) return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    if (known && g4_grid_w4(enc)) {
        if (N % 16 || K % 128) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; a w4 weight takes rows in "
                            "16s and K in whole 128-column scale groups.\n", base,
                    (long long)N, (long long)K);
            return RAD_E_INVAL;
        }
        const char* sn = nm.f("%s.scale", base);
        w  = decl_view(b, wn, nullptr, "codes", cdt, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        ws = decl_view(b, sn, wn, "scale", RAD_F16, { N, K / 128 }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        if (!w || !ws) return RAD_E_INVAL;
        if (in_from && in_from->quant && in_from->q8 && in_from->K == K) {
            q8 = in_from->q8;
            s8 = in_from->s8;
            shared_in = true;
        } else {
            q8 = decl_b(b, nm.f("%s.q8", base), RAD_I8,  { g.max_tok, K });
            s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, 1 });
        }
        if (!q8 || !s8) return RAD_E_INVAL;
        /* quant_act_i8, not quant_act_i8g: the w4a8 and w8a8 GEMMs read codes in the int8 WMMA's
         * step order with one f32 scale a row (docs/OPS.md). quant_act_i8g writes row-major codes
         * with a scale per (row, group) for i8a8; paired with these GEMMs it produces NaN logits.
         * `group` only sizes the optional code sums, which this symmetric grid does not take. */
        if (!shared_in)
            op_q = rw(b, RAD_OP(b, "quant_act_i8",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", 128), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { q8, s8 });
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_INT("group", 128), RAD_STR("dtype", cdt == RAD_I8 ? "w8a8" : "w4a8")),
                    RAD_WEIGHTS(w, ws)),
                { q8, s8 }, { y });
        quant = true;
        return ((op_q || shared_in) && op) ? RAD_OK : RAD_E_INVAL;
    }

    char en[256];
    rad_enc_format(&enc, en, sizeof en);
    fprintf(stderr, "radiance: '%s' is %s, and this model's linears are plain bf16, w4a8, w8a8, "
                    "int8 a 128 (i8a8) or asymmetric int4 or int5 a 128 -- none matches\n", wn, en);
    return RAD_E_DTYPE;
}

inline void G4Linear::step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const {
    if (quant) {
        /* Quantise the bf16 activation into the int8 pair, then run the GEMM. quant_act_i8 takes
         * x, q, scale, asum; `asum` (the per-group row sums asymmetric grids subtract) is absent on
         * this symmetric grid and passed as a null tensor, since operands are positional. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T), RAD_NONE);
        gemm(c, y, T);
    } else if (a4 && T <= a4_split) {
        /* the decode band: f16 copy of the input, the skinny kernel; operands as LinearQ's w4 */
        RAD_ISSUE_N(c, op_a16, T, brows(a, T), brows(a16, T));
        RAD_ISSUE_N(c, op_dec, T, brows(a16, T), RAD_NONE, RAD_W(w), RAD_W(ws), brows(y, T),
                    RAD_NONE, RAD_NONE);
    } else if (grouped()) {
        /* quant_act_i8g takes x, q, scale: no optional `asum`. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T));
        gemm(c, y, T);
    } else
        RAD_ISSUE(c, op, brows(a, T), RAD_W(w), brows(y, T));
}

inline void G4Linear::gemm(RadCtx* c, rad_buf y, int64_t T) const {
    if (a4 && T <= a4_split) {
        /* the decode band reads no pair: the junction wrote the norm output into `in` instead */
        RAD_ISSUE_N(c, op_a16, T, brows(in, T), brows(a16, T));
        RAD_ISSUE_N(c, op_dec, T, brows(a16, T), RAD_NONE, RAD_W(w), RAD_W(ws), brows(y, T),
                    RAD_NONE, RAD_NONE);
        return;
    }
    RAD_ISSUE_N(c, op, T, brows(q8, T), brows(s8, T), RAD_W(w), RAD_W(ws),
                brows(y, T), RAD_NONE, RAD_NONE);
}

inline void G4Linear::step_rows(RadCtx* c, rad_buf a, rad_buf y, int64_t r0, int64_t n) const {
    if (a4 && n <= a4_split) {
        RAD_ISSUE_N(c, op_a16, n, brow_slice(a, r0, n, K), brows(a16, n));
        RAD_ISSUE_N(c, op_dec, n, brows(a16, n), RAD_NONE, RAD_W(w), RAD_W(ws),
                    brow_slice(y, r0, n, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (!quant && !grouped()) {
        RAD_ISSUE_N(c, op, n, brow_slice(a, r0, n, K), RAD_W(w), brow_slice(y, r0, n, N));
        return;
    }
    if (!shared_in && grouped())
        RAD_ISSUE_N(c, op_q, n, brow_slice(a, r0, n, K), brow_slice(q8, r0, n, K),
                    brow_slice(s8, r0, n, scale_w()));
    else if (!shared_in)
        RAD_ISSUE_N(c, op_q, n, brow_slice(a, r0, n, K), brow_slice(q8, r0, n, K),
                    brow_slice(s8, r0, n, 1), RAD_NONE);
    gemm_rows(c, y, r0, n);
}

/* A slice of the prefill overlap only, which is above every decode band (kOverlapRows). */
inline void G4Linear::gemm_rows(RadCtx* c, rad_buf y, int64_t r0, int64_t n) const {
    RAD_ISSUE_N(c, op, n, brow_slice(q8, r0, n, K), brow_slice(s8, r0, n, scale_w()), RAD_W(w),
                RAD_W(ws), brow_slice(y, r0, n, N), RAD_NONE, RAD_NONE);
}


static constexpr int kHeadPosOffset = 0;
static constexpr int kHeadDepth     = 3;


/* The head's per-class geometry, from the trunk's Kind: the head reads the trunk's cache, so its
 * attention must match the trunk's -- the head width (head_dim), the KV head count, the window and
 * the rope (full-width neox over the sliding layers, the proportional "head" ladder over the full
 * layers, K = V). n_head is the trunk's query heads; the head overrides it to its own. */
static ClassGeom class_from_kind(const Kind& k, bool full) {
    ClassGeom c{};
    c.n_head = k.nh;
    c.n_head_kv = k.nkv;
    c.head_dim = k.hd;
    c.n_rot = k.rot_pairs * 2;
    c.window = full ? 0 : k.window;
    c.theta = k.theta;
    c.scale = 1.0f;
    c.head_ladder = full;
    c.has_v = !k.k_eq_v;
    c.kv_rep = k.kv_repl;
    c.kv_off = k.kv_off;
    c.n_head_kv_all = k.nkv_all;
    return c;
}

static int g4_parse_types(const char* lt, int64_t n, std::vector<int>& is_global) {
    RAD_ARCH_TRY(parse_layer_kinds("gemma4", lt, {"full_attention"}, "sliding_attention",
                                   &is_global));
    if ((int64_t)is_global.size() != n) {
        fprintf(stderr, "radiance: gemma4: layer_types names %zu layers and the model has %lld\n",
                is_global.size(), (long long)n);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}


struct Gemma4Head {
    struct HLayer {
        bool        global = false;
        int         trunk_layer = 0;   /* the trunk layer whose cache this layer reads */
        rad_kvgroup kv = 0;
        ClassGeom   cls{};
        rad_buf     q = 0, attn = 0;   /* the class's own rows: the two widths differ */
        rad_weight  w_norm = 0, w_qn = 0, w_post = 0, w_fnorm = 0, w_fpost = 0, w_scalar = 0;
        G4Linear    ln_q{}, ln_o{}, ln_gu{}, ln_down{};
        rad_op      op_norm = 0, op_qn = 0, op_rq = 0, op_attn = 0, op_ar = 0, op_post = 0,
                    op_add = 0, op_fnorm = 0, op_act = 0, op_far = 0, op_fpost = 0, op_fadd = 0,
                    op_scale = 0;
    };

    Geom    g{};            /* the trunk's, with max_tok cut to the head's rows */
    int64_t n = 0;          /* the head's width (1024 on the 31B) */
    int64_t n_trunk = 0;    /* the trunk's width, which the two projections join */
    int64_t n_ff = 0;       /* this rank's share of the feed-forward */
    int64_t rows = 0;       /* the most rows one pass carries: one per drafting sequence */
    double  softcap = 0.0;  /* 0: none, which is what the 31B's head says */
    std::vector<HLayer> layers;

    rad_buf src = 0;        /* the trunk's final-normed hidden state, [max_tok, n_trunk] */
    rad_buf logits = 0;     /* the trunk's logits plane: the sampler reads the head's from it */
    rad_buf b_c = 0, b_e = 0, b_x = 0, b_h = 0, b_gu = 0, b_ffn = 0, b_hd = 0, b_hn = 0;
    rad_weight w_tok = 0, w_norm = 0, w_lm = 0;
    G4Linear   ln_pre{}, ln_post{};
    rad_op op_hsrc = 0, op_hcopy = 0, op_embed = 0, op_embed_ar = 0, op_ecopy = 0, op_norm = 0,
           op_logits = 0, op_cap = 0;

    int  declare(RadBuilder* b, Names& nm, const RadModelMeta* meta, const Geom& trunk,
                 const ClassGeom& cs, const ClassGeom& cg, rad_kvgroup kv_s, rad_kvgroup kv_g,
                 int last_sliding, int last_full, int64_t n_main, rad_weight tok_embd,
                 rad_buf trunk_h, rad_buf trunk_logits);
    void step(RadCtx* c, const RadBatch* batch) const;
};

/* Does the container carry an assistant head at all? The 12B's is the same head under another
 * name (Gemma4UnifiedAssistantForCausalLM). */
static bool has_assistant(const RadModelMeta* m) {
    const char* a = rad_meta_gets(m, "draft.architectures", "");
    return a && (std::strstr(a, "Gemma4Assistant") || std::strstr(a, "Gemma4UnifiedAssistant"));
}

int Gemma4Head::declare(RadBuilder* b, Names& nm, const RadModelMeta* meta, const Geom& trunk,
                        const ClassGeom& cs, const ClassGeom& cg, rad_kvgroup kv_s,
                        rad_kvgroup kv_g, int last_sliding, int last_full, int64_t n_main,
                        rad_weight tok_embd, rad_buf trunk_h, rad_buf trunk_logits) {
    g = trunk;
    rows = trunk.max_seqs > 0 ? trunk.max_seqs : 1;
    g.max_tok = rows;
    src = trunk_h;
    logits = trunk_logits;
    w_tok = tok_embd;

    /* ---- the head's own geometry, and what it must share with the trunk's. */
    n       = meta_text_geti(meta, "hidden_size", 0, "draft.");
    n_trunk = rad_meta_geti(meta, "draft.backbone_hidden_size", 0);
    const long long n_l      = meta_text_geti(meta, "num_hidden_layers", 0, "draft.");
    const long long n_h      = meta_text_geti(meta, "num_attention_heads", 0, "draft.");
    const long long n_kv     = meta_text_geti(meta, "num_key_value_heads", 0, "draft.");
    const long long n_kv_g   = meta_text_geti(meta, "num_global_key_value_heads", n_kv, "draft.");
    const long long head_s   = meta_text_geti(meta, "head_dim", 0, "draft.");
    const long long head_g   = meta_text_geti(meta, "global_head_dim", head_s, "draft.");
    const long long ff       = meta_text_geti(meta, "intermediate_size", 0, "draft.");
    const long long win      = meta_text_geti(meta, "sliding_window", 0, "draft.");
    const long long vocab    = meta_text_geti(meta, "vocab_size", 0, "draft.");
    const long long ordered  = rad_meta_geti(meta, "draft.use_ordered_embeddings", 0);
    const double    eps      = meta_text_getf(meta, "rms_norm_eps", 0.0, "draft.");
    const double    theta_s  = meta_text_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0, "draft.");
    const double    theta_g  = meta_text_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0, "draft.");
    double          prf      = meta_text_getf(meta, "rope_parameters.full_attention.partial_rotary_factor", -1.0, "draft.");
    softcap = meta_text_getf(meta, "final_logit_softcapping", 0.0, "draft.");
    if (!(prf > 0.0)) prf = 1.0;

    if (n <= 0 || n_l <= 0 || n_h <= 0 || ff <= 0 || !(eps > 0.0)) {
        fprintf(stderr, "radiance: gemma4 assistant: the `draft.` config is incomplete (hidden %lld, "
                        "layers %lld, heads %lld, ffn %lld, eps %g)\n",
                (long long)n, n_l, n_h, ff, eps);
        return RAD_E_INVAL;
    }
    /* The head reads the trunk's cache, so everything the cache's shape depends on must agree:
     * the two head widths, the two kv head counts and the window. A head that disagrees was trained
     * with another trunk. The counts are the model's, not a rank's: a replicated kv head (fewer
     * than ranks) is one head on every rank. */
    if (n_trunk != trunk.n_embd || vocab != trunk.n_vocab_all || head_s != cs.head_dim ||
        head_g != cg.head_dim || n_kv != cs.n_head_kv_all || n_kv_g != cg.n_head_kv_all ||
        win != cs.window) {
        fprintf(stderr, "radiance: gemma4 assistant: this head was not made for this trunk -- it "
                        "states width %lld, vocabulary %lld, heads %lld/%lld x %lld/%lld kv over a "
                        "window of %lld; the trunk's cache is %lld/%lld kv heads of %lld/%lld over "
                        "%lld, at width %lld and vocabulary %lld\n",
                (long long)n_trunk, vocab, n_kv, n_kv_g, head_s, head_g, win,
                (long long)cs.n_head_kv_all, (long long)cg.n_head_kv_all,
                (long long)cs.head_dim, (long long)cg.head_dim, (long long)cs.window,
                (long long)trunk.n_embd, (long long)trunk.n_vocab_all);
        return RAD_E_INVAL;
    }
    /* The queries must rotate as the trunk's keys did, or q.k measures the wrong distance. */
    const int64_t rot_g = (int64_t)(head_g * prf + 0.5);
    if (theta_s != cs.theta || theta_g != cg.theta || rot_g != cg.n_rot) {
        fprintf(stderr, "radiance: gemma4 assistant: its rotary (theta %g / %g, %lld of %lld global "
                        "dims) is not the trunk's (%g / %g, %lld): its queries would meet the "
                        "trunk's keys at the wrong angle\n",
                theta_s, theta_g, (long long)rot_g, head_g, cs.theta, cg.theta,
                (long long)cg.n_rot);
        return RAD_E_INVAL;
    }
    /* Centroid masking (use_ordered_embeddings) scores a top-k of vocabulary clusters instead of
     * the whole head. The 31B's head does not use it and it is not implemented, so it is refused. */
    if (ordered) {
        fprintf(stderr, "radiance: gemma4 assistant: use_ordered_embeddings (centroid-masked logits) "
                        "is not implemented\n");
        return RAD_E_UNSUPPORTED;
    }
    int64_t nh = 0;
    if (divide_or_fail(n_h, trunk.world, "draft num_attention_heads", &nh) < 0) return RAD_E_INVAL;
    if (divide_or_fail(ff, trunk.world, "draft intermediate_size", &n_ff) < 0) return RAD_E_INVAL;
    if (nh % cs.n_head_kv || nh % cg.n_head_kv) {
        fprintf(stderr, "radiance: gemma4 assistant: %lld query heads a rank do not group over %lld "
                        "sliding / %lld global kv heads\n",
                (long long)nh, (long long)cs.n_head_kv, (long long)cg.n_head_kv);
        return RAD_E_INVAL;
    }

    /* The layer schedule: `layer_types` names every layer, as the trunk's does. */
    std::vector<int> is_global;
    {
        const char* lt = rad_meta_gets(meta, "draft.text_config.layer_types",
                                       rad_meta_gets(meta, "draft.layer_types", nullptr));
        if (!lt || !*lt || g4_parse_types(lt, n_l, is_global) < 0) {
            fprintf(stderr, "radiance: gemma4 assistant: the `draft.` config's layer_types does not "
                            "name its %lld layers\n", n_l);
            return RAD_E_INVAL;
        }
    }

    const int64_t base = n_main;   /* the head's weight groups, past the trunk's layers */
    g.eps = (float)eps;

    /* ---- the rows. The two classes keep their own query and attention rows, as in the trunk. */
    ClassGeom hs = cs; hs.n_head = nh;
    ClassGeom hgc = cg; hgc.n_head = nh;
    b_c   = decl_b(b, nm.f("mtp_c"),   g.act_dtype, {rows, 2 * n_trunk});
    b_e   = decl_b(b, nm.f("mtp_e"),   g.act_dtype, {rows, n_trunk});
    b_x   = decl_b(b, nm.f("mtp_x"),   g.act_dtype, {rows, n});
    b_h   = decl_b(b, nm.f("mtp_h"),   g.act_dtype, {rows, n});
    b_gu  = decl_b(b, nm.f("mtp_gu"),  g.act_dtype, {rows, 2 * n_ff});
    b_ffn = decl_b(b, nm.f("mtp_ffn"), g.act_dtype, {rows, n_ff});
    b_hd  = decl_b(b, nm.f("mtp_hd"),  g.act_dtype, {rows, n});
    b_hn  = decl_b(b, nm.f("mtp_hn"),  g.act_dtype, {rows, n_trunk});
    const rad_buf q_s = decl_b(b, nm.f("mtp_qs"), g.act_dtype, {rows, hs.q_dim()});
    const rad_buf a_s = decl_b(b, nm.f("mtp_as"), g.act_dtype, {rows, hs.q_dim()});
    const rad_buf q_g = decl_b(b, nm.f("mtp_qg"), g.act_dtype, {rows, hgc.q_dim()});
    const rad_buf a_g = decl_b(b, nm.f("mtp_ag"), g.act_dtype, {rows, hgc.q_dim()});

    /* ---- the entry: the trunk's hidden state into the right half of `c`, the trunk's scaled
     * embedding into the left. A history pass gathers the hidden state from the trunk's rows; a
     * round copies the head's own h' from the previous round. Both write the half in place, since
     * gather and cast take the destination's row pitch from the operand. */
    op_hsrc = rw(b, RAD_OP(b, "gather_rows",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
                 {src}, {b_c});
    op_hcopy = rw(b, RAD_OP(b, "cast",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk),
                               RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                    RAD_NOWEIGHTS),
                  {b_hn}, {b_c});
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_embd", n_trunk),
                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                               RAD_INT("vocab_offset", g.vocab_off),
                               RAD_F64("mult", gemma4::embed_scale(n_trunk))),
                    RAD_WEIGHTS(w_tok)),
                  {}, {b_e});
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                             RAD_PARAMS(RAD_INT("world_size", g.world),
                                        RAD_RANGE("numel", n_trunk, rows * n_trunk),
                                        RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                             RAD_NOWEIGHTS),
                         {b_e}, {b_e});
    op_ecopy = rw(b, RAD_OP(b, "cast",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk),
                               RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                    RAD_NOWEIGHTS),
                  {b_e}, {b_c});
    if (!op_hsrc || !op_hcopy || !op_embed || !op_ecopy) return RAD_E_INVAL;

    /* pre_projection: the joined row to the head's width. Replicated, like the stream it reads
     * and writes. */
    RAD_ARCH_TRY(ln_pre.declare(b, nm, g, nm.f("mtp.pre_proj"), { nm.ckpt("draft.pre_projection.weight") },
                                n, 2 * n_trunk, RAD_SHARD_NONE, grp_layer((int)(base + n_l)),
                                b_c, b_x));

    /* The head's n-wide norm (plain gain), residual add and all-reduce. */
    auto norm = [&](rad_weight w, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w)),
                  {x}, {y});
    };
    auto add = [&]() {
        return rw(b, RAD_OP(b, "add",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                  {b_x, b_h}, {b_x});
    };
    auto all_reduce = [&]() {
        return rw(b, RAD_OP(b, "all_reduce",
                     RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", n, rows * n),
                                RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                RAD_INT("min_bytes", g.wire_min_bytes)),
                     RAD_NOWEIGHTS),
                  {b_h}, {b_h});
    };

    /* ---- the layers. */
    layers.resize((size_t)n_l);
    for (int64_t l = 0; l < n_l; ++l) {
        HLayer& L = layers[(size_t)l];
        const int gl = (int)(base + l);
        L.global = is_global[(size_t)l] == 1;
        L.cls = L.global ? hgc : hs;
        L.trunk_layer = L.global ? last_full : last_sliding;
        L.kv = L.global ? kv_g : kv_s;
        L.q = L.global ? q_g : q_s;
        L.attn = L.global ? a_g : a_s;
        if (L.trunk_layer < 0) {
            fprintf(stderr, "radiance: gemma4 assistant: layer %lld reads a %s trunk layer and the "
                            "trunk has none\n", (long long)l, L.global ? "full" : "sliding");
            return RAD_E_INVAL;
        }
        const char* pre = "draft.model.layers";
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.input_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_q_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.self_attn.q_norm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_post_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.ffn_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.ffn_post_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.layer_scalar", (long long)l),
                              nm.ckpt("%s.%lld.layer_scalar", pre, (long long)l)));
        L.w_norm   = decl_w(b, nm.f("mtp.blk.%lld.attn_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_qn     = decl_w(b, nm.f("mtp.blk.%lld.attn_q_norm.weight", (long long)l), RAD_BF16,
                            {L.cls.head_dim}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_post   = decl_w(b, nm.f("mtp.blk.%lld.attn_post_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_fnorm  = decl_w(b, nm.f("mtp.blk.%lld.ffn_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_fpost  = decl_w(b, nm.f("mtp.blk.%lld.ffn_post_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_scalar = decl_w(b, nm.f("mtp.blk.%lld.layer_scalar", (long long)l), RAD_BF16, {1},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));

        RAD_ARCH_TRY(L.ln_q.declare(b, nm, g, nm.f("mtp.blk.%lld.attn_q", (long long)l),
                                    { nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l) },
                                    L.cls.q_dim(), n, RAD_SHARD_ROW, grp_layer(gl), b_h, L.q));
        RAD_ARCH_TRY(L.ln_o.declare(b, nm, g, nm.f("mtp.blk.%lld.attn_o", (long long)l),
                                    { nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l) },
                                    n, L.cls.q_dim(), RAD_SHARD_COL, grp_layer(gl), L.attn, b_h));
        RAD_ARCH_TRY(L.ln_gu.declare(b, nm, g, nm.f("mtp.blk.%lld.ffn_gate_up", (long long)l),
                                     { nm.ckpt("%s.%lld.mlp.gate_proj.weight", pre, (long long)l),
                                       nm.ckpt("%s.%lld.mlp.up_proj.weight",   pre, (long long)l) },
                                     2 * n_ff, n, RAD_SHARD_ROW, grp_layer(gl), b_h, b_gu,
                                     { n_ff, n_ff }));
        RAD_ARCH_TRY(L.ln_down.declare(b, nm, g, nm.f("mtp.blk.%lld.ffn_down", (long long)l),
                                       { nm.ckpt("%s.%lld.mlp.down_proj.weight", pre, (long long)l) },
                                       n, n_ff, RAD_SHARD_COL, grp_layer(gl), b_ffn, b_h));

        L.op_norm = norm(L.w_norm, b_x, b_h);
        L.op_qn = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows * L.cls.n_head), RAD_INT("n", L.cls.head_dim),
                                  RAD_F64("eps", eps), RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(L.w_qn)),
                     {L.q}, {L.q});
        L.op_rq = rw(b, RAD_OP(b, "rope",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", L.cls.head_dim),
                                  RAD_INT("n_head", L.cls.n_head), RAD_INT("n_head_kv", 0),
                                  RAD_F64("theta", L.cls.theta), RAD_F64("scale", L.cls.scale),
                                  RAD_STR("mode", "neox"), RAD_INT("rotary_dim", L.cls.n_rot),
                                  RAD_STR("ladder", L.cls.head_ladder ? "head" : "rotary")),
                       RAD_NOWEIGHTS),
                     {L.q}, {L.q});
        /* One query row per sequence over the trunk's cache, at scale 1 like the trunk. The
         * context is the committed positions (rad_drafter_shared_kv), so only the decode band
         * exists. */
        L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                         RAD_PARAMS(RAD_RANGE("q_len", 1, 1), RAD_INT("head_dim", L.cls.head_dim),
                                    RAD_INT("gqa", L.cls.gqa()), RAD_INT("causal", 1),
                                    RAD_INT("window", L.cls.window), RAD_INT("n_head", L.cls.n_head),
                                    RAD_INT("block_size", RAD_KV_BLOCK), RAD_INT("max_seqs", rows),
                                    RAD_INT("max_ctx", g.max_ctx), RAD_F64("scale", 1.0),
                                    RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                       {L.q}, {L.attn});
        if (g.world > 1) L.op_ar = all_reduce();
        L.op_post  = norm(L.w_post, b_h, b_h);
        L.op_add   = add();
        L.op_fnorm = norm(L.w_fnorm, b_x, b_h);
        L.op_act = rw(b, RAD_OP(b, "gelu_mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_ff), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                      {b_gu}, {b_ffn});
        if (g.world > 1) L.op_far = all_reduce();
        L.op_fpost = norm(L.w_fpost, b_h, b_h);
        L.op_fadd  = add();
        L.op_scale = rw(b, RAD_OP(b, "scale",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(L.w_scalar)),
                        {b_x}, {b_x});
        if (!L.op_norm || !L.op_qn || !L.op_rq || !L.op_attn || !L.op_post || !L.op_add ||
            !L.op_fnorm || !L.op_act || !L.op_fpost || !L.op_fadd || !L.op_scale ||
            (g.world > 1 && (!L.op_ar || !L.op_far)))
            return RAD_E_INVAL;
    }

    /* ---- the exit: the final norm, the head's own lm_head, and h' for the next round. */
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.output_norm.weight"), nm.ckpt("draft.model.norm.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.output.weight"), nm.ckpt("draft.model.embed_tokens.weight")));
    w_norm = decl_w(b, nm.f("mtp.output_norm.weight"), RAD_BF16, {n}, RAD_ACCESS_PER_TOKEN,
                    RAD_SHARD_NONE, grp_layer((int)(base + n_l)));
    /* Vocab-sharded like the trunk's: the draft's logits go through the same sampler chain, which
     * merges the ranks' candidates. Declared at its codes' dtype (bf16 as shipped, or int8 / block
     * fp8 where a recipe makes it so), selecting the matching logits_gemm kernel. It is a round's
     * largest read, and quantising it only affects proposals, which the trunk verifies. */
    w_lm = decl_w(b, nm.f("mtp.output.weight"), weight_codes_dtype(b, nm.f("mtp.output.weight"), RAD_BF16),
                  {g.n_vocab, n}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_layer((int)(base + n_l)));
    op_norm = norm(w_norm, b_x, b_hd);
    op_logits = rw(b, RAD_OP(b, "logits_gemm",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_vocab", g.n_vocab),
                                RAD_INT("n_embd", n), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_lm)),
                   {b_hd}, {logits});
    if (softcap > 0.0)
        op_cap = rw(b, RAD_OP(b, "logit_softcap",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_vocab),
                                 RAD_F64("cap", softcap), RAD_STR("dtype", "f32")),
                      RAD_NOWEIGHTS),
                    {logits}, {logits});
    if (!op_norm || !op_logits || (softcap > 0.0 && !op_cap)) return RAD_E_INVAL;
    /* post_projection last: h' is the next round's input, read by the head's first op
     * (op_hcopy). Declared here, its buffer lives across the whole head, so nothing in a pass can
     * overwrite it before the next round reads it. */
    RAD_ARCH_TRY(ln_post.declare(b, nm, g, nm.f("mtp.post_proj"), { nm.ckpt("draft.post_projection.weight") },
                                 n_trunk, n, RAD_SHARD_NONE, grp_layer((int)(base + n_l)),
                                 b_hd, b_hn));

    rad_note(b, "the assistant head is declared: %lld layer(s) of width %lld at absolute %lld..%lld, "
                "reading the trunk's cache at layer %d (sliding) and %d (full); no cache of its own, "
                "every round at the committed position",
             (long long)n_l, (long long)n, (long long)base, (long long)(base + n_l - 1),
             last_sliding, last_full);
    return RAD_OK;
}

/* One pass, over the batch staged for a head with no cache of its own: one row per drafting
 * sequence, at the committed position, with the context ending there. */
void Gemma4Head::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    if (batch->draft_pass < 0) {
        /* Every row must be a drafting row, as rad_drafter_shared_kv requested; anything else
         * was built for a head with its own cache. */
        if (batch->n_draft_out != T || batch->n_out != T) {
            rad_step_fail(c, "the gemma4 assistant head was handed a history pass that is not one row "
                             "a drafting sequence; it keeps no cache of its own and drafts from that "
                             "shape only (rad_drafter_shared_kv)");
            return;
        }
        RAD_ISSUE_N(c, op_hsrc, T, RAD_B(src), praw(batch->out_ids, RAD_I32, T),
                    bcol(b_c, n_trunk, n_trunk, T));
    } else {
        RAD_ISSUE_N(c, op_hcopy, T, brows(b_hn, T), bcol(b_c, n_trunk, n_trunk, T));
    }
    RAD_ISSUE(c, op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(b_e, T));
    if (op_embed_ar) RAD_ISSUE_N(c, op_embed_ar, T * n_trunk, brows(b_e, T), RAD_NONE);
    RAD_ISSUE_N(c, op_ecopy, T, brows(b_e, T), bcol(b_c, 0, n_trunk, T));
    ln_pre.step(c, b_c, b_x, T);

    for (const HLayer& L : layers) {
        const RadKVGroupBatch* kvb = kv_batch(batch, L.kv);
        RAD_ISSUE(c, L.op_norm, brows(b_x, T), RAD_W(L.w_norm), brows(b_h, T));
        L.ln_q.step(c, b_h, L.q, T);
        RAD_ISSUE_N(c, L.op_qn, T * L.cls.n_head, brows(L.q, T), RAD_W(L.w_qn), brows(L.q, T));
        RAD_ISSUE(c, L.op_rq, brows(L.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE_N(c, L.op_attn, batch->max_q_len,
                    brows(L.q, T), kv_cache(L.kv, L.trunk_layer),
                    praw2(kvb ? kvb->block_table : nullptr, RAD_I32, batch->n_seq,
                          kvb ? kvb->block_table_pitch : 0),
                    praw(kvb ? kvb->seqused : nullptr, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(L.attn, T));
        L.ln_o.step(c, L.attn, b_h, T);
        if (L.op_ar) RAD_ISSUE_N(c, L.op_ar, T * n, brows(b_h, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_post, T, brows(b_h, T), RAD_W(L.w_post), brows(b_h, T));
        RAD_ISSUE(c, L.op_add, brows(b_x, T), brows(b_h, T), brows(b_x, T));

        RAD_ISSUE(c, L.op_fnorm, brows(b_x, T), RAD_W(L.w_fnorm), brows(b_h, T));
        L.ln_gu.step(c, b_h, b_gu, T);
        RAD_ISSUE(c, L.op_act, brows(b_gu, T), brows(b_ffn, T));
        L.ln_down.step(c, b_ffn, b_h, T);
        if (L.op_far) RAD_ISSUE_N(c, L.op_far, T * n, brows(b_h, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_fpost, T, brows(b_h, T), RAD_W(L.w_fpost), brows(b_h, T));
        RAD_ISSUE(c, L.op_fadd, brows(b_x, T), brows(b_h, T), brows(b_x, T));
        RAD_ISSUE(c, L.op_scale, brows(b_x, T), RAD_W(L.w_scalar), brows(b_x, T));
    }

    RAD_ISSUE(c, op_norm, brows(b_x, T), RAD_W(w_norm), brows(b_hd, T));
    RAD_ISSUE_N(c, op_logits, T, brows(b_hd, T), RAD_W(w_lm), brows(logits, T));
    if (op_cap) RAD_ISSUE_N(c, op_cap, T, brows(logits, T), brows(logits, T));
    ln_post.step(c, b_hd, b_hn, T);
}


struct Model {
    Names   nm{""};
    Geom    g{};             /* the sliding kind's geometry, and everything model-wide */
    Kind    ks{}, kf{};      /* sliding, full */
    int64_t n_layer = 0, E = 0;
    int64_t n_ff = 0;        /* dense arm, this rank */
    bool    moe = false;
    int64_t n_expert = 0, n_local = 0, expert_base = 0, top_k = 0, n_ffe = 0;
    int64_t moe_rows = 0;    /* tokens a routed pass */
    /* A prefill chunk of 2 * moe_rows tokens or more runs as two slices, the trunk's all-reduces
     * on the second lane beside the other slice's compute (trunk_overlap). Set at declare. */
    bool    overlap = false;
    /* In a step that runs whole -- every decode and verify step -- the dense arm rides the second
     * lane beside the routed arm (trunk). Set at declare. */
    bool    arms = false;
    double  escale = 1.0, softcap = 0.0;

    rad_buf b_x = 0, b_h = 0, b_o = 0;
    rad_buf b_qkv = 0, b_qs = 0, b_atts = 0;               /* sliding; q_s where q is lifted */
    int     qrow = -1;               /* attn_paged_qrow resolved: -1 not asked yet, then 0 or 1 */
    rad_buf b_qf = 0, b_kf = 0, b_vf = 0, b_attf = 0;      /* full */
    rad_buf b_ff2 = 0, b_gu = 0, b_act = 0;                /* the arms' join; the dense arm */
    rad_buf b_rn = 0, b_logit = 0, b_ids = 0, b_ew = 0, b_es = 0, b_hx = 0;
    rad_buf b_sorted = 0, b_eoff = 0, b_ecnt = 0, b_egu = 0, b_eact = 0, b_edn = 0;
    /* the experts' E4M3 inputs, declared only where some layer's projection is fp8: gate_up's
     * (the normed block input, a row a token) and down's (the gated product, a row a slot) */
    rad_buf b_hq = 0, b_hqs = 0, b_eq = 0, b_eqs = 0;
    int     n_gu_q = 0, n_dn_q = 0;   /* layers whose projection is quantised */
    /* The quantised experts are rotated int8 against int8 activations (EXP_I8), every layer's:
     * the code pairs are one buffer each for the whole model. */
    bool    exp_i8 = false, exp_seen = false;
    rad_buf b_hout = 0, b_logits = 0;

    rad_weight w_tok = 0, w_onorm = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_onorm = 0, op_gather = 0, op_logits = 0, op_cap = 0;
    rad_op op_mm = 0;
    Gemma4Vision vis{};

    enum { MAX_TAPS = Dflash1::MAX_TAPS };
    int     n_taps = 0;
    int     tap_layer[MAX_TAPS] = {};
    rad_buf b_dfsrc = 0;

    /* The calibration tap: on while `tap_live`, over layers [calib_lo, calib_hi). Written by step,
     * which owns the corpus count. */
    bool       calib = false, tap_live = false;
    int64_t    calib_lo = 0, calib_hi = 0;
    long long  calib_rows = 0;

    Dflash1       df{};
    rad_kvgroup   kv_draft = 0;
    bool          have_df = false;
    const char*   tok_src = nullptr;   /* the checkpoint's table, which the 8-bit draft head copies */

    /* Google's assistant MTP head (ported from arch/gemma4_bf16): a serial drafter over the trunk's
     * own cache, no cache of its own, when the container carries `draft.` Gemma4Assistant weights. */
    Gemma4Head    head{};
    bool          have_head = false;

    std::vector<Layer> layers;
};

static Model g_model[MAX_SLOTS];

/* ============================================================== geometry */

static int split_heads(Kind& k, int64_t nh_all, int world, int rank, const char* what) {
    if (nh_all % world) {
        fprintf(stderr, "radiance: gemma4: %lld %s query heads do not divide over %d ranks\n",
                (long long)nh_all, what, world);
        return RAD_E_INVAL;
    }
    k.nh = nh_all / world;
    if (world <= k.nkv_all) {
        if (k.nkv_all % world) {
            fprintf(stderr, "radiance: gemma4: %lld %s KV heads do not divide over %d ranks\n",
                    (long long)k.nkv_all, what, world);
            return RAD_E_INVAL;
        }
        k.nkv = k.nkv_all / world;
        k.kv_repl = false;
        k.kv_off = rank * k.nkv * k.hd;
    } else {
        /* Replicated: ranks [i*r, (i+1)*r) share KV head i, r = world / nkv_all. A rank's query
         * heads [rank*nh, (rank+1)*nh) all belong to that head, because gqa = nh_all / nkv_all is a
         * multiple of nh exactly when world is a multiple of nkv_all, which is checked. */
        if (world % k.nkv_all) {
            fprintf(stderr, "radiance: gemma4: %d ranks cannot replicate %lld %s KV heads; the "
                            "world must be a multiple of the KV head count\n", world,
                    (long long)k.nkv_all, what);
            return RAD_E_INVAL;
        }
        k.nkv = 1;
        k.kv_repl = true;
        k.kv_off = (rank / (world / k.nkv_all)) * k.hd;
    }
    return RAD_OK;
}

/* The full layers' rotary (see the file header). `rope` rotates n_head + n_head_kv heads of its
 * operand, so a q-only tensor is n_head = nh, n_head_kv = 0 and a k-only one n_head = nkv. */
static rad_op decl_full_rope(RadBuilder* b, const Kind& k, int64_t rows, int64_t heads) {
    const int64_t half = k.hd / 2;
    if (k.rot_pairs >= half)
        return RAD_OP(b, "rope",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", k.hd),
                                 RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                                 RAD_F64("theta", k.theta), RAD_F64("scale", 1.0),
                                 RAD_STR("mode", "neox"), RAD_INT("rotary_dim", k.hd)),
                      RAD_NOWEIGHTS);
    /* frequency(inv, i) = inv / factor for i >= high, inv for i <= low (rad_yarn.h); the
     * magnitude multiplies cos and sin and stays 1. */
    return RAD_OP(b, "rope",
                  RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", k.hd),
                             RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                             RAD_F64("theta", k.theta), RAD_F64("scale", 1.0),
                             RAD_STR("mode", "neox"), RAD_INT("rotary_dim", k.hd),
                             RAD_F64("yarn_factor", 1e30),
                             RAD_F64("yarn_low", (double)k.rot_pairs - 0.5),
                             RAD_F64("yarn_high", (double)k.rot_pairs),
                             RAD_F64("yarn_magnitude", 1.0)),
                  RAD_NOWEIGHTS);
}

/* ============================================================== the drafter */

/* z-lab's DFlash drafter (arch/common/rad_gemma4_dflash.h), past the trunk's layers, reading the
 * taps the trunk copies into `dflash_src` and the trunk's tied table. */
static int declare_dflash(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                          Model& m, int base_layer) {
    Dflash1::Trunk t{};
    t.n_taps     = m.n_taps;
    t.escale     = m.escale;
    t.src        = m.b_dfsrc;
    t.w_tok      = m.w_tok;
    t.tok_src    = m.tok_src;
    t.base_layer = base_layer;
    RAD_ARCH_TRY(Dflash1::declare_drafter(b, meta, ctx, m.nm, m.g, t, &m.df, &m.kv_draft));
    m.have_df = true;
    return RAD_OK;
}

/* ============================================================== the experts' encoding */

/* How one expert projection is stored (the container's entry, or in rad-convert the recipe's
 * plan, rad_weight_encoding), which picks the GEMM: plain bf16 is `moe_gemm`, block fp8 (fp8_e4m3
 * codes, a bf16 scale per 128x128 block) is `moe_gemm_q` at fp8a8. Anything else is refused by
 * name. A builder with no model behind it answers nothing, and the declared dtype bf16 is read. */
enum { EXP_BF16 = 0, EXP_FP8 = 1, EXP_I8 = 2 };
static int expert_format(RadBuilder* b, const char* wn, int* fmt) {
    RadEncoding e{};
    *fmt = EXP_BF16;
    if (!weight_enc(b, wn, &e)) return RAD_OK;
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return RAD_OK;
    if (block_fp8_128(e)) {
        *fmt = EXP_FP8;
        return RAD_OK;
    }
    /* Rotated int8: codes with a bf16 scale per (row, 128 of K), each 128 of K rotated by the
     * Hadamard (libquant's `codes=i8 group=128 scale=bf16 transform=fwht128`). */
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    if (rad_enc_is(&e, "affine") && e.n_planes == 2 && e.plane[0].dtype == RAD_I8 && sc &&
        sc->dtype == RAD_BF16 && sc->block[0] == 1 && sc->block[1] == RAD_SCALE_GROUP &&
        std::strcmp(e.transform, "fwht128") == 0) {
        *fmt = EXP_I8;
        return RAD_OK;
    }
    char en[256];
    rad_enc_format(&e, en, sizeof en);
    fprintf(stderr, "radiance: gemma4: '%s' is %s; the routed experts are served plain bf16, "
                    "block fp8 (fp8_e4m3*bf16[128x128]) or rotated int8 "
                    "(i8*bf16[1x128]/fwht128). Convert with no rule for them, or with `rtn "
                    "codes=fp8_e4m3 block=128x128 scale=bf16` or `rtn codes=i8 group=128 "
                    "scale=bf16 transform=fwht128 clamp=sym rule=search`.\n", wn, en);
    return RAD_E_DTYPE;
}

/* One encoding per projection per layer, since one expert table takes one: the first local
 * expert's decides and every other must match. `what` is "gate_up" or "down". */
static int layer_expert_format(RadBuilder* b, Names& nm, int64_t l, int64_t base, int64_t n,
                               const char* what, int* fmt) {
    for (int64_t e = 0; e < n; ++e) {
        const char* wn = nm.f("blk.%lld.ffn_%s_exps.%lld.weight", (long long)l, what,
                              (long long)(base + e));
        int f = EXP_BF16;
        RAD_ARCH_TRY(expert_format(b, wn, &f));
        if (e == 0) { *fmt = f; continue; }
        if (f != *fmt) {
            static const char* const kName[3] = { "bf16", "block fp8", "rotated int8" };
            fprintf(stderr, "radiance: gemma4: '%s' is %s and expert %lld of the same layer is %s; "
                            "one layer's %s experts take one encoding (a recipe rule for them "
                            "matches all of a layer's or none)\n", wn, kName[f], (long long)base,
                    kName[*fmt], what);
            return RAD_E_DTYPE;
        }
    }
    return RAD_OK;
}

/* ============================================================== declare */

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->is_draft) {
        fprintf(stderr, "radiance: gemma4: this plugin has no draft scope\n");
        return RAD_E_UNSUPPORTED;
    }
    const char* q = meta->quant ? meta->quant : "";
    if (*q) {
        fprintf(stderr, "radiance: gemma4 serves a container converted from the bf16 checkpoint "
                        "(google/gemma-4-*-it), with the recipe choosing what is fp8, and this one "
                        "declares '%s'.\n", q);
        return RAD_E_UNSUPPORTED;
    }

    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    Geom&  g  = m.g;

    /* ---- variants this plugin does not serve, refused by name. */
    if (meta_text_geti(meta, "hidden_size_per_layer_input", 0) > 0 ||
        meta_text_geti(meta, "num_kv_shared_layers", 0) > 0 ||
        meta_text_getb(meta, "use_double_wide_mlp", false) ||
        meta_text_getb(meta, "attention_bias", false)) {
        fprintf(stderr, "radiance: gemma4: this checkpoint uses per-layer embeddings, shared KV "
                        "layers, a double-wide MLP or attention biases (the E2B/E4B shape); this "
                        "plugin serves the 26B-A4B and 31B shape\n");
        return RAD_E_UNSUPPORTED;
    }
    {
        const char* act = meta_text_gets(meta, "hidden_activation", "");
        if (!act || std::strcmp(act, "gelu_pytorch_tanh")) {
            fprintf(stderr, "radiance: gemma4: hidden_activation '%s'; this plugin serves "
                            "gelu_pytorch_tanh\n", act ? act : "");
            return RAD_E_UNSUPPORTED;
        }
    }

    /* ---- the geometry, read rather than derived: head_dim is not hidden / heads here. */
    m.n_layer = meta->n_layers > 0 ? meta->n_layers : meta_text_geti(meta, "num_hidden_layers", 0);
    m.E       = meta_text_geti(meta, "hidden_size", meta->n_embd);
    const int64_t nh_all  = meta_text_geti(meta, "num_attention_heads", meta->n_head);
    const int64_t vocab   = meta_text_geti(meta, "vocab_size", meta->n_vocab);
    const int64_t ff_all  = meta_text_geti(meta, "intermediate_size", meta->n_ff);
    const double  eps     = meta_text_getf(meta, "rms_norm_eps", meta->rms_eps);
    m.ks.hd      = meta_text_geti(meta, "head_dim", meta->head_dim);
    m.ks.nkv_all = meta_text_geti(meta, "num_key_value_heads", meta->n_head_kv);
    m.kf.hd      = meta_text_geti(meta, "global_head_dim", m.ks.hd);
    m.kf.nkv_all = meta_text_geti(meta, "num_global_key_value_heads", m.ks.nkv_all);
    m.kf.k_eq_v  = meta_text_getb(meta, "attention_k_eq_v", false);
    m.ks.theta   = meta_text_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    m.kf.theta   = meta_text_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    m.ks.window  = meta_text_geti(meta, "sliding_window", 0);   /* HF: q - k < window, attn_paged's rule */
    m.softcap    = meta_text_getf(meta, "final_logit_softcapping", 0.0);
    if (m.n_layer <= 0 || m.E <= 0 || nh_all <= 0 || vocab <= 0 || ff_all <= 0 || !(eps > 0) ||
        m.ks.hd <= 0 || m.ks.nkv_all <= 0 || m.kf.hd <= 0 || m.kf.nkv_all <= 0 ||
        !(m.ks.theta > 0) || !(m.kf.theta > 0) || m.ks.window <= 0) {
        fprintf(stderr, "radiance: gemma4: the container does not describe the model: layers "
                        "%lld, hidden %lld, heads %lld, vocab %lld, ffn %lld, eps %g, sliding %lld "
                        "x %lld kv theta %g window %lld, full %lld x %lld kv theta %g\n",
                (long long)m.n_layer, (long long)m.E, (long long)nh_all, (long long)vocab,
                (long long)ff_all, eps, (long long)m.ks.hd, (long long)m.ks.nkv_all, m.ks.theta,
                (long long)m.ks.window, (long long)m.kf.hd, (long long)m.kf.nkv_all, m.kf.theta);
        return RAD_E_INVAL;
    }
    /* The two rotaries. The sliding one is ordinary full-head NeoX; the full one's type and
     * proportion are read and refused if this plugin does not compute them. */
    {
        const char* st = meta_text_gets(meta, "rope_parameters.sliding_attention.rope_type", "default");
        const char* ft = meta_text_gets(meta, "rope_parameters.full_attention.rope_type", "default");
        const double prf = meta_text_getf(meta, "rope_parameters.full_attention.partial_rotary_factor", 1.0);
        const double ff  = meta_text_getf(meta, "rope_parameters.full_attention.factor", 1.0);
        if (std::strcmp(st, "default") || (std::strcmp(ft, "default") && std::strcmp(ft, "proportional")) ||
            ff != 1.0 || !(prf > 0.0) || prf > 1.0) {
            fprintf(stderr, "radiance: gemma4: rotary '%s' (sliding) and '%s' factor %g "
                            "proportion %g (full); this plugin computes default and "
                            "proportional at factor 1\n", st, ft, ff, prf);
            return RAD_E_UNSUPPORTED;
        }
        m.ks.rot_pairs = m.ks.hd / 2;
        /* int(rope_proportion * head_dim // 2), exactly as transformers computes it. */
        m.kf.rot_pairs = !std::strcmp(ft, "proportional")
                       ? (int64_t)std::floor(prf * (double)m.kf.hd / 2.0)
                       : m.kf.hd / 2;
        if (m.kf.rot_pairs <= 0) return RAD_E_INVAL;
    }

    std::vector<int> full;
    const char* lt = meta_text_gets(meta, "layer_types", nullptr);
    if (!lt) return RAD_E_INVAL;
    RAD_ARCH_TRY(parse_layer_kinds("gemma4", lt, {"full_attention"}, "sliding_attention", &full));
    if ((int64_t)full.size() != m.n_layer) {
        fprintf(stderr, "radiance: gemma4: layer_types names %zu layers and the model has %lld\n",
                full.size(), (long long)m.n_layer);
        return RAD_E_INVAL;
    }

    /* The model-wide Geom, which the shared helpers (and the drafter's copy) read. */
    g.n_layer = m.n_layer;
    g.n_embd  = m.E;
    g.head_dim = m.ks.hd;
    g.n_head_all = nh_all;
    g.n_head_kv_all = m.ks.nkv_all;
    g.n_ff_all = ff_all;
    g.n_vocab_all = vocab;
    g.max_tok  = ctx->max_tok;
    g.max_seqs = ctx->max_seqs;
    g.max_ctx  = ctx->max_ctx;
    g.max_spec = ctx->max_spec;
    g.max_logit_rows = ctx->max_seqs * (ctx->max_spec > 0 ? ctx->max_spec + 1 : 1);
    if (g.max_logit_rows > ctx->max_tok) g.max_logit_rows = ctx->max_tok;
    if (ctx->max_out_rows > g.max_logit_rows) g.max_logit_rows = ctx->max_out_rows;
    g.rank  = ctx->rank;
    g.world = ctx->world_size > 0 ? ctx->world_size : 1;
    g.wire_exact = ctx->tp_wire_lossy ? 0 : 1;
    g.wire_min_bytes = ctx->tp_wire_min_bytes;
    g.eps   = (float)eps;
    g.theta = (float)m.ks.theta;
    g.rope_scale = 1.0f;
    g.dtype = "bf16";
    g.act_dtype = RAD_BF16;
    if (g.max_tok <= 0 || g.max_seqs <= 0) return RAD_E_INVAL;
    const int W = g.world, R = g.rank;
    RAD_ARCH_TRY(split_heads(m.ks, nh_all, W, R, "sliding"));
    RAD_ARCH_TRY(split_heads(m.kf, nh_all, W, R, "full"));
    g.n_head = m.ks.nh;
    g.n_head_kv = m.ks.nkv;
    RAD_ARCH_TRY(divide_or_fail(ff_all, W, "intermediate_size", &m.n_ff));
    g.n_ff = m.n_ff;
    g.n_vocab   = shard_extent(vocab, W, R);
    g.vocab_off = shard_begin(vocab, W, R);
    if (g.n_vocab <= 0) return RAD_E_INVAL;
    const int64_t E = m.E, T = g.max_tok;

    /* ---- the routed arm's geometry. */
    m.moe = meta_text_getb(meta, "enable_moe_block", false);
    if (m.moe) {
        m.n_expert = meta_text_geti(meta, "num_experts", meta->n_expert);
        m.top_k    = meta_text_geti(meta, "top_k_experts", meta->n_expert_used);
        m.n_ffe    = meta_text_geti(meta, "moe_intermediate_size", 0);
        if (m.n_expert <= 0 || m.top_k <= 0 || m.top_k > m.n_expert || m.n_ffe <= 0) {
            fprintf(stderr, "radiance: gemma4: enable_moe_block with %lld experts, top %lld, of "
                            "%lld\n", (long long)m.n_expert, (long long)m.top_k,
                    (long long)m.n_ffe);
            return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(divide_or_fail(m.n_expert, W, "num_experts", &m.n_local));
        m.expert_base = (int64_t)R * m.n_local;
        /* A pass bounds the [rows * top_k, ...] expert planes; 2048 tokens is one pass at the
         * default chunk size. */
        m.moe_rows = T < 2048 ? T : 2048;
    }

    /* ---- the embedding scale and the rotary scaling the converter may have written. The scale
     * is rounded to bf16 because the reference holds it in a bf16 buffer: sqrt(2816) = 53.066
     * multiplies as 53.0. */
    m.escale = embed_scale(E);
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.ks.hd));
    if (g.yarn.factor != 0.0f || g.rope_scale != 1.0f) {
        fprintf(stderr, "radiance: gemma4: --rope-scaling is not implemented for this family's "
                        "two rotaries\n");
        return RAD_E_UNSUPPORTED;
    }

    /* ---- the KV groups, one per geometry: the sliding layers' (256-wide heads) and the full
     * layers' (512).
     *
     * The sliding layers' group is RAD_KV_WINDOW, sized to the attention window, which is what
     * the kernel reads: attn_paged takes `window` 1024 on those layers and never looks at a key
     * older than it. As a window group the pool is capped at the window plus one block per
     * sequence (core/mem/kv.cpp), so the sliding layers cost about 1k tokens of cache each
     * instead of the whole context. The full layers' group stays RAD_KV_FULL: it is the only one
     * whose keys outlive a window.
     *
     * A windowed group is not prefix-cached (core/mem/prefix.cpp): its rows below the window are
     * recycled, and adopt() would overfill a table capped at window_blocks. On a prefix hit the
     * full layers' rows come back from the store, and the window resumes from the deepest
     * retained checkpoint at or below the matched end: that checkpoint's slot carries the window
     * behind it, and only the gap between it and the match end is recomputed. Where no retained
     * checkpoint sits at or below the match, the windowed layers start their cache where the hit
     * ended and the prompt is prefilled again. Gemma 3 declares its sliding group the same way.
     *
     * The drafter does not read the trunk's cache. DFlash keeps its own windowed draft KV,
     * rebuilt each context pass from the trunk's residual taps, so a windowed trunk group costs
     * the drafter nothing. */
    {
        const uint32_t kvdt = rad_kv_cache_dtype(ctx, RAD_BF16);
        RadKVGroupDecl d{};
        d.kind = RAD_KV_WINDOW;
        d.window = m.ks.window;   /* the same 1024 the attention op reads */
        d.dtype = kvdt;
        d.n_head_kv = m.ks.nkv;
        d.head_dim = m.ks.hd;
        m.ks.kv = rad_decl_kv_group(b, nm.f("kv_swa"), &d);
        d.kind = RAD_KV_FULL;
        d.window = 0;
        d.n_head_kv = m.kf.nkv;
        d.head_dim = m.kf.hd;
        m.kf.kv = rad_decl_kv_group(b, nm.f("kv_full"), &d);
        g.kv_dtype = rad_dtype_name(kvdt);
    }

    /* Gemma4ForConditionalGeneration keeps the text model under model.language_model;
     * Gemma4ForCausalLM (a text-only export) under model. */
    const char* archs = rad_meta_gets(meta, "architectures", "");
    const char* L = (archs && std::strstr(archs, "Gemma4ForCausalLM")) ? "model"
                                                                       : "model.language_model";

    /* ---- the experts' name maps, and from them each layer's expert format, before any buffer:
     * an fp8 projection reads an E4M3 input a bf16 one does not, and an unread plane would still
     * be charged by the planner. One stacked tensor per projection per layer, sliced by the global
     * expert index: this rank declares its window into the checkpoint's experts. */
    m.layers.resize((size_t)m.n_layer);
    if (m.moe) {
        const int64_t NL = m.n_local;
        for (int64_t l = 0; l < m.n_layer; ++l) {
            Layer& y = m.layers[(size_t)l];
            const long long ll = (long long)l;
            const char* sgu = nm.ckpt("%s.layers.%lld.experts.gate_up_proj", L, ll);
            const char* sdn = nm.ckpt("%s.layers.%lld.experts.down_proj", L, ll);
            for (int64_t e = 0; e < NL; ++e) {
                const long long ge = (long long)(m.expert_base + e);
                RAD_ARCH_TRY(map_slice(b, nm.f("blk.%lld.ffn_gate_up_exps.%lld.weight", ll, ge), sgu, (int)ge));
                RAD_ARCH_TRY(map_slice(b, nm.f("blk.%lld.ffn_down_exps.%lld.weight", ll, ge), sdn, (int)ge));
            }
            int fgu = EXP_BF16, fdn = EXP_BF16;
            RAD_ARCH_TRY(layer_expert_format(b, nm, l, m.expert_base, NL, "gate_up", &fgu));
            RAD_ARCH_TRY(layer_expert_format(b, nm, l, m.expert_base, NL, "down", &fdn));
            y.gu_q = fgu != EXP_BF16;
            y.dn_q = fdn != EXP_BF16;
            /* One quantised format for the model: the experts' code pairs are shared buffers. */
            for (const int f : { fgu, fdn }) {
                if (f == EXP_BF16) continue;
                const bool i8 = f == EXP_I8;
                if (m.exp_seen && i8 != m.exp_i8) {
                    fprintf(stderr, "radiance: gemma4: layer %lld's experts are %s and an earlier "
                                    "projection's %s; a model's quantised experts are all block "
                                    "fp8 or all rotated int8\n", ll,
                            i8 ? "rotated int8" : "block fp8", i8 ? "block fp8" : "rotated int8");
                    return RAD_E_DTYPE;
                }
                m.exp_seen = true;
                m.exp_i8 = i8;
            }
            /* moe_gemm_q's geometry, refused here by name rather than as a kernel miss. It takes
             * experts as two tables by parity, so a rank must hold an even count. */
            if ((y.gu_q || y.dn_q) && NL % 2) {
                fprintf(stderr, "radiance: gemma4: quantised experts come in pairs on a rank (the "
                                "grouped GEMM takes them as two tables by parity); this rank "
                                "holds %lld\n", (long long)NL);
                return RAD_E_INVAL;
            }
            if (y.gu_q && ((2 * m.n_ffe) % 64 || E % RAD_SCALE_GROUP)) {
                fprintf(stderr, "radiance: gemma4: layer %lld's gate_up experts are quantised at "
                                "%lld x %lld; the grouped GEMM takes N in 64s and K in whole "
                                "%d-column blocks\n", ll, (long long)(2 * m.n_ffe), (long long)E,
                        RAD_SCALE_GROUP);
                return RAD_E_INVAL;
            }
            if (y.dn_q && (E % 64 || m.n_ffe % RAD_SCALE_GROUP)) {
                fprintf(stderr, "radiance: gemma4: layer %lld's down experts are quantised, and "
                                "the experts are %lld wide: the grouped GEMM reads K in whole "
                                "%d-column scale blocks. Keep `ffn_down_exps` bf16 (no recipe "
                                "rule for it), or pad the experts to a multiple of %d (zero rows "
                                "of gate and up, zero columns of down) and convert the padded "
                                "checkpoint.\n", ll,
                        (long long)m.n_ffe, RAD_SCALE_GROUP, RAD_SCALE_GROUP);
                return RAD_E_INVAL;
            }
            m.n_gu_q += y.gu_q;
            m.n_dn_q += y.dn_q;
        }
    }

    /* ---- activations. */
    const int64_t qs = m.ks.q(), kvs = m.ks.kv_dim(), qf = m.kf.q(), kvf = m.kf.kv_dim();
    m.b_x    = decl_b(b, nm.f("x"),     RAD_BF16, {T, E});
    m.b_h    = decl_b(b, nm.f("h"),     RAD_BF16, {T, E});
    m.b_o    = decl_b(b, nm.f("o"),     RAD_BF16, {T, E});
    m.b_qkv  = decl_b(b, nm.f("qkv_s"), RAD_BF16, {T, qs + 2 * kvs});
    m.b_atts = decl_b(b, nm.f("att_s"), RAD_BF16, {T, qs});
    m.b_qf   = decl_b(b, nm.f("q_f"),   RAD_BF16, {T, qf});
    m.b_kf   = decl_b(b, nm.f("k_f"),   RAD_BF16, {T, kvf});
    m.b_vf   = decl_b(b, nm.f("v_f"),   RAD_BF16, {T, kvf});
    m.b_attf = decl_b(b, nm.f("att_f"), RAD_BF16, {T, qf});
    m.b_ff2  = decl_b(b, nm.f("ff2"),   RAD_BF16, {2 * T, E});
    m.b_gu   = decl_b(b, nm.f("gate_up"), RAD_BF16, {T, 2 * m.n_ff});
    m.b_act  = decl_b(b, nm.f("ffn"),   RAD_BF16, {T, m.n_ff});
    if (m.moe) {
        const int64_t P = m.moe_rows, K = m.top_k;
        m.b_rn     = decl_b(b, nm.f("moe_rn"),     RAD_BF16, {T, E});
        m.b_logit  = decl_b(b, nm.f("moe_logits"), RAD_BF16, {T, m.n_expert});
        m.b_ids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,  {T, K});
        m.b_ew     = decl_b(b, nm.f("moe_w"),      RAD_BF16, {T, K});
        m.b_es     = decl_b(b, nm.f("moe_es"),     RAD_BF16, {T, K});
        if (m.n_gu_q < m.n_layer)
            m.b_hx = decl_b(b, nm.f("moe_h"),      RAD_BF16, {T, E});
        if (m.n_gu_q > 0) {
            m.b_hq  = decl_b(b, nm.f("moe_hq"),    m.exp_i8 ? RAD_I8 : RAD_F8E4M3, {T, E});
            m.b_hqs = decl_b(b, nm.f("moe_hqs"),   RAD_F32,    {T, scale_groups(E)});
        }
        m.b_sorted = decl_b(b, nm.f("moe_sorted"), RAD_I32,  {P * K});
        m.b_eoff   = decl_b(b, nm.f("moe_eoff"),   RAD_I32,  {m.n_local + 1});
        m.b_ecnt   = decl_b(b, nm.f("moe_ecnt"),   RAD_I32,  {m.n_local});
        m.b_egu    = decl_b(b, nm.f("moe_egu"),    RAD_BF16, {P * K, 2 * m.n_ffe});
        m.b_eact   = decl_b(b, nm.f("moe_eact"),   RAD_BF16, {P * K, m.n_ffe});
        if (m.n_dn_q > 0) {
            m.b_eq  = decl_b(b, nm.f("moe_eq"),    m.exp_i8 ? RAD_I8 : RAD_F8E4M3, {P * K, m.n_ffe});
            m.b_eqs = decl_b(b, nm.f("moe_eqs"),   RAD_F32,    {P * K, scale_groups(m.n_ffe)});
        }
        m.b_edn    = decl_b(b, nm.f("moe_edn"),    RAD_BF16, {P * K, E});
    }
    m.b_hout = decl_b(b, nm.f("h_out"), RAD_BF16, {g.max_logit_rows, E});

    /* ---- the drafter's taps, declared when the container carries a DFlash drafter and this
     * serve speculates. DFlash2 is refused by name: its convolutions and selector are another
     * block. */
    {
        const int n = Dflash1::taps(meta, ctx, m.n_layer, m.tap_layer, Model::MAX_TAPS);
        if (n < 0) return n;
        m.n_taps = n;
        if (n > 0) m.b_dfsrc = decl_b(b, nm.f("dflash_src"), RAD_BF16, {T, (int64_t)n * E});
    }

    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32, {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges: one table, row-sharded with the vocabulary, read as the embedding
     * by `embed_lookup` and as the head by `logits_gemm`. Gemma ties them; a second copy would be
     * 1.4 GiB. */
    m.tok_src = nm.ckpt("%s.embed_tokens.weight", L);
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), m.tok_src));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.norm.weight", L)));
    m.w_tok   = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, E}, RAD_ACCESS_VOCAB,
                       RAD_SHARD_ROW, grp_model());
    m.w_onorm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {E}, RAD_ACCESS_PER_TOKEN,
                       RAD_SHARD_NONE, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                   RAD_INT("vocab_offset", g.vocab_off), RAD_F64("mult", m.escale)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    const bool vision = ctx->max_enc_patches > 0 && Gemma4Vision::present(meta);
    const char* bidi = meta_text_gets(meta, "use_bidirectional_attention", "");
    m.vis.bidirectional = vision && bidi && !std::strcmp(bidi, "vision");
    if (vision)
        m.op_mm = rw(b, RAD_OP(b, "scatter_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                     {m.b_x}, {m.b_x});
    if (W > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", W), RAD_RANGE("numel", E, T * E),
                                          RAD_STR("dtype", "bf16"), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});

    /* ---- small declaration helpers, each banded over rows up to the step's height unless
     * given another bound. */
    auto rmsnorm = [&](int64_t rows, int64_t n, rad_weight w, double mult, rad_buf in, rad_buf out) {
        rad_op h = w ? RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                         RAD_F64("eps", eps), RAD_STR("dtype", "bf16"),
                                         RAD_F64("wadd", 0.0), RAD_F64("mult", mult)),
                              RAD_WEIGHTS(w))
                     : RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                         RAD_F64("eps", eps), RAD_STR("dtype", "bf16"),
                                         RAD_F64("wadd", 0.0), RAD_F64("mult", mult)),
                              RAD_NOWEIGHTS);
        return rw(b, h, {in}, {out});
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w, rad_buf in, rad_buf out) {
        return rw(b, RAD_OP(b, "gemm_nt",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                    RAD_STR("dtype", "bf16")),
                         RAD_WEIGHTS(w)),
                  {in}, {out});
    };
    auto binop = [&](const char* op, int64_t rows, int64_t n, rad_buf a, rad_buf c, rad_buf out) {
        return rw(b, RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                              RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  {a, c}, {out});
    };
    auto act_mul = [&](int64_t rows, int64_t n, rad_buf in, rad_buf out) {
        return rw(b, RAD_OP(b, "act_mul",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                    RAD_STR("act", "gelu_tanh"), RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  {in}, {out});
    };
    auto allreduce = [&](int64_t max_numel, rad_buf buf) {
        return rw(b, RAD_OP(b, "all_reduce",
                         RAD_PARAMS(RAD_INT("world_size", W), RAD_RANGE("numel", E, max_numel),
                                    RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                                    RAD_INT("min_bytes", g.wire_min_bytes)),
                         RAD_NOWEIGHTS),
                  {buf}, {buf});
    };
    auto attn = [&](const Kind& k, rad_buf q, rad_buf out, const char* op = "attn_paged",
                    int causal = 1) {
        return rw(b, RAD_OP(b, op,
                         RAD_PARAMS(RAD_RANGE("q_len", 1, T), RAD_INT("head_dim", k.hd),
                                    RAD_INT("gqa", k.nh / k.nkv), RAD_INT("causal", causal),
                                    RAD_INT("window", k.window), RAD_INT("n_head", k.nh),
                                    RAD_INT("block_size", RAD_KV_BLOCK),
                                    RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                    RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype),
                                    RAD_F64("scale", 1.0)),
                         RAD_NOWEIGHTS),
                  {q}, {out});
    };
    auto kv_store = [&](const Kind& k, rad_buf kb, rad_buf vb) {
        return rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", k.hd),
                                    RAD_INT("n_head_kv", k.nkv),
                                    RAD_INT("block_size", rad_kv_block_size(b, k.kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                  {kb, vb}, {});
    };

    /* The sandwich joins, expert_scale and attn_paged_qrow are each requested as one op and
     * declared as the chain where they do not resolve. Only libr4d declares them (device rows; not
     * libref or libavx), and declaring an op no loaded library declares is an error, so each is
     * checked first with rad_op_declared. A CPU-only build, a --kernels list without libr4d, or
     * the host backend runs the chain. The fused kernels round where the host chain rounds
     * (libr4d/r4d_norm_join_bf16.hip); on a card they differ from libr4d's chain only in reduction
     * order. libref deliberately has no row for them, so the chain path stays tested. */
    const bool has_join_a = rad_op_declared(b, "norm_add_norm") != 0;
    const char* const join_r = m.exp_i8 ? "norm_add_norm_route_i8g" : "norm_add_norm_route";
    const bool has_join_r = rad_op_declared(b, join_r) != 0;
    const bool has_join_f = rad_op_declared(b, "join_add_scale") != 0;
    const bool has_escale = rad_op_declared(b, "expert_scale") != 0;
    const bool has_qrow   = rad_op_declared(b, "attn_paged_qrow") != 0;
    int n_joins = 0, n_route = 0;

    const int A = RAD_ACCESS_PER_TOKEN, N = RAD_SHARD_NONE;
    /* Every layer's input norm gain, declared ahead of the layers: a feed-forward join writes the
     * next layer's input norm (join_add_scale's `y`), naming that gain before its layer is
     * declared. */
    for (int64_t l = 0; l < m.n_layer; ++l) {
        const long long ll = (long long)l;
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_norm.weight", ll),
                              nm.ckpt("%s.layers.%lld.input_layernorm.weight", L, ll)));
        m.layers[(size_t)l].w_anorm = decl_w(b, nm.f("blk.%lld.attn_norm.weight", ll), RAD_F32,
                                             {E}, A, N, grp_layer((int)l));
    }

    for (int64_t l = 0; l < m.n_layer; ++l) {
        Layer& y = m.layers[(size_t)l];
        y.full = full[(size_t)l];
        const Kind& k = y.full ? m.kf : m.ks;
        const long long ll = (long long)l;
        const int li = (int)l;
        const RadWeightGroup gl = grp_layer(li);
        auto src = [&](const char* s) { return nm.ckpt("%s.layers.%lld.%s", L, ll, s); };
        RAD_ARCH_TRY(rad_bind_layer_kv(b, li, k.kv));

        /* ------------------------------------------------ attention: weights */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_q_norm.weight", ll), src("self_attn.q_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_k_norm.weight", ll), src("self_attn.k_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_output.weight", ll), src("self_attn.o_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_post_norm.weight", ll),
                              src("post_attention_layernorm.weight")));
        /* bf16 gains: `qk_norm_rope` reads its gains at the activation's width. */
        y.w_qn    = decl_w(b, nm.f("blk.%lld.attn_q_norm.weight", ll), RAD_BF16, {k.hd}, A, N, gl);
        y.w_kn    = decl_w(b, nm.f("blk.%lld.attn_k_norm.weight", ll), RAD_BF16, {k.hd}, A, N, gl);
        y.w_o     = decl_w(b, nm.f("blk.%lld.attn_output.weight", ll), RAD_BF16, {E, k.q()}, A,
                           RAD_SHARD_COL, gl);
        y.w_apost = decl_w(b, nm.f("blk.%lld.attn_post_norm.weight", ll), RAD_F32, {E}, A, N, gl);
        if (!y.full) {
            /* One projection, q|k|v, three row parts: every rank takes its slice of each. */
            RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.attn_qkv.weight", ll),
                                    { src("self_attn.q_proj.weight"), src("self_attn.k_proj.weight"),
                                      src("self_attn.v_proj.weight") }));
            y.w_qkv = decl_w(b, nm.f("blk.%lld.attn_qkv.weight", ll), RAD_BF16,
                             {k.q() + 2 * k.kv_dim(), E}, A, RAD_SHARD_ROW, gl, 0,
                             {k.q(), k.kv_dim(), k.kv_dim()});
            if (k.kv_repl) {
                const int64_t lo[3] = { R * k.q(), k.kv_off, k.kv_off };
                const int64_t hi[3] = { lo[0] + k.q(), lo[1] + k.kv_dim(), lo[2] + k.kv_dim() };
                const int64_t tot[3] = { nh_all * k.hd, k.nkv_all * k.hd, k.nkv_all * k.hd };
                RAD_ARCH_TRY(rad_weight_shard_span_parts(b, y.w_qkv, lo, hi, tot, 3));
            }
        } else {
            if (!k.k_eq_v) {
                fprintf(stderr, "radiance: gemma4: full-attention layer %lld has a v_proj "
                                "(attention_k_eq_v false); this plugin serves K = V\n", ll);
                return RAD_E_UNSUPPORTED;
            }
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_q.weight", ll), src("self_attn.q_proj.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_k.weight", ll), src("self_attn.k_proj.weight")));
            y.w_q = decl_w(b, nm.f("blk.%lld.attn_q.weight", ll), RAD_BF16, {k.q(), E}, A,
                           RAD_SHARD_ROW, gl);
            y.w_k = decl_w(b, nm.f("blk.%lld.attn_k.weight", ll), RAD_BF16, {k.kv_dim(), E}, A,
                           RAD_SHARD_ROW, gl);
            if (k.kv_repl)
                RAD_ARCH_TRY(rad_weight_shard_span(b, y.w_k, k.kv_off, k.kv_off + k.kv_dim()));
        }

        /* ------------------------------------------------ attention: ops, in issue order (the
         * attention op is declared ahead of its kv_store, which reads the block size off it). */
        if (!(l > 0 && m.layers[(size_t)l - 1].next_norm))
            y.op_anorm = rmsnorm(T, E, y.w_anorm, 1.0, m.b_x, m.b_h);
        if (!y.full) {
            y.op_qkv  = gemm(T, k.q() + 2 * k.kv_dim(), E, y.w_qkv, m.b_h, m.b_qkv);
            y.op_qknr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", k.hd),
                                           RAD_INT("n_head", k.nh), RAD_INT("n_head_kv", k.nkv),
                                           RAD_F64("theta", k.theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "neox"), RAD_INT("rotary_dim", k.hd),
                                           RAD_F64("wadd", 0.0), RAD_INT("v_norm", 1)),
                                RAD_WEIGHTS(y.w_qn, y.w_kn)),
                            {m.b_qkv}, {m.b_qkv});
            /* q is read in place, a column slice of the fused [q|k|v] row, by `attn_paged_qrow`
             * (attn_paged with q at its own row pitch). It is a separate op so no `--kernels`
             * order can hand the slice to a kernel that reads q only dense (which would refuse it
             * with RAD_E_STRIDE and stop the engine). This saves a launch and a [T, q] round trip
             * per sliding layer. Where the op is not declared or does not resolve, q is copied
             * into a dense block for attn_paged. Asked once for all sliding layers; even a miss
             * is the group's first op carrying a block_size, so rad_kv_block_size answers with
             * its RAD_KV_BLOCK. */
            if (has_qrow && m.qrow != 0) {
                y.op_attn = attn(k, m.b_qkv, m.b_atts, "attn_paged_qrow");
                m.qrow = y.op_attn ? 1 : 0;
            }
            if (!y.op_attn) {
                /* Declared here, not with the others: a buffer no op names is live for the whole
                 * program, and when q is read in place nothing names this one. */
                if (!m.b_qs) m.b_qs = decl_b(b, nm.f("q_s"), RAD_BF16, {T, k.q()});
                y.op_lift = cast_op(b, T, k.q(), "bf16", "bf16", m.b_qkv, m.b_qs);
                y.op_attn = attn(k, m.b_qs, m.b_atts);
            }
            if (m.vis.bidirectional) {
                const bool dense_q = y.op_lift != 0;
                const rad_buf q = dense_q ? m.b_qs : m.b_qkv;
                y.op_bidi = attn(k, q, m.b_atts, dense_q ? "attn_paged" : "attn_paged_qrow", 0);
                if (!y.op_bidi) {
                    fprintf(stderr, "radiance: gemma4: no sliding attention kernel for "
                                    "bidirectional image spans\n");
                    return RAD_E_UNSUPPORTED;
                }
                const char* ahead = dense_q ? "attn_paged_ahead" : "attn_paged_qrow_ahead";
                if (T > k.window && rad_op_declared(b, ahead))
                    y.op_ahead = attn(k, q, m.b_atts, ahead, 0);
            }
            y.op_store = kv_store(k, m.b_qkv, m.b_qkv);
            y.op_o     = gemm(T, E, k.q(), y.w_o, m.b_atts, m.b_o);
        } else {
            y.op_q     = gemm(T, k.q(), E, y.w_q, m.b_h, m.b_qf);
            y.op_k     = gemm(T, k.kv_dim(), E, y.w_k, m.b_h, m.b_kf);
            /* V first: the unit-gain norm of the raw k projection, before the key norm writes k. */
            y.op_vnorm = rmsnorm(T * k.nkv, k.hd, 0, 1.0, m.b_kf, m.b_vf);
            y.op_qn    = rmsnorm(T * k.nh, k.hd, y.w_qn, 1.0, m.b_qf, m.b_qf);
            y.op_kn    = rmsnorm(T * k.nkv, k.hd, y.w_kn, 1.0, m.b_kf, m.b_kf);
            y.op_qrope = rw(b, decl_full_rope(b, k, T, k.nh), {m.b_qf}, {m.b_qf});
            y.op_krope = rw(b, decl_full_rope(b, k, T, k.nkv), {m.b_kf}, {m.b_kf});
            y.op_attn  = attn(k, m.b_qf, m.b_attf);
            y.op_store = kv_store(k, m.b_kf, m.b_vf);
            y.op_o     = gemm(T, E, k.q(), y.w_o, m.b_attf, m.b_o);
        }
        if (W > 1) y.op_ar_o = allreduce(T * E, m.b_o);
        /* The post-norm comes after the reduction, since it normalises the whole sum. Declared
         * below with the dense arm's pre-norm, as one `norm_add_norm` where a kernel serves it. */

        /* ------------------------------------------------ the dense arm */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_norm.weight", ll), src("pre_feedforward_layernorm.weight")));
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.ffn_gate_up.weight", ll),
                                { src("mlp.gate_proj.weight"), src("mlp.up_proj.weight") }));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_down.weight", ll), src("mlp.down_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_post_norm.weight", ll),
                              src("post_feedforward_layernorm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.layer_scalar", ll), src("layer_scalar")));
        y.w_fnorm  = decl_w(b, nm.f("blk.%lld.ffn_norm.weight", ll), RAD_F32, {E}, A, N, gl);
        y.w_gu     = decl_w(b, nm.f("blk.%lld.ffn_gate_up.weight", ll), RAD_BF16, {2 * m.n_ff, E},
                            A, RAD_SHARD_ROW, gl, 0, {m.n_ff, m.n_ff});
        y.w_dn     = decl_w(b, nm.f("blk.%lld.ffn_down.weight", ll), RAD_BF16, {E, m.n_ff}, A,
                            RAD_SHARD_COL, gl);
        y.w_fpost  = decl_w(b, nm.f("blk.%lld.ffn_post_norm.weight", ll), RAD_F32, {E}, A, N, gl);
        /* bf16 and one element: the reference multiplies the bf16 stream by it. */
        y.w_scalar = decl_w(b, nm.f("blk.%lld.layer_scalar", ll), RAD_BF16, {1}, A, N, gl);

        /* The routed arm's two input norm gains (the router's and the experts' pre-norm) are
         * declared here, before the attention join, which writes both inputs where
         * `norm_add_norm_route` is served; an op can only name weights already declared. */
        if (m.moe) {
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_norm_exps.weight", ll),
                                  src("pre_feedforward_layernorm_2.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_gate_inp.scale", ll), src("router.scale")));
            y.w_enorm  = decl_w(b, nm.f("blk.%lld.ffn_norm_exps.weight", ll), RAD_F32, {E}, A, N,
                                gl);
            y.w_rscale = decl_w(b, nm.f("blk.%lld.ffn_gate_inp.scale", ll), RAD_F32, {E}, A, N, gl);
        }

        /* The attention join. On a layer whose gate_up is quantised it also writes the routed
         * arm's two inputs: the router's norm (rmsnorm with mult hidden^-0.5) and the experts'
         * codes (as rmsnorm_quant_fp8, or for rotated int8 experts rmsnorm_had_quant_i8g),
         * saving two launches and two reads of x per layer. Otherwise norm_add_norm, else the
         * chain. */
        if (has_join_r && m.moe && y.gu_q)
            y.op_join_r = rw(b, RAD_OP(b, join_r,
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                               RAD_F64("eps", eps), RAD_STR("dtype", "bf16"),
                                               RAD_F64("mult", 1.0 / std::sqrt((double)E)),
                                               RAD_INT("group", RAD_SCALE_GROUP)),
                                    RAD_WEIGHTS(y.w_apost, y.w_fnorm, y.w_rscale, y.w_enorm)),
                             {m.b_o, m.b_x}, {m.b_x, m.b_h, m.b_rn, m.b_hq, m.b_hqs});
        if (has_join_a && !y.op_join_r)
            y.op_join_a = rw(b, RAD_OP(b, "norm_add_norm",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                               RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(y.w_apost, y.w_fnorm)),
                             {m.b_o, m.b_x}, {m.b_x, m.b_h});
        if (!y.op_join_a && !y.op_join_r) {
            y.op_apost = rmsnorm(T, E, y.w_apost, 1.0, m.b_o, m.b_o);
            y.op_add_a = binop("add", T, E, m.b_x, m.b_o, m.b_x);
            y.op_fnorm = rmsnorm(T, E, y.w_fnorm, 1.0, m.b_x, m.b_h);
        }
        y.op_gu    = gemm(T, 2 * m.n_ff, E, y.w_gu, m.b_h, m.b_gu);
        y.op_act   = act_mul(T, m.n_ff, m.b_gu, m.b_act);
        /* The dense partial into rows [0, T) of the join. */
        y.op_dn    = gemm(T, E, m.n_ff, y.w_dn, m.b_act, m.b_ff2);

        /* ------------------------------------------------ the routed arm */
        if (m.moe) {
            const int64_t P = m.moe_rows, K = m.top_k, NL = m.n_local;
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_post_norm1.weight", ll),
                                  src("post_feedforward_layernorm_1.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_post_norm2.weight", ll),
                                  src("post_feedforward_layernorm_2.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_gate_inp.weight", ll), src("router.proj.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_exp_scale", ll), src("router.per_expert_scale")));
            y.w_fpost1 = decl_w(b, nm.f("blk.%lld.ffn_post_norm1.weight", ll), RAD_F32, {E}, A, N, gl);
            y.w_fpost2 = decl_w(b, nm.f("blk.%lld.ffn_post_norm2.weight", ll), RAD_F32, {E}, A, N, gl);
            /* The router is replicated (rad_block_moe_q.h): every rank routes every token the
             * same way from the same residual stream. */
            y.w_router = decl_w(b, nm.f("blk.%lld.ffn_gate_inp.weight", ll), RAD_BF16,
                                {m.n_expert, E}, A, N, gl);
            y.w_escale = decl_w(b, nm.f("blk.%lld.ffn_exp_scale", ll), RAD_BF16, {m.n_expert}, A,
                                N, gl);
            /* The experts: this rank's run of whole experts (name maps and encodings were settled
             * before the buffers). Each projection's handles are declared in its GEMM's table
             * order, since a table is a run of consecutively declared handles: bf16 is one run in
             * expert order; block fp8 or int8 is the even local experts' codes, then the odd ones',
             * then the scale views in the same two runs (`moe_gemm_q` takes two tables by parity;
             * an int8 expert's scale is per row, block fp8's per 128 rows). Codes
             * and scale plane are two views of one stored weight and one movement unit (slot 0
             * gate_up, 1 down). */
            const uint32_t cdt = m.exp_i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3;
            auto experts = [&](bool fp8, const char* what, int slot, int64_t rows, int64_t cols,
                               std::vector<rad_weight>& wc, std::vector<rad_weight>& ws) {
                wc.assign((size_t)NL, 0);
                ws.clear();
                if (!fp8) {
                    for (int64_t e = 0; e < NL; ++e) {
                        const long long ge = (long long)(m.expert_base + e);
                        wc[(size_t)e] = decl_w(b, nm.f("blk.%lld.ffn_%s_exps.%lld.weight", ll, what, ge),
                                               RAD_BF16, {rows, cols}, RAD_ACCESS_CONDITIONAL, N,
                                               grp_expert_slot(li, (int)e, slot));
                        if (!wc[(size_t)e]) return RAD_E_INVAL;
                    }
                    return RAD_OK;
                }
                ws.assign((size_t)NL, 0);
                const int64_t half = NL / 2;
                for (int pass = 0; pass < 2; ++pass)          /* codes, then scales */
                    for (int par = 0; par < 2; ++par)         /* even, then odd */
                        for (int64_t i = 0; i < half; ++i) {
                            const int64_t e = 2 * i + par;
                            const long long ge = (long long)(m.expert_base + e);
                            const char* wn = nm.f("blk.%lld.ffn_%s_exps.%lld.weight", ll, what, ge);
                            rad_weight& h = (pass ? ws : wc)[(size_t)(par * half + i)];
                            h = pass ? decl_view(b, nm.f("blk.%lld.ffn_%s_exps.%lld.scale", ll, what, ge),
                                                 wn, "scale", RAD_BF16,
                                                 {m.exp_i8 ? rows : scale_groups(rows),
                                                  scale_groups(cols)},
                                                 RAD_ACCESS_CONDITIONAL, N,
                                                 grp_expert_slot(li, (int)e, slot))
                                     : decl_view(b, wn, nullptr, "codes", cdt, {rows, cols},
                                                 RAD_ACCESS_CONDITIONAL, N,
                                                 grp_expert_slot(li, (int)e, slot));
                            if (!h) return RAD_E_INVAL;
                        }
                return RAD_OK;
            };
            RAD_ARCH_TRY(experts(y.gu_q, "gate_up", 0, 2 * m.n_ffe, E, y.w_egu, y.w_egus));
            RAD_ARCH_TRY(experts(y.dn_q, "down", 1, E, m.n_ffe, y.w_edn, y.w_edns));
            /* An fp8 projection's op takes its weights in its table operands' order: even codes,
             * even scales, odd codes, odd scales. */
            auto tables = [&](const std::vector<rad_weight>& wc, const std::vector<rad_weight>& ws) {
                const size_t h = wc.size() / 2;
                std::vector<rad_weight> t;
                t.insert(t.end(), wc.begin(), wc.begin() + (long)h);
                t.insert(t.end(), ws.begin(), ws.begin() + (long)h);
                t.insert(t.end(), wc.begin() + (long)h, wc.end());
                t.insert(t.end(), ws.begin() + (long)h, ws.end());
                return t;
            };

            y.op_rnorm  = rmsnorm(P, E, y.w_rscale, 1.0 / std::sqrt((double)E), m.b_x, m.b_rn);
            y.op_router = gemm(P, m.n_expert, E, y.w_router, m.b_rn, m.b_logit);
            y.op_topk   = rw(b, RAD_OP(b, "router_topk",
                                  RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n_expert", m.n_expert),
                                             RAD_INT("top_k", K), RAD_INT("norm", 1),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS),
                              {m.b_logit}, {m.b_ids, m.b_ew});
            /* per_expert_scale[expert] times the routing weight: one launch where `expert_scale` is
             * served, otherwise the [n_expert, 1] table gathered by id and a `mul`; the same bf16
             * product either way (docs/OPS.md). */
            if (has_escale)
                y.op_expert_scale = rw(b, RAD_OP(b, "expert_scale",
                                           RAD_PARAMS(RAD_RANGE("M", 1, P),
                                                      RAD_INT("n_expert", m.n_expert),
                                                      RAD_INT("top_k", K), RAD_STR("dtype", "bf16")),
                                           RAD_WEIGHTS(y.w_escale)),
                                       {m.b_ids, m.b_ew}, {m.b_ew});
            if (!y.op_expert_scale) {
                y.op_escale = rw(b, RAD_OP(b, "embed_lookup",
                                      RAD_PARAMS(RAD_RANGE("M", 1, P * K), RAD_INT("n_embd", 1),
                                                 RAD_INT("n_vocab", m.n_expert), RAD_STR("dtype", "bf16")),
                                      RAD_WEIGHTS(y.w_escale)),
                                  {m.b_ids}, {m.b_es});
                y.op_emul   = binop("mul", P, K, m.b_ew, m.b_es, m.b_ew);
            }
            /* The experts' input, pre_feedforward_layernorm_2 of the stream. For a quantised
             * gate_up it is normed and quantised in one pass (`rmsnorm_quant_fp8`, or for int8
             * experts the rotating `rmsnorm_had_quant_i8g`) with no bf16 copy, since the table
             * GEMM reads only codes and scales. */
            if (y.gu_q) {
                y.op_enorm = RAD_OP(b, m.exp_i8 ? "rmsnorm_had_quant_i8g" : "rmsnorm_quant_fp8",
                                    RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", E),
                                               RAD_F64("eps", eps),
                                               RAD_INT("group", RAD_SCALE_GROUP),
                                               RAD_STR("dtype", "bf16"), RAD_F64("wadd", 0.0)),
                                    RAD_WEIGHTS(y.w_enorm));
                y.op_enorm = rw(b, y.op_enorm, {m.b_x}, {m.b_hq, m.b_hqs});
            } else {
                y.op_enorm = rmsnorm(P, E, y.w_enorm, 1.0, m.b_x, m.b_hx);
            }
            y.op_scatter = rw(b, RAD_OP(b, "moe_scatter",
                                   RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n_expert", NL),
                                              RAD_INT("top_k", K),
                                              RAD_INT("expert_base", m.expert_base)),
                                   RAD_NOWEIGHTS),
                               {m.b_ids}, {m.b_sorted, m.b_eoff, m.b_ecnt});
            /* The router, its top-k and the sort in one launch at decode heights, where each is a
             * small workgroup's work behind a full dispatch. `router_topk_scatter` fuses top-k and
             * sort, bounded by the sort's single-workgroup form (kMoeSortRows routing slots);
             * `router_gemm_topk_scatter` adds the router GEMM, up to the skinny GEMM's decode band.
             * Both are byte-identical to the separate launches (docs/OPS.md). The per-expert scale
             * and the experts' input norm sit between them in issue order and read only the ids
             * and weights the fused op writes. Declared after every buffer they name; a late
             * declaration only lengthens a buffer's planned life (rad_block_moe_q.h). */
            {
                const int64_t sort_rows = P * K <= kMoeSortRows ? P : kMoeSortRows / K;
                if (sort_rows > 0 && rad_op_declared(b, "router_topk_scatter"))
                    y.op_topk_scatter = rw(b, RAD_OP(b, "router_topk_scatter",
                                               RAD_PARAMS(RAD_RANGE("M", 1, sort_rows),
                                                          RAD_INT("n_expert", m.n_expert),
                                                          RAD_INT("n_local", NL), RAD_INT("top_k", K),
                                                          RAD_INT("norm", 1), RAD_STR("dtype", "bf16"),
                                                          RAD_INT("expert_base", m.expert_base)),
                                               RAD_NOWEIGHTS),
                                           {m.b_logit}, {m.b_ids, m.b_ew, m.b_sorted, m.b_eoff, m.b_ecnt});
                if (y.op_topk_scatter) {
                    y.topk_scatter_rows = sort_rows;
                    const int64_t fr = sort_rows < kMoeRouterFusedRows ? sort_rows : kMoeRouterFusedRows;
                    if (rad_op_declared(b, "router_gemm_topk_scatter"))
                        y.op_router_fused = rw(b, RAD_OP(b, "router_gemm_topk_scatter",
                                               RAD_PARAMS(RAD_RANGE("M", 1, fr), RAD_INT("K", E),
                                                          RAD_INT("n_expert", m.n_expert),
                                                          RAD_INT("n_local", NL), RAD_INT("top_k", K),
                                                          RAD_INT("norm", 1), RAD_STR("dtype", "bf16"),
                                                          RAD_INT("expert_base", m.expert_base)),
                                               RAD_WEIGHTS(y.w_router)),
                                           {m.b_rn},
                                           {m.b_logit, m.b_ids, m.b_ew, m.b_sorted, m.b_eoff, m.b_ecnt});
                    if (y.op_router_fused) y.router_fused_rows = fr;
                }
            }
            /* The two grouped GEMMs use different a_order: gate_up reads the block input, one row
             * per token, gathered through sorted_tok; down reads gate_up's product, already one row
             * per (token, slot) in scatter order (rad_block_moe_q.h). */
            if (y.gu_q) {
                const std::vector<rad_weight> tab = tables(y.w_egu, y.w_egus);
                /* `a_group` is stated to the int8 rows alone, last and cut off otherwise. */
                const RadParam ps[] = { RAD_RANGE("M", 1, P), RAD_INT("N", 2 * m.n_ffe),
                                        RAD_INT("K", E), RAD_INT("n_expert", NL),
                                        RAD_INT("top_k", K), RAD_INT("group", RAD_SCALE_GROUP),
                                        RAD_STR("a_order", "token"),
                                        RAD_STR("dtype", m.exp_i8 ? "i8a8h" : "fp8a8"),
                                        RAD_INT("a_group", RAD_SCALE_GROUP) };
                y.op_egu = rw(b, rad_decl_op(b, "moe_gemm_q", ps,
                                     (int)(sizeof ps / sizeof *ps) - (m.exp_i8 ? 0 : 1),
                                     tab.data(), (int)tab.size()),
                              {m.b_hq, m.b_hqs, m.b_sorted, m.b_eoff}, {m.b_egu});
            } else {
                y.op_egu = rw(b, rad_decl_op(b, "moe_gemm",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", 2 * m.n_ffe),
                                                RAD_INT("K", E), RAD_INT("n_expert", NL),
                                                RAD_INT("top_k", K), RAD_STR("a_order", "token"),
                                                RAD_STR("dtype", "bf16")),
                                     y.w_egu.data(), (int)NL),
                              {m.b_hx, m.b_sorted, m.b_eoff}, {m.b_egu});
            }
            /* gelu_tanh(gate) * up over the two column halves. An fp8 down then quantises the
             * product on its own launch: libr4d's fused gated quantiser is compiled for silu. */
            y.op_eact = act_mul(P * K, m.n_ffe, m.b_egu, m.b_eact);
            if (y.dn_q) {
                y.op_eq = rw(b, RAD_OP(b, m.exp_i8 ? "had_quant_act_i8g" : "quant_act_fp8",
                                    RAD_PARAMS(RAD_RANGE("M", 1, P * K), RAD_INT("n", m.n_ffe),
                                               RAD_INT("group", RAD_SCALE_GROUP),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                             {m.b_eact}, {m.b_eq, m.b_eqs});
                const std::vector<rad_weight> tab = tables(y.w_edn, y.w_edns);
                const RadParam ps[] = { RAD_RANGE("M", 1, P), RAD_INT("N", E),
                                        RAD_INT("K", m.n_ffe), RAD_INT("n_expert", NL),
                                        RAD_INT("top_k", K), RAD_INT("group", RAD_SCALE_GROUP),
                                        RAD_STR("a_order", "sorted"),
                                        RAD_STR("dtype", m.exp_i8 ? "i8a8h" : "fp8a8"),
                                        RAD_INT("a_group", RAD_SCALE_GROUP) };
                y.op_edn = rw(b, rad_decl_op(b, "moe_gemm_q", ps,
                                     (int)(sizeof ps / sizeof *ps) - (m.exp_i8 ? 0 : 1),
                                     tab.data(), (int)tab.size()),
                              {m.b_eq, m.b_eqs, m.b_sorted, m.b_eoff}, {m.b_edn});
            } else {
                y.op_edn = rw(b, rad_decl_op(b, "moe_gemm",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", E),
                                                RAD_INT("K", m.n_ffe), RAD_INT("n_expert", NL),
                                                RAD_INT("top_k", K), RAD_STR("a_order", "sorted"),
                                                RAD_STR("dtype", "bf16")),
                                     y.w_edn.data(), (int)NL),
                              {m.b_eact, m.b_sorted, m.b_eoff}, {m.b_edn});
            }
            {
                rad_buf rd[3] = { m.b_edn, m.b_ew, m.b_sorted };
                rad_buf wr[1] = { m.b_ff2 };
                y.op_gather = RAD_OP(b, "moe_gather",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", E),
                                                RAD_INT("top_k", K), RAD_STR("dtype", "bf16"),
                                                RAD_STR("act", "none")),
                                     RAD_NOWEIGHTS);
                if (!y.op_gather) return RAD_E_INVAL;
                rad_op_reads(b, y.op_gather, rd, 3);
                rad_op_writes(b, y.op_gather, wr, 1);
            }
            if (!y.op_enorm || !y.op_egu || !y.op_edn || (y.dn_q && !y.op_eq)) return RAD_E_INVAL;
            if (W > 1) y.op_ar_f = allreduce(2 * T * E, m.b_ff2);
            if (has_join_f)
            {
                /* The join also applies the norm that reads its x (the next layer's input norm, or
                 * the final norm after the last layer), saving a launch and a read of x. */
                const rad_weight w_next = l + 1 < m.n_layer ? m.layers[(size_t)l + 1].w_anorm
                                                            : m.w_onorm;
                y.op_join_f = rw(b, RAD_OP(b, "join_add_scale",
                                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                                   RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                                        RAD_WEIGHTS(y.w_fpost1, y.w_fpost2, y.w_fpost, y.w_scalar,
                                                    w_next)),
                                 {m.b_ff2, m.b_x}, {m.b_x, m.b_h});
                if (y.op_join_f) { y.next_norm = true; y.w_next = w_next; }
            }
            if (!y.op_join_f) {
                y.op_fpost1 = rmsnorm(T, E, y.w_fpost1, 1.0, m.b_ff2, m.b_ff2);
                y.op_fpost2 = rmsnorm(T, E, y.w_fpost2, 1.0, m.b_ff2, m.b_ff2);
                y.op_fsum   = binop("add", T, E, m.b_ff2, m.b_ff2, m.b_h);
                y.op_fpost  = rmsnorm(T, E, y.w_fpost, 1.0, m.b_h, m.b_h);
            }
        } else {
            if (W > 1) y.op_ar_f = allreduce(T * E, m.b_ff2);
            y.op_fpost = rmsnorm(T, E, y.w_fpost, 1.0, m.b_ff2, m.b_h);
        }
        if (!y.op_join_f) {
            y.op_add_f = binop("add", T, E, m.b_x, m.b_h, m.b_x);
            y.op_scale = rw(b, RAD_OP(b, "mul_weight",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                              RAD_STR("dtype", "bf16"), RAD_INT("scalar", 1)),
                                   RAD_WEIGHTS(y.w_scalar)),
                               {m.b_x}, {m.b_x});
        }
        n_joins += (y.op_join_a || y.op_join_r ? 1 : 0) + (y.op_join_f ? 1 : 0);
        n_route += y.op_join_r ? 1 : 0;

        /* The tap: a row copy of the layer's output into its column slice, declared here after
         * the layer's last op because step() issues it here and declaration order is the
         * planner's liveness. Declared after the whole trunk instead, `dflash_src` would look
         * born at the end and could alias buffers every layer writes (e.g. `moe_edn`), corrupting
         * what the drafter reads while the output stays correct. */
        for (int t = 0; t < m.n_taps; ++t)
            if (m.tap_layer[t] == (int)l && !y.op_tap)
                y.op_tap = cast_op(b, T, E, "bf16", "bf16", m.b_x, m.b_dfsrc);
    }

    /* ---- the prefill overlap (trunk_overlap). Requires a reduction to hide, a routed arm whose
     * passes divide a slice, and every join fused (the slices issue only the fused forms). Exact
     * wire only, since nothing shows the lossy wire gives a slice the same bits as the whole; on
     * the exact wire the two slices reproduce the whole chunk's text and logprobs exactly. The two
     * buffers the second lane reduces in place leave the arena's sharing (rad_buf_concurrent);
     * both already live across the trunk, so this only costs sharing with the drafter's and the
     * head's buffers. */
    m.overlap = W > 1 && m.moe && g.wire_exact && T >= 2 * m.moe_rows;
    for (const Layer& y : m.layers)
        m.overlap = m.overlap && (y.op_join_a || y.op_join_r) && y.op_join_f && y.next_norm &&
                    y.op_ar_o && y.op_ar_f;
    /* ---- the two arms side by side (trunk), in a step that runs whole: the dense arm on the
     * second lane beside the routed arm, which needs nothing from it until the feed-forward join.
     * At decode the dense arm is three latency-bound launches (~36 us against the routed arm's
     * ~120 us per layer on 4x R9700), so it hides behind the experts' bandwidth. No collective
     * rides it, so the reductions stay on lane 0. Its buffers leave the arena's sharing. */
    m.arms = m.moe;
    /* ---- the calibration tap. It reads each dense linear's bf16 input where trunk issues the
     * linear, so a calibration runs every step whole and on one lane. The files are named after
     * the weights the 31B's w4a8 container quantises (gemma4_bf16: blk.N.attn_qkv, attn_o,
     * ffn_gate_up, ffn_down), which a GPTQ recipe reads with calib=DIR: the Hessians of the bf16
     * model, whose activations a quantised container's own tap would see through its quantisers.
     * The routed arm is not tapped. */
    if (calib_dir()) {
        m.overlap = false;
        m.arms = false;
        RAD_ARCH_TRY(calib_layers("gemma4", "RADIANCE_G4_CALIB_LAYERS", (int64_t)m.layers.size(),
                                  &m.calib_lo, &m.calib_hi));
        for (int64_t l = m.calib_lo; l < m.calib_hi; ++l) {
            Layer& y = m.layers[(size_t)l];
            const Kind& k = y.full ? m.kf : m.ks;
            const int i = (int)l;
            RAD_ARCH_TRY(y.tap_in.declare(b, nm, T, nm.f("blk.%d.attn_in", i), m.b_h, E));
            RAD_ARCH_TRY(y.tap_o.declare(b, nm, T, nm.f("blk.%d.attn_o", i),
                                         y.full ? m.b_attf : m.b_atts, k.q()));
            RAD_ARCH_TRY(y.tap_gu.declare(b, nm, T, nm.f("blk.%d.ffn_gate_up", i), m.b_h, E));
            RAD_ARCH_TRY(y.tap_dn.declare(b, nm, T, nm.f("blk.%d.ffn_down", i), m.b_act, m.n_ff));
        }
        m.calib = true;
        m.tap_live = true;
    }
    if (m.overlap) RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_o));
    if (m.overlap || m.arms) RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_ff2));
    if (m.arms)
        for (rad_buf bb : { m.b_h, m.b_gu, m.b_act }) RAD_ARCH_TRY(rad_buf_concurrent(b, bb));

    if (m.layers.empty() || !m.layers.back().next_norm)
        m.op_onorm = rmsnorm(T, E, m.w_onorm, 1.0, m.b_x, m.b_h);
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", E),
                                     RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                      {m.b_h}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                     RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", E),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(m.w_tok)),
                      {m.b_hout}, {m.b_logits});
    /* Final logit softcapping, cap * tanh(x / cap) over the f32 logits (`logit_softcap`, served by
     * libref, libavx and libr4d). Without it greedy decoding is unchanged (the cap is strictly
     * increasing) but sampling draws from a sharper distribution, so the startup log warns. */
    if (m.softcap > 0.0) {
        m.op_cap = rw(b, RAD_OP(b, "logit_softcap",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                      RAD_INT("n", g.n_vocab), RAD_F64("cap", m.softcap),
                                      RAD_STR("dtype", "f32")),
                           RAD_NOWEIGHTS),
                       {m.b_logits}, {m.b_logits});
        if (!m.op_cap)
            rad_note(b, "warning: no kernel serves `logit_softcap` (load libr4d or libavx), so "
                        "the logits skip %g * tanh(x / %g): greedy output is unchanged, sampled "
                        "output draws from a sharper distribution", m.softcap, m.softcap);
    }

    /* ---- the assistant MTP head, when the container carries one (rad-convert --draft-model) and the
     * deployment speculates; otherwise its weights stay off the card. It reads the trunk's own cache
     * (no cache of its own) and proposes up to its trained depth. */
    if (ctx->max_spec > 0 && has_assistant(meta)) {
        int last_s = -1, last_f = -1;
        /* this plugin has no KV-sharing: every layer keeps its own cache, so the last layer of each
         * class is simply the last sliding / full layer. */
        for (int64_t l = 0; l < m.n_layer; ++l)
            (m.layers[(size_t)l].full ? last_f : last_s) = (int)l;
        RAD_ARCH_TRY(m.head.declare(b, m.nm, meta, m.g, class_from_kind(m.ks, false), class_from_kind(m.kf, true),
                                    m.ks.kv, m.kf.kv, last_s, last_f, m.n_layer, m.w_tok, m.b_h, m.b_logits));
        m.have_head = true;

        /* Serial drafter: `max_spec` dependent passes, each embedding the last one's proposal,
         * through the ordinary sampler chain (proposal 0). The head has no cache of its own, so
         * the core stages every pass at the committed position. */
        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec;
        dd.proposal       = 0;
        dd.proposal_pitch = 1;
        dd.mask_token     = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        RAD_ARCH_TRY(rad_drafter_shared_kv(b, kHeadPosOffset));
    }

    /* The drafter follows the trunk: declaration order is the buffer planner's liveness. */
    if (m.n_taps > 0)
        RAD_ARCH_TRY(declare_dflash(b, meta, ctx, m, (int)m.n_layer));
    if (vision) {
        /* Reserve the draft groups even when speculation is disabled. */
        const int64_t draft_layers = dflash_kind(meta) == 1
            ? rad_meta_geti(meta, "draft.num_hidden_layers", 0) : 0;
        RAD_ARCH_TRY(m.vis.declare(b, meta, ctx, nm, E, (int)(m.n_layer + draft_layers),
                                  "model.", "gemma4"));
        /* A declaration without a weight source is valid for graph tools and tests. When
         * the trunk is backed by weights, check the tower's entry and exit before loading. */
        RadEncoding enc{};
        if (rad_weight_encoding(b, nm.f("token_embd.weight"), &enc, nullptr, nullptr) == RAD_OK) {
            for (const char* name : { "v.patch.weight", "v.embed.weight" }) {
                if (rad_weight_encoding(b, nm.f("%s", name), &enc, nullptr, nullptr) != RAD_E_NOTFOUND)
                    continue;
                fprintf(stderr, "radiance: gemma4: vision is enabled but the model is missing "
                                "vision weight '%s'; use --mm-max-patches 0 for text-only serving\n",
                        name);
                return RAD_E_NOTFOUND;
            }
        }
    }

    int nfull = 0;
    for (int f : full) nfull += f;
    rad_note(b, "rank %d/%d: %lld layers (%d full attention %lld x %lld over %lld kv%s, %lld "
                "sliding %lld x %lld over %lld kv, window %lld); dense ffn %lld; %s; vocab "
                "[%lld,%lld) tied",
             R, W, (long long)m.n_layer, nfull, (long long)m.kf.nh, (long long)m.kf.hd,
             (long long)m.kf.nkv, m.kf.kv_repl ? " replicated" : "",
             (long long)(m.n_layer - nfull), (long long)m.ks.nh, (long long)m.ks.hd,
             (long long)m.ks.nkv, (long long)m.ks.window, (long long)m.n_ff,
             m.moe ? "routed: experts [base, base + local) of the model's, top-k" : "no routed arm",
             (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "sandwich joins: %d of %lld fused (norm_add_norm / join_add_scale)%s",
             n_joins, (long long)(m.moe ? 2 * m.n_layer : m.n_layer),
             n_joins == 0 ? " -- not served, the rmsnorm / add / mul_weight chain" : "");
    if (m.moe && m.n_gu_q > 0)
        rad_note(b, "routed inputs: %d of %d quantised layers take the router's norm and the "
                    "experts' codes from the attention join (%s)%s",
                 n_route, m.n_gu_q, join_r,
                 n_route ? "" : m.exp_i8 ? " -- the rmsnorm and rmsnorm_had_quant_i8g launches instead"
                                         : " -- the rmsnorm and rmsnorm_quant_fp8 launches instead");
    if (m.ks.nh > 0)
        rad_note(b, "sliding attention: q %s", m.qrow == 1
                    ? "read in the fused qkv row (attn_paged_qrow)"
                    : "lifted into a dense block for attn_paged (attn_paged_qrow not served)");
    if (m.moe) {
        auto how = [&](int n_fp8) {
            return n_fp8 == 0 ? "bf16" : n_fp8 < m.n_layer ? "mixed"
                 : m.exp_i8 ? "rotated int8" : "block fp8";
        };
        rad_note(b, "routed arm: %lld local experts from %lld of %lld, top %lld, %lld wide; "
                    "gate_up %s (%d of %lld layers quantised), down %s (%d quantised); passes of "
                    "%lld tokens",
                 (long long)m.n_local, (long long)m.expert_base, (long long)m.n_expert,
                 (long long)m.top_k, (long long)m.n_ffe, how(m.n_gu_q), m.n_gu_q,
                 (long long)m.n_layer, how(m.n_dn_q), m.n_dn_q, (long long)m.moe_rows);
        if (W > 1)
            rad_note(b, "prefill overlap: %s", m.overlap
                        ? "a chunk of two routed passes or more runs as two slices, each slice's "
                          "all-reduces on a second stream beside the other's compute"
                        : !g.wire_exact ? "off (the lossy all-reduce wire)"
                        : T < 2 * m.moe_rows ? "off (a step holds fewer than two routed passes)"
                        : "off (the sandwich joins are not fused)");
        rad_note(b, "the two arms: in a step that runs whole, the dense arm on a second stream "
                    "beside the routed");
    }
    if (m.kf.rot_pairs < m.kf.hd / 2)
        rad_note(b, "full layers: proportional rope, %lld of %lld frequency pairs rotating at "
                    "theta %g over the whole %lld-wide head and the rest at frequency zero",
                 (long long)m.kf.rot_pairs, (long long)(m.kf.hd / 2), m.kf.theta,
                 (long long)m.kf.hd);
    return RAD_OK;
}

/* ============================================================== step */

/* The routed arm over rows [first, first + rows) of a T-token step, in passes of m.moe_rows,
 * writing its partial into rows [dst, dst + rows) of the join: [T, 2T) for a whole step, or right
 * after the slice's dense rows for a slice (trunk_overlap). */
static void routed(RadCtx* c, const Model& m, const Layer& y, int64_t T, int64_t first,
                   int64_t rows, int64_t dst) {
    const int64_t E = m.E, K = m.top_k, NL = m.n_local, NE = m.n_expert;
    const int64_t end = first + rows;
    for (int64_t r0 = first; r0 < end; r0 += m.moe_rows) {
        const int64_t n = (end - r0 < m.moe_rows) ? (end - r0) : m.moe_rows;
        int32_t* const counts = rad_route_counts(c, (int)(&y - m.layers.data()), NL);
        const RadOperand ecnt = counts ? praw(counts, RAD_I32, NL) : brows(m.b_ecnt, NL);
        if (!y.op_join_r)   /* the attention join wrote the router's input otherwise */
            RAD_ISSUE_N(c, y.op_rnorm, n, brow_slice(m.b_x, r0, n, E), RAD_W(y.w_rscale),
                        brow_slice(m.b_rn, r0, n, E));
        /* The router, its top-k and the sort in one, two or three launches (see declare). */
        const bool rfused = y.op_router_fused && n <= y.router_fused_rows;
        const bool pair = !rfused && y.op_topk_scatter && n <= y.topk_scatter_rows;
        if (rfused) {
            RAD_ISSUE_N(c, y.op_router_fused, n, brow_slice(m.b_rn, r0, n, E), RAD_W(y.w_router),
                        RAD_NONE, brow_slice(m.b_logit, r0, n, NE),
                        brow_slice(m.b_ids, r0, n, K), brow_slice(m.b_ew, r0, n, K),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1), ecnt, RAD_NONE);
        } else {
            RAD_ISSUE_N(c, y.op_router, n, brow_slice(m.b_rn, r0, n, E), RAD_W(y.w_router),
                        brow_slice(m.b_logit, r0, n, NE));
            if (pair)
                RAD_ISSUE_N(c, y.op_topk_scatter, n, brow_slice(m.b_logit, r0, n, NE),
                            brow_slice(m.b_ids, r0, n, K), brow_slice(m.b_ew, r0, n, K),
                            brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1), ecnt);
            else
                RAD_ISSUE_N(c, y.op_topk, n, brow_slice(m.b_logit, r0, n, NE),
                            brow_slice(m.b_ids, r0, n, K), brow_slice(m.b_ew, r0, n, K));
        }
        /* The ids and gathered scales are read as flat [n * K] vectors over the same bytes. */
        int32_t* ids = (int32_t*)rad_buf_ptr(c, m.b_ids);
        uint16_t* es = (uint16_t*)rad_buf_ptr(c, m.b_es);
        if (y.op_expert_scale) {
            RAD_ISSUE_N(c, y.op_expert_scale, n, brow_slice(m.b_ids, r0, n, K), RAD_W(y.w_escale),
                        brow_slice(m.b_ew, r0, n, K));
        } else {
            RAD_ISSUE_N(c, y.op_escale, n * K, praw(ids ? ids + r0 * K : nullptr, RAD_I32, n * K),
                        RAD_W(y.w_escale), praw2(es ? es + r0 * K : nullptr, RAD_BF16, n * K, 1));
            RAD_ISSUE_N(c, y.op_emul, n, brow_slice(m.b_ew, r0, n, K), brow_slice(m.b_es, r0, n, K),
                        brow_slice(m.b_ew, r0, n, K));
        }
        /* rmsnorm_quant_fp8's operands: x, residual (absent), w, q, scale, out_bf16 (absent);
         * issued only where the attention join did not already write the codes. */
        if (y.gu_q && !y.op_join_r)
            RAD_ISSUE_N(c, y.op_enorm, n, brow_slice(m.b_x, r0, n, E), RAD_NONE, RAD_W(y.w_enorm),
                        brow_slice(m.b_hq, r0, n, E),
                        brow_slice(m.b_hqs, r0, n, E / RAD_SCALE_GROUP),
                        RAD_NONE);
        else if (!y.gu_q)
            RAD_ISSUE_N(c, y.op_enorm, n, brow_slice(m.b_x, r0, n, E), RAD_W(y.w_enorm),
                        brow_slice(m.b_hx, r0, n, E));
        if (!rfused && !pair)
            RAD_ISSUE_N(c, y.op_scatter, n, brow_slice(m.b_ids, r0, n, K), brows(m.b_sorted, n * K),
                        brows(m.b_eoff, NL + 1), ecnt);
        /* An fp8 projection's tables: the first handle of each parity's run and its length --
         * even codes, even scales, odd codes, odd scales (declare). */
        const int64_t H = NL / 2;
        if (y.gu_q)
            RAD_ISSUE_N(c, y.op_egu, n, brow_slice(m.b_hq, r0, n, E),
                        brow_slice(m.b_hqs, r0, n, E / RAD_SCALE_GROUP),
                        RAD_WTAB(y.w_egu[0], H), RAD_WTAB(y.w_egus[0], H),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1),
                        RAD_WTAB(y.w_egu[(size_t)H], H), RAD_WTAB(y.w_egus[(size_t)H], H),
                        brows(m.b_egu, n * K));
        else
            RAD_ISSUE_N(c, y.op_egu, n, brow_slice(m.b_hx, r0, n, E), RAD_WTAB(y.w_egu[0], NL),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1), brows(m.b_egu, n * K));
        RAD_ISSUE_N(c, y.op_eact, n * K, bcol(m.b_egu, 0, m.n_ffe, n * K),
                    bcol(m.b_egu, m.n_ffe, m.n_ffe, n * K), brows(m.b_eact, n * K));
        if (y.dn_q) {
            RAD_ISSUE_N(c, y.op_eq, n * K, brows(m.b_eact, n * K), brows(m.b_eq, n * K),
                        brows(m.b_eqs, n * K));
            RAD_ISSUE_N(c, y.op_edn, n, brows(m.b_eq, n * K), brows(m.b_eqs, n * K),
                        RAD_WTAB(y.w_edn[0], H), RAD_WTAB(y.w_edns[0], H),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1),
                        RAD_WTAB(y.w_edn[(size_t)H], H), RAD_WTAB(y.w_edns[(size_t)H], H),
                        brows(m.b_edn, n * K));
        } else {
            RAD_ISSUE_N(c, y.op_edn, n, brows(m.b_eact, n * K), RAD_WTAB(y.w_edn[0], NL),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1), brows(m.b_edn, n * K));
        }
        /* The routed partial into its rows of the join. */
        RAD_ISSUE_N(c, y.op_gather, n, brows(m.b_edn, n * K), brow_slice(m.b_ew, r0, n, K),
                    brows(m.b_sorted, n * K), RAD_NONE, RAD_NONE,
                    brow_slice(m.b_ff2, dst + (r0 - first), n, E));
        if (r0 + n >= T) {
            RadRouting rt{};
            rt.expert_ids    = ids;
            rt.expert_w      = nullptr;   /* bf16 weights; the field is const float* (rad_block_moe_q.h) */
            rt.expert_count  = counts ? counts : (const int32_t*)rad_buf_ptr(c, m.b_ecnt);
            rt.sorted_tok    = (const int32_t*)rad_buf_ptr(c, m.b_sorted);
            rt.expert_offset = (const int32_t*)rad_buf_ptr(c, m.b_eoff);
            rt.top_k         = K;
            rt.n_expert      = NL;
            rad_route_report(c, (int)(&y - m.layers.data()), &rt);
        }
    }
}

/* A layer's attention between its projections and its output projection, over the whole step:
 * the norms and the rotaries, the cache write, the attention. */
static void attend(RadCtx* c, const Model& m, const Layer& y, size_t li, const RadBatch* batch) {
    const int64_t T = batch->n_tok;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu  = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);
    const Kind& k = y.full ? m.kf : m.ks;
    const RadKVGroupBatch* kvb = kv_batch(batch, k.kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;
    const int64_t pitch = kvb ? kvb->block_table_pitch : 0;
    const RadOperand bt  = praw2(table, RAD_I32, batch->n_seq, pitch);
    const RadOperand su  = praw(used, RAD_I32, batch->n_seq);

    if (!y.full) {
        const int64_t qd = k.q(), kvd = k.kv_dim();
        RAD_ISSUE(c, y.op_qknr, brows(m.b_qkv, T), pos, RAD_W(y.w_qn), RAD_W(y.w_kn));
        RAD_ISSUE(c, y.op_store, bcol(m.b_qkv, qd, kvd, T), bcol(m.b_qkv, qd + kvd, kvd, T),
                  praw(slot, RAD_I32, T), kv_cache(k.kv, (int)li));
        if (y.op_lift) RAD_ISSUE(c, y.op_lift, bcol(m.b_qkv, 0, qd, T), brows(m.b_qs, T));
        RAD_ISSUE_N(c, y.op_attn, batch->max_q_len,
                    y.op_lift ? brows(m.b_qs, T) : bcol(m.b_qkv, 0, qd, T),
                    kv_cache(k.kv, (int)li), bt, su, RAD_NONE, RAD_NONE, cu, brows(m.b_atts, T));
        /* Recompute only the image rows, with the cache ending at the span's last row.
         * The window still limits lookback; for a long span it must not limit lookahead. */
        for (int64_t i = 0; y.op_bidi && i < batch->n_bidi; ++i) {
            const int64_t first = batch->bidi_row[i], n = batch->bidi_n[i], s = batch->bidi_seq[i];
            const rad_op op = n > k.window ? y.op_ahead : y.op_bidi;
            if (!op) {
                rad_step_fail(c, "gemma4: an image span exceeds the sliding window and needs "
                                 "a paged attention kernel with unrestricted lookahead");
                return;
            }
            RAD_ISSUE_N(c, op, n,
                        y.op_lift ? brow_slice(m.b_qs, first, n, qd)
                                  : bcol_at(m.b_qkv, first, qd + 2 * kvd, 0, qd, n),
                        kv_cache(k.kv, (int)li),
                        praw2(table ? table + s * pitch : nullptr, RAD_I32, 1, pitch),
                        praw(batch->bidi_used + i, RAD_I32, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE, brow_slice(m.b_atts, first, n, qd));
        }
    } else {
        RAD_ISSUE_N(c, y.op_vnorm, T * k.nkv, brows(m.b_kf, T), RAD_NONE, brows(m.b_vf, T));
        RAD_ISSUE_N(c, y.op_qn, T * k.nh, brows(m.b_qf, T), RAD_W(y.w_qn), brows(m.b_qf, T));
        RAD_ISSUE_N(c, y.op_kn, T * k.nkv, brows(m.b_kf, T), RAD_W(y.w_kn), brows(m.b_kf, T));
        RAD_ISSUE(c, y.op_qrope, brows(m.b_qf, T), pos);
        RAD_ISSUE(c, y.op_krope, brows(m.b_kf, T), pos);
        RAD_ISSUE(c, y.op_store, brows(m.b_kf, T), brows(m.b_vf, T), praw(slot, RAD_I32, T),
                  kv_cache(k.kv, (int)li));
        RAD_ISSUE_N(c, y.op_attn, batch->max_q_len, brows(m.b_qf, T), kv_cache(k.kv, (int)li),
                    bt, su, RAD_NONE, RAD_NONE, cu, brows(m.b_attf, T));
    }
}

/* The trunk over the whole step, the dense arm beside the routed one where the model says so. */
static void trunk(RadCtx* c, const Model& m, const RadBatch* batch) {
    const int64_t T = batch->n_tok, E = m.E;
    bool lanes_ok = true;
    for (size_t li = 0; li < m.layers.size(); ++li) {
        const Layer& y = m.layers[li];
        if (y.op_anorm) RAD_ISSUE(c, y.op_anorm, brows(m.b_x, T), RAD_W(y.w_anorm), brows(m.b_h, T));
        if (!y.full) {
            RAD_ISSUE(c, y.op_qkv, brows(m.b_h, T), RAD_W(y.w_qkv), brows(m.b_qkv, T));
        } else {
            RAD_ISSUE(c, y.op_q, brows(m.b_h, T), RAD_W(y.w_q), brows(m.b_qf, T));
            RAD_ISSUE(c, y.op_k, brows(m.b_h, T), RAD_W(y.w_k), brows(m.b_kf, T));
        }
        y.tap_in.step(c, m.b_h, T, m.tap_live);
        attend(c, m, y, li, batch);
        RAD_ISSUE(c, y.op_o, brows(y.full ? m.b_attf : m.b_atts, T), RAD_W(y.w_o), brows(m.b_o, T));
        y.tap_o.step(c, y.full ? m.b_attf : m.b_atts, T, m.tap_live);
        if (y.op_ar_o) RAD_ISSUE_N(c, y.op_ar_o, T * E, brows(m.b_o, T), RAD_NONE);
        if (y.op_join_r) {
            RAD_ISSUE(c, y.op_join_r, brows(m.b_o, T), RAD_W(y.w_apost), brows(m.b_x, T),
                      RAD_W(y.w_fnorm), brows(m.b_h, T), RAD_W(y.w_rscale), brows(m.b_rn, T),
                      RAD_W(y.w_enorm), brows(m.b_hq, T), brows(m.b_hqs, T));
        } else if (y.op_join_a) {
            RAD_ISSUE(c, y.op_join_a, brows(m.b_o, T), RAD_W(y.w_apost), brows(m.b_x, T),
                      RAD_W(y.w_fnorm), brows(m.b_h, T));
        } else {
            RAD_ISSUE(c, y.op_apost, brows(m.b_o, T), RAD_W(y.w_apost), brows(m.b_o, T));
            RAD_ISSUE(c, y.op_add_a, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
            RAD_ISSUE(c, y.op_fnorm, brows(m.b_x, T), RAD_W(y.w_fnorm), brows(m.b_h, T));
        }

        /* The dense arm into rows [0, T) of the join. With side-by-side arms it runs on lane 1
         * behind the attention join, and lane 0 joins it back before the reduction. */
        if (m.arms)
            lanes_ok = lanes_ok && rad_lane_join(c, 0, 1) == RAD_OK && rad_lane(c, 1) == RAD_OK;
        RAD_ISSUE(c, y.op_gu, brows(m.b_h, T), RAD_W(y.w_gu), brows(m.b_gu, T));
        y.tap_gu.step(c, m.b_h, T, m.tap_live);
        RAD_ISSUE(c, y.op_act, bcol(m.b_gu, 0, m.n_ff, T), bcol(m.b_gu, m.n_ff, m.n_ff, T),
                  brows(m.b_act, T));
        RAD_ISSUE(c, y.op_dn, brows(m.b_act, T), RAD_W(y.w_dn), brows(m.b_ff2, T));
        y.tap_dn.step(c, m.b_act, T, m.tap_live);
        if (m.arms) lanes_ok = rad_lane(c, 0) == RAD_OK && lanes_ok;
        if (m.moe) {
            routed(c, m, y, T, 0, T, T);
            if (m.arms) lanes_ok = rad_lane_join(c, 1, 0) == RAD_OK && lanes_ok;
            if (y.op_ar_f) RAD_ISSUE_N(c, y.op_ar_f, 2 * T * E, brows(m.b_ff2, 2 * T), RAD_NONE);
            if (y.op_join_f) {
                RAD_ISSUE(c, y.op_join_f, brows(m.b_ff2, T), RAD_W(y.w_fpost1),
                          brow_slice(m.b_ff2, T, T, E), RAD_W(y.w_fpost2), RAD_W(y.w_fpost),
                          brows(m.b_x, T), RAD_W(y.w_scalar), RAD_W(y.w_next), brows(m.b_h, T));
            } else {
                RAD_ISSUE(c, y.op_fpost1, brows(m.b_ff2, T), RAD_W(y.w_fpost1), brows(m.b_ff2, T));
                RAD_ISSUE(c, y.op_fpost2, brow_slice(m.b_ff2, T, T, E), RAD_W(y.w_fpost2),
                          brow_slice(m.b_ff2, T, T, E));
                RAD_ISSUE(c, y.op_fsum, brows(m.b_ff2, T), brow_slice(m.b_ff2, T, T, E), brows(m.b_h, T));
                RAD_ISSUE(c, y.op_fpost, brows(m.b_h, T), RAD_W(y.w_fpost), brows(m.b_h, T));
            }
        } else {
            if (y.op_ar_f) RAD_ISSUE_N(c, y.op_ar_f, T * E, brows(m.b_ff2, T), RAD_NONE);
            RAD_ISSUE(c, y.op_fpost, brows(m.b_ff2, T), RAD_W(y.w_fpost), brows(m.b_h, T));
        }
        if (!y.op_join_f) {
            RAD_ISSUE(c, y.op_add_f, brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));
            RAD_ISSUE(c, y.op_scale, brows(m.b_x, T), RAD_W(y.w_scalar), brows(m.b_x, T));
        }

        /* The tap comes after the layer scalar: HF's hidden_states[i + 1] is the decoder layer's
         * return value, which includes it. */
        for (int t = 0; t < m.n_taps && y.op_tap; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, y.op_tap, brows(m.b_x, T),
                      bcol(m.b_dfsrc, (int64_t)t * E, E, T));
        }
    }
    if (!lanes_ok)
        rad_step_fail(c, "gemma4: the dense arm's second lane could not be ordered against the "
                         "first");
}

/* ============================================================== the prefill overlap */
/* A prefill chunk in two slices, each slice's all-reduces on the second lane (rad_lane) beside
 * the other slice's compute. A layer reduces T x E after attention and 2T x E (both arms) after the
 * feed-forward; at tp 4 on 4x R9700 the reductions were about half of a long prefill (~16.5 GB/s
 * per rank on the link against ~8 ms of compute per layer). Run whole, compute and reduction
 * alternate; cut in two, one slice reduces while the other computes:
 *
 *   lane 0  o(A) o(B) . nan(A) ff(A) . nan(B) ff(B) . jas(A) qkv(A) . jas(B) qkv(B) attn o(A) ..
 *   lane 1       ar_o(A)    ar_o(B)       ar_f(A)         ar_f(B)
 *
 * Attention runs whole: it reads every row's keys, and its batch bounds describe the whole step.
 * Everything else (projections, joins, both arms, and the reduction, which folds each word in fp32
 * in ascending rank order) is row-wise.
 *
 * Same kernels, so same bits (core/runtime/ctx.h, the second lane's fourth rule). The cut is a
 * whole number of routed passes from the first row, so the routed arm makes the same passes as the
 * whole step, and both slices are at least one pass (2048 rows), above every row band of the
 * projections, joins, activation and reduction (the GEMM's at 16 and 64, the reduction's one-shot
 * edge at 524288 elements). No row result depends on the row count, so the output matches the
 * whole step's.
 *
 * Ordering: each reduction rides lane 1 behind a join from lane 0, which holds its producer; lane 0
 * waits for lane 1 before the reduction's consumer. A join orders behind everything its lane has
 * issued, so each wait goes after the lane-0 work that does not need it and before the next
 * reduction, which fixes the issue order below. The embedding's reduction stays on lane 0, before
 * lane 1's first join, so the collectives keep one order on every rank; the runtime joins lane 1
 * back at the end of the step. */

/* The first row of slice B, or 0 for a step that runs whole: the whole number of routed passes
 * nearest half the step, so both slices are at least one pass. */
static int64_t split_at(const Model& m, int64_t T) {
    if (!m.overlap || T < 2 * m.moe_rows) return 0;
    const int64_t P = m.moe_rows;
    return P * ((T + P) / (2 * P));
}

/* Rows [r0, r0 + n) of a layer's projections: q|k|v on a sliding layer, q and k on a full one. */
static void project(RadCtx* c, const Model& m, const Layer& y, int64_t r0, int64_t n) {
    const RadOperand h = brow_slice(m.b_h, r0, n, m.E);
    if (!y.full) {
        RAD_ISSUE_N(c, y.op_qkv, n, h, RAD_W(y.w_qkv),
                    brow_slice(m.b_qkv, r0, n, m.ks.q() + 2 * m.ks.kv_dim()));
    } else {
        RAD_ISSUE_N(c, y.op_q, n, h, RAD_W(y.w_q), brow_slice(m.b_qf, r0, n, m.kf.q()));
        RAD_ISSUE_N(c, y.op_k, n, h, RAD_W(y.w_k), brow_slice(m.b_kf, r0, n, m.kf.kv_dim()));
    }
}

/* Rows [r0, r0 + n) of the output projection. */
static void out_proj(RadCtx* c, const Model& m, const Layer& y, int64_t r0, int64_t n) {
    const int64_t q = y.full ? m.kf.q() : m.ks.q();
    RAD_ISSUE_N(c, y.op_o, n, brow_slice(y.full ? m.b_attf : m.b_atts, r0, n, q), RAD_W(y.w_o),
                brow_slice(m.b_o, r0, n, m.E));
}

/* Rows [r0, r0 + n) from the attention join to the arms' partials, once those rows' attention
 * reduction is done. The partials land in rows [2 r0, 2 r0 + 2n) of the join, dense then routed,
 * so the slice's feed-forward reduction is one run. */
static void arms(RadCtx* c, const Model& m, const Layer& y, int64_t T, int64_t r0, int64_t n) {
    const int64_t E = m.E, F = m.n_ff;
    if (y.op_join_r)
        RAD_ISSUE_N(c, y.op_join_r, n, brow_slice(m.b_o, r0, n, E), RAD_W(y.w_apost),
                    brow_slice(m.b_x, r0, n, E), RAD_W(y.w_fnorm), brow_slice(m.b_h, r0, n, E),
                    RAD_W(y.w_rscale), brow_slice(m.b_rn, r0, n, E), RAD_W(y.w_enorm),
                    brow_slice(m.b_hq, r0, n, E), brow_slice(m.b_hqs, r0, n, E / RAD_SCALE_GROUP));
    else
        RAD_ISSUE_N(c, y.op_join_a, n, brow_slice(m.b_o, r0, n, E), RAD_W(y.w_apost),
                    brow_slice(m.b_x, r0, n, E), RAD_W(y.w_fnorm), brow_slice(m.b_h, r0, n, E));
    RAD_ISSUE_N(c, y.op_gu, n, brow_slice(m.b_h, r0, n, E), RAD_W(y.w_gu),
                brow_slice(m.b_gu, r0, n, 2 * F));
    RAD_ISSUE_N(c, y.op_act, n, bcol_at(m.b_gu, r0, 2 * F, 0, F, n),
                bcol_at(m.b_gu, r0, 2 * F, F, F, n), brow_slice(m.b_act, r0, n, F));
    RAD_ISSUE_N(c, y.op_dn, n, brow_slice(m.b_act, r0, n, F), RAD_W(y.w_dn),
                brow_slice(m.b_ff2, 2 * r0, n, E));
    routed(c, m, y, T, r0, n, 2 * r0 + n);
}

/* Rows [r0, r0 + n) of layer li's feed-forward join (with the next norm it writes), once their
 * reduction is done, and the drafter's taps of those rows. */
static void join_ff(RadCtx* c, const Model& m, size_t li, int64_t r0, int64_t n) {
    const Layer& y = m.layers[li];
    const int64_t E = m.E;
    RAD_ISSUE_N(c, y.op_join_f, n, brow_slice(m.b_ff2, 2 * r0, n, E), RAD_W(y.w_fpost1),
                brow_slice(m.b_ff2, 2 * r0 + n, n, E), RAD_W(y.w_fpost2), RAD_W(y.w_fpost),
                brow_slice(m.b_x, r0, n, E), RAD_W(y.w_scalar), RAD_W(y.w_next),
                brow_slice(m.b_h, r0, n, E));
    for (int t = 0; t < m.n_taps && y.op_tap; ++t) {
        if (m.tap_layer[t] != (int)li) continue;
        RAD_ISSUE_N(c, y.op_tap, n, brow_slice(m.b_x, r0, n, E),
                    bcol_at(m.b_dfsrc, r0, (int64_t)m.n_taps * E, (int64_t)t * E, E, n));
    }
}

static void trunk_overlap(RadCtx* c, const Model& m, const RadBatch* batch, int64_t cut) {
    const int64_t T = batch->n_tok, E = m.E;
    const int64_t a0 = 0, an = cut, b0 = cut, bn = T - cut;
    bool lanes_ok = true;
    /* An in-place all-reduce over `rows` rows of `buf` from `row`, on lane 1 behind everything
     * lane 0 has issued, including its producer. */
    auto reduce = [&](rad_op op, rad_buf buf, int64_t row, int64_t rows) {
        lanes_ok = lanes_ok && rad_lane_join(c, 0, 1) == RAD_OK && rad_lane(c, 1) == RAD_OK;
        RAD_ISSUE_N(c, op, rows * E, brow_slice(buf, row, rows, E), RAD_NONE);
        lanes_ok = rad_lane(c, 0) == RAD_OK && lanes_ok;
    };
    /* Lane 0 behind every reduction issued so far. */
    auto join_back = [&] { lanes_ok = rad_lane_join(c, 1, 0) == RAD_OK && lanes_ok; };

    const size_t L = m.layers.size();
    for (size_t li = 0; li < L; ++li) {
        const Layer& y = m.layers[li];
        if (li == 0) {
            RAD_ISSUE(c, y.op_anorm, brows(m.b_x, T), RAD_W(y.w_anorm), brows(m.b_h, T));
            project(c, m, y, a0, an);
            project(c, m, y, b0, bn);
        } else {
            join_ff(c, m, li - 1, a0, an);   /* its reduction was waited for last layer */
            project(c, m, y, a0, an);
            join_back();                     /* the last layer's feed-forward, slice B */
            join_ff(c, m, li - 1, b0, bn);
            project(c, m, y, b0, bn);
        }
        attend(c, m, y, li, batch);
        out_proj(c, m, y, a0, an);
        reduce(y.op_ar_o, m.b_o, a0, an);
        out_proj(c, m, y, b0, bn);
        join_back();                         /* attention, slice A */
        reduce(y.op_ar_o, m.b_o, b0, bn);
        arms(c, m, y, T, a0, an);
        join_back();                         /* attention, slice B */
        reduce(y.op_ar_f, m.b_ff2, 2 * a0, 2 * an);
        arms(c, m, y, T, b0, bn);
        join_back();                         /* the feed-forward, slice A */
        reduce(y.op_ar_f, m.b_ff2, 2 * b0, 2 * bn);
    }
    join_ff(c, m, L - 1, a0, an);
    join_back();                             /* the last feed-forward, slice B */
    join_ff(c, m, L - 1, b0, bn);
    if (!lanes_ok)
        rad_step_fail(c, "gemma4: the prefill overlap could not order its second lane against "
                         "the first");
}

/* The tap's rows, and its files once the corpus reached RADIANCE_CALIB_TOKENS. The input of q|k|v
 * and of gate|up is whole on every rank, so rank 0 writes it, under the name of the linear that
 * reads it. The column-split linears (o, down) see this rank's columns only: each rank writes its
 * block as `<stem>.part`, which libquant's calib_sum places on the diagonal in rank order (the
 * cross-rank products are never formed on one card). */
static void calib_count(RadCtx* c, Model& m, int64_t T) {
    m.calib_rows += T;
    if (m.calib_rows < calib_target()) return;
    const int rk = rad_rank(c);
    bool ok = true;
    for (int64_t l = m.calib_lo; l < m.calib_hi; ++l) {
        const Layer& y = m.layers[(size_t)l];
        char stem[96];
        if (rk == 0) {
            std::snprintf(stem, sizeof stem, "blk.%d.attn_qkv", (int)l);
            ok = calib_write(c, y.tap_in.h, y.tap_in.K, m.calib_rows, stem, 0) && ok;
            std::snprintf(stem, sizeof stem, "blk.%d.ffn_gate_up", (int)l);
            ok = calib_write(c, y.tap_gu.h, y.tap_gu.K, m.calib_rows, stem, 0) && ok;
        }
        std::snprintf(stem, sizeof stem, "blk.%d.attn_o.part", (int)l);
        ok = calib_write(c, y.tap_o.h, y.tap_o.K, m.calib_rows, stem, 0) && ok;
        std::snprintf(stem, sizeof stem, "blk.%d.ffn_down.part", (int)l);
        ok = calib_write(c, y.tap_dn.h, y.tap_dn.K, m.calib_rows, stem, 0) && ok;
    }
    m.tap_live = false;
    std::fprintf(stderr, "radiance: gemma4: rank %d wrote the Hessians of layers %lld:%lld after "
                         "%lld tokens to %s%s\n", rk, (long long)m.calib_lo, (long long)m.calib_hi,
                 m.calib_rows, calib_dir(), ok ? "" : ", with errors");
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (batch->draft_pass != 0) {
        if (m.have_head) {
            m.head.step(c, batch);
            return;
        }
        if (!m.have_df) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries neither an assistant `draft.` head nor a DFlash "
                             "block, or declare ran at max_spec 0");
            return;
        }
        if (batch->draft_pass > 0) m.df.step(c, batch);
        else                       m.df.context(c, batch);
        return;
    }
    if (batch->enc) {
        if (batch->enc & ~(RAD_MM_IMAGE | RAD_MM_VIDEO))
            rad_step_fail(c, "gemma4: unsupported encoder modality");
        else if (!m.vis.on)
            rad_step_fail(c, "gemma4: no vision tower was declared; image and video encoding "
                             "require vision weights and --mm-max-patches greater than 0");
        else
            m.vis.step(c, batch);
        return;
    }
    const int64_t T = batch->n_tok, E = m.E;
    if (T <= 0) return;

    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    /* The batch supplies image rows on rank 0 and zeros on the other ranks. */
    if (m.op_mm && batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_mm, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, E),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(m.b_x, T));
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * E, brows(m.b_x, T), RAD_NONE);

    const int64_t cut = split_at(m, T);
    if (cut > 0) trunk_overlap(c, m, batch, cut);
    else         trunk(c, m, batch);
    if (m.calib && m.tap_live) calib_count(c, g_model[rad_slot(c)], T);

    if (m.op_onorm) RAD_ISSUE(c, m.op_onorm, brows(m.b_x, T), RAD_W(m.w_onorm), brows(m.b_h, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.b_h, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out, brows(m.b_hout, batch->n_out), RAD_W(m.w_tok),
                    brows(m.b_logits, batch->n_out));
        if (m.op_cap)
            RAD_ISSUE_N(c, m.op_cap, batch->n_out, brows(m.b_logits, batch->n_out),
                        brows(m.b_logits, batch->n_out));
    }
}

/* ============================================================== probe */
/* DFlash proposes its whole block in one pass, so its depth is fixed at the trained block size
 * minus the anchor: 15 for z-lab's Gemma 4 drafters (block_size 16). */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    /* the assistant head: a serial drafter, depth an operating point (not a trained block). */
    if (has_assistant(meta)) {
        out->draft_depth = kHeadDepth;
        out->draft_depth_fixed = 0;
    }
    (void)dflash_probe(meta, out);
    if (Gemma4Vision::present(meta)) {
        const int64_t k = rad_meta_geti(meta, "vision_config.pooling_kernel_size", 0);
        if (k > 0) out->enc_merge = k * k;
    }
    return RAD_OK;
}

/* ============================================================== the reply format
 * Gemma 4's thought channel and calls written as a name and one object in Gemma's JSON dialect,
 * shared with gemma4_bf16 (arch/rad_gemma4_chat.h). The container's template lives in the vocab
 * section, not the metadata, so it is not visible here; this answers for every Gemma 4 container. */
static const RadChatFormat* chat_format(const RadModelMeta*) { return gemma4_chat_format(); }

}  /* namespace gemma4 */

#ifndef RAD_ARCH_NO_EXPORTS
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(gemma4, "gemma4", "", "0.1.0",
                "Gemma 4 decoders (26B-A4B MoE, dense 31B) with vision, bf16: sliding and full attention "
                "with proportional rope and K = V, a dense and a routed feed-forward side by side, "
                "and z-lab's DFlash block-diffusion drafter")
RAD_ARCH_PROBE(gemma4)
RAD_ARCH_CHAT_FORMAT(gemma4)
#endif
