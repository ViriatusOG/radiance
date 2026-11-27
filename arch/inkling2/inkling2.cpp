/* inkling2 -- Inkling 2's text backbone as the loop checkpoint lays it out (`inkling2_loop`, served
 * as the arch id `inkling2`), at one rank or split over a world that divides the heads (TENSOR
 * PARALLELISM, below).
 *
 * ============================== THE SHAPE ==============================
 *
 * 24 physical layers run as 48 layer applications: the prelude 0..5 once, the core 6..17 three
 * times (each pass restarts from the prelude's output and the prelude's blocks), the coda 18..23
 * once. The weights are shared by the applications. Each application is one bound layer of the
 * attention group of its kind (40 local, 8 full) and of each of the four convolution streams, so
 * every application keeps its own KV and convolution state. tests/mkinkling2golden.py is the
 * reference this follows, op for op.
 *
 * A layer's attention input is a softmax over depth of the stream and the blocks closed so far
 * (attn_res_mix, one query at a time, so each mix is one [1, H] weight). A core layer in a later
 * pass mixes its four messages from the previous pass's bank and passes each input through a
 * controller, (1 - tanh(F g)) x + tanh(G g) msg with g = D [rms x | rms msg]. q, k and v go through
 * the stateful short convolutions (sconv_res; a full layer's K and V add input-dependent taps), the
 * L2 clips and, on full layers, the position tau. Attention is attn_paged_rel: a bias from d_rel and
 * a table of `extent` distances, and a sink per query head. The experts are SiTU over the gate and up
 * rows, which the importer regroups to [gate; up] (core/format/checkpoint.cpp). An expert is served
 * in one of two forms, read from its tensors: an IQR weight (the rotated computed trellis under one
 * f32 scale a row, IQR-V4's form, read at `i8a8h` by moe_gemm_iqr at its codes' width;
 * data/recipes/inkling2-mini-iq4r-v4.recipe), or bf16, the calibration container's form
 * (data/recipes/inkling2-mini-bf16.recipe), over which --calib-dir records the Hessians the IQR
 * recipe's GPTQ reads (CALIBRATION, below). The activation is `situ_mul` in both; an IQR layer's
 * product is then rotated and quantised for its down projection.
 *
 * ============================== WHAT IT DOES NOT SERVE ==============================
 *
 * Routed experts in any other form (the release's NVFP4 and MXFP8 as stored, rotated int8 or int4),
 * attention linears other than bf16 or rotated int8, a tree verify, a loop schedule other than the
 * release's, and the release's own DSpark drafter (`mtp.safetensors`, draft.model_type
 * `inkling2_dspark`). Each is refused by name. The vision tower (rad_block_inkling2_tower.h) is
 * served when the container carries it; its rows replace the placeholders after the embedding
 * norm. The drafter is incoai's DFlash2 (Inkling-2-Mini-DFlash2, inkling2_dflash2.h): the trunk
 * copies the streams it reads (InkDflash2::tap) and verifies its block as a multi-row decode, whose
 * short convolutions roll back to `num_accepted` (sconv_res) and whose rejected K and V the next
 * step overwrites. The release's full-attention `static_w` convolutions are rewritten to the conv1d
 * layout at import (core/format/checkpoint.cpp, regroup_inkling_static_w), so the plugin reads one
 * layout.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_inkling2_tower.h>
#include <arch/rad_inkling2_chat.h>

#include "inkling2_dflash2.h"

#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace inkling2 {

using namespace rad::arch;

static const int kApps = 48;          /* layer applications: 6 + 3 * 12 + 6 */
static const int kLayers = 24;        /* physical layers */
static const int kWidth = 4;          /* sconv_kernel_size of the release */
static const int kBlocks = 4;         /* block planes: the embedding and three pushes */
static const int kExpGroup = 128;     /* an IQR expert's int8 activation: one scale per 128 */

/* How a layer's routed experts are stored, read from the container's own tensors: bf16 (the
 * calibration container) or IQR (a recipe's computed-trellis rules). */
enum ExpertForm { kExpBf16 = 0, kExpIqr = 1 };

struct Layer {
    bool    full = false;
    bool    core = false;
    int     n_mix = 1;                  /* attention input queries: 3 on a full layer, 1 otherwise */
    int     n_state_attn = 0;           /* blocks read by the attention input (before the push) */
    int     n_state_mlp = 0;            /* blocks read by the feed-forward input (after it) */
    int     push_slot = -1;             /* the block plane this layer's push writes, or -1 */

    rad_weight w_norm[3] = { 0, 0, 0 };     /* q/k/v input norms; on a local layer w_norm[0] */
    rad_weight w_mix[3] = { 0, 0, 0 };      /* the attention input queries */
    rad_op op_mix[3] = { 0, 0, 0 };
    rad_op op_mix_norm = 0, op_mix3 = 0, op_mix3_norm = 0, op_msg4 = 0;
    rad_op op_norm[3] = { 0, 0, 0 };        /* from the mixed input */
    rad_op op_norm_ctl[3] = { 0, 0, 0 };    /* from the controlled input */
    rad_op op_norm_hq[3] = { 0, 0, 0 };     /* fused rmsnorm + had_quant_act_i8g (ip == 0) */
    rad_op op_norm_ctl_hq[3] = { 0, 0, 0 }; /* fused rmsnorm + had_quant_act_i8g (control) */

    rad_weight w_q = 0, w_k = 0, w_v = 0, w_r = 0, w_gate = 0, w_sink = 0, w_o = 0;
    rad_op op_q = 0, op_k = 0, op_v = 0, op_r = 0, op_gate = 0, op_sink = 0, op_o = 0;
    /* q, k, v, r and the gate's int8 linears in one launch, at steps of up to 16 rows */
    rad_op op_proj = 0;
    rad_weight w_dk = 0, w_dv = 0;          /* input-dependent taps (full layers) */
    rad_op op_dk = 0, op_dv = 0;
    rad_weight w_kconv = 0, w_vconv = 0, w_aconv = 0, w_mconv = 0;
    rad_op op_kconv = 0, op_vconv = 0, op_aconv = 0, op_mconv = 0;
    rad_op op_aconv_add = 0, op_mconv_add = 0;
    rad_weight w_qscale = 0, w_rscale = 0, w_rel = 0;
    rad_op op_qclip = 0, op_rclip = 0, op_qr_clip = 0, op_kclip = 0, op_vnorm = 0;
    rad_op op_store = 0, op_kv_store_fused = 0, op_attn = 0, op_sig = 0, op_gated = 0, op_act_mul = 0;

    /* the core's messages and controllers, per input j: attn (q), k, v, mlp */
    rad_weight w_msg[4] = { 0, 0, 0, 0 };
    rad_op op_msg[4] = { 0, 0, 0, 0 };
    rad_weight w_dx[4] = { 0, 0, 0, 0 };
    rad_weight w_cgain[4] = { 0, 0, 0, 0 }, w_cforget[4] = { 0, 0, 0, 0 };
    rad_op ctl[4][12] = {};
    rad_op op_ctl_fused[4] = {}, op_ctl_blend[4] = {};

    rad_weight w_mlp_q = 0, w_mlp_norm = 0, w_router = 0, w_bias = 0;
    rad_op op_mlp_mix = 0, op_mlp_norm = 0, op_mlp_norm_ctl = 0, op_router = 0, op_topk = 0;
    rad_op op_mlp_norm_q8 = 0, op_mlp_norm_ctl_q8 = 0, op_router_scatter = 0;
    rad_op op_scatter = 0, op_egu = 0, op_situ = 0, op_edn = 0, op_gather = 0;
    rad_op op_push = 0, op_resid = 0, op_push_copy = 0, op_resid_copy = 0, op_attn_add = 0, op_mlp_add = 0;
    /* the experts' stored form (ExpertForm), its codes and scale planes, and for an IQR expert the
     * quantiser of the routed input and of the SiTU product */
    int ex_form = 0;
    /* an IQR expert's codes dtype (the stored width) and its trellis's run (rows), gate_up's and
     * down's: a recipe may give the two projections different widths and runs */
    uint32_t ex_cdt[2] = { 0, 0 };
    int64_t  ex_run[2] = { 0, 0 };
    std::vector<rad_weight> w_egu, w_edn;   /* the expert codes (or the bf16 experts), by expert */
    std::vector<rad_weight> w_egus, w_edns; /* an IQR expert's scales, by expert */
    rad_op op_q8 = 0;
    /* this rank's slice of each expert by parity class (TENSOR PARALLELISM), for the layer's form */
    int64_t fq[2] = { 0, 0 }, flo[2] = { 0, 0 };
    rad_op op_dq = 0;   /* an IQR expert's quantiser of the bf16 SiTU product */
    /* The attention linears (q, k, v, r, gate, o) in their stored form: rotated int8 at a bf16
     * scale per (row, 128 of K), read by gemm_nt_q at i8a8h, or bf16. The sink and the dynamic
     * taps stay bf16 (a few rows each). */
    bool lin_i8 = false;
    rad_weight w_qs = 0, w_ks = 0, w_vs = 0, w_rs = 0, w_gates = 0, w_os = 0;
    rad_op op_hq[3] = { 0, 0, 0 };   /* had_quant_act_i8g of each projection input */
    rad_op op_gq = 0;                /* gate_had_quant_i8g: the gated output in wo's codes */
    /* An int8 wr whose rank share is not whole 128-row tiles (12 heads of d_rel 16 at two ranks is
     * 192 rows, and gemm_nt_q's prefill tiles want N in 128s) is held whole on every rank: the GEMM
     * writes every head's r (Model::b_rf) and the clips read this rank's columns. 0.9 MB a layer. */
    bool r_full = false;
    /* tensor parallelism: the all-reduces of the output projection's and the experts' partial sums */
    rad_op op_ar_o = 0, op_ar_m = 0;
    /* --calib-dir (CALIBRATION), on the rank that records the layer: the pooled Grams of the
     * experts' gate_up input and of the attention linears' three inputs, f32 [H, H] on the card,
     * and every expert's down Gram, summed on the host. The ops are 0 where nothing is recorded. */
    bool c_own = false;
    rad_buf c_gu = 0, c_at[3] = { 0, 0, 0 };
    rad_op op_c_gu = 0, op_c_at[3] = { 0, 0, 0 };
    mutable bool c_zeroed = false;
    mutable long long c_rows_gu = 0, c_rows_at = 0;
    mutable CalibExperts c_dn;
};

struct Model {
    Names   nm{""};
    Geom    g{};
    int64_t H = 0, F = 0, E = 0, K = 0, HD = 0, NH = 0, KVH = 0, DR = 0, EXT = 0, WIN = 0;
    int64_t VR = 0, VN = 0, RANK = 0;
    double  qclip = 0, rclip = 0, alpha = 0, floorv = 0, route_scale = 1;
    double  gcap = 0, ucap = 0, fcap = 0;

    rad_kvgroup kv_full = 0, kv_local = 0;
    uint32_t kv_dt = RAD_BF16;
    rad_kvgroup cv_k = 0, cv_v = 0, cv_a = 0, cv_m = 0;

    rad_buf b_s = 0, b_pre = 0, b_bank1 = 0, b_ones = 0, b_minus1 = 0;
    rad_buf b_blk = 0;                       /* [kBlocks, max_tok, H]: the block planes */
    rad_buf b_bank0 = 0;                     /* [1, max_tok, H]: the previous pass's block */
    rad_buf b_y[3] = { 0, 0, 0 }, b_h[3] = { 0, 0, 0 }, b_cy[3] = { 0, 0, 0 }, b_msg[4] = { 0, 0, 0, 0 };
    rad_buf b_gxm = 0, b_g1 = 0, b_g2 = 0, b_g = 0, b_gp = 0, b_gt = 0;
    rad_buf b_fp = 0, b_ft = 0, b_nft = 0, b_t1 = 0, b_t2 = 0, b_t3 = 0;
    rad_buf b_q = 0, b_k = 0, b_v = 0, b_r = 0, b_gate = 0, b_sink = 0;
    rad_buf b_rf = 0;   /* every head's r, under tensor parallelism (Layer::r_full) */
    rad_buf b_dk = 0, b_dv = 0, b_kcv = 0, b_vcv = 0, b_qc = 0, b_kc = 0, b_vc = 0, b_rc = 0;
    rad_buf b_att = 0, b_sig = 0, b_attg = 0, b_ao = 0, b_ax = 0;
    rad_buf b_mx = 0, b_cym = 0, b_fm = 0, b_rl = 0, b_ids = 0, b_wts = 0;
    rad_buf b_sorted = 0, b_eoff = 0, b_ecnt = 0, b_gu = 0, b_act = 0, b_dn = 0, b_mo = 0, b_zc = 0;
    rad_buf b_yo = 0, b_yn = 0, b_hout = 0, b_logits = 0;
    rad_buf b_q8 = 0, b_s8 = 0, b_aq = 0, b_as = 0;   /* an IQR expert's int8 input and product */
    /* an int8 trunk's rotated activation codes: the three projection inputs and the gated output */
    rad_buf b_hq8[3] = { 0, 0, 0 }, b_hs8[3] = { 0, 0, 0 }, b_oq8 = 0, b_os8 = 0;

    rad_weight w_embed = 0, w_embed_norm = 0, w_out_norm = 0, w_out_q = 0, w_head = 0;
    rad_op op_embed = 0, op_embed_norm = 0, op_out_mix = 0, op_out_norm = 0, op_out_mix_norm = 0;
    rad_op op_rows = 0;
    rad_op op_head = 0, op_softcap = 0, op_pad = 0, op_ones = 0, op_minus = 0;
    rad_op op_pre = 0, op_bank_blk = 0, op_bank_cur = 0, op_reset = 0;
    rad_op op_pre_copy = 0, op_bank_blk_copy = 0, op_bank_cur_copy = 0, op_reset_copy = 0;
    bool   use_plane_copy = false, all_ctl_fused = false, all_ctl_blend = false;

    /* Tensor parallelism (TENSOR PARALLELISM): FW, the width of this rank's slice of a routed
     * expert in the layers' stored form (the wider parity class of an IQR layer's), which the
     * buffers between the two expert GEMMs take; F at one rank. The head's padding columns that
     * fall on this rank: [pad_lo, pad_lo + pad_n) of its logits. */
    int64_t FW = 0;
    int64_t pad_lo = 0, pad_n = 0;
    rad_op  op_embed_ar = 0;

    std::vector<Layer> layers;

    /* The vision tower (arch/common/rad_block_inkling2_tower.h), declared last, and the scatter
     * of its rows over the placeholders: after the embedding norm, so a picture's rows go in raw
     * (the reference norms token rows only). */
    Inkling2Tower vit;
    rad_op op_media_rows = 0;

    /* The DFlash2 drafter (inkling2_dflash2.h), declared when the container carries it and the
     * serve speculates; it reads four of the trunk's streams, copied into a buffer of its own. */
    InkDflash2 df2{};

    /* --calib-dir (CALIBRATION): the recorded layers [lo, hi), the corpus's rows, and the experts'
     * down input as the owner reads it: `Fg` columns a row (every rank's FW under tensor
     * parallelism, gathered), expert column a of parity class q at gathered column
     * f2g[split][q][a] (`split` 0 for IQR's block split, 1 for bf16's even one). */
    struct Calib {
        bool        on = false;
        const char* dir = nullptr;
        int64_t     lo = 0, hi = 0, Fg = 0;
        long long   target = 0;
        std::vector<int32_t> f2g[2][2];
        rad_buf     gathered = 0, scratch = 0;
        rad_op      op_gather = 0, op_dn = 0;
        mutable long long seen = 0;
        mutable bool      written = false, failed = false;
    } cal;
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

/* The physical layer of application a. */
static int phys_of(int a) {
    if (a < 6) return a;
    if (a < 42) return 6 + (a - 6) % 12;
    return a - 24;
}

/* The blocks a layer reads: the attention input sees the blocks before the layer's own push, the
 * feed-forward input the blocks after it. The embedding is block 0 (pushed by layer 0); the core
 * pushes its input at layers 6 and 12 of every pass, and the coda at 18. A pass starts over with the
 * embedding alone, so the count is 1 + the pushes before the layer, within a pass. */
static void block_counts(int ip, int* before, int* after, int* slot) {
    *slot = (ip % 6 == 0) ? ip / 6 : -1;
    if (ip == 0) *before = 0;
    else {
        int pushes = 0;
        for (int p : { 6, 12, 18 }) pushes += p < ip ? 1 : 0;
        *before = 1 + pushes;
    }
    *after = *before + (*slot >= 0 ? 1 : 0);
}

/* The checkpoint's prefix for the model's tensors: the release's `model.llm` (the names its
 * hf_quant_config.json and the golden reference use) or the test model's `model`. The release's
 * embedding is the probe: only the release holds it, and a builder with no model behind it reads as
 * the test model's names. */
static const char* ckpt_prefix(RadBuilder* b, Names& nm) {
    RadEncoding e{};
    if (map_copy(b, nm.f("probe.ckpt_prefix"), "model.llm.embed.weight") == RAD_OK &&
        weight_enc(b, nm.f("probe.ckpt_prefix"), &e))
        return "model.llm";
    return "model";
}

/* The form the expert tensor `src` is held in, and for an IQR weight its codes' dtype in `cdt` and
 * its trellis's run in `run`. No encoding (a builder with no model behind it) is bf16, as the tiny
 * model's experts are. Anything else is refused by name: the release's NVFP4 and MXFP8 experts are
 * read by rad-convert, dequantised, and written in one of the two forms by a recipe. */
static int expert_form(RadBuilder* b, const char* src, int* form, uint32_t* cdt, int64_t* run) {
    RadEncoding e{};
    *form = kExpBf16;
    *cdt = 0;
    *run = 0;
    if (!weight_enc(b, src, &e)) return RAD_OK;
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return RAD_OK;
    /* the rotated computed trellis (libquant tcq_code=mad) under one f32 scale a row and one f32
     * a 128-K block for every row (`scale` [1 x whole], `scale.col` [whole x 128]): IQR-V4's form,
     * kept at the codes' stored width and decoded by moe_gemm_iqr */
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    if (rad_enc_is(&e, "trellis") && std::strcmp(e.transform, "fwht128") == 0 &&
        rad_enc_plane(&e, "trellis.mad") && !rad_enc_plane(&e, "scale.1")) {
        const RadEncPlane* cf = rad_enc_plane(&e, "scale.col");
        const RadEncPlane* tp = rad_enc_plane(&e, "trellis");
        if (sc && cf && tp && sc->kind == RAD_PLANE_TILED && sc->dtype == RAD_F32 &&
            sc->block[0] == 1 && sc->block[1] == 0 && cf->kind == RAD_PLANE_TILED &&
            cf->dtype == RAD_F32 && cf->block[0] == 0 && cf->block[1] == kExpGroup) {
            *form = kExpIqr;
            *cdt = e.plane[0].dtype;
            *run = tp->block[0];
            return RAD_OK;
        }
    }
    char en[256];
    rad_enc_format(&e, en, sizeof en);
    fprintf(stderr, "radiance: inkling2: expert '%s' is %s, and the routed experts are served as an "
                    "IQR weight (the computed trellis under one f32 scale a row, /fwht128; "
                    "data/recipes/inkling2-mini-iq4r-v4.recipe) or as bf16 for calibration "
                    "(data/recipes/inkling2-mini-bf16.recipe); convert the release with one of "
                    "them\n", src, en);
    return RAD_E_DTYPE;
}

/* The form an attention linear `src` is held in: bf16 as the release ships it, or rotated int8 at a
 * bf16 scale per (row, 128 of K), what data/recipes/inkling2-mini-iq4r-v4.recipe writes (read by
 * gemm_nt_q at i8a8h). */
static int lin_form(RadBuilder* b, const char* src, bool* i8) {
    RadEncoding e{};
    *i8 = false;
    if (!weight_enc(b, src, &e)) return RAD_OK;
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return RAD_OK;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    if (rad_enc_is(&e, "affine") && e.n_planes == 2 && e.plane[0].dtype == RAD_I8 && sc &&
        sc->dtype == RAD_BF16 && sc->block[0] == 1 && sc->block[1] == kExpGroup &&
        std::strcmp(e.transform, "fwht128") == 0) {
        *i8 = true;
        return RAD_OK;
    }
    char en[256];
    rad_enc_format(&e, en, sizeof en);
    fprintf(stderr, "radiance: inkling2: linear '%s' is %s, and the attention linears are served as "
                    "bf16 or rotated int8 (i8*bf16[1x128]/fwht128)\n", src, en);
    return RAD_E_DTYPE;
}

/* ============================== TENSOR PARALLELISM ==============================
 *
 * Every rank runs the whole loop over the whole residual stream; only the attention's heads, the
 * experts' intermediate columns and the vocabulary are split. A layer application issues two
 * all-reduces, the fewest its two row-parallel GEMMs allow:
 *   - the attention: q, k, v, r, the gate and the sink are rows of this rank's heads (q, r, the gate
 *     and the sink per query head, k and v and a full layer's dynamic taps per KV head), so the K
 *     and V short convolutions, the clips, the cache and the attention itself run on the rank's own
 *     heads and channels. The output projection takes the matching columns, and its partial sum is
 *     reduced before the attention's short convolution, which runs over the whole stream as on one
 *     rank.
 *   - the experts: the router is replicated, so every rank routes every token alike, and each rank
 *     computes its slice of every routed expert's columns (below). moe_gather weighs the partial
 *     down projections, and the sum is reduced before the feed-forward's short convolution.
 * The depth mixes, the controllers, the norms and the router are replicated. The embedding and the
 * head are split by vocabulary as everywhere (Geom::vocab_off): the embedding's rows are disjoint,
 * so its all-reduce (once a step) is exact; the sampler merges the ranks' candidates, and the head's
 * padding columns are filled on the rank that holds them.
 *
 * An IQR expert is F columns, F / 128 blocks of the int8 activation's scale group and the
 * rotation. When the blocks do not divide by the ranks (1152 is nine), they are split by parity as
 * qwen4exp's are: each parity's blocks are cut contiguously and the rank taking the extra block
 * alternates with the expert's parity, so at two ranks rank r computes five blocks of the experts
 * of parity r and four of the others. moe_gemm_q takes the two classes' widths (N_odd, K_odd) and writes gate_up as two
 * parts of the wider class's width. Whole experts per rank (expert parallelism) would leave one rank
 * a whole expert ahead whenever a token's eight picks split unevenly, which the all-reduce turns into
 * waiting; the parity split differs by a block. bf16 experts are split evenly (moe_gemm takes one
 * table of one width), which needs F to divide by the ranks: 576 columns a rank at two. The buffers
 * between the two expert GEMMs are one width (Model::FW) for every layer, so a container's layers
 * split alike (one form throughout, as each recipe writes them). */
static void block_slice(int64_t nb, int world, int64_t place, int64_t* lo, int64_t* hi) {
    const int64_t base = nb / world, extra = nb % world;
    const int64_t b0 = place * base + std::min<int64_t>(place, extra);
    *lo = b0 * kExpGroup;
    *hi = (b0 + base + (place < extra ? 1 : 0)) * kExpGroup;
}

/* This rank's slice of an expert by parity class: columns [lo[q], lo[q] + w[q]) of F. False when
 * the form cannot be split over the world. */
static bool expert_slice(int64_t F, int world, int rank, bool quant, int64_t lo[2], int64_t w[2]) {
    for (int q = 0; q < 2; ++q) {
        lo[q] = 0;
        w[q] = F;
    }
    if (world <= 1) return true;
    if (!quant) {
        if (F % world) return false;
        for (int q = 0; q < 2; ++q) {
            lo[q] = rank * (F / world);
            w[q] = F / world;
        }
        return true;
    }
    if (F % kExpGroup) return false;
    const int64_t nb = F / kExpGroup;
    if (nb < world) return false;
    for (int q = 0; q < 2; ++q) {
        int64_t hi = 0;
        block_slice(nb, world, nb % world ? (rank - q + world) % world : rank, &lo[q], &hi);
        w[q] = hi - lo[q];
    }
    return true;
}

/* ============================== CALIBRATION ==============================
 *
 * --calib-corpus with --calib-dir: the Hessians H = sum x x^T of the inputs a GPTQ rule's `calib=`
 * reads, as RADGRAM2 files (arch/rad_arch.h, calib_write_host_to) named by the container's weights,
 * in the model's own domain (domain 0, from the bf16 planes beside the codes; libquant rotates a
 * file into a rotated grid's domain):
 *   - the routed experts' gate_up input, `blk.<l>.gu`, pooled over the layer's experts and its
 *     applications: every expert reads the same normed stream, so libquant's pooled stem of
 *     `blk.<l>.e<e>.gu` finds it. A Gram per expert would be 128 files of 21 MB a layer;
 *   - each expert's down input, `blk.<l>.e<e>.dn` (its own SiTU product, whose units mean something
 *     different in every expert), and their sum `blk.<l>.dn`, which libquant reads for an expert
 *     routed too few rows (lq_hess.h, calib_resolve);
 *   - the attention linears' three inputs: the query input under `blk.<l>.wq`, `.wr` and `.gate`
 *     (they read it alike), `.wk` and `.wv`. The output projection's input is not recorded: an
 *     int8 trunk never writes it as bf16.
 * A layer's weights are shared by its applications, so its files sum every application's rows. One
 * rank records each layer (layer % world). Under tensor parallelism an expert's down input is split
 * over the ranks (evenly for bf16 experts, by parity class for IQR's; TENSOR PARALLELISM), so every
 * rank's SiTU product is gathered and the owner puts the columns back in the expert's order on the
 * host (f2g). The pooled Grams accumulate on the card and come back once; the per-expert ones are
 * summed into one scratch a layer application, copied back and added on the host, as MoeQ's
 * per-expert tap does. The SiTU product is situ_mul's bf16 in both expert forms, so the taps read
 * the same buffers whichever the container holds; the calibration container's experts are bf16
 * (data/recipes/inkling2-mini-bf16.recipe), the release's dequantised and rounded to bf16. */
static int calib_declare(RadBuilder* b, Model& m, const RadBuildCtx* ctx) {
    Model::Calib& cl = m.cal;
    if (!ctx->calib_dir || !*ctx->calib_dir) return RAD_OK;
    const Geom& g = m.g;
    Names& nm = m.nm;
    cl.on = true;
    cl.dir = ctx->calib_dir;
    cl.lo = ctx->calib_layer_hi ? ctx->calib_layer_lo : 0;
    cl.hi = ctx->calib_layer_hi ? std::min<int64_t>(ctx->calib_layer_hi, kLayers) : kLayers;
    cl.target = ctx->calib_tokens;
    if (cl.lo >= cl.hi || cl.target <= 0) {
        fprintf(stderr, "radiance: inkling2: --calib-dir records layers %lld:%lld of %d over a "
                        "--calib-corpus run of %lld tokens; that names nothing to record\n",
                (long long)cl.lo, (long long)cl.hi, kLayers, cl.target);
        return RAD_E_INVAL;
    }
    const int world = g.world;
    cl.Fg = world > 1 ? world * m.FW : m.FW;
    for (int split = 0; split < 2; ++split)
        for (int q = 0; q < 2; ++q) {
            std::vector<int32_t>& f = cl.f2g[split][q];
            f.assign((size_t)m.F, -1);
            for (int r = 0; r < world; ++r) {
                int64_t lo[2], w[2];
                if (!expert_slice(m.F, world, r, split == 0, lo, w)) break;
                for (int64_t a = lo[q]; a < lo[q] + w[q]; ++a)
                    f[(size_t)a] = (int32_t)(r * m.FW + a - lo[q]);
            }
        }
    for (int64_t l = cl.lo; l < cl.hi; ++l) {
        const int split = m.layers[(size_t)l].ex_form == kExpBf16 ? 1 : 0;
        for (int q = 0; q < 2; ++q)
            for (int32_t x : cl.f2g[split][q])
                if (x < 0) return RAD_E_UNSUPPORTED;
    }
    const int64_t T = g.max_tok, R = T * m.K;
    auto gram = [&](int64_t rows, int64_t n, std::initializer_list<rad_buf> reads, rad_buf acc) {
        return rw(b, RAD_OP(b, "gram_accum", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                       RAD_INT("group", kExpGroup),
                                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS), reads, {acc});
    };
    /* Every buffer here is PERSIST: the taps are issued in the middle of a layer application, where
     * a transient one could share bytes with a buffer still in use. */
    if (world > 1) {
        cl.gathered = decl_b(b, nm.f("calib.gathered"), RAD_BF16, {R, cl.Fg}, RAD_BUF_PERSIST);
        if (!cl.gathered) return RAD_E_INVAL;
        cl.op_gather = rw(b, RAD_OP(b, "all_gather",
                                    RAD_PARAMS(RAD_INT("world_size", world),
                                               RAD_RANGE("numel", m.FW, R * m.FW),
                                               RAD_STR("dtype", "bf16"), RAD_INT("row", m.FW)),
                                    RAD_NOWEIGHTS),
                          {m.b_act}, {cl.gathered});
        if (!cl.op_gather) return RAD_E_UNSUPPORTED;
    }
    bool any = false;
    for (int64_t l = cl.lo; l < cl.hi; ++l) {
        Layer& L = m.layers[(size_t)l];
        L.c_own = l % world == g.rank;
        any = any || L.c_own;
    }
    if (!any) return RAD_OK;
    cl.scratch = decl_b(b, nm.f("calib.scratch"), RAD_F32, {m.E * cl.Fg, cl.Fg}, RAD_BUF_PERSIST);
    if (!cl.scratch) return RAD_E_INVAL;
    const rad_buf src = cl.gathered ? cl.gathered : m.b_act;
    cl.op_dn = gram(R, cl.Fg, {src, cl.scratch}, cl.scratch);
    if (!cl.op_dn) return RAD_E_UNSUPPORTED;
    for (int64_t l = cl.lo; l < cl.hi; ++l) {
        Layer& L = m.layers[(size_t)l];
        if (!L.c_own) continue;
        L.c_gu = decl_b(b, nm.f("calib.blk.%lld.gu", (long long)l), RAD_F32, {m.H, m.H},
                        RAD_BUF_PERSIST);
        if (!L.c_gu) return RAD_E_INVAL;
        L.op_c_gu = gram(T, m.H, {m.b_fm, L.c_gu}, L.c_gu);
        if (!L.op_c_gu) return RAD_E_UNSUPPORTED;
        for (int j = 0; j < 3; ++j) {
            L.c_at[j] = decl_b(b, nm.f("calib.blk.%lld.at%d", (long long)l, j), RAD_F32, {m.H, m.H},
                               RAD_BUF_PERSIST);
            if (!L.c_at[j]) return RAD_E_INVAL;
            L.op_c_at[j] = gram(T, m.H, {m.b_h[0], m.b_h[1], m.b_h[2], L.c_at[j]}, L.c_at[j]);
            if (!L.op_c_at[j]) return RAD_E_UNSUPPORTED;
        }
    }
    rad_note(b, "inkling2: --calib-dir records layers %lld:%lld into %s (rank %d owns layer %% %d "
                "== %d)", (long long)cl.lo, (long long)cl.hi, cl.dir, g.rank, world, g.rank);
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    const char* pfx = ckpt_prefix(b, nm);

    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom& g = m.g;
    m.H = g.n_embd;
    m.HD = g.head_dim;
    m.NH = g.n_head;
    m.KVH = g.n_head_kv;
    m.DR = meta_int(meta, "d_rel", 0);
    m.EXT = meta_int(meta, "rel_extent", 0);
    m.WIN = meta_int(meta, "sliding_window_size", 0);
    m.F = meta_int(meta, "moe_intermediate_size", meta_int(meta, "intermediate_size", 0));
    m.E = meta_int(meta, "n_routed_experts", 0);
    m.K = meta_int(meta, "num_experts_per_tok", 0);
    m.RANK = meta_int(meta, "loop_config.gate_rank", 0);
    m.VR = meta_int(meta, "unpadded_vocab_size", g.n_vocab_all);
    m.VN = g.n_vocab_all;
    m.qclip = meta_f64(meta, "qk_l2_clip_threshold", 0);
    m.rclip = meta_f64(meta, "rel_l2_clip_threshold", 0);
    m.alpha = meta_f64(meta, "log_scaling_alpha", 0);
    m.floorv = meta_f64(meta, "log_scaling_n_floor", 0);
    m.route_scale = meta_f64(meta, "route_scale", 1.0);
    m.gcap = meta_f64(meta, "mlp_gate_softcap", 0);
    m.ucap = meta_f64(meta, "mlp_up_softcap", 0);
    m.fcap = meta_f64(meta, "final_logit_softcapping", 0);
    const int64_t core_s = meta_int(meta, "loop_config.core_start", 0);
    const int64_t core_e = meta_int(meta, "loop_config.core_end", 0);
    const int64_t passes = meta_int(meta, "loop_config.num_passes", 0);

    if (g.n_layer != kLayers || m.KVH <= 0 || m.NH <= 0 || m.NH % m.KVH != 0 || m.HD <= 0 ||
        m.E <= 0 || m.K <= 0 || m.F <= 0 || m.DR <= 0 || m.EXT <= 0 || m.WIN <= 0 || m.VR <= 0 ||
        m.VR > g.n_vocab_all ||
        m.RANK <= 0 || !(m.qclip > 0) || !(m.rclip > 0) || m.alpha < 0 || !(m.floorv > 0)) {
        fprintf(stderr, "radiance: inkling2: the checkpoint's geometry is not this loop model's "
                        "(layers %lld, heads %lld over %lld, head_dim %lld, experts %lld top-%lld, "
                        "d_rel %lld, extent %lld, window %lld, vocab %lld, rank %lld)\n",
                (long long)g.n_layer, (long long)m.NH, (long long)m.KVH, (long long)m.HD,
                (long long)m.E, (long long)m.K, (long long)m.DR, (long long)m.EXT,
                (long long)m.WIN, (long long)m.VR, (long long)m.RANK);
        return RAD_E_UNSUPPORTED;
    }
    if (core_s != 6 || core_e != 18 || passes != 3) {
        fprintf(stderr, "radiance: inkling2: this plugin serves the release's loop schedule (core "
                        "6..18, three passes); the checkpoint states core %lld..%lld and %lld passes\n",
                (long long)core_s, (long long)core_e, (long long)passes);
        return RAD_E_UNSUPPORTED;
    }
    if (meta_int(meta, "n_shared_experts", 0) != 0) {
        fprintf(stderr, "radiance: inkling2: shared experts are not in this model\n");
        return RAD_E_UNSUPPORTED;
    }
    /* The release's own DSpark drafter (mtp.safetensors; a container converted with it names it
     * draft.model_type `inkling2_dspark`) is not served, with or without speculation. */
    if (!std::strcmp(rad_meta_gets(meta, "draft.model_type", ""), "inkling2_dspark")) {
        fprintf(stderr, "radiance: inkling2: this container carries Inkling 2's DSpark drafter "
                        "(draft.model_type inkling2_dspark), which is not served; convert the model "
                        "with incoai's DFlash2 drafter (data/recipes/inkling2-mini-iq4r-v4.recipe, "
                        "rad-convert --draft-model <Inkling-2-Mini-DFlash2 directory>)\n");
        return RAD_E_UNSUPPORTED;
    }

    /* the split (see TENSOR PARALLELISM): geom_from divided the heads and the vocabulary; a KV head
     * is not replicated over ranks, so the world divides the KV heads */
    if (g.world > 1 && (g.n_head_kv_all % g.world || g.n_head_all % g.world)) {
        fprintf(stderr, "radiance: inkling2: %d ranks do not divide the %lld query and %lld KV heads; "
                        "serve it at a --tp that divides both\n", g.world, (long long)g.n_head_all,
                (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }
    /* the head's padding (never sampled) on the rank whose columns hold it */
    {
        const int64_t lo = std::max<int64_t>(m.VR, g.vocab_off);
        const int64_t hi = std::min<int64_t>(m.VN, g.vocab_off + g.n_vocab);
        m.pad_lo = hi > lo ? lo - g.vocab_off : 0;
        m.pad_n = hi > lo ? hi - lo : 0;
    }

    /* full attention on the layers the checkpoint does not list as local */
    std::vector<int> local(kLayers, 0);
    {
        int ids[64];
        const int n = parse_layer_ids(rad_meta_gets(meta, "local_layer_ids", ""), kLayers, ids, 64);
        for (int i = 0; i < n; ++i)
            if (ids[i] >= 0 && ids[i] < kLayers) local[(size_t)ids[i]] = 1;
    }
    m.layers.assign(kLayers, Layer{});
    for (int l = 0; l < kLayers; ++l) {
        Layer& L = m.layers[(size_t)l];
        L.full = !local[(size_t)l];
        L.core = l >= 6 && l < 18;
        L.n_mix = L.full ? 3 : 1;
        block_counts(l, &L.n_state_attn, &L.n_state_mlp, &L.push_slot);
        /* The experts' maps and their stored form, read from the first expert's tensors (every
         * expert of a layer is one form), ahead of the buffers: the form decides the width of this
         * rank's slice, which the buffers between the expert GEMMs take. */
        const std::string ex = std::string(pfx) + ".layers." + std::to_string(l) + ".mlp.experts";
        for (int64_t e = 0; e < m.E; ++e)
            RAD_ARCH_TRY(map_slice(b, nm.f("blk.%d.e%lld.gu", l, (long long)e), nm.ckpt("%s.w13_weight", ex.c_str()), (int)e));
        for (int64_t e = 0; e < m.E; ++e)
            RAD_ARCH_TRY(map_slice(b, nm.f("blk.%d.e%lld.dn", l, (long long)e), nm.ckpt("%s.w2_weight", ex.c_str()), (int)e));
        int dn_form = 0;
        RAD_ARCH_TRY(expert_form(b, nm.f("blk.%d.e0.gu", l), &L.ex_form, &L.ex_cdt[0], &L.ex_run[0]));
        RAD_ARCH_TRY(expert_form(b, nm.f("blk.%d.e0.dn", l), &dn_form, &L.ex_cdt[1], &L.ex_run[1]));
        if (dn_form != L.ex_form) {
            fprintf(stderr, "radiance: inkling2: layer %d's gate_up and down experts are stored in "
                            "different forms\n", l);
            return RAD_E_DTYPE;
        }
    }
    /* this rank's slice of an expert (TENSOR PARALLELISM): the widest any layer's form takes */
    {
        bool any_bf16 = false, any_iqr = false;
        for (const Layer& L : m.layers) {
            if (L.ex_form == kExpBf16) any_bf16 = true;
            else                       any_iqr = true;
        }
        int64_t qlo[2] = { 0, 0 }, qw[2] = { 0, 0 }, blo[2] = { 0, 0 }, bw[2] = { 0, 0 };
        const bool qok = any_iqr && expert_slice(m.F, g.world, g.rank, true, qlo, qw);
        const bool bok = any_bf16 && expert_slice(m.F, g.world, g.rank, false, blo, bw);
        m.FW = std::max(qok ? std::max(qw[0], qw[1]) : 0, bok ? bw[0] : 0);
        if ((any_iqr && !qok) || (any_bf16 && !bok) || m.FW <= 0) {
            fprintf(stderr, "radiance: inkling2: an expert's %lld columns do not split over %d ranks\n",
                    (long long)m.F, g.world);
            return RAD_E_UNSUPPORTED;
        }
        if (g.world > 1)
            rad_note(b, "inkling2 rank %d/%d: query heads [%lld, %lld) of %lld, KV heads [%lld, %lld) "
                        "of %lld, vocabulary [%lld, %lld) of %lld; an expert's columns "
                        "[%lld, %lld) when even and [%lld, %lld) when odd, of %lld (%s)",
                     g.rank, g.world, (long long)(g.rank * m.NH), (long long)((g.rank + 1) * m.NH),
                     (long long)g.n_head_all, (long long)(g.rank * m.KVH),
                     (long long)((g.rank + 1) * m.KVH), (long long)g.n_head_kv_all,
                     (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab), (long long)m.VN,
                     (long long)(qok ? qlo[0] : blo[0]), (long long)(qok ? qlo[0] + qw[0] : blo[0] + bw[0]),
                     (long long)(qok ? qlo[1] : blo[1]), (long long)(qok ? qlo[1] + qw[1] : blo[1] + bw[1]),
                     (long long)m.F, qok ? "IQR" : "bf16");
    }

    const int64_t T = g.max_tok;
    const int64_t NHD = m.NH * m.HD, KVD = m.KVH * m.HD, KVC = m.KVH * kWidth;

    /* ---- state: one bound layer per application of each stateful kind */
    const uint32_t kvd = rad_kv_cache_dtype(ctx, RAD_BF16);
    if (kvd != RAD_BF16 && kvd != RAD_I8) {
        fprintf(stderr, "radiance: inkling2: unsupported kv_cache_dtype %u (expected bf16 or i8)\n",
                (unsigned)kvd);
        return RAD_E_UNSUPPORTED;
    }
    m.kv_dt = kvd;
    {
        RadKVGroupDecl fd{};
        fd.kind = RAD_KV_FULL;
        fd.dtype = kvd;
        fd.n_head_kv = m.KVH;
        fd.head_dim = m.HD;
        m.kv_full = rad_decl_kv_group(b, nm.f("kv_full"), &fd);
        RadKVGroupDecl ld{};
        ld.kind = RAD_KV_WINDOW;
        ld.dtype = kvd;
        ld.window = m.WIN;
        ld.n_head_kv = m.KVH;
        ld.head_dim = m.HD;
        m.kv_local = rad_decl_kv_group(b, nm.f("kv_local"), &ld);
        if (!m.kv_full || !m.kv_local) return RAD_E_INVAL;
        const int64_t widths[4] = { KVD, KVD, m.H, m.H };
        rad_kvgroup* gs[4] = { &m.cv_k, &m.cv_v, &m.cv_a, &m.cv_m };
        const char* names[4] = { "conv_k", "conv_v", "conv_a", "conv_m" };
        for (int s = 0; s < 4; ++s) {
            RadKVGroupDecl cd{};
            cd.kind = RAD_KV_CONV;
            cd.dtype = RAD_BF16;
            cd.n_head_kv = 1;
            cd.head_dim = widths[s];
            cd.conv_width = kWidth;
            *gs[s] = rad_decl_kv_group(b, nm.f(names[s]), &cd);
            if (!*gs[s]) return RAD_E_INVAL;
        }
        for (int a = 0; a < kApps; ++a) {
            const bool full = m.layers[(size_t)phys_of(a)].full;
            RAD_ARCH_TRY(rad_bind_layer_kv(b, a, full ? m.kv_full : m.kv_local));
            for (rad_kvgroup cg : { m.cv_k, m.cv_v, m.cv_a, m.cv_m })
                RAD_ARCH_TRY(rad_bind_layer_kv(b, a, cg));
        }
    }

    /* ---- activations: [max_tok, width] unless stated */
    m.b_s     = decl_b(b, nm.f("S"),     RAD_BF16, {T, m.H});
    m.b_pre   = decl_b(b, nm.f("pre"),   RAD_BF16, {T, m.H});
    m.b_bank1 = decl_b(b, nm.f("bank1"), RAD_BF16, {T, m.H});
    m.b_bank0 = decl_b(b, nm.f("bank0"), RAD_BF16, {1, T, m.H});
    m.b_blk   = decl_b(b, nm.f("blocks"), RAD_BF16, {kBlocks, T, m.H});
    m.b_ones  = decl_b(b, nm.f("ones"),  RAD_BF16, {T, m.H});
    m.b_minus1 = decl_b(b, nm.f("minus1"), RAD_BF16, {T, m.H});
    for (int j = 0; j < 3; ++j) {
        m.b_y[j]  = decl_b(b, nm.f("mix%d", j), RAD_BF16, {T, m.H});
        m.b_h[j]  = decl_b(b, nm.f("norm%d", j), RAD_BF16, {T, m.H});
        m.b_cy[j] = decl_b(b, nm.f("ctl%d", j), RAD_BF16, {T, m.H});
    }
    for (int j = 0; j < 4; ++j) m.b_msg[j] = decl_b(b, nm.f("msg%d", j), RAD_BF16, {T, m.H});
    m.b_gxm = decl_b(b, nm.f("gxm"), RAD_BF16, {T, 2 * m.H});
    m.b_g1  = decl_b(b, nm.f("g1"), RAD_BF16, {T, m.RANK});
    m.b_g2  = decl_b(b, nm.f("g2"), RAD_BF16, {T, m.RANK});
    m.b_g   = decl_b(b, nm.f("g"),  RAD_BF16, {T, m.RANK});
    m.b_gp  = decl_b(b, nm.f("gp"), RAD_BF16, {T, m.H});
    m.b_gt  = decl_b(b, nm.f("gt"), RAD_BF16, {T, m.H});
    m.b_fp  = decl_b(b, nm.f("fp"), RAD_BF16, {T, m.H});
    m.b_ft  = decl_b(b, nm.f("ft"), RAD_BF16, {T, m.H});
    m.b_nft = decl_b(b, nm.f("nft"), RAD_BF16, {T, m.H});
    m.b_t1  = decl_b(b, nm.f("t1"), RAD_BF16, {T, m.H});
    m.b_t2  = decl_b(b, nm.f("t2"), RAD_BF16, {T, m.H});
    m.b_t3  = decl_b(b, nm.f("t3"), RAD_BF16, {T, m.H});
    m.b_q   = decl_b(b, nm.f("q"),  RAD_BF16, {T, NHD});
    m.b_k   = decl_b(b, nm.f("k"),  RAD_BF16, {T, KVD});
    m.b_v   = decl_b(b, nm.f("v"),  RAD_BF16, {T, KVD});
    m.b_r   = decl_b(b, nm.f("r"),  RAD_BF16, {T, m.NH * m.DR});
    if (g.world > 1) m.b_rf = decl_b(b, nm.f("r_all"), RAD_BF16, {T, m.NH * m.DR * g.world});
    m.b_gate = decl_b(b, nm.f("gate"), RAD_BF16, {T, NHD});
    m.b_sink = decl_b(b, nm.f("sink"), RAD_BF16, {T, m.NH});
    m.b_dk  = decl_b(b, nm.f("dk"), RAD_BF16, {T, KVC});
    m.b_dv  = decl_b(b, nm.f("dv"), RAD_BF16, {T, KVC});
    m.b_kcv = decl_b(b, nm.f("kcv"), RAD_BF16, {T, KVD});
    m.b_vcv = decl_b(b, nm.f("vcv"), RAD_BF16, {T, KVD});
    m.b_qc  = decl_b(b, nm.f("qc"), RAD_BF16, {T, NHD});
    m.b_kc  = decl_b(b, nm.f("kc"), RAD_BF16, {T, KVD});
    m.b_vc  = decl_b(b, nm.f("vc"), RAD_BF16, {T, KVD});
    m.b_rc  = decl_b(b, nm.f("rc"), RAD_BF16, {T, m.NH * m.DR});
    m.b_att = decl_b(b, nm.f("att"), RAD_BF16, {T, NHD});
    m.b_sig = decl_b(b, nm.f("sig"), RAD_BF16, {T, NHD});
    m.b_attg = decl_b(b, nm.f("attg"), RAD_BF16, {T, NHD});
    m.b_ao  = decl_b(b, nm.f("ao"), RAD_BF16, {T, m.H});
    m.b_ax  = decl_b(b, nm.f("ax"), RAD_BF16, {T, m.H});
    m.b_mx  = decl_b(b, nm.f("mx"), RAD_BF16, {T, m.H});
    m.b_cym = decl_b(b, nm.f("cym"), RAD_BF16, {T, m.H});
    m.b_fm  = decl_b(b, nm.f("fm"), RAD_BF16, {T, m.H});
    m.b_rl  = decl_b(b, nm.f("rl"), RAD_F32, {T, m.E});
    m.b_ids = decl_b(b, nm.f("ids"), RAD_I32, {T, m.K});
    m.b_wts = decl_b(b, nm.f("wts"), RAD_F32, {T, m.K});
    m.b_sorted = decl_b(b, nm.f("sorted"), RAD_I32, {T * m.K});
    m.b_eoff = decl_b(b, nm.f("eoff"), RAD_I32, {m.E + 1});
    m.b_ecnt = decl_b(b, nm.f("ecnt"), RAD_I32, {m.E});
    m.b_gu  = decl_b(b, nm.f("gu"), RAD_BF16, {T * m.K, 2 * m.FW});
    m.b_act = decl_b(b, nm.f("act"), RAD_BF16, {T * m.K, m.FW});
    /* an IQR expert's routed input and SiTU product: int8 codes with an f32 scale per 128 */
    m.b_q8  = decl_b(b, nm.f("q8"), RAD_I8, {T, m.H});
    m.b_s8  = decl_b(b, nm.f("s8"), RAD_F32, {T, (m.H + kExpGroup - 1) / kExpGroup});
    m.b_aq  = decl_b(b, nm.f("aq"), RAD_I8, {T * m.K, m.FW});
    m.b_as  = decl_b(b, nm.f("as"), RAD_F32, {T * m.K, (m.FW + kExpGroup - 1) / kExpGroup});
    m.b_dn  = decl_b(b, nm.f("dn"), RAD_BF16, {T * m.K, m.H});
    m.b_mo  = decl_b(b, nm.f("mo"), RAD_BF16, {T, m.H});
    m.b_zc  = decl_b(b, nm.f("zc"), RAD_BF16, {T, m.H});
    m.b_yo  = decl_b(b, nm.f("yo"), RAD_BF16, {T, m.H});
    m.b_yn  = decl_b(b, nm.f("yn"), RAD_BF16, {T, m.H});
    m.b_hout = decl_b(b, nm.f("hout"), RAD_BF16, {g.max_logit_rows, m.H});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32, {g.max_logit_rows, g.n_vocab});
    /* The planner bounds a buffer's life by the op range its declared uses span, in declaration
     * order, but a step runs the 24 physical layers' ops 48 times. A buffer that carries values
     * across applications (the stream, the prelude, the banks, the block planes, the constants)
     * would get a short life there and share bytes with a layer's buffers that run in between:
     * the prelude written at application 6 and read at 30, overwritten by the controllers at 18.
     * Which buffer gets clobbered depends on the arena's packing, so --max-num-seqs changed the
     * output. These are live for the whole step. */
    for (rad_buf h : { m.b_s, m.b_pre, m.b_bank0, m.b_bank1, m.b_blk, m.b_ones, m.b_minus1 })
        RAD_ARCH_TRY(rad_buf_concurrent(b, h));
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- weights: the vocabulary edges */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s.embed.weight", pfx)));
    const int vsh = g.world > 1 ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    /* bf16 as shipped, or a table on the int8 grid where a recipe quantised it (int8 codes and a
     * bf16 scale per row and 128 columns): declared over all its planes at the codes' dtype, so
     * embed_lookup's IQR gather (libr4d embed_lookup_iqr) reads the codes where they are held and
     * decodes the rows it gathers. */
    {
        RadEncoding te{};
        const bool q = weight_enc(b, nm.f("token_embd.weight"), &te) && !rad_enc_is(&te, "plain");
        m.w_embed = q ? decl_view(b, nm.f("token_embd.weight"), nullptr, nullptr, te.plane[0].dtype,
                                  {g.n_vocab, m.H}, RAD_ACCESS_VOCAB, vsh, grp_model())
                      : decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, m.H},
                               RAD_ACCESS_VOCAB, vsh, grp_model());
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("embed_norm.weight"), nm.ckpt("%s.embed_norm.weight", pfx)));
    m.w_embed_norm = decl_w(b, nm.f("embed_norm.weight"), RAD_BF16, {m.H}, RAD_ACCESS_PER_TOKEN,
                            RAD_SHARD_NONE, grp_model());
    /* the head covers the padded table, since the engine samples over n_vocab; op_pad fills the
     * logits past unpadded_vocab_size so the padding is never sampled. */
    RAD_ARCH_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("%s.unembed.weight", pfx)));
    /* bf16 as shipped, or the int8 grid (a bf16 scale per row per 128) where a recipe quantised it:
     * logits_gemm reads either, and the head is a decode step's widest GEMM */
    m.w_head = decl_w(b, nm.f("output.weight"), weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                      {g.n_vocab, m.H}, RAD_ACCESS_VOCAB, vsh, grp_model());
    if (!m.w_embed || !m.w_head) return RAD_E_INVAL;
    if (g.world > 1 && g.n_vocab_all % g.world) {
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_embed, g.vocab_off, g.vocab_off + g.n_vocab));
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_head, g.vocab_off, g.vocab_off + g.n_vocab));
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.norm.weight", pfx)));
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_BF16, {m.H}, RAD_ACCESS_PER_TOKEN,
                          RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("output_query"), nm.ckpt("%s.attn_res.output_query", pfx)));
    m.w_out_q = decl_w(b, nm.f("output_query"), RAD_F32, {1, m.H}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                       grp_model());

    /* ---- weights: the layers, shared by the applications that name their physical layer */
    for (int l = 0; l < kLayers; ++l) {
        Layer& L = m.layers[(size_t)l];
        const std::string pre = std::string(pfx) + ".layers." + std::to_string(l) + ".";
        const std::string at = pre + "attn.";
        /* `shard`: this rank's heads or channels (see TENSOR PARALLELISM); replicated at one rank */
        auto one = [&](rad_weight* w, const char* name, const std::string& src, uint32_t dt,
                       std::initializer_list<int64_t> shape, int shard = RAD_SHARD_NONE) -> int {
            RAD_ARCH_TRY(map_copy(b, nm.f(name, l), nm.ckpt("%s", src.c_str())));
            *w = decl_w(b, nm.f(name, l), dt, shape, RAD_ACCESS_PER_TOKEN,
                        g.world > 1 ? shard : RAD_SHARD_NONE, grp_layer(l));
            return *w ? RAD_OK : RAD_E_INVAL;
        };
        const int ROW = RAD_SHARD_ROW, COL = RAD_SHARD_COL;
        if (L.full) {
            static const char* nn[3] = { "q_input_norm", "k_input_norm", "v_input_norm" };
            for (int j = 0; j < 3; ++j) {
                RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.norm%d", l, j), nm.ckpt("%s.weight", (pre + nn[j]).c_str())));
                L.w_norm[j] = decl_w(b, nm.f("blk.%d.norm%d", l, j), RAD_BF16, {m.H}, RAD_ACCESS_PER_TOKEN,
                                     RAD_SHARD_NONE, grp_layer(l));
                if (!L.w_norm[j]) return RAD_E_INVAL;
            }
        } else {
            RAD_ARCH_TRY(one(&L.w_norm[0], "blk.%d.attn_norm", pre + "attn_norm.weight", RAD_BF16, {m.H}));
        }
        /* the attention linears, all in one form (the layer's first decides; a mixed layer is
         * refused): bf16 weights, or rotated int8 codes and their scale planes as two views */
        {
            /* `shard`: q, k, v, r and the gate by rows (this rank's heads), wo by columns. An int8
             * linear's 128-groups of K never straddle two ranks: K is the stream for the rows'
             * linears, and wo's K is whole heads of 128. */
            struct LinSpec { rad_weight* w; rad_weight* ws; const char* name; std::string src; int64_t N, K; int shard; };
            const LinSpec specs[] = {
                { &L.w_q, &L.w_qs, "blk.%d.wq", at + "wq_du.weight", NHD, m.H, ROW },
                { &L.w_k, &L.w_ks, "blk.%d.wk", at + "wk_dv.weight", KVD, m.H, ROW },
                { &L.w_v, &L.w_vs, "blk.%d.wv", at + "wv_dv.weight", KVD, m.H, ROW },
                { &L.w_r, &L.w_rs, "blk.%d.wr", at + "wr_du.weight", m.NH * m.DR, m.H, ROW },
                { &L.w_gate, &L.w_gates, "blk.%d.gate", at + "gate_attn.weight", NHD, m.H, ROW },
                { &L.w_o, &L.w_os, "blk.%d.wo", at + "wo_ud.weight", m.H, NHD, COL },
            };
            for (size_t i = 0; i < sizeof specs / sizeof specs[0]; ++i) {
                const LinSpec& sp = specs[i];
                int sh = g.world > 1 ? sp.shard : RAD_SHARD_NONE;
                int64_t N = sp.N;
                const std::string wn = nm.f(sp.name, l);
                RAD_ARCH_TRY(map_copy(b, wn.c_str(), nm.ckpt("%s", sp.src.c_str())));
                bool i8 = false;
                RAD_ARCH_TRY(lin_form(b, wn.c_str(), &i8));
                if (i == 0) L.lin_i8 = i8;
                if (i8 != L.lin_i8) {
                    fprintf(stderr, "radiance: inkling2: layer %d's attention linears are stored in "
                                    "different forms ('%s')\n", l, wn.c_str());
                    return RAD_E_DTYPE;
                }
                if (i8 && sp.w == &L.w_r && sh != RAD_SHARD_NONE && N % kExpGroup) {
                    L.r_full = true;
                    sh = RAD_SHARD_NONE;
                    N *= g.world;
                }
                if (!i8) {
                    *sp.w = decl_w(b, wn.c_str(), RAD_BF16, {N, sp.K}, RAD_ACCESS_PER_TOKEN,
                                   sh, grp_layer(l));
                } else {
                    if (N % kExpGroup || sp.K % kExpGroup) {
                        fprintf(stderr, "radiance: inkling2: int8 linear '%s' [%lld, %lld] is not whole "
                                        "%d-blocks\n", wn.c_str(), (long long)N, (long long)sp.K, kExpGroup);
                        return RAD_E_INVAL;
                    }
                    *sp.w = decl_view(b, wn.c_str(), nullptr, "codes", RAD_I8, {N, sp.K},
                                      RAD_ACCESS_PER_TOKEN, sh, grp_layer(l));
                    const std::string sn = wn + ".scale";
                    *sp.ws = decl_view(b, sn.c_str(), wn.c_str(), "scale", RAD_BF16, {N, sp.K / kExpGroup},
                                       RAD_ACCESS_PER_TOKEN, sh, grp_layer(l));
                    if (!*sp.ws) return RAD_E_INVAL;
                }
                if (!*sp.w) return RAD_E_INVAL;
            }
        }
        RAD_ARCH_TRY(one(&L.w_sink, "blk.%d.sink", at + "attn_sink_proj.weight", RAD_BF16, {m.NH, m.H}, ROW));
        /* The clip scales are stored [heads, width]; one rank reads them as one vector, and a rank's
         * share is its heads' rows, so a split declares the rows. */
        if (g.world > 1) {
            RAD_ARCH_TRY(one(&L.w_qscale, "blk.%d.qscale", at + "q_l2_clip.scale", RAD_BF16, {m.NH, m.HD}, ROW));
            RAD_ARCH_TRY(one(&L.w_rscale, "blk.%d.rscale", at + "rel_l2_clip.scale", RAD_BF16, {m.NH, m.DR}, ROW));
        } else {
            RAD_ARCH_TRY(one(&L.w_qscale, "blk.%d.qscale", at + "q_l2_clip.scale", RAD_BF16, {NHD}));
            RAD_ARCH_TRY(one(&L.w_rscale, "blk.%d.rscale", at + "rel_l2_clip.scale", RAD_BF16, {m.NH * m.DR}));
        }
        RAD_ARCH_TRY(one(&L.w_rel, "blk.%d.rel", at + "rel_logits_proj.proj", RAD_BF16, {m.DR, m.EXT}));
        if (L.full) {
            /* the dynamic taps' rows are head-major (head * width + lag), so a rank's KV heads are
             * a contiguous run of rows */
            RAD_ARCH_TRY(one(&L.w_dk, "blk.%d.dk", at + "k_sconv.weight_proj", RAD_BF16, {KVC, m.H}, ROW));
            RAD_ARCH_TRY(one(&L.w_dv, "blk.%d.dv", at + "v_sconv.weight_proj", RAD_BF16, {KVC, m.H}, ROW));
        }
        RAD_ARCH_TRY(one(&L.w_kconv, "blk.%d.kconv", at + "k_sconv.weight", RAD_BF16, {KVD, kWidth}, ROW));
        RAD_ARCH_TRY(one(&L.w_vconv, "blk.%d.vconv", at + "v_sconv.weight", RAD_BF16, {KVD, kWidth}, ROW));
        RAD_ARCH_TRY(one(&L.w_aconv, "blk.%d.aconv", pre + "attn_sconv.weight", RAD_BF16, {m.H, kWidth}));
        RAD_ARCH_TRY(one(&L.w_mconv, "blk.%d.mconv", pre + "mlp_sconv.weight", RAD_BF16, {m.H, kWidth}));
        /* the attention input queries: one row of the layer's table each, a [1, H] weight. The
         * release stores a local layer's single query as a plain [H] vector (its full layers' tables
         * are [3, H]), so a one-query layer copies its table whole. */
        if (l > 0) {
            for (int q = 0; q < L.n_mix; ++q) {
                const char* qsrc = nm.ckpt("%s.attn_res.attn_queries.%d", pfx, l - 1);
                if (L.n_mix == 1)
                    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.mixq%d", l, q), qsrc));
                else
                    RAD_ARCH_TRY(map_rows(b, nm.f("blk.%d.mixq%d", l, q), { { qsrc, q, 1 } }));
                L.w_mix[q] = decl_w(b, nm.f("blk.%d.mixq%d", l, q), RAD_F32, {1, m.H},
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
                if (!L.w_mix[q]) return RAD_E_INVAL;
            }
        }
        /* the feed-forward: its input query, norm and router */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.mlpq", l), nm.ckpt("%s.attn_res.mlp_queries.%d", pfx, l)));
        L.w_mlp_q = decl_w(b, nm.f("blk.%d.mlpq", l), RAD_F32, {1, m.H}, RAD_ACCESS_PER_TOKEN,
                           RAD_SHARD_NONE, grp_layer(l));
        if (!L.w_mlp_q) return RAD_E_INVAL;
        RAD_ARCH_TRY(one(&L.w_mlp_norm, "blk.%d.mlp_norm", pre + "mlp_norm.weight", RAD_BF16, {m.H}));
        RAD_ARCH_TRY(one(&L.w_router, "blk.%d.router", pre + "mlp.gate.weight", RAD_F32, {m.E, m.H}));
        RAD_ARCH_TRY(one(&L.w_bias, "blk.%d.router_bias", pre + "mlp.gate.bias", RAD_F32, {m.E}));
        /* the experts (mapped above): the handles consecutively, as a table needs them */
        {
            /* this rank's columns of each expert (see TENSOR PARALLELISM): gate_up's rows of both
             * parts and down's columns, by parity class */
            const bool split = g.world > 1;
            if (!expert_slice(m.F, g.world, g.rank, L.ex_form == kExpIqr, L.flo, L.fq) ||
                std::max(L.fq[0], L.fq[1]) != m.FW) {
                fprintf(stderr, "radiance: inkling2: layer %d's %s experts (%lld columns) do not split "
                                "over %d ranks as the other layers' do; a container's experts are "
                                "one form throughout\n", l, L.ex_form == kExpBf16 ? "bf16" : "IQR",
                        (long long)m.F, g.world);
                return RAD_E_UNSUPPORTED;
            }
            /* An IQR weight's trellis runs down its rows, so a rank's rows of gate_up (each part's
             * [flo, flo + fq) under tensor parallelism, all 2F at one rank) and down's H are whole
             * runs; moe_gemm_iqr refuses a part that is not. Inkling's 1152 rows split at 128-row
             * blocks, so gate_up takes runs of 128 rows wherever it is split. */
            if (L.ex_form == kExpIqr) {
                bool ok = L.ex_run[0] > 0 && L.ex_run[1] > 0 && m.H % L.ex_run[1] == 0;
                for (int q = 0; q < 2; ++q)
                    ok = ok && (split ? L.fq[q] % L.ex_run[0] == 0 && L.flo[q] % L.ex_run[0] == 0
                                      : (2 * m.F) % L.ex_run[0] == 0);
                if (!ok) {
                    fprintf(stderr, "radiance: inkling2: layer %d's IQR experts run their trellis "
                                    "over %lld (gate_up) and %lld (down) rows, and this rank's "
                                    "rows are gate_up's [%lld, +%lld) and [%lld, +%lld) of each "
                                    "part and down's %lld; convert gate_up with tcq_rows=128\n", l,
                            (long long)L.ex_run[0], (long long)L.ex_run[1], (long long)L.flo[0],
                            (long long)L.fq[0], (long long)L.flo[1], (long long)L.fq[1],
                            (long long)m.H);
                    return RAD_E_SHAPE;
                }
            }
            const bool even = L.fq[0] == L.fq[1] && L.flo[0] == L.flo[1] &&
                              L.flo[0] == g.rank * L.fq[0] && L.fq[0] * g.world == m.F;
            auto span = [&](rad_weight h, int q, int64_t unit) -> int {
                if (!split || even || !h) return RAD_OK;
                return rad_weight_shard_span(b, h, L.flo[q] / unit, (L.flo[q] + L.fq[q]) / unit);
            };
            const int sh_row = split ? RAD_SHARD_ROW : RAD_SHARD_NONE;
            const int sh_col = split ? RAD_SHARD_COL : RAD_SHARD_NONE;
            if (L.ex_form == kExpBf16) {
                const int64_t fw = L.fq[0], parts[2] = { fw, fw };
                for (int64_t e = 0; e < m.E; ++e)
                    L.w_egu.push_back(decl_w_n(b, nm.f("blk.%d.e%lld.gu", l, (long long)e), RAD_BF16, {2 * fw, m.H},
                                               RAD_ACCESS_PER_TOKEN, sh_row, grp_expert(l, (int)e), 0,
                                               split ? parts : nullptr, split ? 2 : 0));
                for (int64_t e = 0; e < m.E; ++e)
                    L.w_edn.push_back(decl_w(b, nm.f("blk.%d.e%lld.dn", l, (long long)e), RAD_BF16, {m.H, fw},
                                             RAD_ACCESS_PER_TOKEN, sh_col, grp_expert_slot(l, (int)e, 1)));
            } else {
                /* An IQR expert: its codes as one view and its scales as another, declared by
                 * parity class (moe_gemm_q takes the even experts' table, then the odd's), so each
                 * class's codes are consecutive handles and its scales are too. */
                if (m.E % 2 || m.H % kExpGroup || m.F % kExpGroup) {
                    fprintf(stderr, "radiance: inkling2: an IQR expert needs an even expert count and "
                                    "H and F multiples of %d (E %lld, H %lld, F %lld)\n", kExpGroup,
                            (long long)m.E, (long long)m.H, (long long)m.F);
                    return RAD_E_INVAL;
                }
                /* The codes view is at the codes' stored width with the code's planes (moe_gemm_iqr's
                 * layout reads the computed trellis's constants beside them), and the row and block
                 * scales are one f32 view at the logical extent (MoeQ's views of the same form).
                 * Under tensor parallelism gate_up is two stacked parts of the class's width, cut by
                 * rows, and down is cut by columns; the scale views take the same cut. */
                auto decl_class = [&](const char* leg, bool gu, std::vector<rad_weight>& wc,
                                      std::vector<rad_weight>& ws, int slot) -> int {
                    wc.assign((size_t)m.E, 0);
                    ws.assign((size_t)m.E, 0);
                    for (int q = 0; q < 2; ++q)
                        for (int64_t e = q; e < m.E; e += 2) {
                            const int64_t N = gu ? 2 * L.fq[q] : m.H, K = gu ? m.H : L.fq[q];
                            const int64_t parts[2] = { L.fq[q], L.fq[q] };
                            const char* w = nm.f("blk.%d.e%lld.%s", l, (long long)e, leg);
                            wc[(size_t)e] = decl_view(b, w, nullptr, "codes,trellis,trellis.mad",
                                                      L.ex_cdt[gu ? 0 : 1], {N, K},
                                                      RAD_ACCESS_PER_TOKEN, gu ? sh_row : sh_col,
                                                      grp_expert_slot(l, (int)e, slot),
                                                      gu && split ? parts : nullptr, gu && split ? 2 : 0);
                            if (!wc[(size_t)e]) return RAD_E_INVAL;
                            RAD_ARCH_TRY(span(wc[(size_t)e], q, 1));
                        }
                    for (int q = 0; q < 2; ++q)
                        for (int64_t e = q; e < m.E; e += 2) {
                            const int64_t N = gu ? 2 * L.fq[q] : m.H, K = gu ? m.H : L.fq[q];
                            const int64_t parts[2] = { L.fq[q], L.fq[q] };
                            const char* w = nm.f("blk.%d.e%lld.%s", l, (long long)e, leg);
                            ws[(size_t)e] = decl_view(b, nm.f("%s.scale", w), w, "scale,scale.col", RAD_F32,
                                                      {N, K}, RAD_ACCESS_PER_TOKEN, gu ? sh_row : sh_col,
                                                      grp_expert_slot(l, (int)e, slot),
                                                      gu && split ? parts : nullptr, gu && split ? 2 : 0);
                            if (!ws[(size_t)e]) return RAD_E_INVAL;
                            RAD_ARCH_TRY(span(ws[(size_t)e], q, 1));
                        }
                    return RAD_OK;
                };
                RAD_ARCH_TRY(decl_class("gu", true, L.w_egu, L.w_egus, 0));
                RAD_ARCH_TRY(decl_class("dn", false, L.w_edn, L.w_edns, 1));
            }
            for (int64_t e = 0; e < m.E; ++e)
                if (!L.w_egu[(size_t)e] || !L.w_edn[(size_t)e]) return RAD_E_INVAL;
        }
        /* the core: four controllers, and the messages' queries from the cross-pass table */
        if (L.core) {
            static const char* cn[4] = { "attn", "k", "v", "mlp" };
            const int i = l - 6;
            for (int j = 0; j < 4; ++j) {
                const std::string cp = pre + "controllers." + cn[j] + ".";
                RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.c%d.down", l, j), nm.ckpt("%s", (cp + "down_weight").c_str())));
                L.w_dx[j] = decl_w(b, nm.f("blk.%d.c%d.down", l, j), RAD_BF16, {m.RANK, 2 * m.H},
                                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
                RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.c%d.gain", l, j), nm.ckpt("%s", (cp + "gain_weight").c_str())));
                L.w_cgain[j] = decl_w(b, nm.f("blk.%d.c%d.gain", l, j), RAD_BF16, {m.H, m.RANK},
                                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
                RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.c%d.forget", l, j), nm.ckpt("%s", (cp + "forget_weight").c_str())));
                L.w_cforget[j] = decl_w(b, nm.f("blk.%d.c%d.forget", l, j), RAD_BF16, {m.H, m.RANK},
                                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
                /* the message query j: the cross-pass table's rows [q[2i], kv[2i], kv[2i+1], q[2i+1]] */
                static const int which[4] = { 0, 1, 1, 0 };          /* 0: q, 1: kv */
                static const int row[4] = { 0, 0, 1, 1 };            /* the core layer's pair of rows */
                const char* src = which[j] == 0 ? "%s.middle_cycle_cross_pass_queries" : "%s.middle_cycle_cross_pass_kv_queries";
                RAD_ARCH_TRY(map_rows(b, nm.f("blk.%d.msgq%d", l, j),
                                      { { nm.ckpt(src, pfx), 2 * i + row[j], 1 } }));
                L.w_msg[j] = decl_w(b, nm.f("blk.%d.msgq%d", l, j), RAD_F32, {1, m.H}, RAD_ACCESS_PER_TOKEN,
                                    RAD_SHARD_NONE, grp_layer(l));
                if (!L.w_msg[j]) return RAD_E_INVAL;
            }
        }
    }
    {
        bool any_i8 = false;
        for (const Layer& L : m.layers) any_i8 = any_i8 || L.lin_i8;
        if (any_i8) {
            for (int j = 0; j < 3; ++j) {
                m.b_hq8[j] = decl_b(b, nm.f("hq8_%d", j), RAD_I8, {T, m.H});
                m.b_hs8[j] = decl_b(b, nm.f("hs8_%d", j), RAD_F32, {T, m.H / kExpGroup});
                if (!m.b_hq8[j] || !m.b_hs8[j]) return RAD_E_INVAL;
            }
            m.b_oq8 = decl_b(b, nm.f("oq8"), RAD_I8, {T, NHD});
            m.b_os8 = decl_b(b, nm.f("os8"), RAD_F32, {T, NHD / kExpGroup});
            if (!m.b_oq8 || !m.b_os8) return RAD_E_INVAL;
        }
    }
    /* ---- ops: the global ones, then each physical layer's, declared once and issued at every
     * application that runs the layer (the weights and the buffers are the same; only the state
     * an application reads, which is named at issue, differs) */
    {
        const int64_t NHD = m.NH * m.HD, KVD = m.KVH * m.HD, KVC = m.KVH * kWidth;
        const int64_t NR = m.NH * m.DR;
        auto gemm = [&](int64_t N, int64_t K, rad_weight w, rad_buf in, rad_buf out) {
            return rw(b, RAD_OP(b, "gemm_nt", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N),
                                                    RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(w)), {in}, {out});
        };
        /* an int8 attention linear over input j's rotated codes */
        auto gemm_q = [&](int64_t N, int64_t K, rad_weight w, rad_weight ws, int j, rad_buf out) {
            return decl_gemm_q(b, T, N, K, kExpGroup, "i8a8h", w, ws, { m.b_hq8[j], m.b_hs8[j] },
                               out);
        };
        auto rms = [&](int64_t rows, int64_t n, rad_weight w, rad_buf in, rad_buf out) {
            if (w)
                return rw(b, RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                        RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(w)), {in}, {out});
            return rw(b, RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                    RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS), {in}, {out});
        };
        auto ew = [&](const char* op, int64_t rows, int64_t n, std::initializer_list<rad_buf> reads, rad_buf out) {
            return rw(b, RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                             RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS), reads, {out});
        };
        auto mix = [&](int64_t n_state, rad_buf cur, rad_weight q, rad_buf out, rad_buf states) {
            return rw(b, RAD_OP(b, "attn_res_mix", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                         RAD_INT("n_state", n_state), RAD_INT("n_query", 1),
                                                         RAD_STR("dtype", "bf16"), RAD_F64("eps", g.eps)),
                               RAD_WEIGHTS(q)), {states, cur}, {out});
        };
        auto sconv = [&](rad_buf x, rad_weight w, rad_buf y, int64_t n, int64_t heads, rad_buf dyn) {
            if (dyn)
                return rw(b, RAD_OP(b, "sconv_res", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                                          RAD_INT("width", kWidth), RAD_INT("heads", heads),
                                                          RAD_STR("dtype", "bf16")), RAD_WEIGHTS(w)),
                          {x, dyn}, {y});
            return rw(b, RAD_OP(b, "sconv_res", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                                      RAD_INT("width", kWidth), RAD_INT("heads", heads),
                                                      RAD_STR("dtype", "bf16")), RAD_WEIGHTS(w)),
                      {x}, {y});
        };
        auto try_op = [&](const char* name, const RadParam* ps, int np,
                          const rad_weight* ws, int nw,
                          std::initializer_list<rad_buf> reads,
                          std::initializer_list<rad_buf> writes) -> rad_op {
            if (!rad_op_declared(b, name)) return 0;
            rad_op h = rad_decl_op(b, name, ps, np, ws, nw);
            if (!h || !rad_op_resolved(b, h)) return 0;
            return rw(b, h, reads, writes);
        };
        auto try_copy = [&](rad_buf src, rad_buf ref, rad_buf dst) -> rad_op {
            const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H), RAD_STR("dtype", "bf16") };
            if (src == ref)
                return try_op("plane_copy", ps, 3, nullptr, 0, {src}, {dst});
            return try_op("plane_copy", ps, 3, nullptr, 0, {src, ref}, {dst});
        };

        /* the stream's copies, and the state the pass boundaries move */
        m.op_ones = rw(b, RAD_OP(b, "fill", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                  RAD_STR("dtype", "bf16"), RAD_F64("value", 1.0)),
                              RAD_NOWEIGHTS), {}, {m.b_ones});
        m.op_minus = rw(b, RAD_OP(b, "fill", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                   RAD_STR("dtype", "bf16"), RAD_F64("value", -1.0)),
                               RAD_NOWEIGHTS), {}, {m.b_minus1});
        if (g.world > 1) {
            /* this rank's rows of the table; the others read zero, so the sum is exact */
            m.op_embed = rw(b, RAD_OP(b, "embed_lookup", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", m.H),
                                                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                                               RAD_INT("vocab_offset", g.vocab_off)),
                                     RAD_WEIGHTS(m.w_embed)), {}, {m.b_s});
            m.op_embed_ar = decl_all_reduce_exact(b, g, m.b_s, m.H, T, "bf16");
            if (!m.op_embed_ar) return RAD_E_UNSUPPORTED;
        } else {
            m.op_embed = rw(b, RAD_OP(b, "embed_lookup", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", m.H),
                                                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16")),
                                     RAD_WEIGHTS(m.w_embed)), {}, {m.b_s});
        }
        /* a row-parallel GEMM's partial sums, reduced in place (tp-wire states whether a lossy wire
         * is accepted) */
        auto all_reduce = [&](rad_buf x) -> rad_op {
            if (g.world <= 1) return 0;
            return decl_all_reduce(b, g, x, m.H, T, "bf16");
        };
        m.op_embed_norm = rms(T, m.H, m.w_embed_norm, m.b_s, m.b_s);
        m.op_pre = ew("mul", T, m.H, {m.b_s, m.b_ones}, m.b_pre);
        m.op_bank_blk = ew("mul", T, m.H, {m.b_blk, m.b_ones}, m.b_bank0);
        m.op_bank_cur = ew("mul", T, m.H, {m.b_s, m.b_ones}, m.b_bank1);
        m.op_reset = ew("mul", T, m.H, {m.b_pre, m.b_ones}, m.b_s);
        m.op_pre_copy      = try_copy(m.b_s,   m.b_s,   m.b_pre);
        m.op_bank_blk_copy = try_copy(m.b_blk, m.b_s,   m.b_bank0);
        m.op_bank_cur_copy = try_copy(m.b_s,   m.b_s,   m.b_bank1);
        m.op_reset_copy    = try_copy(m.b_pre, m.b_pre, m.b_s);
        bool all_copy = (m.op_pre_copy && m.op_bank_blk_copy && m.op_bank_cur_copy && m.op_reset_copy);
        bool all_fused = true, all_blend = true;
        m.op_out_mix = mix(4, m.b_s, m.w_out_q, m.b_yo, m.b_blk);
        m.op_out_norm = rms(T, m.H, m.w_out_norm, m.b_yo, m.b_yn);
        {
            const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H), RAD_INT("n_state", 4),
                                    RAD_INT("n_query", 1), RAD_STR("dtype", "bf16"),
                                    RAD_F64("eps", g.eps) };
            const rad_weight ws[] = { m.w_out_q, m.w_out_norm };
            m.op_out_mix_norm = try_op("attn_res_mix", ps, 6, ws, 2, {m.b_blk, m.b_s}, {m.b_yn});
        }
        m.op_rows = rw(b, RAD_OP(b, "gather_rows", RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                                         RAD_INT("n", m.H), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS), {m.b_yn}, {m.b_hout});
        m.op_head = rw(b, RAD_OP(b, "logits_gemm", RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                                         RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", m.H),
                                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.w_head)), {m.b_hout}, {m.b_logits});
        m.op_softcap = rw(b, RAD_OP(b, "logit_softcap", RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                                              RAD_INT("n", g.n_vocab), RAD_F64("cap", m.fcap),
                                                              RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS),
                         {m.b_logits}, {m.b_logits});
        /* The head's rows past unpadded_vocab_size are padding the checkpoint never trained, and
         * the reference drops them. Scored, they take real mass after a control token
         * (<|message_model|> puts ~40% on them) and sample as tokens that decode to nothing. */
        if (m.pad_n > 0)
            m.op_pad = rw(b, RAD_OP(b, "fill", RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                                       RAD_INT("n", m.pad_n),
                                                       RAD_STR("dtype", "f32"), RAD_F64("value", -1e30)),
                                     RAD_NOWEIGHTS),
                          {}, {m.b_logits});

        for (int ip = 0; ip < kLayers; ++ip) {
            Layer& L = m.layers[(size_t)ip];
            const rad_kvgroup grp = L.full ? m.kv_full : m.kv_local;
            const int64_t n_mix = L.n_mix;
            /* attention inputs */
            for (int q = 0; q < n_mix && ip > 0; ++q)
                L.op_mix[q] = mix(L.n_state_attn, m.b_s, L.w_mix[q], m.b_y[q], m.b_blk);
            if (ip > 0) {
                if (L.full) {
                    const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                            RAD_INT("n_state", L.n_state_attn),
                                            RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16") };
                    const rad_weight ws3[] = { L.w_mix[0], L.w_mix[1], L.w_mix[2] };
                    L.op_mix3 = try_op("attn_res_mix3", ps, 5, ws3, 3, {m.b_blk, m.b_s},
                                       {m.b_y[0], m.b_y[1], m.b_y[2]});
                    const rad_weight ws6[] = { L.w_mix[0], L.w_mix[1], L.w_mix[2],
                                               L.w_norm[0], L.w_norm[1], L.w_norm[2] };
                    L.op_mix3_norm = try_op("attn_res_mix3", ps, 5, ws6, 6, {m.b_blk, m.b_s},
                                            {m.b_h[0], m.b_h[1], m.b_h[2]});
                } else {
                    const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                            RAD_INT("n_state", L.n_state_attn),
                                            RAD_INT("n_query", 1), RAD_STR("dtype", "bf16"),
                                            RAD_F64("eps", g.eps) };
                    const rad_weight ws[] = { L.w_mix[0], L.w_norm[0] };
                    L.op_mix_norm = try_op("attn_res_mix", ps, 6, ws, 2, {m.b_blk, m.b_s},
                                           {m.b_h[0]});
                }
            }
            for (int j = 0; j < 3; ++j) {
                if (ip == 0) {
                    if (j == 0) L.op_norm[0] = rms(T, m.H, L.w_norm[0], m.b_s, m.b_h[0]);
                } else if (L.full) {
                    L.op_norm[j] = rms(T, m.H, L.w_norm[j], m.b_y[j], m.b_h[j]);
                    L.op_norm_ctl[j] = rms(T, m.H, L.w_norm[j], m.b_cy[j], m.b_h[j]);
                } else if (j == 0) {
                    L.op_norm[0] = rms(T, m.H, L.w_norm[0], m.b_y[0], m.b_h[0]);
                    L.op_norm_ctl[0] = rms(T, m.H, L.w_norm[0], m.b_cy[0], m.b_h[0]);
                }
                if (!L.full && j > 0 && ip > 0) {
                    L.op_norm_ctl[j] = rms(T, m.H, L.w_norm[0], m.b_cy[j], m.b_h[j]);
                }
            }
            if (L.lin_i8) {
                auto try_attn_norm_hq = [&](rad_buf src, rad_weight w, int j) -> rad_op {
                    const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                            RAD_F64("eps", g.eps), RAD_INT("group", kExpGroup),
                                            RAD_STR("dtype", "bf16") };
                    return try_op("rmsnorm_had_quant_i8g", ps, 5, &w, 1, {src},
                                  {m.b_hq8[j], m.b_hs8[j], m.b_h[j]});
                };
                if (ip == 0) {
                    L.op_norm_hq[0] = try_attn_norm_hq(m.b_s, L.w_norm[0], 0);
                } else {
                    for (int j = 0; j < 3; ++j)
                        L.op_norm_ctl_hq[j] = try_attn_norm_hq(m.b_cy[j], L.w_norm[L.full ? j : 0], j);
                }
                for (int j = 0; j < 3; ++j)
                    L.op_hq[j] = rw(b, RAD_OP(b, "had_quant_act_i8g",
                                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                         RAD_INT("group", kExpGroup), RAD_STR("dtype", "bf16")),
                                              RAD_NOWEIGHTS), {m.b_h[j]}, {m.b_hq8[j], m.b_hs8[j]});
                L.op_q = gemm_q(NHD, m.H, L.w_q, L.w_qs, 0, m.b_q);
                L.op_k = gemm_q(KVD, m.H, L.w_k, L.w_ks, 1, m.b_k);
                L.op_v = gemm_q(KVD, m.H, L.w_v, L.w_vs, 2, m.b_v);
                L.op_r = L.r_full ? gemm_q(NR * g.world, m.H, L.w_r, L.w_rs, 0, m.b_rf)
                                  : gemm_q(NR, m.H, L.w_r, L.w_rs, 0, m.b_r);
                L.op_gate = gemm_q(NHD, m.H, L.w_gate, L.w_gates, 0, m.b_gate);
                {
                    /* the five in one launch where a step is a decode or verify's few rows; q, r
                     * and the gate read input 0, k input 1 and v input 2, as the five above */
                    const int64_t T_proj = std::min<int64_t>(T, 16);
                    const RadParam ps[] = {
                        RAD_RANGE("M", 1, T_proj), RAD_INT("K", m.H), RAD_INT("group", kExpGroup),
                        RAD_STR("dtype", "i8a8h"),
                        RAD_INT("N0", NHD), RAD_INT("N1", KVD), RAD_INT("N2", KVD),
                        RAD_INT("N3", L.r_full ? NR * g.world : NR), RAD_INT("N4", NHD),
                        RAD_INT("src0", 0), RAD_INT("src1", 1), RAD_INT("src2", 2),
                        RAD_INT("src3", 0), RAD_INT("src4", 0) };
                    const rad_weight ws[] = { L.w_q, L.w_qs, L.w_k, L.w_ks, L.w_v, L.w_vs,
                                              L.w_r, L.w_rs, L.w_gate, L.w_gates };
                    L.op_proj = try_op("gemm_nt_q_multi", ps, (int)(sizeof ps / sizeof ps[0]),
                                       ws, 10,
                                       { m.b_hq8[0], m.b_hs8[0], m.b_hq8[1], m.b_hs8[1],
                                         m.b_hq8[2], m.b_hs8[2] },
                                       { m.b_q, m.b_k, m.b_v, L.r_full ? m.b_rf : m.b_r,
                                         m.b_gate });
                }
            } else {
                L.op_q = gemm(NHD, m.H, L.w_q, m.b_h[0], m.b_q);
                L.op_k = gemm(KVD, m.H, L.w_k, m.b_h[1], m.b_k);
                L.op_v = gemm(KVD, m.H, L.w_v, m.b_h[2], m.b_v);
                L.op_r = gemm(NR, m.H, L.w_r, m.b_h[0], m.b_r);
                L.op_gate = gemm(NHD, m.H, L.w_gate, m.b_h[0], m.b_gate);
            }
            L.op_sink = gemm(m.NH, m.H, L.w_sink, m.b_h[0], m.b_sink);
            if (L.full) {
                L.op_dk = gemm(KVC, m.H, L.w_dk, m.b_h[1], m.b_dk);
                L.op_dv = gemm(KVC, m.H, L.w_dv, m.b_h[2], m.b_dv);
            }
            L.op_kconv = sconv(m.b_k, L.w_kconv, m.b_kcv, KVD, L.full ? m.KVH : 0, L.full ? m.b_dk : 0);
            L.op_vconv = sconv(m.b_v, L.w_vconv, m.b_vcv, KVD, L.full ? m.KVH : 0, L.full ? m.b_dv : 0);
            L.op_aconv = sconv(m.b_ao, L.w_aconv, m.b_ax, m.H, 0, 0);
            L.op_mconv = sconv(m.b_mo, L.w_mconv, m.b_zc, m.H, 0, 0);
            if (L.push_slot < 0) {
                const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                        RAD_INT("width", kWidth), RAD_INT("heads", 0),
                                        RAD_STR("dtype", "bf16") };
                L.op_aconv_add = try_op("sconv_res_add", ps, 5, &L.w_aconv, 1,
                                        {m.b_ao, m.b_s}, {m.b_s});
            }
            {
                const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                        RAD_INT("width", kWidth), RAD_INT("heads", 0),
                                        RAD_STR("dtype", "bf16") };
                L.op_mconv_add = try_op("sconv_res_add", ps, 5, &L.w_mconv, 1,
                                        {m.b_mo, m.b_s}, {m.b_s});
            }
            /* the clips: the position tau rides on full layers only */
            const rad_buf rsrc = L.r_full ? m.b_rf : m.b_r;
            if (L.full) {
                L.op_qclip = rw(b, RAD_OP(b, "l2_clip", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", NHD),
                                                              RAD_INT("group", m.HD), RAD_F64("threshold", m.qclip),
                                                              RAD_STR("dtype", "bf16"), RAD_F64("floor", m.floorv),
                                                              RAD_F64("alpha", m.alpha)),
                                                 RAD_WEIGHTS(L.w_qscale)), {m.b_q}, {m.b_qc});
                L.op_rclip = rw(b, RAD_OP(b, "l2_clip", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", NR),
                                                              RAD_INT("group", m.DR), RAD_F64("threshold", m.rclip),
                                                              RAD_STR("dtype", "bf16"), RAD_F64("floor", m.floorv),
                                                              RAD_F64("alpha", m.alpha)),
                                                 RAD_WEIGHTS(L.w_rscale)), {rsrc}, {m.b_rc});
                const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("nq", NHD), RAD_INT("gq", m.HD),
                                        RAD_F64("thrq", m.qclip), RAD_INT("nr", NR),
                                        RAD_INT("gr", m.DR), RAD_F64("thrr", m.rclip),
                                        RAD_STR("dtype", "bf16"), RAD_F64("floor", m.floorv),
                                        RAD_F64("alpha", m.alpha) };
                const rad_weight ws[] = { L.w_qscale, L.w_rscale };
                L.op_qr_clip = try_op("qr_l2_clip", ps, 10, ws, 2, {m.b_q, rsrc}, {m.b_qc, m.b_rc});
            } else {
                L.op_qclip = rw(b, RAD_OP(b, "l2_clip", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", NHD),
                                                              RAD_INT("group", m.HD), RAD_F64("threshold", m.qclip),
                                                              RAD_STR("dtype", "bf16")),
                                                 RAD_WEIGHTS(L.w_qscale)), {m.b_q}, {m.b_qc});
                L.op_rclip = rw(b, RAD_OP(b, "l2_clip", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", NR),
                                                              RAD_INT("group", m.DR), RAD_F64("threshold", m.rclip),
                                                              RAD_STR("dtype", "bf16")),
                                                 RAD_WEIGHTS(L.w_rscale)), {rsrc}, {m.b_rc});
                const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("nq", NHD), RAD_INT("gq", m.HD),
                                        RAD_F64("thrq", m.qclip), RAD_INT("nr", NR),
                                        RAD_INT("gr", m.DR), RAD_F64("thrr", m.rclip),
                                        RAD_STR("dtype", "bf16") };
                const rad_weight ws[] = { L.w_qscale, L.w_rscale };
                L.op_qr_clip = try_op("qr_l2_clip", ps, 8, ws, 2, {m.b_q, rsrc}, {m.b_qc, m.b_rc});
            }
            L.op_kclip = rw(b, RAD_OP(b, "l2_clip", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", KVD),
                                                          RAD_INT("group", m.HD), RAD_F64("threshold", m.qclip),
                                                          RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS),
                            {m.b_kcv}, {m.b_kc});
            L.op_vnorm = rms(T * m.KVH, m.HD, 0, m.b_vcv, m.b_vc);
            const char* kv_s = m.kv_dt == RAD_I8 ? "i8" : "bf16";
            L.op_attn = rw(b, RAD_OP(b, "attn_paged_rel", RAD_PARAMS(RAD_RANGE("q_len", 1, T), RAD_INT("head_dim", m.HD),
                                                                RAD_INT("gqa", m.NH / m.KVH),
                                                                RAD_INT("block_size", RAD_KV_BLOCK),
                                                                RAD_INT("causal", 1),
                                                                RAD_INT("window", L.full ? 0 : m.WIN),
                                                                RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", kv_s),
                                                                RAD_INT("max_ctx", g.max_ctx), RAD_INT("n_head", m.NH),
                                                                RAD_INT("max_seqs", g.max_seqs),
                                                                RAD_INT("d_rel", m.DR),
                                                                RAD_INT("extent", L.full ? m.EXT : m.WIN)),
                               RAD_WEIGHTS(L.w_rel)), {m.b_qc, m.b_rc, m.b_sink}, {m.b_att});
            /* after the attention: its block size is the attention kernel's constraint */
            const int64_t kv_bs = rad_kv_block_size(b, grp);
            L.op_store = rw(b, RAD_OP(b, "kv_store", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", m.HD),
                                                           RAD_INT("n_head_kv", m.KVH),
                                                           RAD_INT("block_size", kv_bs),
                                                           RAD_STR("kv_dtype", kv_s)), RAD_NOWEIGHTS),
                            {m.b_kc, m.b_vc}, {});
            {
                const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("head_dim", m.HD),
                                        RAD_INT("n_head_kv", m.KVH), RAD_INT("block_size", kv_bs),
                                        RAD_F64("threshold", m.qclip), RAD_F64("eps", g.eps),
                                        RAD_STR("kv_dtype", kv_s) };
                L.op_kv_store_fused = try_op("kv_clip_norm_store", ps, 7, nullptr, 0,
                                             {m.b_kcv, m.b_vcv}, {});
            }
            L.op_sig = ew("sigmoid", T, NHD, {m.b_gate}, m.b_sig);
            L.op_gated = ew("mul", T, NHD, {m.b_att, m.b_sig}, m.b_attg);
            {
                const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", NHD),
                                        RAD_STR("act", "sigmoid"), RAD_STR("dtype", "bf16") };
                L.op_act_mul = try_op("act_mul", ps, 4, nullptr, 0, {m.b_gate, m.b_att}, {m.b_attg});
            }
            if (L.lin_i8) {
                L.op_gq = rw(b, RAD_OP(b, "gate_had_quant_i8g",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", NHD),
                                                  RAD_INT("group", kExpGroup), RAD_STR("dtype", "bf16"),
                                                  RAD_STR("act", "sigmoid")),
                                       RAD_NOWEIGHTS), {m.b_gate, m.b_att}, {m.b_oq8, m.b_os8});
                L.op_o = decl_gemm_q(b, T, m.H, NHD, kExpGroup, "i8a8h", L.w_o, L.w_os,
                                     { m.b_oq8, m.b_os8 }, m.b_ao);
            } else {
                L.op_o = gemm(m.H, NHD, L.w_o, m.b_attg, m.b_ao);
            }
            L.op_ar_o = all_reduce(m.b_ao);
            /* the residual: a pushed block, or the stream plus the attention */
            if (L.push_slot >= 0) {
                L.op_push = ew("mul", T, m.H, {m.b_s, m.b_ones}, m.b_blk);
                L.op_resid = ew("mul", T, m.H, {m.b_ax, m.b_ones}, m.b_s);
                L.op_push_copy  = try_copy(m.b_s,  m.b_s,  m.b_blk);
                L.op_resid_copy = try_copy(m.b_ax, m.b_ax, m.b_s);
                all_copy = all_copy && L.op_push_copy && L.op_resid_copy;
            } else {
                L.op_attn_add = ew("add", T, m.H, {m.b_s, m.b_ax}, m.b_s);
            }
            /* the feed-forward */
            L.op_mlp_mix = mix(L.n_state_mlp, m.b_s, L.w_mlp_q, m.b_mx, m.b_blk);
            L.op_mlp_norm = rms(T, m.H, L.w_mlp_norm, m.b_mx, m.b_fm);
            L.op_mlp_norm_ctl = rms(T, m.H, L.w_mlp_norm, m.b_cym, m.b_fm);
            if (L.ex_form == kExpIqr) {
                auto try_norm_q8 = [&](rad_buf src) -> rad_op {
                    const rad_weight w[] = { L.w_mlp_norm };
                    const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                            RAD_F64("eps", g.eps), RAD_INT("group", kExpGroup),
                                            RAD_STR("dtype", "bf16") };
                    return try_op("rmsnorm_had_quant_i8g", ps, 5, w, 1, {src}, {m.b_q8, m.b_s8, m.b_fm});
                };
                L.op_mlp_norm_q8     = try_norm_q8(m.b_mx);
                L.op_mlp_norm_ctl_q8 = try_norm_q8(m.b_cym);
            }
            /* The routing is f32, as GlmRouter's: the card's router_sigmoid_topk reads f32 logits
             * only. A bf16 router resolves a host kernel alone, and the planner then keeps the
             * layer's whole weight group in the container's mapping, so the layer's ops run on the
             * host against buffers that live on the card. The weight is widened at load. */
            L.op_router = rw(b, RAD_OP(b, "gemm_nt_f32", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", m.E),
                                                                RAD_INT("K", m.H), RAD_STR("dtype", "bf16")),
                                       RAD_WEIGHTS(L.w_router)), {m.b_fm}, {m.b_rl});
            L.op_topk = rw(b, RAD_OP(b, "router_sigmoid_topk", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_expert", m.E),
                                                                     RAD_INT("top_k", m.K), RAD_INT("norm", 1),
                                                                     RAD_F64("route_scale", m.route_scale),
                                                                     RAD_STR("dtype", "f32")),
                                        RAD_WEIGHTS(L.w_bias)), {m.b_rl}, {m.b_ids, m.b_wts});
            L.op_scatter = rw(b, RAD_OP(b, "moe_scatter", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_expert", m.E),
                                                                RAD_INT("top_k", m.K)), RAD_NOWEIGHTS),
                              {m.b_ids}, {m.b_sorted, m.b_eoff, m.b_ecnt});
            {
                const int64_t T_rs = std::min<int64_t>(T, 1024 / m.K);
                if (T_rs >= 1) {
                    const RadParam ps[] = { RAD_RANGE("M", 1, T_rs), RAD_INT("n_expert", m.E),
                                            RAD_INT("top_k", m.K), RAD_INT("norm", 1),
                                            RAD_F64("route_scale", m.route_scale), RAD_STR("dtype", "f32") };
                    const rad_weight w[] = { L.w_bias };
                    L.op_router_scatter = try_op("router_sigmoid_topk_scatter", ps, 6, w, 1,
                                                 {m.b_rl}, {m.b_ids, m.b_wts, m.b_sorted, m.b_eoff, m.b_ecnt});
                }
            }
            if (L.ex_form == kExpBf16) {
                const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("N", 2 * L.fq[0]), RAD_INT("K", m.H),
                                        RAD_INT("n_expert", m.E), RAD_INT("top_k", m.K),
                                        RAD_STR("a_order", "token"), RAD_STR("dtype", "bf16") };
                L.op_egu = rw(b, rad_decl_op(b, "moe_gemm", ps, (int)(sizeof ps / sizeof ps[0]),
                                             L.w_egu.data(), (int)L.w_egu.size()),
                              {m.b_fm, m.b_sorted, m.b_eoff}, {m.b_gu});
                const RadParam pd[] = { RAD_RANGE("M", 1, T), RAD_INT("N", m.H), RAD_INT("K", L.fq[0]),
                                        RAD_INT("n_expert", m.E), RAD_INT("top_k", m.K),
                                        RAD_STR("a_order", "sorted"), RAD_STR("dtype", "bf16") };
                L.op_edn = rw(b, rad_decl_op(b, "moe_gemm", pd, (int)(sizeof pd / sizeof pd[0]),
                                             L.w_edn.data(), (int)L.w_edn.size()),
                              {m.b_act, m.b_sorted, m.b_eoff}, {m.b_dn});
                L.op_situ = rw(b, RAD_OP(b, "situ_mul", RAD_PARAMS(RAD_RANGE("M", 1, T * m.K), RAD_INT("n", m.FW),
                                                              RAD_STR("dtype", "bf16"), RAD_F64("gate_cap", m.gcap),
                                                              RAD_F64("up_cap", m.ucap)), RAD_NOWEIGHTS),
                               {m.b_gu}, {m.b_act});
            } else {
                /* An IQR weight is read at `i8a8h` against the rotated int8 activation; the layout hook
                 * resolves moe_gemm_iqr from its encoding. */
                const char* ex_dt = "i8a8h";
                L.op_q8 = decl_quant_act(b, "had_quant_act_i8g", T, m.H, kExpGroup, m.b_fm, m.b_q8, m.b_s8);
                /* moe_gemm_q's table: the even experts' codes, their scales, then the odd experts' */
                auto parity_table = [&](const std::vector<rad_weight>& wc, const std::vector<rad_weight>& ws) {
                    std::vector<rad_weight> tab;
                    for (int q = 0; q < 2; ++q) {
                        for (int64_t e = q; e < m.E; e += 2) tab.push_back(wc[(size_t)e]);
                        for (int64_t e = q; e < m.E; e += 2) tab.push_back(ws[(size_t)e]);
                    }
                    return tab;
                };
                /* Split over ranks, each class has its own width (N_odd, K_odd), and gate_up is two
                 * parts of the wider class's: a narrower class fills each part's leading columns and
                 * zeros the rest, which its down projection (K_odd) never reads. One rank states
                 * neither: both its classes are F wide. */
                const bool tp = g.world > 1;
                {
                    const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("N", 2 * L.fq[0]), RAD_INT("K", m.H),
                                            RAD_INT("n_expert", m.E), RAD_INT("top_k", m.K),
                                            RAD_INT("group", kExpGroup), RAD_STR("a_order", "token"),
                                            RAD_STR("dtype", ex_dt), RAD_INT("a_group", kExpGroup),
                                            RAD_INT("N_odd", 2 * L.fq[1]), RAD_INT("K_odd", m.H),
                                            RAD_INT("parts", 2) };
                    const std::vector<rad_weight> tab = parity_table(L.w_egu, L.w_egus);
                    L.op_egu = rw(b, rad_decl_op(b, "moe_gemm_q", ps,
                                                 (int)(sizeof ps / sizeof ps[0]) - (tp ? 0 : 3),
                                                 tab.data(), (int)tab.size()),
                                  {m.b_q8, m.b_s8, m.b_sorted, m.b_eoff}, {m.b_gu});
                }
                /* The SiTU product in the down projection's activation form, had_quant_act_i8g's: a
                 * rotation per 128 with nothing folded and a scale per (row, 128). gated_had_quant_i8
                 * is the per-row form (one scale a row carrying 1/sqrt(128), codes in the step order),
                 * and the fused per-group form, gated_had_quant_i8g, is a card op without SiTU, so the
                 * product is situ_mul's bf16 and then that quantiser, the pair gated_had_quant_i8g is
                 * defined as. */
                L.op_situ = rw(b, RAD_OP(b, "situ_mul",
                                         RAD_PARAMS(RAD_RANGE("M", 1, T * m.K), RAD_INT("n", m.FW),
                                                    RAD_STR("dtype", "bf16"), RAD_F64("gate_cap", m.gcap),
                                                    RAD_F64("up_cap", m.ucap)), RAD_NOWEIGHTS),
                               {m.b_gu}, {m.b_act});
                L.op_dq = rw(b, RAD_OP(b, "had_quant_act_i8g",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T * m.K), RAD_INT("n", m.FW),
                                                  RAD_INT("group", kExpGroup), RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS),
                             {m.b_act}, {m.b_aq, m.b_as});
                {
                    const RadParam pd[] = { RAD_RANGE("M", 1, T), RAD_INT("N", m.H), RAD_INT("K", L.fq[0]),
                                            RAD_INT("n_expert", m.E), RAD_INT("top_k", m.K),
                                            RAD_INT("group", kExpGroup), RAD_STR("a_order", "sorted"),
                                            RAD_STR("dtype", ex_dt), RAD_INT("a_group", kExpGroup),
                                            RAD_INT("N_odd", m.H), RAD_INT("K_odd", L.fq[1]) };
                    const std::vector<rad_weight> tab = parity_table(L.w_edn, L.w_edns);
                    L.op_edn = rw(b, rad_decl_op(b, "moe_gemm_q", pd,
                                                 (int)(sizeof pd / sizeof pd[0]) - (tp ? 0 : 2),
                                                 tab.data(), (int)tab.size()),
                                  {m.b_aq, m.b_as, m.b_sorted, m.b_eoff}, {m.b_dn});
                }
            }
            L.op_gather = rw(b, RAD_OP(b, "moe_gather", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                              RAD_INT("top_k", m.K), RAD_STR("dtype", "bf16"),
                                                              RAD_STR("act", "none")), RAD_NOWEIGHTS),
                             {m.b_dn, m.b_wts, m.b_sorted}, {m.b_mo});
            L.op_ar_m = all_reduce(m.b_mo);
            if (g.world > 1 && (!L.op_ar_o || !L.op_ar_m)) return RAD_E_UNSUPPORTED;
            L.op_mlp_add = ew("add", T, m.H, {m.b_s, m.b_zc}, m.b_s);
            /* the controllers: (1 - tanh(F g)) x + tanh(G g) msg, g = D [rms x | rms msg] */
            if (L.core) {
                {
                    const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                            RAD_INT("n_state", 1), RAD_F64("eps", g.eps),
                                            RAD_STR("dtype", "bf16") };
                    const rad_weight ws4[] = { L.w_msg[0], L.w_msg[1], L.w_msg[2], L.w_msg[3] };
                    L.op_msg4 = try_op("attn_res_mix4", ps, 5, ws4, 4, {m.b_bank0, m.b_bank1},
                                       {m.b_msg[0], m.b_msg[1], m.b_msg[2], m.b_msg[3]});
                }
                for (int j = 0; j < 4; ++j) {
                    /* the input a controller reads: the mix of its query (a local layer has one) and
                     * the message of its own pass bank; the mlp's input is the feed-forward mix */
                    const rad_buf x = j < 3 ? m.b_y[L.full ? j : 0] : m.b_mx;
                    const rad_buf y = j < 3 ? m.b_cy[j] : m.b_cym;
                    const rad_buf msg = m.b_msg[j];
                    L.op_msg[j] = mix(1, m.b_bank1, L.w_msg[j], msg, m.b_bank0);
                    rad_op* o = L.ctl[j];
                    /* g = D [rms x | rms msg]: both norms land in one [T, 2H] buffer, so the down
                     * projection is one GEMM over the checkpoint's own down_weight */
                    o[0] = rms(T, m.H, 0, x, m.b_gxm);
                    o[1] = rms(T, m.H, 0, msg, m.b_gxm);
                    o[2] = gemm(m.RANK, 2 * m.H, L.w_dx[j], m.b_gxm, m.b_g1);
                    o[3] = gemm(m.H, m.RANK, L.w_cgain[j], m.b_g1, m.b_gp);
                    o[4] = rw(b, RAD_OP(b, "logit_softcap", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                                  RAD_F64("cap", 1.0), RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS), {m.b_gp}, {m.b_gt});
                    o[5] = gemm(m.H, m.RANK, L.w_cforget[j], m.b_g1, m.b_fp);
                    o[6] = rw(b, RAD_OP(b, "logit_softcap", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                                  RAD_F64("cap", 1.0), RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS), {m.b_fp}, {m.b_ft});
                    o[7] = ew("mul", T, m.H, {m.b_ft, m.b_minus1}, m.b_nft);
                    o[8] = ew("mul", T, m.H, {m.b_nft, x}, m.b_t1);
                    o[9] = ew("add", T, m.H, {m.b_t1, x}, m.b_t2);
                    o[10] = ew("mul", T, m.H, {m.b_gt, msg}, m.b_t3);
                    o[11] = ew("add", T, m.H, {m.b_t2, m.b_t3}, y);
                    {
                        const int64_t T_ctl = std::min<int64_t>(T, 16);
                        const RadParam ps[] = { RAD_RANGE("M", 1, T_ctl), RAD_INT("n", m.H),
                                                RAD_INT("rank", m.RANK), RAD_F64("eps", g.eps),
                                                RAD_STR("dtype", "bf16") };
                        const rad_weight ws[] = { L.w_dx[j], L.w_cgain[j], L.w_cforget[j] };
                        L.op_ctl_fused[j] = try_op("loop_ctl", ps, 5, ws, 3, {x, msg}, {y});
                    }
                    {
                        const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                RAD_STR("dtype", "bf16") };
                        L.op_ctl_blend[j] = try_op("loop_ctl_blend", ps, 3, nullptr, 0,
                                                   {x, msg, m.b_gp, m.b_fp}, {y});
                    }
                    if (!L.op_ctl_fused[j]) all_fused = false;
                    if (!L.op_ctl_blend[j]) all_blend = false;
                }
            }
        }
        m.use_plane_copy = all_copy;
        m.all_ctl_fused  = all_fused;
        m.all_ctl_blend  = all_blend;
    }
    /* ---- calibration (CALIBRATION): after every layer op, so the buffers its taps read stay live
     * to the end of the layers in the plan */
    RAD_ARCH_TRY(calib_declare(b, m, ctx));
    /* ---- media: the vision tower, after the trunk so its activations share the arena with every
     * buffer above; its groups follow the 24 physical layers. */
    RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, m.H, kLayers));
    if (m.vit.on) {
        m.op_media_rows = rw(b, RAD_OP(b, "scatter_rows",
                                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", m.H),
                                                  RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS),
                             {m.b_s}, {m.b_s});
        if (!m.op_media_rows) return RAD_E_INVAL;
    }

    /* The drafter last: its passes run after the whole trunk, and the planner lays lifetimes out in
     * declaration order (its tap buffer is live for the whole step, as the trunk's own). */
    if (dflash2_present(meta) && ctx->max_spec > 0) {
        RAD_ARCH_TRY(m.df2.declare(b, nm, meta, ctx, g, kApps, kApps, m.b_s, m.b_yn, m.w_embed,
                                   m.w_embed_norm, m.w_head, m.VR));
    }
    return RAD_OK;
}


/* A controller on one input: the twelve ops of Layer::ctl, in order. */
static void run_control(RadCtx* c, const Model& m, const Layer& L, int j, rad_buf x, rad_buf msg,
                        rad_buf y, int64_t T) {
    if (T <= 16 && L.op_ctl_fused[j]) {
        RAD_ISSUE_N(c, L.op_ctl_fused[j], T, brows(x, T), brows(msg, T),
                    RAD_W(L.w_dx[j]), RAD_W(L.w_cgain[j]), RAD_W(L.w_cforget[j]), brows(y, T));
        return;
    }
    const int64_t MT = T;   /* the step's rows: the mixes' outputs are [max_tok, H], trimmed */
    const rad_op* o = L.ctl[j];
    /* the two norm halves: rms x in columns [0, H), rms msg in [H, 2H) */
    RAD_ISSUE_N(c, o[0], MT, brows(x, MT), RAD_NONE, bcol(m.b_gxm, 0, m.H, MT));
    RAD_ISSUE_N(c, o[1], MT, brows(msg, MT), RAD_NONE, bcol(m.b_gxm, m.H, m.H, MT));
    RAD_ISSUE_N(c, o[2], MT, brows(m.b_gxm, MT), RAD_W(L.w_dx[j]), brows(m.b_g1, MT));
    RAD_ISSUE_N(c, o[3], MT, brows(m.b_g1, MT), RAD_W(L.w_cgain[j]), brows(m.b_gp, MT));
    if (L.op_ctl_blend[j]) {
        RAD_ISSUE_N(c, o[5], MT, brows(m.b_g1, MT), RAD_W(L.w_cforget[j]), brows(m.b_fp, MT));
        RAD_ISSUE_N(c, L.op_ctl_blend[j], MT, brows(x, MT), brows(msg, MT),
                    brows(m.b_gp, MT), brows(m.b_fp, MT), brows(y, MT));
        return;
    }
    RAD_ISSUE_N(c, o[4], MT, brows(m.b_gp, MT), brows(m.b_gt, MT));
    RAD_ISSUE_N(c, o[5], MT, brows(m.b_g1, MT), RAD_W(L.w_cforget[j]), brows(m.b_fp, MT));
    RAD_ISSUE_N(c, o[6], MT, brows(m.b_fp, MT), brows(m.b_ft, MT));
    RAD_ISSUE_N(c, o[7], MT, brows(m.b_ft, MT), brows(m.b_minus1, MT), brows(m.b_nft, MT));
    RAD_ISSUE_N(c, o[8], MT, brows(m.b_nft, MT), brows(x, MT), brows(m.b_t1, MT));
    RAD_ISSUE_N(c, o[9], MT, brows(m.b_t1, MT), brows(x, MT), brows(m.b_t2, MT));
    RAD_ISSUE_N(c, o[10], MT, brows(m.b_gt, MT), brows(msg, MT), brows(m.b_t3, MT));
    RAD_ISSUE_N(c, o[11], MT, brows(m.b_t2, MT), brows(m.b_t3, MT), brows(y, MT));
}

/* A short convolution over one stream, issued as LFM2's is: the decode rows [0, DT) with
 * num_accepted, then the prefill rows without it. The prefill's per-sequence arrays start at +D and
 * its token operands stay at their base, since the kernel reads rows at the absolute offsets in cu. */
static void issue_sconv(RadCtx* c, const RadBatch* batch, rad_op op, rad_weight w,
                        rad_kvgroup cg, int a, rad_buf x, rad_buf dyn, rad_buf y) {
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const RadKVGroupBatch* cv = kv_batch(batch, cg);
    const int32_t* idx = cv ? cv->state_index : nullptr;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t P = S - D;
    if (D > 0)
        RAD_ISSUE_N(c, op, DT, brows(x, DT), RAD_W(w), kv_cache(cg, a),
                    praw(batch->cu_seqlens, RAD_I32, D + 1), dyn ? brows(dyn, DT) : RAD_NONE,
                    praw(idx, RAD_I32, D), praw(batch->ctx_lens, RAD_I32, D),
                    praw(batch->num_accepted, RAD_I32, D), brows(y, DT));
    if (P > 0)
        RAD_ISSUE_N(c, op, T, brows(x, T), RAD_W(w), kv_cache(cg, a),
                    praw(batch->cu_seqlens + D, RAD_I32, P + 1), dyn ? brows(dyn, T) : RAD_NONE,
                    praw(idx ? idx + D : nullptr, RAD_I32, P), praw(batch->ctx_lens + D, RAD_I32, P),
                    RAD_NONE, brows(y, T));
}

static void issue_sconv_add(RadCtx* c, const RadBatch* batch, rad_op op, rad_weight w,
                            rad_kvgroup cg, int a, rad_buf x, rad_buf add_resid,
                            rad_buf y) {
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const RadKVGroupBatch* cv = kv_batch(batch, cg);
    const int32_t* idx = cv ? cv->state_index : nullptr;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t P = S - D;
    if (D > 0)
        RAD_ISSUE_N(c, op, DT, brows(x, DT), RAD_W(w), kv_cache(cg, a),
                    praw(batch->cu_seqlens, RAD_I32, D + 1), RAD_NONE,
                    praw(idx, RAD_I32, D), praw(batch->ctx_lens, RAD_I32, D),
                    praw(batch->num_accepted, RAD_I32, D), brows(y, DT), brows(add_resid, DT));
    if (P > 0)
        RAD_ISSUE_N(c, op, T, brows(x, T), RAD_W(w), kv_cache(cg, a),
                    praw(batch->cu_seqlens + D, RAD_I32, P + 1), RAD_NONE,
                    praw(idx ? idx + D : nullptr, RAD_I32, P), praw(batch->ctx_lens + D, RAD_I32, P),
                    RAD_NONE, brows(y, T), brows(add_resid, T));
}

/* ============================== the calibration taps (CALIBRATION) */

/* The owner's pooled accumulators, zeroed before their first sum (arena memory). */
static void calib_zero(RadCtx* c, const Model& m, const Layer& L) {
    if (L.c_zeroed) return;
    L.c_zeroed = true;
    for (rad_buf h : { L.c_gu, L.c_at[0], L.c_at[1], L.c_at[2] })
        rad_memset_async(rad_buf_ptr(c, h), 0, m.H * m.H * 4, rad_stream(c));
}

/* The attention linears' inputs, after the projections that read them. */
static void calib_attn(RadCtx* c, const Model& m, const Layer& L, const rad_buf hin[3], int64_t T) {
    if (!L.op_c_gu) return;
    calib_zero(c, m, L);
    for (int j = 0; j < 3; ++j)
        RAD_ISSUE_N(c, L.op_c_at[j], T, brows(hin[j], T), RAD_NONE, brows(L.c_at[j], m.H));
    L.c_rows_at += T;
}

/* `full` is the scratch's `ne` row-major Fg x Fg Grams; expert e's columns of its own order are
 * f2g[e % 2]'s, added into its packed upper triangle. An expert with no rows this pass is
 * skipped. */
static void calib_add(CalibExperts& ce, const float* full, int64_t Fg,
                      const std::vector<int32_t>* f2g, const std::vector<int32_t>& off) {
    const int64_t k = ce.k, tri = k * (k + 1) / 2;
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for (int64_t e = t; e < ce.ne; e += nt) {
                if (off[(size_t)e + 1] <= off[(size_t)e]) continue;
                const int32_t* map = f2g[e & 1].data();
                const float* S = full + (size_t)(e * Fg * Fg);
                float* d = ce.acc.data() + (size_t)(e * tri);
                for (int64_t a = 0; a < k; ++a) {
                    const float* row = S + (size_t)map[a] * (size_t)Fg;
                    for (int64_t x = a; x < k; ++x) *d++ += row[map[x]];
                }
            }
        });
    for (std::thread& x : th) x.join();
}

/* The experts' inputs, after the down projection: the gate_up input's pooled sum, and each expert's
 * run of the sorted SiTU rows into its square of the scratch, which comes back to the host. The
 * gather is a collective, so every rank issues it whatever it records. */
static void calib_ffn(RadCtx* c, const Model& m, const Layer& L, int ip, int64_t T) {
    const Model::Calib& cl = m.cal;
    if (!cl.on || ip < cl.lo || ip >= cl.hi) return;
    const int64_t R = T * m.K;
    if (cl.op_gather)
        RAD_ISSUE_N(c, cl.op_gather, R * m.FW, brows(m.b_act, R), brows(cl.gathered, R));
    if (!L.op_c_gu || cl.failed) return;
    calib_zero(c, m, L);
    RAD_ISSUE_N(c, L.op_c_gu, T, brows(m.b_fm, T), RAD_NONE, brows(L.c_gu, m.H));
    L.c_rows_gu += T;

    const RadStream s = rad_stream(c);
    const int64_t E = m.E, Fg = cl.Fg;
    std::vector<int32_t> off((size_t)(E + 1));
    rad_memcpy_async(off.data(), rad_buf_ptr(c, m.b_eoff), (E + 1) * 4, s);
    rad_stream_sync(s);
    if (off[0] != 0 || off[(size_t)E] < 0 || off[(size_t)E] > R) {
        fprintf(stderr, "radiance: inkling2: layer %d's expert offsets end at %d of %lld sorted "
                        "rows; the calibration stops\n", ip, off[(size_t)E], (long long)R);
        cl.failed = true;
        return;
    }
    const int64_t bytes = E * Fg * Fg * 4;
    float* staging = calib_staging(c, bytes);
    if (!staging) {
        cl.failed = true;
        return;
    }
    void* scratch = rad_buf_ptr(c, cl.scratch);
    rad_memset_async(scratch, 0, bytes, s);
    const rad_buf src = cl.gathered ? cl.gathered : m.b_act;
    if (L.c_dn.ne == 0) L.c_dn.init(E, m.F);
    for (int64_t e = 0; e < E; ++e) {
        const int64_t r0 = off[(size_t)e], n = off[(size_t)e + 1] - r0;
        if (n <= 0) continue;
        RAD_ISSUE_N(c, cl.op_dn, n, brow_slice(src, r0, n, Fg), RAD_NONE,
                    brow_slice(cl.scratch, e * Fg, Fg, Fg));
        L.c_dn.rows[(size_t)e] += n;
    }
    rad_memcpy_async(staging, scratch, bytes, s);
    rad_stream_sync(s);
    calib_add(L.c_dn, staging, Fg, cl.f2g[L.ex_form == kExpBf16 ? 1 : 0], off);
}

/* After a trunk step: once the corpus is through, every recorded layer this rank owns to its
 * files. */
static void calib_after_step(RadCtx* c, const Model& m, int64_t T) {
    const Model::Calib& cl = m.cal;
    if (!cl.on || cl.written) return;
    cl.seen += T;
    if (cl.seen < cl.target) return;
    cl.written = true;
    int n = 0, bad = cl.failed ? 1 : 0;
    std::vector<float> h((size_t)(m.H * m.H));
    auto put = [&](const std::vector<float>& x, int64_t k, long long rows, const char* what) {
        ++n;
        if (!calib_write_host_to(cl.dir, x.data(), k, rows, what, 0, rad_rank(c))) ++bad;
    };
    for (int64_t l = cl.lo; l < cl.hi; ++l) {
        const Layer& L = m.layers[(size_t)l];
        if (!L.op_c_gu || !L.c_zeroed) continue;
        char what[96];
        auto back = [&](rad_buf b) {
            rad_memcpy_async(h.data(), rad_buf_ptr(c, b), m.H * m.H * 4, rad_stream(c));
            rad_stream_sync(rad_stream(c));
        };
        back(L.c_gu);
        std::snprintf(what, sizeof what, "blk.%lld.gu", (long long)l);
        put(h, m.H, L.c_rows_gu, what);
        static const char* const at[3][3] = { { "wq", "wr", "gate" }, { "wk", nullptr, nullptr },
                                              { "wv", nullptr, nullptr } };
        for (int j = 0; j < 3; ++j) {
            back(L.c_at[j]);
            for (const char* w : at[j]) {
                if (!w) continue;
                std::snprintf(what, sizeof what, "blk.%lld.%s", (long long)l, w);
                put(h, m.H, L.c_rows_at, what);
            }
        }
        if (L.c_dn.ne == 0) continue;
        const int64_t F = m.F;
        std::vector<float> one((size_t)(F * F)), sum((size_t)(F * F), 0.0f);
        long long total = 0;
        for (int64_t e = 0; e < L.c_dn.ne; ++e) {
            L.c_dn.unpack(e, one.data());
            for (size_t i = 0; i < one.size(); ++i) sum[i] += one[i];
            total += L.c_dn.rows[(size_t)e];
            std::snprintf(what, sizeof what, "blk.%lld.e%lld.dn", (long long)l, (long long)e);
            put(one, F, L.c_dn.rows[(size_t)e], what);
        }
        std::snprintf(what, sizeof what, "blk.%lld.dn", (long long)l);
        put(sum, F, total, what);
        L.c_dn.acc = {};
        L.c_dn.rows = {};
    }
    fprintf(stderr, "radiance: inkling2: calibration: rank %d wrote %d Hessian(s) of layers "
                    "%lld:%lld after %lld rows to %s%s\n", rad_rank(c), n, (long long)cl.lo,
            (long long)cl.hi, cl.seen, cl.dir, bad ? ", with errors" : "");
    if (cl.seen != cl.target)
        fprintf(stderr, "radiance: inkling2: calibration: %lld rows seen and the corpus has %lld\n",
                cl.seen, cl.target);
}

/* One layer application: application a runs physical layer phys_of(a), its state is a's, and a core
 * application in a later pass takes the controllers. */
static void run_layer(RadCtx* c, const Model& m, const RadBatch* batch, int a, bool control) {
    const int ip = phys_of(a);
    const Layer& L = m.layers[(size_t)ip];
    const int64_t T = batch->n_tok;
    const int64_t MT = m.g.max_tok;
    const rad_kvgroup grp = L.full ? m.kv_full : m.kv_local;
    const RadKVGroupBatch* kvb = kv_batch(batch, grp);
    const int64_t n_seq = batch->n_seq;

    /* the attention inputs: the mixes, then the norms, with the controllers in between on a core
     * application of a later pass */
    rad_buf hin[3] = { m.b_h[0], m.b_h[1], m.b_h[2] };
    bool hq_done[3] = { false, false, false };
    if (ip == 0) {
        if (L.lin_i8 && L.op_norm_hq[0]) {
            RAD_ISSUE_N(c, L.op_norm_hq[0], T, brows(m.b_s, T), RAD_NONE, RAD_W(L.w_norm[0]),
                        brows(m.b_hq8[0], T), brows(m.b_hs8[0], T), brows(m.b_h[0], T));
            hq_done[0] = true;
        } else {
            RAD_ISSUE_N(c, L.op_norm[0], T, brows(m.b_s, T), RAD_W(L.w_norm[0]), brows(m.b_h[0], T));
        }
        hin[1] = hin[2] = m.b_h[0];
    } else if (!control) {
        if (L.full && L.op_mix3_norm) {
            RAD_ISSUE_N(c, L.op_mix3_norm, T, RAD_B_N(m.b_blk, 0, L.n_state_attn),
                        brows(m.b_s, T), RAD_W(L.w_mix[0]), RAD_W(L.w_mix[1]), RAD_W(L.w_mix[2]),
                        brows(m.b_h[0], T), brows(m.b_h[1], T), brows(m.b_h[2], T),
                        RAD_W(L.w_norm[0]), RAD_W(L.w_norm[1]), RAD_W(L.w_norm[2]));
        } else if (!L.full && L.op_mix_norm) {
            RAD_ISSUE_N(c, L.op_mix_norm, T, RAD_B_N(m.b_blk, 0, L.n_state_attn),
                        brows(m.b_s, T), RAD_NONE, RAD_W(L.w_mix[0]), RAD_W(L.w_norm[0]),
                        brows(m.b_h[0], T));
            hin[1] = hin[2] = m.b_h[0];
        } else {
            for (int q = 0; q < L.n_mix; ++q)
                RAD_ISSUE_N(c, L.op_mix[q], T, RAD_B_N(m.b_blk, 0, L.n_state_attn),
                            brows(m.b_s, T), RAD_NONE, RAD_W(L.w_mix[q]), RAD_NONE,
                            brows(m.b_y[q], T));
            if (L.full) {
                for (int j = 0; j < 3; ++j)
                    RAD_ISSUE_N(c, L.op_norm[j], T, brows(m.b_y[j], T), RAD_W(L.w_norm[j]),
                                brows(m.b_h[j], T));
            } else {
                RAD_ISSUE_N(c, L.op_norm[0], T, brows(m.b_y[0], T), RAD_W(L.w_norm[0]),
                            brows(m.b_h[0], T));
                hin[1] = hin[2] = m.b_h[0];
            }
        }
    } else {
        if (L.full && L.op_mix3) {
            RAD_ISSUE_N(c, L.op_mix3, T, RAD_B_N(m.b_blk, 0, L.n_state_attn), brows(m.b_s, T),
                        RAD_W(L.w_mix[0]), RAD_W(L.w_mix[1]), RAD_W(L.w_mix[2]),
                        brows(m.b_y[0], T), brows(m.b_y[1], T), brows(m.b_y[2], T));
        } else {
            for (int q = 0; q < L.n_mix; ++q)
                RAD_ISSUE_N(c, L.op_mix[q], T, RAD_B_N(m.b_blk, 0, L.n_state_attn),
                            brows(m.b_s, T), RAD_NONE, RAD_W(L.w_mix[q]), RAD_NONE,
                            brows(m.b_y[q], T));
        }
        if (L.op_msg4) {
            RAD_ISSUE_N(c, L.op_msg4, T, RAD_B_N(m.b_bank0, 0, 1), brows(m.b_bank1, T),
                        RAD_W(L.w_msg[0]), RAD_W(L.w_msg[1]), RAD_W(L.w_msg[2]), RAD_W(L.w_msg[3]),
                        brows(m.b_msg[0], T), brows(m.b_msg[1], T), brows(m.b_msg[2], T),
                        brows(m.b_msg[3], T));
        } else {
            for (int j = 0; j < 4; ++j)
                RAD_ISSUE_N(c, L.op_msg[j], T, RAD_B_N(m.b_bank0, 0, 1), brows(m.b_bank1, T),
                            RAD_NONE, RAD_W(L.w_msg[j]), RAD_NONE, brows(m.b_msg[j], T));
        }
        for (int j = 0; j < 3; ++j) {
            const rad_buf x = m.b_y[L.full ? j : 0];
            run_control(c, m, L, j, x, m.b_msg[j], m.b_cy[j], T);
            if (L.lin_i8 && L.op_norm_ctl_hq[j]) {
                RAD_ISSUE_N(c, L.op_norm_ctl_hq[j], T, brows(m.b_cy[j], T), RAD_NONE,
                            RAD_W(L.w_norm[L.full ? j : 0]), brows(m.b_hq8[j], T),
                            brows(m.b_hs8[j], T), brows(m.b_h[j], T));
                hq_done[j] = true;
            } else {
                RAD_ISSUE_N(c, L.op_norm_ctl[j], T, brows(m.b_cy[j], T),
                            RAD_W(L.w_norm[L.full ? j : 0]), brows(m.b_h[j], T));
            }
        }
    }

    /* the projections and the dynamic taps */
    if (L.lin_i8) {
        /* each distinct input's rotated codes once: a local layer outside the controllers reads
         * one norm for all three */
        int hj[3] = { 0, 0, 0 };
        for (int j = 0; j < 3; ++j) {
            hj[j] = hin[j] == m.b_h[1] ? 1 : hin[j] == m.b_h[2] ? 2 : 0;
            bool seen = false;
            for (int i = 0; i < j; ++i) seen = seen || hj[i] == hj[j];
            if (!seen && !hq_done[hj[j]])
                RAD_ISSUE_N(c, L.op_hq[hj[j]], T, brows(m.b_h[hj[j]], T), brows(m.b_hq8[hj[j]], T),
                            brows(m.b_hs8[hj[j]], T));
        }
        auto lin = [&](rad_op op, int j, rad_weight w, rad_weight ws, rad_buf y) {
            RAD_ISSUE_N(c, op, T, brows(m.b_hq8[hj[j]], T), brows(m.b_hs8[hj[j]], T), RAD_W(w), RAD_W(ws),
                        brows(y, T));
        };
        if (T <= 16 && L.op_proj) {
            RAD_ISSUE_N(c, L.op_proj, T,
                        brows(m.b_hq8[hj[0]], T), brows(m.b_hs8[hj[0]], T),
                        brows(m.b_hq8[hj[1]], T), brows(m.b_hs8[hj[1]], T),
                        brows(m.b_hq8[hj[2]], T), brows(m.b_hs8[hj[2]], T),
                        RAD_W(L.w_q), RAD_W(L.w_qs), RAD_W(L.w_k), RAD_W(L.w_ks),
                        RAD_W(L.w_v), RAD_W(L.w_vs), RAD_W(L.w_r), RAD_W(L.w_rs),
                        RAD_W(L.w_gate), RAD_W(L.w_gates),
                        brows(m.b_q, T), brows(m.b_k, T), brows(m.b_v, T),
                        brows(L.r_full ? m.b_rf : m.b_r, T), brows(m.b_gate, T));
        } else {
            lin(L.op_q, 0, L.w_q, L.w_qs, m.b_q);
            lin(L.op_k, 1, L.w_k, L.w_ks, m.b_k);
            lin(L.op_v, 2, L.w_v, L.w_vs, m.b_v);
            lin(L.op_r, 0, L.w_r, L.w_rs, L.r_full ? m.b_rf : m.b_r);
            lin(L.op_gate, 0, L.w_gate, L.w_gates, m.b_gate);
        }
    } else {
        RAD_ISSUE_N(c, L.op_q, T, brows(hin[0], T), RAD_W(L.w_q), brows(m.b_q, T));
        RAD_ISSUE_N(c, L.op_k, T, brows(hin[1], T), RAD_W(L.w_k), brows(m.b_k, T));
        RAD_ISSUE_N(c, L.op_v, T, brows(hin[2], T), RAD_W(L.w_v), brows(m.b_v, T));
        RAD_ISSUE_N(c, L.op_r, T, brows(hin[0], T), RAD_W(L.w_r), brows(m.b_r, T));
        RAD_ISSUE_N(c, L.op_gate, T, brows(hin[0], T), RAD_W(L.w_gate), brows(m.b_gate, T));
    }
    RAD_ISSUE_N(c, L.op_sink, T, brows(hin[0], T), RAD_W(L.w_sink), brows(m.b_sink, T));
    if (L.full) {
        RAD_ISSUE_N(c, L.op_dk, T, brows(hin[1], T), RAD_W(L.w_dk), brows(m.b_dk, T));
        RAD_ISSUE_N(c, L.op_dv, T, brows(hin[2], T), RAD_W(L.w_dv), brows(m.b_dv, T));
    }
    calib_attn(c, m, L, hin, T);
    issue_sconv(c, batch, L.op_kconv, L.w_kconv, m.cv_k, a, m.b_k, L.full ? m.b_dk : 0, m.b_kcv);
    issue_sconv(c, batch, L.op_vconv, L.w_vconv, m.cv_v, a, m.b_v, L.full ? m.b_dv : 0, m.b_vcv);

    /* the clips; the position tau on full layers */
    const RadOperand pos = L.full ? praw(batch->positions, RAD_I32, T) : RAD_NONE;
    /* this rank's heads' r: its columns of every head's when wr is held whole */
    const int64_t NR = m.NH * m.DR;
    const RadOperand rin = L.r_full ? bcol(m.b_rf, m.g.rank * NR, NR, T) : brows(m.b_r, T);
    if (L.op_qr_clip) {
        RAD_ISSUE_N(c, L.op_qr_clip, T, brows(m.b_q, T), RAD_W(L.w_qscale), rin,
                    RAD_W(L.w_rscale), brows(m.b_qc, T), brows(m.b_rc, T), pos);
    } else {
        RAD_ISSUE_N(c, L.op_qclip, T, brows(m.b_q, T), RAD_W(L.w_qscale), pos, brows(m.b_qc, T));
        RAD_ISSUE_N(c, L.op_rclip, T, rin, RAD_W(L.w_rscale), pos, brows(m.b_rc, T));
    }

    /* the cache, then the attention, the output gate and the output projection */
    if (L.op_kv_store_fused) {
        RAD_ISSUE_N(c, L.op_kv_store_fused, T, brows(m.b_kcv, T), brows(m.b_vcv, T),
                    praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T), kv_cache(grp, a));
    } else {
        RAD_ISSUE_N(c, L.op_kclip, T, brows(m.b_kcv, T), RAD_NONE, RAD_NONE, brows(m.b_kc, T));
        RAD_ISSUE_N(c, L.op_vnorm, T * m.KVH, brows(m.b_vcv, T), RAD_NONE, brows(m.b_vc, T));
        RAD_ISSUE(c, L.op_store, brows(m.b_kc, T), brows(m.b_vc, T),
                  praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T), kv_cache(grp, a));
    }
    RAD_ISSUE_N(c, L.op_attn, batch->max_q_len, brows(m.b_qc, T), kv_cache(grp, a),
                praw2(kvb ? kvb->block_table : nullptr, RAD_I32, n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(kvb ? kvb->seqused : nullptr, RAD_I32, n_seq), RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, n_seq + 1), brows(m.b_rc, T), RAD_W(L.w_rel),
                brows(m.b_sink, T), brows(m.b_att, T));
    if (L.lin_i8) {
        RAD_ISSUE_N(c, L.op_gq, T, brows(m.b_gate, T), brows(m.b_att, T), brows(m.b_oq8, T),
                    brows(m.b_os8, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_o, T, brows(m.b_oq8, T), brows(m.b_os8, T), RAD_W(L.w_o), RAD_W(L.w_os),
                    brows(m.b_ao, T));
    } else {
        if (L.op_act_mul) {
            RAD_ISSUE_N(c, L.op_act_mul, T, brows(m.b_gate, T), brows(m.b_att, T), brows(m.b_attg, T));
        } else {
            RAD_ISSUE(c, L.op_sig, brows(m.b_gate, T), brows(m.b_sig, T));
            RAD_ISSUE(c, L.op_gated, brows(m.b_att, T), brows(m.b_sig, T), brows(m.b_attg, T));
        }
        RAD_ISSUE_N(c, L.op_o, T, brows(m.b_attg, T), RAD_W(L.w_o), brows(m.b_ao, T));
    }
    if (L.op_ar_o) RAD_ISSUE_N(c, L.op_ar_o, T * m.H, brows(m.b_ao, T), RAD_NONE);
    if (L.push_slot < 0 && L.op_aconv_add) {
        issue_sconv_add(c, batch, L.op_aconv_add, L.w_aconv, m.cv_a, a, m.b_ao, m.b_s, m.b_s);
    } else {
        issue_sconv(c, batch, L.op_aconv, L.w_aconv, m.cv_a, a, m.b_ao, 0, m.b_ax);

        /* the residual: a block push when this layer closes one, then the stream is the attention */
        if (L.push_slot >= 0) {
            if (m.use_plane_copy) {
                RAD_ISSUE_N(c, L.op_push_copy, T, brows(m.b_s, T), brows(m.b_s, T),
                            RAD_B_N(m.b_blk, (int64_t)L.push_slot * MT * m.H, 1));
                RAD_ISSUE_N(c, L.op_resid_copy, T, brows(m.b_ax, T), brows(m.b_ax, T), brows(m.b_s, T));
            } else {
                RAD_ISSUE_N(c, L.op_push, MT, brows(m.b_s, MT), brows(m.b_ones, MT),
                          RAD_B_N(m.b_blk, (int64_t)L.push_slot * MT * m.H, 1));
                RAD_ISSUE(c, L.op_resid, brows(m.b_ax, T), brows(m.b_ones, T), brows(m.b_s, T));
            }
        } else {
            RAD_ISSUE(c, L.op_attn_add, brows(m.b_s, T), brows(m.b_ax, T), brows(m.b_s, T));
        }
    }

    /* the feed-forward: its mix, the controller, the norm, the router and the experts */
    RAD_ISSUE_N(c, L.op_mlp_mix, T, RAD_B_N(m.b_blk, 0, L.n_state_mlp), brows(m.b_s, T), RAD_NONE, RAD_W(L.w_mlp_q),
                RAD_NONE, brows(m.b_mx, T));
    if (control) run_control(c, m, L, 3, m.b_mx, m.b_msg[3], m.b_cym, T);
    const rad_buf msrc = control ? m.b_cym : m.b_mx;
    const rad_op fused_nq = control ? L.op_mlp_norm_ctl_q8 : L.op_mlp_norm_q8;
    if (L.ex_form == kExpIqr && fused_nq) {
        RAD_ISSUE_N(c, fused_nq, T, brows(msrc, T), RAD_NONE, RAD_W(L.w_mlp_norm),
                    brows(m.b_q8, T), brows(m.b_s8, T), brows(m.b_fm, T));
    } else {
        RAD_ISSUE_N(c, control ? L.op_mlp_norm_ctl : L.op_mlp_norm, T, brows(msrc, T),
                    RAD_W(L.w_mlp_norm), brows(m.b_fm, T));
        if (L.ex_form == kExpIqr)
            RAD_ISSUE_N(c, L.op_q8, T, brows(m.b_fm, T), brows(m.b_q8, T), brows(m.b_s8, T));
    }
    RAD_ISSUE_N(c, L.op_router, T, brows(m.b_fm, T), RAD_W(L.w_router), brows(m.b_rl, T));
    if (L.op_router_scatter && T * m.K <= 1024) {
        RAD_ISSUE_N(c, L.op_router_scatter, T, brows(m.b_rl, T), RAD_W(L.w_bias),
                    brows(m.b_ids, T), brows(m.b_wts, T), RAD_B(m.b_sorted), RAD_B(m.b_eoff),
                    RAD_B(m.b_ecnt), RAD_NONE, RAD_NONE);
    } else {
        RAD_ISSUE_N(c, L.op_topk, T, brows(m.b_rl, T), RAD_W(L.w_bias), brows(m.b_ids, T), brows(m.b_wts, T));
        RAD_ISSUE_N(c, L.op_scatter, T, brows(m.b_ids, T), RAD_B(m.b_sorted), RAD_B(m.b_eoff), RAD_B(m.b_ecnt));
    }
    if (L.ex_form == kExpBf16) {
        RAD_ISSUE_N(c, L.op_egu, T, brows(m.b_fm, T), RAD_WTAB(L.w_egu[0], m.E), RAD_B(m.b_sorted),
                    RAD_B(m.b_eoff), brows(m.b_gu, T * m.K));
        RAD_ISSUE_N(c, L.op_situ, T * m.K, brows(m.b_gu, T * m.K), brows(m.b_act, T * m.K));
        RAD_ISSUE_N(c, L.op_edn, T, brows(m.b_act, T * m.K), RAD_WTAB(L.w_edn[0], m.E), RAD_B(m.b_sorted),
                    RAD_B(m.b_eoff), brows(m.b_dn, T * m.K));
    } else {
        /* the even experts' table, then the odd's (see the declaration): codes, then scales */
        const int64_t h = m.E / 2;
        RAD_ISSUE_N(c, L.op_egu, T, brows(m.b_q8, T), brows(m.b_s8, T), RAD_WTAB(L.w_egu[0], h),
                    RAD_WTAB(L.w_egus[0], h), RAD_B(m.b_sorted), RAD_B(m.b_eoff),
                    RAD_WTAB(L.w_egu[1], h), RAD_WTAB(L.w_egus[1], h), brows(m.b_gu, T * m.K));
        RAD_ISSUE_N(c, L.op_situ, T * m.K, brows(m.b_gu, T * m.K), brows(m.b_act, T * m.K));
        RAD_ISSUE_N(c, L.op_dq, T * m.K, brows(m.b_act, T * m.K), brows(m.b_aq, T * m.K),
                    brows(m.b_as, T * m.K));
        RAD_ISSUE_N(c, L.op_edn, T, brows(m.b_aq, T * m.K), brows(m.b_as, T * m.K), RAD_WTAB(L.w_edn[0], h),
                    RAD_WTAB(L.w_edns[0], h), RAD_B(m.b_sorted), RAD_B(m.b_eoff),
                    RAD_WTAB(L.w_edn[1], h), RAD_WTAB(L.w_edns[1], h), brows(m.b_dn, T * m.K));
    }
    calib_ffn(c, m, L, ip, T);
    RAD_ISSUE_N(c, L.op_gather, T, brows(m.b_dn, T * m.K), RAD_B(m.b_wts), RAD_B(m.b_sorted), RAD_NONE,
                RAD_NONE, brows(m.b_mo, T));
    if (L.op_ar_m) RAD_ISSUE_N(c, L.op_ar_m, T * m.H, brows(m.b_mo, T), RAD_NONE);
    if (L.op_mconv_add) {
        issue_sconv_add(c, batch, L.op_mconv_add, L.w_mconv, m.cv_m, a, m.b_mo, m.b_s, m.b_s);
    } else {
        issue_sconv(c, batch, L.op_mconv, L.w_mconv, m.cv_m, a, m.b_mo, 0, m.b_zc);
        RAD_ISSUE(c, L.op_mlp_add, brows(m.b_s, T), brows(m.b_zc, T), brows(m.b_s, T));
    }
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc && m.vit.on) {
        m.vit.step(c, batch);
        return;
    }
    /* A drafter pass: negative is the context pass over the trunk step just run, positive the
     * query pass over the blocks (inkling2_dflash2.h). */
    if (batch->draft_pass != 0 && m.df2.on) {
        if (batch->draft_pass < 0) m.df2.context(c, batch);
        else                       m.df2.query(c, batch);
        return;
    }
    if (batch->draft_pass || batch->spec_parent || batch->enc ||
        (batch->n_mm_rows > 0 && !m.op_media_rows)) {
        rad_step_fail(c, "inkling2: this step asks for an encoder, a draft pass, a tree verify or "
                         "media rows, and the program declares none of them");
        return;
    }
    const int64_t T = batch->n_tok;
    const int64_t MT = m.g.max_tok;
    if (!m.use_plane_copy) RAD_ISSUE_N(c, m.op_ones, T, brows(m.b_ones, T));
    if (!((T <= 16 && m.all_ctl_fused) || (T > 16 && m.all_ctl_blend)))
        RAD_ISSUE_N(c, m.op_minus, T, brows(m.b_minus1, T));
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_embed), brows(m.b_s, T));
    if (m.op_embed_ar) RAD_ISSUE_N(c, m.op_embed_ar, T * m.H, brows(m.b_s, T), RAD_NONE);
    RAD_ISSUE(c, m.op_embed_norm, brows(m.b_s, T), RAD_W(m.w_embed_norm), brows(m.b_s, T));
    /* A picture's rows replace its placeholders' after the norm: image rows enter raw. */
    if (batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_media_rows, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, m.H),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(m.b_s, T));

    for (int a = 0; a < kApps; ++a) {
        /* the prelude's output is kept for every pass to restart from; a pass boundary banks the
         * previous pass's block 2 and output, and restarts the stream from the prelude */
        if (a == 6) {
            if (m.use_plane_copy)
                RAD_ISSUE_N(c, m.op_pre_copy, T, brows(m.b_s, T), brows(m.b_s, T), brows(m.b_pre, T));
            else
                RAD_ISSUE(c, m.op_pre, brows(m.b_s, T), brows(m.b_ones, T), brows(m.b_pre, T));
        }
        if (a == 18 || a == 30) {
            if (m.use_plane_copy) {
                RAD_ISSUE_N(c, m.op_bank_cur_copy, T, brows(m.b_s, T), brows(m.b_s, T), brows(m.b_bank1, T));
                RAD_ISSUE_N(c, m.op_bank_blk_copy, T, RAD_B_N(m.b_blk, 2 * MT * m.H, 1), brows(m.b_s, T),
                            RAD_B_N(m.b_bank0, 0, 1));
                RAD_ISSUE_N(c, m.op_reset_copy, T, brows(m.b_pre, T), brows(m.b_pre, T), brows(m.b_s, T));
            } else {
                RAD_ISSUE(c, m.op_bank_cur, brows(m.b_s, T), brows(m.b_ones, T), brows(m.b_bank1, T));
                RAD_ISSUE_N(c, m.op_bank_blk, MT, RAD_B_N(m.b_blk, 2 * MT * m.H, 1), brows(m.b_ones, MT),
                          RAD_B_N(m.b_bank0, 0, 1));
                RAD_ISSUE(c, m.op_reset, brows(m.b_pre, T), brows(m.b_ones, T), brows(m.b_s, T));
            }
        }
        const bool control = a >= 18 && a < 42;
        m.df2.tap(c, m.b_s, T, a);
        run_layer(c, m, batch, a, control);
    }

    /* the output: a mix over the four blocks and the stream, the final norm, and the rows read */
    if (m.op_out_mix_norm) {
        RAD_ISSUE_N(c, m.op_out_mix_norm, T, RAD_B_N(m.b_blk, 0, 4), brows(m.b_s, T), RAD_NONE,
                    RAD_W(m.w_out_q), RAD_W(m.w_out_norm), brows(m.b_yn, T));
    } else {
        RAD_ISSUE_N(c, m.op_out_mix, T, RAD_B_N(m.b_blk, 0, 4), brows(m.b_s, T), RAD_NONE, RAD_W(m.w_out_q), RAD_NONE,
                    brows(m.b_yo, T));
        RAD_ISSUE_N(c, m.op_out_norm, T, brows(m.b_yo, T), RAD_W(m.w_out_norm), brows(m.b_yn, T));
    }
    m.df2.tap(c, m.b_yn, T, -1);
    if (batch->n_out > 0) {
        const int64_t N = batch->n_out;
        RAD_ISSUE_N(c, m.op_rows, N, brows(m.b_yn, T), praw(batch->out_ids, RAD_I32, N), brows(m.b_hout, N));
        if (!batch->score_only) {
            RAD_ISSUE_N(c, m.op_head, N, brows(m.b_hout, N), RAD_W(m.w_head), brows(m.b_logits, N));
            RAD_ISSUE_N(c, m.op_softcap, N, brows(m.b_logits, N), brows(m.b_logits, N));
            if (m.op_pad)
                RAD_ISSUE_N(c, m.op_pad, N, bcol(m.b_logits, m.pad_lo, m.pad_n, N));
        }
    }
    calib_after_step(c, m, T);
}

/* ============================================================== the reply format
 * The model's, as its template writes it: reasoning, the answer and tool calls, each a message
 * opening with <|message_model|> (arch/rad_inkling2_chat.h). */
static const RadChatFormat* chat_format(const RadModelMeta*) { return inkling2_chat_format(); }

/* The tower's merge factor (4 patches per merged row) and the DFlash2 drafter's fixed depth (one
 * proposal per mask row). */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    out->chain_pays = RAD_CHAIN_ALWAYS;
    if (meta && Inkling2Tower::present(meta)) out->enc_merge = 4;
    out->draft_depth = 0;
    out->draft_depth_fixed = 0;
    if (meta && dflash2_present(meta)) {
        const int64_t d = dflash2_depth(meta);
        if (d < 1) {
            fprintf(stderr, "radiance: inkling2: the container carries a DFlash2 drafter with no "
                            "dflash_config.block_size\n");
            return RAD_E_FORMAT;
        }
        out->draft_depth = (int)d;
        out->draft_depth_fixed = 1;
    }
    return RAD_OK;
}

}  /* namespace inkling2 */

#ifndef RAD_ARCH_NO_EXPORTS
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(inkling2, "inkling2", "", "0.1.0",
                "Inkling 2 Mini's text backbone: the loop checkpoint's 48 layer applications over 24 "
                "physical layers, block attention residuals, relative-position attention with a sink, "
                "stateful short convolutions, controllers and SiTU experts (IQR-V4, or bf16 for "
                "calibration), a DFlash2 drafter; tensor parallel")
RAD_ARCH_CHAT_FORMAT(inkling2)
RAD_ARCH_PROBE(inkling2)
#endif
