/* inkling2 -- Inkling 2's text backbone as the loop checkpoint lays it out (`inkling2_loop`, served
 * as the arch id `inkling2`): bf16 weights, one rank, the host kernels.
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
 * in the form its tensors are stored in: bf16, NVFP4 (the release's layers 0..19, kept as shipped,
 * `nvfp4a8i8`), or rotated int8 (`i8a8h`, the form the MXFP8 layers of the recipe become). The
 * activation is SiTU in every form (`gated_quant_i8` / `gated_had_quant_i8` in place of `situ_mul`).
 *
 * ============================== WHAT IT DOES NOT SERVE ==============================
 *
 * Quantised linears (NVFP4, MXFP8, IQ8R), tensor parallelism, the vision tower, the
 * DSpark draft and a loop schedule other than the release's. Each is refused by name. The release's
 * full-attention `static_w` convolutions are rewritten to the conv1d layout at import
 * (core/format/checkpoint.cpp, regroup_inkling_static_w), so the plugin reads one layout.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_dspark_inkling.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace inkling2 {

using namespace rad::arch;

static const int kApps = 48;          /* layer applications: 6 + 3 * 12 + 6 */
static const int kLayers = 24;        /* physical layers */
static const int kWidth = 4;          /* sconv_kernel_size of the release */
static const int kBlocks = 4;         /* block planes: the embedding and three pushes */
static const int kExpGroup = 128;     /* a quantised expert's int8 activation: one scale per 128 */

/* How a layer's routed experts are stored, read from the checkpoint's own tensors (the importer
 * takes each tensor's form from hf_quant_config.json's parameter groups, and a recipe's rules name
 * the form they produce). */
enum ExpertForm { kExpBf16 = 0, kExpNvfp4 = 1, kExpI8 = 2 };

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
    rad_op op_norm[3] = { 0, 0, 0 };        /* from the mixed input */
    rad_op op_norm_ctl[3] = { 0, 0, 0 };    /* from the controlled input */

    rad_weight w_q = 0, w_k = 0, w_v = 0, w_r = 0, w_gate = 0, w_sink = 0, w_o = 0;
    rad_op op_q = 0, op_k = 0, op_v = 0, op_r = 0, op_gate = 0, op_sink = 0, op_o = 0;
    rad_weight w_dk = 0, w_dv = 0;          /* input-dependent taps (full layers) */
    rad_op op_dk = 0, op_dv = 0;
    rad_weight w_kconv = 0, w_vconv = 0, w_aconv = 0, w_mconv = 0;
    rad_op op_kconv = 0, op_vconv = 0, op_aconv = 0, op_mconv = 0;
    rad_weight w_qscale = 0, w_rscale = 0, w_rel = 0;
    rad_op op_qclip = 0, op_rclip = 0, op_kclip = 0, op_vnorm = 0;
    rad_op op_store = 0, op_attn = 0, op_sig = 0, op_gated = 0;

    /* the core's messages and controllers, per input j: attn (q), k, v, mlp */
    rad_weight w_msg[4] = { 0, 0, 0, 0 };
    rad_op op_msg[4] = { 0, 0, 0, 0 };
    rad_weight w_dx[4] = { 0, 0, 0, 0 }, w_dm[4] = { 0, 0, 0, 0 };
    rad_weight w_cgain[4] = { 0, 0, 0, 0 }, w_cforget[4] = { 0, 0, 0, 0 };
    rad_op ctl[4][14] = {};

    rad_weight w_mlp_q = 0, w_mlp_norm = 0, w_router = 0, w_bias = 0;
    rad_op op_mlp_mix = 0, op_mlp_norm = 0, op_mlp_norm_ctl = 0, op_router = 0, op_topk = 0;
    rad_op op_scatter = 0, op_egu = 0, op_situ = 0, op_edn = 0, op_gather = 0, op_mconv_out = 0;
    rad_op op_push = 0, op_resid = 0, op_attn_add = 0, op_mlp_add = 0;
    /* the experts' stored form (ExpertForm), its codes and scale planes, and for a quantised
     * expert the quantiser of the routed input and of the SiTU product */
    int ex_form = 0;
    std::vector<rad_weight> w_egu, w_edn;   /* the expert codes (or the bf16 experts), by expert */
    std::vector<rad_weight> w_egus, w_edns; /* a quantised expert's scale plane, by expert */
    rad_op op_q8 = 0;
    rad_op op_dq = 0;   /* the rotated experts' quantiser of the bf16 SiTU product */
};

struct Model {
    Names   nm{""};
    Geom    g{};
    int64_t H = 0, F = 0, E = 0, K = 0, HD = 0, NH = 0, KVH = 0, DR = 0, EXT = 0, WIN = 0;
    int64_t VR = 0, VN = 0, RANK = 0;
    double  qclip = 0, rclip = 0, alpha = 0, floorv = 0, route_scale = 1;
    double  gcap = 0, ucap = 0, fcap = 0;

    rad_kvgroup kv_full = 0, kv_local = 0;
    rad_kvgroup cv_k = 0, cv_v = 0, cv_a = 0, cv_m = 0;

    rad_buf b_s = 0, b_pre = 0, b_bank1 = 0, b_ones = 0, b_minus1 = 0;
    rad_buf b_blk = 0;                       /* [kBlocks, max_tok, H]: the block planes */
    rad_buf b_bank0 = 0;                     /* [1, max_tok, H]: the previous pass's block */
    rad_buf b_y[3] = { 0, 0, 0 }, b_h[3] = { 0, 0, 0 }, b_cy[3] = { 0, 0, 0 }, b_msg[4] = { 0, 0, 0, 0 };
    rad_buf b_gxm = 0, b_g1 = 0, b_g2 = 0, b_g = 0, b_gp = 0, b_gt = 0;
    rad_buf b_fp = 0, b_ft = 0, b_nft = 0, b_t1 = 0, b_t2 = 0, b_t3 = 0;
    rad_buf b_q = 0, b_k = 0, b_v = 0, b_r = 0, b_gate = 0, b_sink = 0;
    rad_buf b_dk = 0, b_dv = 0, b_kcv = 0, b_vcv = 0, b_qc = 0, b_kc = 0, b_vc = 0, b_rc = 0;
    rad_buf b_att = 0, b_sig = 0, b_attg = 0, b_ao = 0, b_ax = 0;
    rad_buf b_mx = 0, b_cym = 0, b_fm = 0, b_rl = 0, b_ids = 0, b_wts = 0;
    rad_buf b_sorted = 0, b_eoff = 0, b_ecnt = 0, b_gu = 0, b_act = 0, b_dn = 0, b_mo = 0, b_zc = 0;
    rad_buf b_yo = 0, b_yn = 0, b_hout = 0, b_logits = 0;
    rad_buf b_q8 = 0, b_s8 = 0, b_aq = 0, b_as = 0;   /* a quantised expert's int8 input and product */

    rad_weight w_embed = 0, w_embed_norm = 0, w_out_norm = 0, w_out_q = 0, w_head = 0;
    rad_op op_embed = 0, op_embed_norm = 0, op_out_mix = 0, op_out_norm = 0, op_rows = 0;
    rad_op op_head = 0, op_softcap = 0, op_pad = 0, op_ones = 0, op_minus = 0;
    rad_op op_pre = 0, op_bank_blk = 0, op_bank_cur = 0, op_reset = 0;

    /* The DSpark drafter (mtp.safetensors), when the container carries it and the serve drafts:
     * the stream at the tapped applications (`draft.target_layer_ids`, an id of kApps being the
     * final normed output), one bf16 column slice a tap of b_taps, and the drafter's graph. */
    bool    have_ds = false;
    int     n_taps = 0, tap_at[DsparkInkling::N_TAPS] = {};
    rad_buf b_taps = 0;
    rad_op  op_tap = 0, op_tap_out = 0;
    DsparkInkling ds{};
    DsparkInkling::Wire dsw{};

    std::vector<Layer> layers;
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

/* The form the expert tensor `src` is held in. No encoding (a builder with no model behind it) is
 * bf16, as the tiny model's experts are. Anything else is refused by name. */
static int expert_form(RadBuilder* b, const char* src, int* form) {
    RadEncoding e{};
    *form = kExpBf16;
    if (!weight_enc(b, src, &e)) return RAD_OK;
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return RAD_OK;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    const RadEncPlane* s2 = rad_enc_plane(&e, "scale.1");
    /* NVFP4: e2m1 codes, an e4m3 scale per (row, 16 of K) and the expert's f32 level */
    if (rad_enc_is(&e, "affine") && e.n_planes == 3 && !e.transform[0] &&
        e.plane[0].dtype == RAD_FP4E2M1 && sc && sc->dtype == RAD_F8E4M3 &&
        sc->block[0] == 1 && sc->block[1] == 16 && s2 && s2->dtype == RAD_F32 &&
        s2->block[0] == 0 && s2->block[1] == 0) {
        *form = kExpNvfp4;
        return RAD_OK;
    }
    /* rotated int8 codes at a bf16 scale per (row, 128 of K): what a recipe's i8a8h rule writes */
    if (rad_enc_is(&e, "affine") && e.n_planes == 2 && e.plane[0].dtype == RAD_I8 && sc &&
        sc->dtype == RAD_BF16 && sc->block[0] == 1 && sc->block[1] == kExpGroup &&
        std::strcmp(e.transform, "fwht128") == 0) {
        *form = kExpI8;
        return RAD_OK;
    }
    char en[256];
    rad_enc_format(&e, en, sizeof en);
    fprintf(stderr, "radiance: inkling2: expert '%s' is %s, and the routed experts are served as bf16, "
                    "NVFP4 (fp4_e2m1*fp8_e4m3[1x16]*f32[*x*]) or rotated int8 (i8*bf16[1x128]/fwht128); "
                    "convert it with a recipe that makes one of them\n", src, en);
    return RAD_E_DTYPE;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: inkling2 serves one rank; run it with --tp 1\n");
        return RAD_E_UNSUPPORTED;
    }
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
    m.VR = meta_int(meta, "unpadded_vocab_size", g.n_vocab);
    m.VN = g.n_vocab;
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
        m.VR > g.n_vocab ||
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
    }

    const int64_t T = g.max_tok;
    const int64_t NHD = m.NH * m.HD, KVD = m.KVH * m.HD, KVC = m.KVH * kWidth;

    /* ---- state: one bound layer per application of each stateful kind */
    {
        RadKVGroupDecl fd{};
        fd.kind = RAD_KV_FULL;
        fd.dtype = RAD_BF16;
        fd.n_head_kv = m.KVH;
        fd.head_dim = m.HD;
        m.kv_full = rad_decl_kv_group(b, nm.f("kv_full"), &fd);
        RadKVGroupDecl ld{};
        ld.kind = RAD_KV_WINDOW;
        ld.dtype = RAD_BF16;
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
    m.b_gu  = decl_b(b, nm.f("gu"), RAD_BF16, {T * m.K, 2 * m.F});
    m.b_act = decl_b(b, nm.f("act"), RAD_BF16, {T * m.K, m.F});
    /* a quantised expert's routed input and SiTU product: int8 codes with an f32 scale per 128 */
    m.b_q8  = decl_b(b, nm.f("q8"), RAD_I8, {T, m.H});
    m.b_s8  = decl_b(b, nm.f("s8"), RAD_F32, {T, (m.H + kExpGroup - 1) / kExpGroup});
    m.b_aq  = decl_b(b, nm.f("aq"), RAD_I8, {T * m.K, m.F});
    m.b_as  = decl_b(b, nm.f("as"), RAD_F32, {T * m.K, (m.F + kExpGroup - 1) / kExpGroup});
    m.b_dn  = decl_b(b, nm.f("dn"), RAD_BF16, {T * m.K, m.H});
    m.b_mo  = decl_b(b, nm.f("mo"), RAD_BF16, {T, m.H});
    m.b_zc  = decl_b(b, nm.f("zc"), RAD_BF16, {T, m.H});
    m.b_yo  = decl_b(b, nm.f("yo"), RAD_BF16, {T, m.H});
    m.b_yn  = decl_b(b, nm.f("yn"), RAD_BF16, {T, m.H});
    m.b_hout = decl_b(b, nm.f("hout"), RAD_BF16, {g.max_logit_rows, m.H});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32, {g.max_logit_rows, g.n_vocab});
    m.have_ds = ctx->max_spec > 0 && rad_meta_geti(meta, "draft.block_size", 0) > 0;
    if (m.have_ds) {
        m.n_taps = parse_layer_ids(rad_meta_gets(meta, "draft.target_layer_ids", nullptr), kApps + 1,
                                   m.tap_at, DsparkInkling::N_TAPS);
        if (m.n_taps != DsparkInkling::N_TAPS) {
            fprintf(stderr, "radiance: inkling2: the drafter names %d trunk taps in 0..%d; its "
                            "graph mixes %d\n", m.n_taps, kApps, (int)DsparkInkling::N_TAPS);
            return RAD_E_UNSUPPORTED;
        }
        m.b_taps = decl_b(b, nm.f("draft_src"), RAD_BF16, {T, (int64_t)m.n_taps * m.H});
        if (!m.b_taps) return RAD_E_INVAL;
        /* written at four points of the step and read by the drafter's passes after it */
        RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_taps));
    }
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
    m.w_embed = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, m.H}, RAD_ACCESS_VOCAB,
                       RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("embed_norm.weight"), nm.ckpt("%s.embed_norm.weight", pfx)));
    m.w_embed_norm = decl_w(b, nm.f("embed_norm.weight"), RAD_BF16, {m.H}, RAD_ACCESS_PER_TOKEN,
                            RAD_SHARD_NONE, grp_model());
    /* the head covers the padded table, since the engine samples over n_vocab; op_pad fills the
     * logits past unpadded_vocab_size so the padding is never sampled. */
    RAD_ARCH_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("%s.unembed.weight", pfx)));
    m.w_head = decl_w(b, nm.f("output.weight"), RAD_BF16, {g.n_vocab, m.H}, RAD_ACCESS_VOCAB,
                      RAD_SHARD_NONE, grp_model());
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
        auto one = [&](rad_weight* w, const char* name, const std::string& src, uint32_t dt,
                       std::initializer_list<int64_t> shape) -> int {
            RAD_ARCH_TRY(map_copy(b, nm.f(name, l), nm.ckpt("%s", src.c_str())));
            *w = decl_w(b, nm.f(name, l), dt, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
            return *w ? RAD_OK : RAD_E_INVAL;
        };
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
        RAD_ARCH_TRY(one(&L.w_q, "blk.%d.wq", at + "wq_du.weight", RAD_BF16, {NHD, m.H}));
        RAD_ARCH_TRY(one(&L.w_k, "blk.%d.wk", at + "wk_dv.weight", RAD_BF16, {KVD, m.H}));
        RAD_ARCH_TRY(one(&L.w_v, "blk.%d.wv", at + "wv_dv.weight", RAD_BF16, {KVD, m.H}));
        RAD_ARCH_TRY(one(&L.w_r, "blk.%d.wr", at + "wr_du.weight", RAD_BF16, {m.NH * m.DR, m.H}));
        RAD_ARCH_TRY(one(&L.w_gate, "blk.%d.gate", at + "gate_attn.weight", RAD_BF16, {NHD, m.H}));
        RAD_ARCH_TRY(one(&L.w_sink, "blk.%d.sink", at + "attn_sink_proj.weight", RAD_BF16, {m.NH, m.H}));
        RAD_ARCH_TRY(one(&L.w_o, "blk.%d.wo", at + "wo_ud.weight", RAD_BF16, {m.H, NHD}));
        RAD_ARCH_TRY(one(&L.w_qscale, "blk.%d.qscale", at + "q_l2_clip.scale", RAD_BF16, {NHD}));
        RAD_ARCH_TRY(one(&L.w_rscale, "blk.%d.rscale", at + "rel_l2_clip.scale", RAD_BF16, {m.NH * m.DR}));
        RAD_ARCH_TRY(one(&L.w_rel, "blk.%d.rel", at + "rel_logits_proj.proj", RAD_BF16, {m.DR, m.EXT}));
        if (L.full) {
            RAD_ARCH_TRY(one(&L.w_dk, "blk.%d.dk", at + "k_sconv.weight_proj", RAD_BF16, {KVC, m.H}));
            RAD_ARCH_TRY(one(&L.w_dv, "blk.%d.dv", at + "v_sconv.weight_proj", RAD_BF16, {KVC, m.H}));
        }
        RAD_ARCH_TRY(one(&L.w_kconv, "blk.%d.kconv", at + "k_sconv.weight", RAD_BF16, {KVD, kWidth}));
        RAD_ARCH_TRY(one(&L.w_vconv, "blk.%d.vconv", at + "v_sconv.weight", RAD_BF16, {KVD, kWidth}));
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
        RAD_ARCH_TRY(one(&L.w_mlp_norm, "blk.%d.mlp_norm", pre + "mlp_norm.weight", RAD_BF16, {m.H}));
        RAD_ARCH_TRY(one(&L.w_router, "blk.%d.router", pre + "mlp.gate.weight", RAD_F32, {m.E, m.H}));
        RAD_ARCH_TRY(one(&L.w_bias, "blk.%d.router_bias", pre + "mlp.gate.bias", RAD_F32, {m.E}));
        /* the experts: the maps first, then the handles consecutively, as a table needs them */
        {
            const std::string ex = pre + "mlp.experts";
            for (int64_t e = 0; e < m.E; ++e)
                RAD_ARCH_TRY(map_slice(b, nm.f("blk.%d.e%lld.gu", l, (long long)e), nm.ckpt("%s.w13_weight", ex.c_str()), (int)e));
            for (int64_t e = 0; e < m.E; ++e)
                RAD_ARCH_TRY(map_slice(b, nm.f("blk.%d.e%lld.dn", l, (long long)e), nm.ckpt("%s.w2_weight", ex.c_str()), (int)e));
            /* the stored form, read from the first expert's tensors: every expert of a layer is one form */
            int dn_form = 0;
            RAD_ARCH_TRY(expert_form(b, nm.f("blk.%d.e0.gu", l), &L.ex_form));
            RAD_ARCH_TRY(expert_form(b, nm.f("blk.%d.e0.dn", l), &dn_form));
            if (dn_form != L.ex_form) {
                fprintf(stderr, "radiance: inkling2: layer %d's gate_up and down experts are stored in "
                                "different forms\n", l);
                return RAD_E_DTYPE;
            }
            if (L.ex_form == kExpBf16) {
                for (int64_t e = 0; e < m.E; ++e)
                    L.w_egu.push_back(decl_w(b, nm.f("blk.%d.e%lld.gu", l, (long long)e), RAD_BF16, {2 * m.F, m.H},
                                             RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_expert(l, (int)e)));
                for (int64_t e = 0; e < m.E; ++e)
                    L.w_edn.push_back(decl_w(b, nm.f("blk.%d.e%lld.dn", l, (long long)e), RAD_BF16, {m.H, m.F},
                                             RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_expert_slot(l, (int)e, 1)));
            } else {
                /* A quantised expert: its codes as one view and its scale plane as another, declared
                 * by parity class (moe_gemm_q takes the even experts' table, then the odd's), so each
                 * class's codes are consecutive handles and its scales are too. */
                if (m.E % 2 || m.H % kExpGroup || m.F % kExpGroup) {
                    fprintf(stderr, "radiance: inkling2: a quantised expert needs an even expert count and "
                                    "H and F multiples of %d (E %lld, H %lld, F %lld)\n", kExpGroup,
                            (long long)m.E, (long long)m.H, (long long)m.F);
                    return RAD_E_INVAL;
                }
                const bool nv = L.ex_form == kExpNvfp4;
                const uint32_t cdt = nv ? RAD_FP4E2M1 : RAD_I8;
                const uint32_t sdt = nv ? RAD_F8E4M3 : RAD_BF16;
                const char* splanes = nv ? "scale,scale.1" : "scale";
                /* A scale view over both planes (NVFP4's e4m3 and its level) is declared at the logical
                 * extent, as the codebook forms' are: the layout hook reads the stored [N, K/16] plane.
                 * The i8 form's one scale plane is declared at its own [N, K/128]. */
                const int64_t sg = nv ? 1 : kExpGroup;
                auto decl_class = [&](const char* leg, int64_t N, int64_t K, std::vector<rad_weight>& wc,
                                      std::vector<rad_weight>& ws, int slot) -> int {
                    wc.assign((size_t)m.E, 0);
                    ws.assign((size_t)m.E, 0);
                    for (int q = 0; q < 2; ++q)
                        for (int64_t e = q; e < m.E; e += 2) {
                            const char* w = nm.f("blk.%d.e%lld.%s", l, (long long)e, leg);
                            wc[(size_t)e] = decl_view(b, w, nullptr, "codes", cdt, {N, K},
                                                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_expert_slot(l, (int)e, slot));
                            if (!wc[(size_t)e]) return RAD_E_INVAL;
                        }
                    for (int q = 0; q < 2; ++q)
                        for (int64_t e = q; e < m.E; e += 2) {
                            const char* w = nm.f("blk.%d.e%lld.%s", l, (long long)e, leg);
                            ws[(size_t)e] = decl_view(b, nm.f("%s.scale", w), w, splanes, sdt, {N, (K + sg - 1) / sg},
                                                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_expert_slot(l, (int)e, slot));
                            if (!ws[(size_t)e]) return RAD_E_INVAL;
                        }
                    return RAD_OK;
                };
                RAD_ARCH_TRY(decl_class("gu", 2 * m.F, m.H, L.w_egu, L.w_egus, 0));
                RAD_ARCH_TRY(decl_class("dn", m.H, m.F, L.w_edn, L.w_edns, 1));
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

        /* the stream's copies, and the state the pass boundaries move */
        m.op_ones = rw(b, RAD_OP(b, "fill", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                  RAD_STR("dtype", "bf16"), RAD_F64("value", 1.0)),
                              RAD_NOWEIGHTS), {}, {m.b_ones});
        m.op_minus = rw(b, RAD_OP(b, "fill", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                   RAD_STR("dtype", "bf16"), RAD_F64("value", -1.0)),
                               RAD_NOWEIGHTS), {}, {m.b_minus1});
        m.op_embed = rw(b, RAD_OP(b, "embed_lookup", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", m.H),
                                                           RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(m.w_embed)), {}, {m.b_s});
        m.op_embed_norm = rms(T, m.H, m.w_embed_norm, m.b_s, m.b_s);
        m.op_pre = ew("mul", T, m.H, {m.b_s, m.b_ones}, m.b_pre);
        m.op_bank_blk = ew("mul", T, m.H, {m.b_blk, m.b_ones}, m.b_bank0);
        m.op_bank_cur = ew("mul", T, m.H, {m.b_s, m.b_ones}, m.b_bank1);
        m.op_reset = ew("mul", T, m.H, {m.b_pre, m.b_ones}, m.b_s);
        m.op_out_mix = mix(4, m.b_s, m.w_out_q, m.b_yo, m.b_blk);
        m.op_out_norm = rms(T, m.H, m.w_out_norm, m.b_yo, m.b_yn);
        if (m.have_ds) {
            /* a tap is a copy (times one, exact) of the stream into its column slice */
            m.op_tap = ew("mul", T, m.H, {m.b_s, m.b_ones}, m.b_taps);
            m.op_tap_out = ew("mul", T, m.H, {m.b_yn, m.b_ones}, m.b_taps);
            if (!m.op_tap || !m.op_tap_out) return RAD_E_INVAL;
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
        if (m.VR < g.n_vocab)
            m.op_pad = rw(b, RAD_OP(b, "fill", RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                                       RAD_INT("n", g.n_vocab - m.VR),
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
            L.op_q = gemm(NHD, m.H, L.w_q, m.b_h[0], m.b_q);
            L.op_k = gemm(KVD, m.H, L.w_k, m.b_h[1], m.b_k);
            L.op_v = gemm(KVD, m.H, L.w_v, m.b_h[2], m.b_v);
            L.op_r = gemm(NR, m.H, L.w_r, m.b_h[0], m.b_r);
            L.op_gate = gemm(NHD, m.H, L.w_gate, m.b_h[0], m.b_gate);
            L.op_sink = gemm(m.NH, m.H, L.w_sink, m.b_h[0], m.b_sink);
            if (L.full) {
                L.op_dk = gemm(KVC, m.H, L.w_dk, m.b_h[1], m.b_dk);
                L.op_dv = gemm(KVC, m.H, L.w_dv, m.b_h[2], m.b_dv);
            }
            L.op_kconv = sconv(m.b_k, L.w_kconv, m.b_kcv, KVD, L.full ? m.KVH : 0, L.full ? m.b_dk : 0);
            L.op_vconv = sconv(m.b_v, L.w_vconv, m.b_vcv, KVD, L.full ? m.KVH : 0, L.full ? m.b_dv : 0);
            L.op_aconv = sconv(m.b_ao, L.w_aconv, m.b_ax, m.H, 0, 0);
            L.op_mconv = sconv(m.b_mo, L.w_mconv, m.b_zc, m.H, 0, 0);
            /* the clips: the position tau rides on full layers only */
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
                                                 RAD_WEIGHTS(L.w_rscale)), {m.b_r}, {m.b_rc});
            } else {
                L.op_qclip = rw(b, RAD_OP(b, "l2_clip", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", NHD),
                                                              RAD_INT("group", m.HD), RAD_F64("threshold", m.qclip),
                                                              RAD_STR("dtype", "bf16")),
                                                 RAD_WEIGHTS(L.w_qscale)), {m.b_q}, {m.b_qc});
                L.op_rclip = rw(b, RAD_OP(b, "l2_clip", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", NR),
                                                              RAD_INT("group", m.DR), RAD_F64("threshold", m.rclip),
                                                              RAD_STR("dtype", "bf16")),
                                                 RAD_WEIGHTS(L.w_rscale)), {m.b_r}, {m.b_rc});
            }
            L.op_kclip = rw(b, RAD_OP(b, "l2_clip", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", KVD),
                                                          RAD_INT("group", m.HD), RAD_F64("threshold", m.qclip),
                                                          RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS),
                            {m.b_kcv}, {m.b_kc});
            L.op_vnorm = rms(T * m.KVH, m.HD, 0, m.b_vcv, m.b_vc);
            L.op_attn = rw(b, RAD_OP(b, "attn_paged_rel", RAD_PARAMS(RAD_RANGE("q_len", 1, T), RAD_INT("head_dim", m.HD),
                                                                RAD_INT("gqa", m.NH / m.KVH),
                                                                RAD_INT("block_size", RAD_KV_BLOCK),
                                                                RAD_INT("causal", 1),
                                                                RAD_INT("window", L.full ? 0 : m.WIN),
                                                                RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "bf16"),
                                                                RAD_INT("max_ctx", g.max_ctx), RAD_INT("n_head", m.NH),
                                                                RAD_INT("max_seqs", g.max_seqs),
                                                                RAD_INT("d_rel", m.DR),
                                                                RAD_INT("extent", L.full ? m.EXT : m.WIN)),
                               RAD_WEIGHTS(L.w_rel)), {m.b_qc, m.b_rc, m.b_sink}, {m.b_att});
            /* after the attention: its block size is the attention kernel's constraint */
            L.op_store = rw(b, RAD_OP(b, "kv_store", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", m.HD),
                                                           RAD_INT("n_head_kv", m.KVH),
                                                           RAD_INT("block_size", rad_kv_block_size(b, grp)),
                                                           RAD_STR("kv_dtype", "bf16")), RAD_NOWEIGHTS),
                            {m.b_kc, m.b_vc}, {});
            L.op_sig = ew("sigmoid", T, NHD, {m.b_gate}, m.b_sig);
            L.op_gated = ew("mul", T, NHD, {m.b_att, m.b_sig}, m.b_attg);
            L.op_o = gemm(m.H, NHD, L.w_o, m.b_attg, m.b_ao);
            /* the residual: a pushed block, or the stream plus the attention */
            if (L.push_slot >= 0) {
                L.op_push = ew("mul", T, m.H, {m.b_s, m.b_ones}, m.b_blk);
                L.op_resid = ew("mul", T, m.H, {m.b_ax, m.b_ones}, m.b_s);
            } else {
                L.op_attn_add = ew("add", T, m.H, {m.b_s, m.b_ax}, m.b_s);
            }
            /* the feed-forward */
            L.op_mlp_mix = mix(L.n_state_mlp, m.b_s, L.w_mlp_q, m.b_mx, m.b_blk);
            L.op_mlp_norm = rms(T, m.H, L.w_mlp_norm, m.b_mx, m.b_fm);
            L.op_mlp_norm_ctl = rms(T, m.H, L.w_mlp_norm, m.b_cym, m.b_fm);
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
            if (L.ex_form == kExpBf16) {
                const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("N", 2 * m.F), RAD_INT("K", m.H),
                                        RAD_INT("n_expert", m.E), RAD_INT("top_k", m.K),
                                        RAD_STR("a_order", "token"), RAD_STR("dtype", "bf16") };
                L.op_egu = rw(b, rad_decl_op(b, "moe_gemm", ps, (int)(sizeof ps / sizeof ps[0]),
                                             L.w_egu.data(), (int)L.w_egu.size()),
                              {m.b_fm, m.b_sorted, m.b_eoff}, {m.b_gu});
                const RadParam pd[] = { RAD_RANGE("M", 1, T), RAD_INT("N", m.H), RAD_INT("K", m.F),
                                        RAD_INT("n_expert", m.E), RAD_INT("top_k", m.K),
                                        RAD_STR("a_order", "sorted"), RAD_STR("dtype", "bf16") };
                L.op_edn = rw(b, rad_decl_op(b, "moe_gemm", pd, (int)(sizeof pd / sizeof pd[0]),
                                             L.w_edn.data(), (int)L.w_edn.size()),
                              {m.b_act, m.b_sorted, m.b_eoff}, {m.b_dn});
                L.op_situ = rw(b, RAD_OP(b, "situ_mul", RAD_PARAMS(RAD_RANGE("M", 1, T * m.K), RAD_INT("n", m.F),
                                                              RAD_STR("dtype", "bf16"), RAD_F64("gate_cap", m.gcap),
                                                              RAD_F64("up_cap", m.ucap)), RAD_NOWEIGHTS),
                               {m.b_gu}, {m.b_act});
            } else {
                /* NVFP4 (kept as stored) reads int8 activations quantised without rotation; the rotated
                 * int8 layers take the Hadamard quantisers. The SiTU product is quantised in the same
                 * pass, so the down projection reads its codes and scales, not a bf16 plane. */
                const bool nv = L.ex_form == kExpNvfp4;
                const char* ex_dt = nv ? "nvfp4a8i8" : "i8a8h";
                L.op_q8 = rw(b, RAD_OP(b, nv ? "quant_act_i8g" : "had_quant_act_i8g",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                  RAD_INT("group", kExpGroup), RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS), {m.b_fm}, {m.b_q8, m.b_s8});
                /* moe_gemm_q's table: the even experts' codes, their scales, then the odd experts' */
                auto parity_table = [&](const std::vector<rad_weight>& wc, const std::vector<rad_weight>& ws) {
                    std::vector<rad_weight> tab;
                    for (int q = 0; q < 2; ++q) {
                        for (int64_t e = q; e < m.E; e += 2) tab.push_back(wc[(size_t)e]);
                        for (int64_t e = q; e < m.E; e += 2) tab.push_back(ws[(size_t)e]);
                    }
                    return tab;
                };
                {
                    const RadParam ps[] = { RAD_RANGE("M", 1, T), RAD_INT("N", 2 * m.F), RAD_INT("K", m.H),
                                            RAD_INT("n_expert", m.E), RAD_INT("top_k", m.K),
                                            RAD_INT("group", kExpGroup), RAD_STR("a_order", "token"),
                                            RAD_STR("dtype", ex_dt), RAD_INT("a_group", kExpGroup) };
                    const std::vector<rad_weight> tab = parity_table(L.w_egu, L.w_egus);
                    L.op_egu = rw(b, rad_decl_op(b, "moe_gemm_q", ps, (int)(sizeof ps / sizeof ps[0]),
                                                 tab.data(), (int)tab.size()),
                                  {m.b_q8, m.b_s8, m.b_sorted, m.b_eoff}, {m.b_gu});
                }
                /* The SiTU product in the down projection's activation form. NVFP4 reads
                 * quant_act_i8g's, which gated_quant_i8 writes in the same pass. The rotated int8
                 * experts read had_quant_act_i8g's: a rotation per 128 with nothing folded and a
                 * scale per (row, 128). gated_had_quant_i8 is the per-row form (one scale a row
                 * carrying 1/sqrt(128), codes in the step order), and the fused per-group form,
                 * gated_had_quant_i8g, is a card op without SiTU, so the product is situ_mul's bf16
                 * and then that quantiser, the pair gated_had_quant_i8g is defined as. */
                if (nv) {
                    L.op_situ = rw(b, RAD_OP(b, "gated_quant_i8",
                                             RAD_PARAMS(RAD_RANGE("M", 1, T * m.K), RAD_INT("n", m.F),
                                                        RAD_INT("group", kExpGroup), RAD_STR("dtype", "bf16"),
                                                        RAD_STR("act", "situ"), RAD_F64("gate_cap", m.gcap),
                                                        RAD_F64("up_cap", m.ucap)), RAD_NOWEIGHTS),
                                   {m.b_gu}, {m.b_aq, m.b_as});
                } else {
                    L.op_situ = rw(b, RAD_OP(b, "situ_mul",
                                             RAD_PARAMS(RAD_RANGE("M", 1, T * m.K), RAD_INT("n", m.F),
                                                        RAD_STR("dtype", "bf16"), RAD_F64("gate_cap", m.gcap),
                                                        RAD_F64("up_cap", m.ucap)), RAD_NOWEIGHTS),
                                   {m.b_gu}, {m.b_act});
                    L.op_dq = rw(b, RAD_OP(b, "had_quant_act_i8g",
                                           RAD_PARAMS(RAD_RANGE("M", 1, T * m.K), RAD_INT("n", m.F),
                                                      RAD_INT("group", kExpGroup), RAD_STR("dtype", "bf16")),
                                           RAD_NOWEIGHTS),
                                 {m.b_act}, {m.b_aq, m.b_as});
                }
                {
                    const RadParam pd[] = { RAD_RANGE("M", 1, T), RAD_INT("N", m.H), RAD_INT("K", m.F),
                                            RAD_INT("n_expert", m.E), RAD_INT("top_k", m.K),
                                            RAD_INT("group", kExpGroup), RAD_STR("a_order", "sorted"),
                                            RAD_STR("dtype", ex_dt), RAD_INT("a_group", kExpGroup) };
                    const std::vector<rad_weight> tab = parity_table(L.w_edn, L.w_edns);
                    L.op_edn = rw(b, rad_decl_op(b, "moe_gemm_q", pd, (int)(sizeof pd / sizeof pd[0]),
                                                 tab.data(), (int)tab.size()),
                                  {m.b_aq, m.b_as, m.b_sorted, m.b_eoff}, {m.b_dn});
                }
            }
            L.op_gather = rw(b, RAD_OP(b, "moe_gather", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.H),
                                                              RAD_INT("top_k", m.K), RAD_STR("dtype", "bf16"),
                                                              RAD_STR("act", "none")), RAD_NOWEIGHTS),
                             {m.b_dn, m.b_wts, m.b_sorted}, {m.b_mo});
            L.op_mlp_add = ew("add", T, m.H, {m.b_s, m.b_zc}, m.b_s);
            /* the controllers: (1 - tanh(F g)) x + tanh(G g) msg, g = D [rms x | rms msg] */
            if (L.core) {
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
                }
            }
        }
    }

    /* The drafter last: its passes run after the whole trunk, and the planner lays lifetimes out in
     * declaration order. Its layers take the absolute indices past the 48 applications. */
    if (m.have_ds) {
        DsparkInkling::Target t{};
        t.embed = m.w_embed;
        t.embed_norm = m.w_embed_norm;
        t.head = m.w_head;
        t.ck_head = nm.ckpt("%s.unembed.weight", pfx);
        RAD_ARCH_TRY(declare_dspark_inkling(&m.ds, &m.dsw, b, nm, meta, ctx, g, m.b_taps, kApps, t,
                                            m.VR, m.fcap));
    }
    return RAD_OK;
}


/* A controller on one input: the fourteen ops of Layer::ctl, in order. */

static void run_control(RadCtx* c, const Model& m, const Layer& L, int j, rad_buf x, rad_buf msg,
                        rad_buf y, int64_t T) {
    const int64_t MT = T;   /* the step's rows: the mixes' outputs are [max_tok, H], trimmed */
    const rad_op* o = L.ctl[j];
    /* the two norm halves: rms x in columns [0, H), rms msg in [H, 2H) */
    RAD_ISSUE_N(c, o[0], MT, brows(x, MT), RAD_NONE, bcol(m.b_gxm, 0, m.H, MT));
    RAD_ISSUE_N(c, o[1], MT, brows(msg, MT), RAD_NONE, bcol(m.b_gxm, m.H, m.H, MT));
    RAD_ISSUE_N(c, o[2], MT, brows(m.b_gxm, MT), RAD_W(L.w_dx[j]), brows(m.b_g1, MT));
    RAD_ISSUE_N(c, o[3], MT, brows(m.b_g1, MT), RAD_W(L.w_cgain[j]), brows(m.b_gp, MT));
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
static void issue_sconv(RadCtx* c, const Model& m, const RadBatch* batch, rad_op op, rad_weight w,
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
    (void)m;
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
    if (ip == 0) {
        RAD_ISSUE_N(c, L.op_norm[0], T, brows(m.b_s, T), RAD_W(L.w_norm[0]), brows(m.b_h[0], T));
        hin[1] = hin[2] = m.b_h[0];
    } else {
        for (int q = 0; q < L.n_mix; ++q)
            RAD_ISSUE_N(c, L.op_mix[q], T, RAD_B_N(m.b_blk, 0, L.n_state_attn), brows(m.b_s, T), RAD_NONE,
                        RAD_W(L.w_mix[q]), RAD_NONE, brows(m.b_y[q], T));
        if (!control) {
            if (L.full) {
                for (int j = 0; j < 3; ++j)
                    RAD_ISSUE_N(c, L.op_norm[j], T, brows(m.b_y[j], T), RAD_W(L.w_norm[j]), brows(m.b_h[j], T));
            } else {
                RAD_ISSUE_N(c, L.op_norm[0], T, brows(m.b_y[0], T), RAD_W(L.w_norm[0]), brows(m.b_h[0], T));
                hin[1] = hin[2] = m.b_h[0];
            }
        } else {
            for (int j = 0; j < 4; ++j)
                RAD_ISSUE_N(c, L.op_msg[j], T, RAD_B_N(m.b_bank0, 0, 1), brows(m.b_bank1, T), RAD_NONE,
                            RAD_W(L.w_msg[j]), RAD_NONE, brows(m.b_msg[j], T));
            for (int j = 0; j < 3; ++j) {
                const rad_buf x = m.b_y[L.full ? j : 0];
                run_control(c, m, L, j, x, m.b_msg[j], m.b_cy[j], T);
                RAD_ISSUE_N(c, L.op_norm_ctl[j], T, brows(m.b_cy[j], T), RAD_W(L.w_norm[L.full ? j : 0]),
                            brows(m.b_h[j], T));
            }
        }
    }

    /* the projections and the dynamic taps */
    RAD_ISSUE_N(c, L.op_q, T, brows(hin[0], T), RAD_W(L.w_q), brows(m.b_q, T));
    RAD_ISSUE_N(c, L.op_k, T, brows(hin[1], T), RAD_W(L.w_k), brows(m.b_k, T));
    RAD_ISSUE_N(c, L.op_v, T, brows(hin[2], T), RAD_W(L.w_v), brows(m.b_v, T));
    RAD_ISSUE_N(c, L.op_r, T, brows(hin[0], T), RAD_W(L.w_r), brows(m.b_r, T));
    RAD_ISSUE_N(c, L.op_gate, T, brows(hin[0], T), RAD_W(L.w_gate), brows(m.b_gate, T));
    RAD_ISSUE_N(c, L.op_sink, T, brows(hin[0], T), RAD_W(L.w_sink), brows(m.b_sink, T));
    if (L.full) {
        RAD_ISSUE_N(c, L.op_dk, T, brows(hin[1], T), RAD_W(L.w_dk), brows(m.b_dk, T));
        RAD_ISSUE_N(c, L.op_dv, T, brows(hin[2], T), RAD_W(L.w_dv), brows(m.b_dv, T));
    }
    issue_sconv(c, m, batch, L.op_kconv, L.w_kconv, m.cv_k, a, m.b_k, L.full ? m.b_dk : 0, m.b_kcv);
    issue_sconv(c, m, batch, L.op_vconv, L.w_vconv, m.cv_v, a, m.b_v, L.full ? m.b_dv : 0, m.b_vcv);

    /* the clips; the position tau on full layers */
    const RadOperand pos = L.full ? praw(batch->positions, RAD_I32, T) : RAD_NONE;
    RAD_ISSUE_N(c, L.op_qclip, T, brows(m.b_q, T), RAD_W(L.w_qscale), pos, brows(m.b_qc, T));
    RAD_ISSUE_N(c, L.op_rclip, T, brows(m.b_r, T), RAD_W(L.w_rscale), pos, brows(m.b_rc, T));
    RAD_ISSUE_N(c, L.op_kclip, T, brows(m.b_kcv, T), RAD_NONE, RAD_NONE, brows(m.b_kc, T));
    RAD_ISSUE_N(c, L.op_vnorm, T * m.KVH, brows(m.b_vcv, T), RAD_NONE, brows(m.b_vc, T));

    /* the cache, then the attention, the output gate and the output projection */
    RAD_ISSUE(c, L.op_store, brows(m.b_kc, T), brows(m.b_vc, T),
              praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T), kv_cache(grp, a));
    RAD_ISSUE_N(c, L.op_attn, batch->max_q_len, brows(m.b_qc, T), kv_cache(grp, a),
                praw2(kvb ? kvb->block_table : nullptr, RAD_I32, n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(kvb ? kvb->seqused : nullptr, RAD_I32, n_seq), RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, n_seq + 1), brows(m.b_rc, T), RAD_W(L.w_rel),
                brows(m.b_sink, T), brows(m.b_att, T));
    RAD_ISSUE(c, L.op_sig, brows(m.b_gate, T), brows(m.b_sig, T));
    RAD_ISSUE(c, L.op_gated, brows(m.b_att, T), brows(m.b_sig, T), brows(m.b_attg, T));
    RAD_ISSUE_N(c, L.op_o, T, brows(m.b_attg, T), RAD_W(L.w_o), brows(m.b_ao, T));
    issue_sconv(c, m, batch, L.op_aconv, L.w_aconv, m.cv_a, a, m.b_ao, 0, m.b_ax);

    /* the residual: a block push when this layer closes one, then the stream is the attention */
    if (L.push_slot >= 0) {
        RAD_ISSUE_N(c, L.op_push, MT, brows(m.b_s, MT), brows(m.b_ones, MT),
                  RAD_B_N(m.b_blk, (int64_t)L.push_slot * MT * m.H, 1));
        RAD_ISSUE(c, L.op_resid, brows(m.b_ax, T), brows(m.b_ones, T), brows(m.b_s, T));
    } else {
        RAD_ISSUE(c, L.op_attn_add, brows(m.b_s, T), brows(m.b_ax, T), brows(m.b_s, T));
    }

    /* the feed-forward: its mix, the controller, the norm, the router and the experts */
    RAD_ISSUE_N(c, L.op_mlp_mix, T, RAD_B_N(m.b_blk, 0, L.n_state_mlp), brows(m.b_s, T), RAD_NONE, RAD_W(L.w_mlp_q),
                RAD_NONE, brows(m.b_mx, T));
    if (control) {
        run_control(c, m, L, 3, m.b_mx, m.b_msg[3], m.b_cym, T);
        RAD_ISSUE_N(c, L.op_mlp_norm_ctl, T, brows(m.b_cym, T), RAD_W(L.w_mlp_norm), brows(m.b_fm, T));
    } else {
        RAD_ISSUE_N(c, L.op_mlp_norm, T, brows(m.b_mx, T), RAD_W(L.w_mlp_norm), brows(m.b_fm, T));
    }
    if (L.ex_form != kExpBf16)
        RAD_ISSUE_N(c, L.op_q8, T, brows(m.b_fm, T), brows(m.b_q8, T), brows(m.b_s8, T));
    RAD_ISSUE_N(c, L.op_router, T, brows(m.b_fm, T), RAD_W(L.w_router), brows(m.b_rl, T));
    RAD_ISSUE_N(c, L.op_topk, T, brows(m.b_rl, T), RAD_W(L.w_bias), brows(m.b_ids, T), brows(m.b_wts, T));
    RAD_ISSUE_N(c, L.op_scatter, T, brows(m.b_ids, T), RAD_B(m.b_sorted), RAD_B(m.b_eoff), RAD_B(m.b_ecnt));
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
        if (L.ex_form == kExpNvfp4) {
            /* `b` absent: gate_up is the packed [gate | up] row (the quantiser's a, b?, q, scale) */
            RAD_ISSUE_N(c, L.op_situ, T * m.K, brows(m.b_gu, T * m.K), RAD_NONE, brows(m.b_aq, T * m.K),
                        brows(m.b_as, T * m.K));
        } else {
            RAD_ISSUE_N(c, L.op_situ, T * m.K, brows(m.b_gu, T * m.K), brows(m.b_act, T * m.K));
            RAD_ISSUE_N(c, L.op_dq, T * m.K, brows(m.b_act, T * m.K), brows(m.b_aq, T * m.K),
                        brows(m.b_as, T * m.K));
        }
        RAD_ISSUE_N(c, L.op_edn, T, brows(m.b_aq, T * m.K), brows(m.b_as, T * m.K), RAD_WTAB(L.w_edn[0], h),
                    RAD_WTAB(L.w_edns[0], h), RAD_B(m.b_sorted), RAD_B(m.b_eoff),
                    RAD_WTAB(L.w_edn[1], h), RAD_WTAB(L.w_edns[1], h), brows(m.b_dn, T * m.K));
    }
    RAD_ISSUE_N(c, L.op_gather, T, brows(m.b_dn, T * m.K), RAD_B(m.b_wts), RAD_B(m.b_sorted), RAD_NONE,
                RAD_NONE, brows(m.b_mo, T));
    issue_sconv(c, m, batch, L.op_mconv, L.w_mconv, m.cv_m, a, m.b_mo, 0, m.b_zc);
    RAD_ISSUE(c, L.op_mlp_add, brows(m.b_s, T), brows(m.b_zc, T), brows(m.b_s, T));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    /* A drafter pass: positive is the query pass over [anchor, masks], negative the context pass
     * over the rows the trunk step just ran. */
    if (batch->draft_pass != 0 && m.have_ds) {
        if (batch->draft_pass > 0) m.ds.step(c, batch);
        else                       m.ds.context(c, batch);
        return;
    }
    if (batch->draft_pass || batch->spec_parent || batch->enc || batch->n_mm_rows > 0) {
        rad_step_fail(c, "inkling2: this step asks for an encoder, a draft pass, a tree verify or "
                         "media rows, and the program declares none of them");
        return;
    }
    const int64_t T = batch->n_tok;
    const int64_t MT = m.g.max_tok;
    RAD_ISSUE_N(c, m.op_ones, T, brows(m.b_ones, T));
    RAD_ISSUE_N(c, m.op_minus, T, brows(m.b_minus1, T));
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_embed), brows(m.b_s, T));
    RAD_ISSUE(c, m.op_embed_norm, brows(m.b_s, T), RAD_W(m.w_embed_norm), brows(m.b_s, T));

    for (int a = 0; a < kApps; ++a) {
        /* the prelude's output is kept for every pass to restart from; a pass boundary banks the
         * previous pass's block 2 and output, and restarts the stream from the prelude */
        if (a == 6) RAD_ISSUE(c, m.op_pre, brows(m.b_s, T), brows(m.b_ones, T), brows(m.b_pre, T));
        if (a == 18 || a == 30) {
            RAD_ISSUE(c, m.op_bank_cur, brows(m.b_s, T), brows(m.b_ones, T), brows(m.b_bank1, T));
            RAD_ISSUE_N(c, m.op_bank_blk, MT, RAD_B_N(m.b_blk, 2 * MT * m.H, 1), brows(m.b_ones, MT),
                      RAD_B_N(m.b_bank0, 0, 1));
            RAD_ISSUE(c, m.op_reset, brows(m.b_pre, T), brows(m.b_ones, T), brows(m.b_s, T));
        }
        const bool control = a >= 18 && a < 42;
        /* a drafter's tap at application a reads the stream entering it */
        for (int t = 0; t < m.n_taps; ++t)
            if (m.op_tap && m.tap_at[t] == a)
                RAD_ISSUE_N(c, m.op_tap, T, brows(m.b_s, T), brows(m.b_ones, T),
                            bcol(m.b_taps, (int64_t)t * m.H, m.H, T));
        run_layer(c, m, batch, a, control);
    }

    /* the output: a mix over the four blocks and the stream, the final norm, and the rows read */
    RAD_ISSUE_N(c, m.op_out_mix, T, RAD_B_N(m.b_blk, 0, 4), brows(m.b_s, T), RAD_NONE, RAD_W(m.w_out_q), RAD_NONE,
                brows(m.b_yo, T));
    RAD_ISSUE_N(c, m.op_out_norm, T, brows(m.b_yo, T), RAD_W(m.w_out_norm), brows(m.b_yn, T));
    for (int t = 0; t < m.n_taps; ++t)
        if (m.op_tap_out && m.tap_at[t] == kApps)
            RAD_ISSUE_N(c, m.op_tap_out, T, brows(m.b_yn, T), brows(m.b_ones, T),
                        bcol(m.b_taps, (int64_t)t * m.H, m.H, T));
    if (batch->n_out > 0) {
        const int64_t N = batch->n_out;
        RAD_ISSUE_N(c, m.op_rows, N, brows(m.b_yn, T), praw(batch->out_ids, RAD_I32, N), brows(m.b_hout, N));
        if (!batch->score_only) {
            RAD_ISSUE_N(c, m.op_head, N, brows(m.b_hout, N), RAD_W(m.w_head), brows(m.b_logits, N));
            RAD_ISSUE_N(c, m.op_softcap, N, brows(m.b_logits, N), brows(m.b_logits, N));
            if (m.op_pad)
                RAD_ISSUE_N(c, m.op_pad, N, bcol(m.b_logits, m.VR, m.VN - m.VR, N));
        }
    }
}

/* The speculation depth this container supports: zero without a drafter, else the DSpark block's
 * mask positions (block_size - 1), fixed because the drafter denoises exactly that many. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    out->draft_depth_fixed = 0;
    const long long bs = rad_meta_geti(meta, "draft.block_size", 0);
    if (bs <= 0) return RAD_OK;
    if (bs < 2) {
        fprintf(stderr, "radiance: inkling2: the drafter's block_size is %lld\n", bs);
        return RAD_E_FORMAT;
    }
    out->draft_depth = (int)(bs - 1);
    out->draft_depth_fixed = 1;
    return RAD_OK;
}

/* ============================================================== the reply format
 * The template writes every message as <|message_ROLE|><|content_KIND|>...<|end_message|>, and a
 * turn as the model's messages then <|content_model_end_sampling|> (the eos). A reply reads
 *   <|message_model|><|content_thinking|>REASONING<|end_message|><|message_model|><|content_text|>
 *   ANSWER<|end_message|><|content_model_end_sampling|>
 * where <|message_model|> and <|content_text|> render as nothing and the eos ends the turn. The
 * template analysis finds the reasoning block but reads the answer as plain text, so the answer's
 * own <|end_message|> stayed in it. With reasoning_anywhere a close with no block open is not part
 * of the answer, and a block opened later (the template interleaves thinking) is reasoning. No tool
 * calls: they are JSON in a tag, which the reply parser does not read. */
static const RadChatFormat* chat_format(const RadModelMeta*) {
    static const RadChatFormat f = [] {
        RadChatFormat c{};
        c.struct_size         = (uint32_t)sizeof(RadChatFormat);
        c.name                = "plugin:inkling2";
        c.reasoning_open      = "<|content_thinking|>";
        c.reasoning_close     = "<|end_message|>";
        c.reasoning_in_prompt = RAD_CHAT_THINK_FROM_PROMPT;
        c.reasoning_anywhere  = 1;
        return c;
    }();
    return &f;
}

}  /* namespace inkling2 */

#ifndef RAD_ARCH_NO_EXPORTS
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PROBE(inkling2)
RAD_ARCH_PLUGIN(inkling2, "inkling2", "", "0.1.0",
                "Inkling 2 Mini's text backbone: the loop checkpoint's 48 layer applications over 24 "
                "physical layers, block attention residuals, relative-position attention with a sink, "
                "stateful short convolutions, controllers and SiTU experts; bf16, one rank")
RAD_ARCH_CHAT_FORMAT(inkling2)
#endif
