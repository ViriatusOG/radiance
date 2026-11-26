/* qwen3asr_bf16 -- Qwen3-ASR: a Whisper-style audio encoder over a dense Qwen3 decoder, bf16
 * weights and bf16 activations.
 *
 * A speech recogniser served as the generative model it is: the clip goes through the audio
 * encoder (RadEncoderDecl, RAD_MM_AUDIO), its rows replace the prompt's <|audio_pad|>
 * placeholders, and the decoder writes the transcript, after "language <Name><asr_text>" when the
 * request forces a language and after naming it when it does not. POST /v1/audio/transcriptions
 * builds the prompt from rad_arch_audio_format (below); /v1/chat/completions takes the clip as an
 * input_audio part. The reference is transformers' modeling_qwen3_asr (5.18) with its
 * Qwen3ASRFeatureExtractor, which core/mm reproduces (AudioFamily::Qwen3ASR).
 *
 * ============================== The shape, at 1.7B ==============================
 *
 *   audio encoder: log-mel [frames, 128], convolved one second (100 frames) at a time: three
 *     Conv2d 3x3 / stride 2 / pad 1 over [mel, time] with 480 channels and exact gelu, 128 mel ->
 *     16 and 100 frames -> 13 rows; a row's 16 x 480 features through a 7680 -> 1024 linear (no
 *     bias); sinusoidal positions 0..12 added to each second's rows; 24 pre-LayerNorm layers (eps
 *     1e-5), d_model 1024, 16 heads of 64, biased q/k/v/o and FFN linears, FFN 4096 with exact
 *     gelu, attention bidirectional in blocks of 104 rows (n_window_infer 800 frames = 8 s) from
 *     the clip's start; a final LayerNorm.
 *   projector: 1024 -> 1024, gelu, 1024 -> 2048.
 *   decoder: Qwen3-1.7B: 28 layers, hidden 2048, 16 heads over 8 KV heads of 128 (GQA 2), per-head
 *     q/k RMS norms, NeoX rope over the whole head at theta 1e6, SwiGLU 6144, RMS eps 1e-6,
 *     vocab 151936, the lm_head tied to the embedding.
 *
 * ============================== What is easy to get wrong ==============================
 *
 * Each second is convolved on its own. The core lays a clip out so one convolution over it does
 * that (core/mm/processor.h, AudioFamily::Qwen3ASR): a whole second is staged as 104 frames whose
 * last four read as zero (RadBatch::enc_coord plane 1), and the last, shorter second as 8 x the
 * rows the reference keeps of it. `conv2d`'s `valid` at strides 1, 2 and 4 keeps one second's
 * convolution out of the next's, and every row the tower writes is a row the reference keeps, so
 * nothing is compacted.
 *
 * The convolutions run over [mel, time], so PyTorch's kernel is [cout, cin, mel, time] with time
 * last, where `conv2d` convolves (time, width); `t_last` reads the checkpoint as it is. The 7680 ->
 * 1024 linear takes its input channel-major (c * 16 + f) where the convolution writes
 * frequency-major (f * 480 + c); as a 1 x 16 convolution over the frequency axis, the checkpoint's
 * [1024, 7680] is the kernel [1024, 480, 1, 16] in PyTorch's layout, and nothing is permuted.
 *
 * The 480-channel convolutions are GEMMs. Two of them and the 7680 -> 1024 linear are 95% of the
 * tower's arithmetic outside its transformer (68 GMAC for 30 s), and a direct convolution re-reads
 * the weights for every output. `conv2d_unfold` writes each output position's receptive field as
 * a row, its columns in the weight's own order, so `gemm_nt_bias` against the checkpoint's kernel
 * read as [cout, cin * 9] (a reshape, not a relayout) is the convolution, with the bias and the
 * gelu in its epilogue. The first convolution has one input channel and nine taps, too narrow a K
 * for a GEMM, so it stays a `conv2d`.
 *
 * Positions restart every 13 rows (`seg_sinusoid`, per clip), and attention runs in blocks of 104
 * rows from the clip's start, the last one short (`attn_dense`'s `block`). A clip is one encoder
 * segment, so the segment bounds (in frames, read at cu_div 8) are the clip's.
 *
 * The decoder's positions are the tokens' indices. The checkpoint's text_config is plain Qwen3
 * (no mrope_section), and the reference's position ids count the placeholders like text.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_attn.h>
#include <arch/rad_block_mlp.h>
#include <arch/rad_block_vit.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace qwen3asr_bf16 {

using namespace rad::arch;

/* The longest clip one encoder pass carries, in seconds: the reference extractor's own chunk
 * length (`chunk_length`), and the clip the transcription route cuts longer audio into. */
constexpr int64_t kPassSeconds = 30;

/* Rows a chunk of `frames` keeps: three convolutions of kernel 3, stride 2, padding 1. */
static int64_t post_cnn(int64_t frames) {
    for (int i = 0; i < 3; ++i) frames = frames > 0 ? (frames - 1) / 2 + 1 : 0;
    return frames;
}

/* ============================== THE AUDIO ENCODER ============================== */
struct AudioTower {
    struct Layer {
        rad_weight ln1w = 0, ln1b = 0, qkv_w = 0, qkv_b = 0, o_w = 0, o_b = 0;
        rad_weight ln2w = 0, ln2b = 0, fc1_w = 0, fc1_b = 0, fc2_w = 0, fc2_b = 0;
        rad_op op_ln1 = 0, op_qkv = 0, op_attn = 0, op_o = 0, op_ln2 = 0, op_fc1 = 0, op_fc2 = 0;
    };
    bool on = false;
    int64_t P = 0, R = 0, H = 0, heads = 0, hd = 0, ffn = 0, C = 0, nmel = 0, f3 = 0, w3 = 0;
    int64_t chunk_rows = 0, block = 0, n_embd = 0;
    static constexpr int64_t kMerge = 8;   /* three stride-2 convolutions: 8 frames a row */
    rad_weight w_c1 = 0, b_c1 = 0, w_c2 = 0, b_c2 = 0, w_c3 = 0, b_c3 = 0, w_co = 0;
    rad_weight ln_w = 0, ln_b = 0, p1_w = 0, p1_b = 0, p2_w = 0, p2_b = 0;
    rad_op op_c1 = 0, op_u2 = 0, op_c2 = 0, op_u3 = 0, op_c3 = 0, op_uo = 0, op_co = 0;
    rad_op op_pos = 0, op_add_pos = 0;
    rad_op op_ln = 0, op_p1 = 0, op_p2 = 0;
    rad_buf b_s1 = 0, b_u = 0, b_s2 = 0, b_s3 = 0, b_uo = 0, b_x = 0, b_h = 0, b_pos = 0;
    rad_buf b_qkv = 0, b_att = 0;
    rad_buf b_f = 0, b_p1 = 0, b_out = 0;
    std::vector<Layer> L;

    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "audio_config.model_type", "");
        return t && !std::strcmp(t, "qwen3_asr_encoder");
    }
    /* The frames a pass of kPassSeconds stages: whole seconds of merge x 13 rows each. */
    static int64_t frames_for(const RadModelMeta* m) {
        const int64_t nw = rad_meta_geti(m, "audio_config.n_window", 0);
        if (nw <= 0) return 0;
        return kPassSeconds * kMerge * post_cnn(2 * nw);
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group) {
        on = false;
        if (!present(m) || ctx->max_audio_frames <= 0) return RAD_OK;
        auto gi = [&](const char* k, int64_t d) {
            char key[96];
            std::snprintf(key, sizeof key, "audio_config.%s", k);
            return rad_meta_geti(m, key, d);
        };
        H     = gi("d_model", 0);
        heads = gi("encoder_attention_heads", 0);
        ffn   = gi("encoder_ffn_dim", 0);
        C     = gi("downsample_hidden_size", 0);
        nmel  = gi("num_mel_bins", 0);
        const int64_t depth  = gi("encoder_layers", 0);
        const int64_t nw     = gi("n_window", 0);
        const int64_t nwi    = gi("n_window_infer", 0);
        const int64_t out    = gi("output_dim", 0);
        const int64_t kvh    = gi("num_key_value_heads", heads);
        const char*   act    = rad_meta_gets(m, "audio_config.activation_function", "");
        hd = heads > 0 ? H / heads : 0;
        f3 = ((((nmel + 1) / 2) + 1) / 2 + 1) / 2;
        chunk_rows = post_cnn(2 * nw);
        block = nw > 0 && nwi % (2 * nw) == 0 ? chunk_rows * (nwi / (2 * nw)) : 0;
        /* What this tower does not carry is refused by name: a scaled embedding, grouped heads,
         * an activation other than exact gelu, a projector into some other width. */
        if (H <= 0 || heads <= 0 || hd * heads != H || ffn <= 0 || C <= 0 || nmel <= 0 ||
            depth <= 0 || nw <= 0 || block <= 0 || kvh != heads || out != model_embd ||
            std::strcmp(act ? act : "", "gelu") ||
            rad_meta_geti(m, "audio_config.scale_embedding", 0)) {
            fprintf(stderr, "radiance: qwen3asr: the container's audio_config is not the encoder "
                            "this plugin serves (d_model %lld, heads %lld, ffn %lld, channels %lld, "
                            "mel %lld, layers %lld, n_window %lld/%lld, output %lld for a %lld-wide "
                            "decoder, act %s)\n", (long long)H, (long long)heads, (long long)ffn,
                    (long long)C, (long long)nmel, (long long)depth, (long long)nw,
                    (long long)nwi, (long long)out, (long long)model_embd, act ? act : "");
            return RAD_E_UNSUPPORTED;
        }
        n_embd = model_embd;
        P = ctx->max_audio_frames;
        if (P % kMerge) return RAD_E_SHAPE;
        R = P / kMerge;
        const int AC = RAD_ACCESS_PER_REQUEST, N = RAD_SHARD_NONE;
        const char* T = "model.audio_tower";

        auto w = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                     RadWeightGroup g) -> rad_weight {
            if (map_copy(b, dn, src) < 0) return 0;
            return decl_w(b, dn, RAD_BF16, shp, AC, N, g);
        };
        /* q, k and v into one projection, weights and biases alike: one GEMM a layer, and the
         * attention reads three column slices of its row. */
        auto wcat = [&](const char* dn, std::initializer_list<const char*> src,
                        std::initializer_list<int64_t> shp, RadWeightGroup g) -> rad_weight {
            if (map_concat(b, dn, src, 0) < 0) return 0;
            return decl_w(b, dn, RAD_BF16, shp, AC, N, g);
        };
        w_c1 = w(nm.f("a.conv1.w"), nm.ckpt("%s.conv2d1.weight", T), {C, 1, 3, 3}, grp_model());
        b_c1 = w(nm.f("a.conv1.b"), nm.ckpt("%s.conv2d1.bias", T),   {C}, grp_model());
        /* The two 480-channel kernels as the [cout, cin * 9] matrices their GEMMs read: the
         * checkpoint's [C, C, 3, 3], row-major, is that matrix. */
        w_c2 = w(nm.f("a.conv2.w"), nm.ckpt("%s.conv2d2.weight", T), {C, C * 9}, grp_model());
        b_c2 = w(nm.f("a.conv2.b"), nm.ckpt("%s.conv2d2.bias", T),   {C}, grp_model());
        w_c3 = w(nm.f("a.conv3.w"), nm.ckpt("%s.conv2d3.weight", T), {C, C * 9}, grp_model());
        b_c3 = w(nm.f("a.conv3.b"), nm.ckpt("%s.conv2d3.bias", T),   {C}, grp_model());
        w_co = w(nm.f("a.conv_out.w"), nm.ckpt("%s.conv_out.weight", T), {H, C * f3}, grp_model());
        ln_w = w(nm.f("a.ln_post.w"), nm.ckpt("%s.ln_post.weight", T), {H}, grp_model());
        ln_b = w(nm.f("a.ln_post.b"), nm.ckpt("%s.ln_post.bias", T),   {H}, grp_model());
        p1_w = w(nm.f("a.proj1.w"), nm.ckpt("model.multi_modal_projector.linear_1.weight"), {H, H}, grp_model());
        p1_b = w(nm.f("a.proj1.b"), nm.ckpt("model.multi_modal_projector.linear_1.bias"),   {H}, grp_model());
        p2_w = w(nm.f("a.proj2.w"), nm.ckpt("model.multi_modal_projector.linear_2.weight"), {n_embd, H}, grp_model());
        p2_b = w(nm.f("a.proj2.b"), nm.ckpt("model.multi_modal_projector.linear_2.bias"),   {n_embd}, grp_model());
        if (!w_c1 || !b_c1 || !w_c2 || !b_c2 || !w_c3 || !b_c3 || !w_co || !ln_w || !ln_b ||
            !p1_w || !p1_b || !p2_w || !p2_b)
            return RAD_E_INVAL;
        L.assign((size_t)depth, Layer{});
        for (int64_t i = 0; i < depth; ++i) {
            Layer& l = L[(size_t)i];
            const RadWeightGroup g = grp_layer(first_group + (int)i);
            const int li = (int)i;
            const char* S = "model.audio_tower.layers";
            l.ln1w  = w(nm.f("a.blk.%d.ln1.w", li), nm.ckpt("%s.%d.self_attn_layer_norm.weight", S, li), {H}, g);
            l.ln1b  = w(nm.f("a.blk.%d.ln1.b", li), nm.ckpt("%s.%d.self_attn_layer_norm.bias", S, li),   {H}, g);
            l.qkv_w = wcat(nm.f("a.blk.%d.qkv.w", li),
                           { nm.ckpt("%s.%d.self_attn.q_proj.weight", S, li),
                             nm.ckpt("%s.%d.self_attn.k_proj.weight", S, li),
                             nm.ckpt("%s.%d.self_attn.v_proj.weight", S, li) }, {3 * H, H}, g);
            l.qkv_b = wcat(nm.f("a.blk.%d.qkv.b", li),
                           { nm.ckpt("%s.%d.self_attn.q_proj.bias", S, li),
                             nm.ckpt("%s.%d.self_attn.k_proj.bias", S, li),
                             nm.ckpt("%s.%d.self_attn.v_proj.bias", S, li) }, {3 * H}, g);
            l.o_w   = w(nm.f("a.blk.%d.o.w", li),   nm.ckpt("%s.%d.self_attn.out_proj.weight", S, li), {H, H}, g);
            l.o_b   = w(nm.f("a.blk.%d.o.b", li),   nm.ckpt("%s.%d.self_attn.out_proj.bias", S, li),   {H}, g);
            l.ln2w  = w(nm.f("a.blk.%d.ln2.w", li), nm.ckpt("%s.%d.final_layer_norm.weight", S, li),   {H}, g);
            l.ln2b  = w(nm.f("a.blk.%d.ln2.b", li), nm.ckpt("%s.%d.final_layer_norm.bias", S, li),     {H}, g);
            l.fc1_w = w(nm.f("a.blk.%d.fc1.w", li), nm.ckpt("%s.%d.fc1.weight", S, li), {ffn, H}, g);
            l.fc1_b = w(nm.f("a.blk.%d.fc1.b", li), nm.ckpt("%s.%d.fc1.bias", S, li),   {ffn}, g);
            l.fc2_w = w(nm.f("a.blk.%d.fc2.w", li), nm.ckpt("%s.%d.fc2.weight", S, li), {H, ffn}, g);
            l.fc2_b = w(nm.f("a.blk.%d.fc2.b", li), nm.ckpt("%s.%d.fc2.bias", S, li),   {H}, g);
            if (!l.ln1w || !l.ln1b || !l.qkv_w || !l.qkv_b || !l.o_w || !l.o_b || !l.ln2w ||
                !l.ln2b || !l.fc1_w || !l.fc1_b || !l.fc2_w || !l.fc2_b)
                return RAD_E_INVAL;
        }

        /* ---- activations: the convolutions' planes at their own widths, (time, mel, channel)
         * with the channel innermost; the rest at R rows. */
        /* The second and third convolutions' outputs are their GEMMs' [positions, C], which is the
         * same bytes as [time rows, width * C]: the next unfold reads them that way. */
        const int64_t w2 = (nmel + 1) / 2;
        w3 = (w2 + 1) / 2;
        b_s1  = decl_b(b, nm.f("a.s1"),  RAD_BF16, {P / 2, w2 * C});
        b_u   = decl_b(b, nm.f("a.unfold"), RAD_BF16, {P / 4 * w3, 9 * C});
        b_s2  = decl_b(b, nm.f("a.s2"),  RAD_BF16, {P / 4 * w3, C});
        b_s3  = decl_b(b, nm.f("a.s3"),  RAD_BF16, {R * f3, C});
        b_uo  = decl_b(b, nm.f("a.unfold_out"), RAD_BF16, {R, f3 * C});
        b_x   = decl_b(b, nm.f("a.x"),   RAD_BF16, {R, H});
        b_h   = decl_b(b, nm.f("a.h"),   RAD_BF16, {R, H});
        b_pos = decl_b(b, nm.f("a.pos"), RAD_BF16, {R, H});
        b_qkv = decl_b(b, nm.f("a.qkv"), RAD_BF16, {R, 3 * H});
        b_att = decl_b(b, nm.f("a.att"), RAD_BF16, {R, H});
        b_f   = decl_b(b, nm.f("a.f"),   RAD_BF16, {R, ffn});
        b_p1  = decl_b(b, nm.f("a.p1"),  RAD_BF16, {R, H});
        b_out = decl_b(b, nm.f("a.out"), RAD_BF16, {R, n_embd});

        /* The first convolution has one input channel, so it is direct: nine taps are too narrow
         * a K for a GEMM. */
        auto conv1 = [&](rad_weight wt, rad_weight bias) {
            return RAD_OP(b, "conv2d",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("width", nmel),
                                     RAD_INT("cin", 1), RAD_INT("cout", C), RAD_INT("kh", 3),
                                     RAD_INT("kw", 3), RAD_INT("stride", 2), RAD_INT("pad", 1),
                                     RAD_INT("cu_div", 1), RAD_INT("valid_stride", 1),
                                     RAD_STR("dtype", "bf16"), RAD_INT("t_last", 1),
                                     RAD_STR("act", "gelu")),
                          RAD_WEIGHTS(wt, bias));
        };
        auto gemm_m = [&](int64_t rows, int64_t Nn, int64_t K, const char* act_, rad_weight wt,
                          rad_weight bias) {
            return RAD_OP(b, "gemm_nt_bias",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", Nn), RAD_INT("K", K),
                                     RAD_STR("dtype", "bf16"), RAD_STR("act", act_)),
                          RAD_WEIGHTS(wt, bias));
        };
        auto gemm = [&](int64_t Nn, int64_t K, const char* act_, rad_weight wt, rad_weight bias) {
            return gemm_m(R, Nn, K, act_, wt, bias);
        };
        auto unfold = [&](int64_t rows, int64_t width, int64_t kh, int64_t kw, int64_t stride,
                          int64_t pad, int64_t cu_div, int tlast) {
            return RAD_OP(b, "conv2d_unfold",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("width", width),
                                     RAD_INT("cin", C), RAD_INT("kh", kh), RAD_INT("kw", kw),
                                     RAD_INT("stride", stride), RAD_INT("pad", pad),
                                     RAD_INT("cu_div", cu_div), RAD_INT("valid_stride", cu_div),
                                     RAD_INT("t_last", tlast), RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS);
        };
        auto lnorm = [&](rad_weight wt, rad_weight bias) {
            return RAD_OP(b, "layernorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H), RAD_F64("eps", 1e-5),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(wt, bias));
        };

        op_c1 = rw(b, conv1(w_c1, b_c1), {}, {b_s1});
        op_u2 = rw(b, unfold(P / 2, w2, 3, 3, 2, 1, 2, 1), {b_s1}, {b_u});
        op_c2 = rw(b, gemm_m(P / 4 * w3, C, 9 * C, "gelu", w_c2, b_c2), {b_u}, {b_s2});
        op_u3 = rw(b, unfold(P / 4, w3, 3, 3, 2, 1, 4, 1), {b_s2}, {b_u});
        op_c3 = rw(b, gemm_m(R * f3, C, 9 * C, "gelu", w_c3, b_c3), {b_u}, {b_s3});
        /* The 7680 -> 1024 linear as a 1 x f3 convolution over the frequency axis, unfolded: its
         * rows are (channel, frequency), the checkpoint's own [H, C * f3] column order. */
        op_uo = rw(b, unfold(R, f3, 1, f3, 1, 0, kMerge, 0), {b_s3}, {b_uo});
        op_co = rw(b, RAD_OP(b, "gemm_nt",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("N", H),
                                        RAD_INT("K", C * f3), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w_co)),
                   {b_uo}, {b_x});
        op_pos = rw(b, RAD_OP(b, "seg_sinusoid",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H),
                                         RAD_INT("period", chunk_rows),
                                         RAD_F64("max_timescale", 10000.0),
                                         RAD_INT("cu_div", kMerge), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    {}, {b_pos});
        op_add_pos = rw(b, RAD_OP(b, "add",
                                  RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS),
                        {b_x, b_pos}, {b_x});
        for (Layer& l : L) {
            l.op_ln1 = rw(b, lnorm(l.ln1w, l.ln1b), {b_x}, {b_h});
            l.op_qkv = rw(b, gemm(3 * H, H, "none", l.qkv_w, l.qkv_b), {b_h}, {b_qkv});
            l.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                     RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                                RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                                RAD_STR("dtype", "bf16"),
                                                RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                                RAD_INT("max_seqlen", R),
                                                RAD_INT("cu_div", kMerge),
                                                RAD_INT("block", block)),
                                     RAD_NOWEIGHTS),
                           {b_qkv}, {b_att});
            /* The output projection and the second FFN linear add the residual in their epilogue,
             * rounded after the biased product as the reference's two tensors are. */
            l.op_o   = rw(b, gemm(H, H, "none", l.o_w, l.o_b), {b_att, b_x}, {b_x});
            l.op_ln2 = rw(b, lnorm(l.ln2w, l.ln2b), {b_x}, {b_h});
            l.op_fc1 = rw(b, gemm(ffn, H, "gelu", l.fc1_w, l.fc1_b), {b_h}, {b_f});
            l.op_fc2 = rw(b, gemm(H, ffn, "none", l.fc2_w, l.fc2_b), {b_f, b_x}, {b_x});
        }
        op_ln = rw(b, lnorm(ln_w, ln_b), {b_x}, {b_h});
        op_p1 = rw(b, gemm(H, H, "gelu", p1_w, p1_b), {b_h}, {b_p1});
        op_p2 = rw(b, gemm(n_embd, H, "none", p2_w, p2_b), {b_p1}, {b_out});

        if (!ctx->shape_probe) {
            RadEncoderDecl e{};
            e.name        = "audio";
            e.modalities  = RAD_MM_AUDIO;
            e.patch_dim   = nmel;
            e.merge       = kMerge;
            e.max_patches = P;
            e.n_embd      = n_embd;
            e.out         = b_out;
            RAD_ARCH_TRY(rad_declare_encoder(b, &e));
            rad_note(b, "audio encoder: 3 convolutions of %lld channels over %lld mel bins, %lld "
                        "layers of %lld (%lld heads x %lld), ffn %lld, attention in blocks of %lld "
                        "rows; %lld frames a pass (%lld s) -> %lld rows of %lld",
                     (long long)C, (long long)nmel, (long long)depth, (long long)H,
                     (long long)heads, (long long)hd, (long long)ffn, (long long)block,
                     (long long)P, (long long)kPassSeconds, (long long)R, (long long)n_embd);
        }
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
        if (!on || Pn <= 0 || Pn > P || Pn % kMerge || S <= 0) {
            rad_step_fail(c, "an audio pass this program's audio encoder cannot serve: none "
                             "declared, more frames than it was declared for, or not whole rows");
            return;
        }
        const int64_t Rn = Pn / kMerge;
        const RadOperand frames = praw2(batch->enc_pixels, RAD_BF16, Pn, nmel);
        /* Plane 1: whether a frame takes part in the convolution (item.cpp). */
        const RadOperand valid = praw(batch->enc_coord + Pn, RAD_I32, Pn);
        const RadOperand cu = praw(batch->enc_cu, RAD_I32, S + 1);

        RAD_ISSUE_N(c, op_c1, Pn, frames, RAD_W(w_c1), cu, valid, brows(b_s1, Pn / 2),
                    RAD_W(b_c1));
        const int64_t n2 = Pn / 4 * w3, n3 = Rn * f3;
        RAD_ISSUE_N(c, op_u2, Pn / 2, brows(b_s1, Pn / 2), cu, valid, brows(b_u, n2));
        RAD_ISSUE_N(c, op_c2, n2, brows(b_u, n2), RAD_W(w_c2), RAD_W(b_c2), RAD_NONE,
                    brows(b_s2, n2));
        RAD_ISSUE_N(c, op_u3, Pn / 4, brows(b_s2, n2), cu, valid, brows(b_u, n3));
        RAD_ISSUE_N(c, op_c3, n3, brows(b_u, n3), RAD_W(w_c3), RAD_W(b_c3), RAD_NONE,
                    brows(b_s3, n3));
        RAD_ISSUE_N(c, op_uo, Rn, brows(b_s3, n3), cu, RAD_NONE, brows(b_uo, Rn));
        RAD_ISSUE_N(c, op_co, Rn, brows(b_uo, Rn), RAD_W(w_co), brows(b_x, Rn));
        RAD_ISSUE_N(c, op_pos, Rn, cu, brows(b_pos, Rn));
        RAD_ISSUE_N(c, op_add_pos, Rn, brows(b_x, Rn), brows(b_pos, Rn), brows(b_x, Rn));
        for (const Layer& l : L) {
            RAD_ISSUE_N(c, l.op_ln1, Rn, brows(b_x, Rn), RAD_W(l.ln1w), RAD_W(l.ln1b),
                        brows(b_h, Rn));
            RAD_ISSUE_N(c, l.op_qkv, Rn, brows(b_h, Rn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                        brows(b_qkv, Rn));
            RAD_ISSUE_N(c, l.op_attn, Rn, bcol(b_qkv, 0, H, Rn), bcol(b_qkv, H, H, Rn),
                        bcol(b_qkv, 2 * H, H, Rn), cu, brows(b_att, Rn));
            RAD_ISSUE_N(c, l.op_o, Rn, brows(b_att, Rn), RAD_W(l.o_w), RAD_W(l.o_b),
                        brows(b_x, Rn), brows(b_x, Rn));
            RAD_ISSUE_N(c, l.op_ln2, Rn, brows(b_x, Rn), RAD_W(l.ln2w), RAD_W(l.ln2b),
                        brows(b_h, Rn));
            RAD_ISSUE_N(c, l.op_fc1, Rn, brows(b_h, Rn), RAD_W(l.fc1_w), RAD_W(l.fc1_b), RAD_NONE,
                        brows(b_f, Rn));
            RAD_ISSUE_N(c, l.op_fc2, Rn, brows(b_f, Rn), RAD_W(l.fc2_w), RAD_W(l.fc2_b),
                        brows(b_x, Rn), brows(b_x, Rn));
        }
        RAD_ISSUE_N(c, op_ln, Rn, brows(b_x, Rn), RAD_W(ln_w), RAD_W(ln_b), brows(b_h, Rn));
        RAD_ISSUE_N(c, op_p1, Rn, brows(b_h, Rn), RAD_W(p1_w), RAD_W(p1_b), RAD_NONE,
                    brows(b_p1, Rn));
        RAD_ISSUE_N(c, op_p2, Rn, brows(b_p1, Rn), RAD_W(p2_w), RAD_W(p2_b), RAD_NONE,
                    brows(b_out, Rn));
    }
};

/* ============================== THE DECODER ============================== */
struct Layer {
    AttnBF16 attn;
    MlpBF16  mlp;
};

struct Model {
    Names nm{""};
    Geom  g{};
    rad_kvgroup kv = 0;
    rad_buf b_x = 0, b_h = 0, b_q = 0, b_k = 0, b_v = 0, b_attn = 0;
    rad_buf b_gate_up = 0, b_ffn = 0, b_hout = 0, b_logits = 0;
    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;
    bool media = false;
    MediaRows mrows{};
    AudioTower aud{};
    std::vector<Layer> layers;
};

static Model g_model[MAX_SLOTS];

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

    /* Plain Qwen3: every layer full attention, no sliding window, plain RMS gains. */
    if (meta->n_expert > 0 || rad_meta_geti(meta, "text_config.use_sliding_window", 0)) {
        fprintf(stderr, "radiance: qwen3asr: the decoder has experts or a sliding window; this "
                        "plugin serves the dense, full-attention Qwen3 decoder\n");
        return RAD_E_UNSUPPORTED;
    }
    if (!(g.theta > 0.0f)) g.theta = (float)rad_meta_getf(meta, "rope_parameters.rope_theta", 0.0);
    if (!(g.theta > 0.0f) || g.head_dim <= 0 || g.n_head % g.n_head_kv) {
        fprintf(stderr, "radiance: qwen3asr: the decoder's geometry is incomplete (theta %g, "
                        "head_dim %lld, heads %lld over %lld)\n", (double)g.theta,
                (long long)g.head_dim, (long long)g.n_head, (long long)g.n_head_kv);
        return RAD_E_INVAL;
    }
    g.wadd = 0.0f;
    g.wadd_qk = 0.0f;

    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    m.b_x       = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h       = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_q       = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, g.q_dim()});
    m.b_k       = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, g.kv_dim()});
    m.b_v       = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, g.kv_dim()});
    m.b_attn    = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, g.q_dim()});
    m.b_gate_up = decl_b(b, nm.f("gate_up"),  g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn     = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});
    m.b_hout    = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits  = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both row-sharded; the head is tied to the embedding
     * (tie_word_embeddings, and the checkpoint ships no lm_head.weight). */
    const char* pre = "model.language_model";
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", pre)));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.norm.weight", pre)));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("%s.embed_tokens.weight", pre)));
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
    /* The encoder's rows, over the lookup and before its reduction (MediaRows says why). */
    m.media = AudioTower::present(meta) && ctx->max_audio_frames > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, g.n_embd, m.b_x));
    if (g.world > 1)
        m.op_embed_ar = decl_all_reduce_exact(b, g, m.b_x);

    AttnBF16::Wire aw{};
    aw.x = m.b_x; aw.h = m.b_h; aw.q = m.b_q; aw.k = m.b_k; aw.v = m.b_v; aw.attn = m.b_attn;
    MlpBF16::Wire mw{};
    mw.x = m.b_x; mw.h = m.b_h; mw.gate_up = m.b_gate_up; mw.ffn = m.b_ffn;
    m.layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        const char* lp = "model.language_model.layers";
        const long long li = (long long)l;
        AttnBF16::Src as{};
        as.norm   = nm.ckpt("%s.%lld.input_layernorm.weight",  lp, li);
        as.q      = nm.ckpt("%s.%lld.self_attn.q_proj.weight", lp, li);
        as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight", lp, li);
        as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight", lp, li);
        as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight", lp, li);
        as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight", lp, li);
        as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight", lp, li);
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv));
        RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv, aw, as));
        MlpBF16::Src ms{};
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", lp, li);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            lp, li);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              lp, li);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            lp, li);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms));
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

    rad_note(b, "rank %d/%d: %lld decoder layers, n_head=%lld n_head_kv=%lld head_dim=%lld "
                "n_ff=%lld theta=%g vocab=[%lld,%lld), lm_head tied",
             g.rank, g.world, (long long)g.n_layer, (long long)g.n_head, (long long)g.n_head_kv,
             (long long)g.head_dim, (long long)g.n_ff, (double)g.theta, (long long)g.vocab_off,
             (long long)(g.vocab_off + g.n_vocab));

    /* The audio tower last, so its activations share the arena with every buffer above. */
    if (m.media) RAD_ARCH_TRY(m.aud.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];
    /* An encoder pass runs the audio tower and nothing else (RadBatch::enc). */
    if (batch->enc) {
        m.aud.step(c, batch);
        return;
    }
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    m.mrows.step(c, batch, m.b_x, T);
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * m.g.n_embd, brows(m.b_x, T), RAD_NONE);
    for (const Layer& l : m.layers) {
        l.attn.step(c, batch);
        l.mlp.step(c, batch);
    }
    RAD_ISSUE(c, m.op_final_norm, brows(m.b_x, T), RAD_W(m.w_out_norm), brows(m.b_h, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.b_h, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out, brows(m.b_hout, batch->n_out),
                    RAD_W(m.w_lm_head), brows(m.b_logits, batch->n_out));
    }
}

/* Before declare: the frames a pass carries, so the core sizes the audio encoder's pass. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (AudioTower::present(meta)) out->audio_frames = AudioTower::frames_for(meta);
    return RAD_OK;
}

/* Qwen3-ASR's audio format (RadAudioFormat): its clip markers; its transcription turn, which is
 * the reference processor's chat template with the request's prompt as the system text and the
 * clip as the user turn; and the forced-language prefix the model was trained with,
 * "language <Name><asr_text>". The language names are the reference processor's
 * LANGUAGE_CODE_TO_NAME. */
static const char* const kLanguages[] = {
    "ar", "Arabic",     "yue", "Cantonese",  "zh", "Chinese",   "cs", "Czech",
    "da", "Danish",     "nl", "Dutch",       "en", "English",   "fil", "Filipino",
    "fi", "Finnish",    "fr", "French",      "de", "German",    "el", "Greek",
    "hi", "Hindi",      "hu", "Hungarian",   "id", "Indonesian", "it", "Italian",
    "ja", "Japanese",   "ko", "Korean",      "mk", "Macedonian", "ms", "Malay",
    "fa", "Persian",    "pl", "Polish",      "pt", "Portuguese", "ro", "Romanian",
    "ru", "Russian",    "es", "Spanish",     "sv", "Swedish",   "th", "Thai",
    "tr", "Turkish",    "vi", "Vietnamese",
};

static const RadAudioFormat* audio_format(const RadModelMeta* meta) {
    if (!AudioTower::present(meta)) return nullptr;
    static const RadAudioFormat f = {
        (uint32_t)sizeof(RadAudioFormat),
        "<|audio_start|>",
        "<|audio_end|>",
        "<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n{audio}<|im_end|>\n"
        "<|im_start|>assistant\n{language}",
        "language {language}<asr_text>",
        "<asr_text>",
        "language ",
        kLanguages,
        (uint32_t)(sizeof(kLanguages) / sizeof(kLanguages[0]) / 2),
    };
    return &f;
}

}  /* namespace qwen3asr_bf16 */

#ifndef RAD_ARCH_NO_EXPORTS
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(qwen3asr_bf16, "qwen3asr", "", "0.1.0",
                "Qwen3-ASR: a Whisper-style audio encoder over a dense Qwen3 decoder; bf16 weights "
                "and activations")
RAD_ARCH_PROBE(qwen3asr_bf16)
RAD_ARCH_AUDIO_FORMAT(qwen3asr_bf16)
#endif
