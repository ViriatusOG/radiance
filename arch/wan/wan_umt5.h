/* The UMT5-XXL conditioning encoder shared by Wan text-to-video and image-to-video.
 * Prompts are packed through the encoder. The transformer pads the encoded rows
 * separately for each prompt before its text projection. */
#pragma once

#include <arch/rad_arch.h>

#include "rad_umt5_attention.h"

#include <array>

namespace wan {

struct Umt5Encoder {
    static constexpr int64_t width = 4096;
    static constexpr int64_t heads = 64;
    static constexpr int64_t ffn_width = 10240;
    static constexpr int64_t vocab = 256384;
    static constexpr int64_t max_rows = 2 * rad_umt5::max_tokens;
    static constexpr int layer_count = 24;
    // Keep request-only UMT5 units apart from the DiT's per-step layers and the VAE.
    static constexpr int first_weight_group = 64;

    rad_buf encoded = 0;

    int declare(RadBuilder* b, rad::arch::Names& names) {
        using namespace rad::arch;
        state = decl_b(b, names.f("umt5.state"), RAD_BF16, {max_rows, width});
        norm = decl_b(b, names.f("umt5.norm"), RAD_BF16, {max_rows, width});
        scaled = decl_b(b, names.f("umt5.scaled"), RAD_BF16, {max_rows, width});
        qkv = decl_b(b, names.f("umt5.qkv"), RAD_BF16, {max_rows, 3 * width});
        attended = decl_b(b, names.f("umt5.attended"), RAD_BF16, {max_rows, width});
        residual = decl_b(b, names.f("umt5.residual"), RAD_BF16, {max_rows, width});
        gate_up = decl_b(b, names.f("umt5.gate_up"), RAD_BF16, {max_rows, 2 * ffn_width});
        activated = decl_b(b, names.f("umt5.activated"), RAD_BF16, {max_rows, ffn_width});
        gated = decl_b(b, names.f("umt5.gated"), RAD_BF16, {max_rows, ffn_width});
        encoded = decl_b(b, names.f("umt5.encoded"), RAD_BF16, {max_rows, width});
        if (!state || !norm || !scaled || !qkv || !attended || !residual || !gate_up ||
            !activated || !gated || !encoded) return RAD_E_INVAL;

        // The Diffusers loader flattens text_encoder into the checkpoint's root namespace.
        token_weight = mapped(b, names.f("umt5.token.weight"), {"shared.weight"},
                              {vocab, width}, grp_model());
        if (!token_weight) return RAD_E_INVAL;
        embed = rw(b, RAD_OP(b, "embed_lookup",
                             RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n_embd", width),
                                        RAD_INT("n_vocab", vocab), RAD_STR("dtype", "bf16"),
                                        RAD_INT("vocab_offset", 0)),
                             RAD_WEIGHTS(token_weight)), {}, {state});
        if (!embed) return RAD_E_INVAL;

        for (int i = 0; i < layer_count; ++i) {
            Layer& l = layers[i];
            const auto group = grp_layer(first_weight_group + i);
            const char* source = names.ckpt("encoder.block.%d", i);
            const char* attn = names.ckpt("%s.layer.0.SelfAttention", source);
            const char* ffn = names.ckpt("%s.layer.1.DenseReluDense", source);
            auto weight = [&](const char* suffix, std::initializer_list<const char*> sources,
                              std::initializer_list<int64_t> shape) {
                return mapped(b, names.f("umt5.block.%d.%s.weight", i, suffix), sources, shape, group);
            };

            l.attn_norm = weight("attn_norm", {names.ckpt("%s.layer.0.layer_norm.weight", source)}, {width});
            l.qkv = weight("qkv", {names.ckpt("%s.q.weight", attn), names.ckpt("%s.k.weight", attn),
                                   names.ckpt("%s.v.weight", attn)}, {3 * width, width});
            l.bias = weight("relative_bias", {names.ckpt("%s.relative_attention_bias.weight", attn)},
                            {rad_umt5::buckets, heads});
            l.out = weight("attn_out", {names.ckpt("%s.o.weight", attn)}, {width, width});
            l.ffn_norm = weight("ffn_norm", {names.ckpt("%s.layer.1.layer_norm.weight", source)}, {width});
            l.gate_up = weight("gate_up", {names.ckpt("%s.wi_0.weight", ffn), names.ckpt("%s.wi_1.weight", ffn)},
                               {2 * ffn_width, width});
            l.down = weight("ffn_down", {names.ckpt("%s.wo.weight", ffn)}, {width, ffn_width});
            if (!l.attn_norm || !l.qkv || !l.bias || !l.out || !l.ffn_norm ||
                !l.gate_up || !l.down) return RAD_E_INVAL;

            l.before_attn = declare_norm(b, l.attn_norm, scaled);
            l.project_qkv = linear(b, scaled, qkv, width, 3 * width, l.qkv);
            l.attention = rw(b, RAD_OP(b, "attn_umt5",
                                      RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("heads", heads),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_WEIGHTS(l.bias)), {qkv}, {attended});
            l.project_out = linear(b, attended, residual, width, width, l.out);
            l.after_attn = declare_rows(b, "add", width, {state, residual}, state);

            l.before_ffn = declare_norm(b, l.ffn_norm, scaled);
            l.project_gate_up = linear(b, scaled, gate_up, width, 2 * ffn_width, l.gate_up);
            l.gelu = declare_rows(b, "gelu_umt5", ffn_width, {gate_up}, activated);
            l.multiply = declare_rows(b, "mul", ffn_width, {activated, gate_up}, gated);
            l.project_down = linear(b, gated, residual, ffn_width, width, l.down);
            l.after_ffn = declare_rows(b, "add", width, {state, residual}, state);
            if (!l.before_attn.valid() || !l.project_qkv || !l.attention || !l.project_out ||
                !l.after_attn || !l.before_ffn.valid() || !l.project_gate_up || !l.gelu ||
                !l.multiply || !l.project_down || !l.after_ffn) return RAD_E_NOKERNEL;
        }
        final_weight = mapped(b, names.f("umt5.final_norm.weight"),
                              {"encoder.final_layer_norm.weight"}, {width},
                              grp_layer(first_weight_group + layer_count));
        if (!final_weight) return RAD_E_INVAL;
        final_norm = declare_norm(b, final_weight, encoded);
        return final_norm.valid() ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, const RadBatch& batch) const {
        using namespace rad::arch;
        const int64_t rows = batch.n_tok, prompts = batch.n_seq;
        if (prompts < 1 || prompts > 2 || rows < prompts || rows > prompts * rad_umt5::max_tokens ||
            !batch.token_ids || !batch.cu_seqlens) {
            rad_step_fail(c, "wan: invalid packed UMT5 prompt geometry");
            return;
        }
        const RadOperand cu = praw(batch.cu_seqlens, RAD_I32, prompts + 1);
        RAD_ISSUE_N(c, embed, rows, praw(batch.token_ids, RAD_I32, rows), RAD_W(token_weight),
                    brows(state, rows));
        for (const Layer& l : layers) {
            step_norm(c, l.before_attn, l.attn_norm, scaled, rows);
            RAD_ISSUE_N(c, l.project_qkv, rows, brows(scaled, rows), RAD_W(l.qkv), brows(qkv, rows));
            RAD_ISSUE_N(c, l.attention, rows, bcol(qkv, 0, width, rows), bcol(qkv, width, width, rows),
                        bcol(qkv, 2 * width, width, rows), cu, RAD_W(l.bias), brows(attended, rows));
            RAD_ISSUE_N(c, l.project_out, rows, brows(attended, rows), RAD_W(l.out), brows(residual, rows));
            RAD_ISSUE_N(c, l.after_attn, rows, brows(state, rows), brows(residual, rows), brows(state, rows));

            step_norm(c, l.before_ffn, l.ffn_norm, scaled, rows);
            RAD_ISSUE_N(c, l.project_gate_up, rows, brows(scaled, rows), RAD_W(l.gate_up), brows(gate_up, rows));
            RAD_ISSUE_N(c, l.gelu, rows, bcol(gate_up, 0, ffn_width, rows), brows(activated, rows));
            RAD_ISSUE_N(c, l.multiply, rows, brows(activated, rows), bcol(gate_up, ffn_width, ffn_width, rows),
                        brows(gated, rows));
            RAD_ISSUE_N(c, l.project_down, rows, brows(gated, rows), RAD_W(l.down), brows(residual, rows));
            RAD_ISSUE_N(c, l.after_ffn, rows, brows(state, rows), brows(residual, rows), brows(state, rows));
        }
        step_norm(c, final_norm, final_weight, encoded, rows);
    }

private:
    static_assert(width == heads * rad_umt5::head_width);

    struct Norm {
        rad_op rms = 0, gain = 0;
        bool valid() const { return rms && gain; }
    };

    struct Layer {
        rad_weight attn_norm = 0, qkv = 0, bias = 0, out = 0;
        rad_weight ffn_norm = 0, gate_up = 0, down = 0;
        Norm before_attn, before_ffn;
        rad_op project_qkv = 0, attention = 0, project_out = 0, after_attn = 0;
        rad_op project_gate_up = 0, gelu = 0, multiply = 0, project_down = 0, after_ffn = 0;
    };

    std::array<Layer, layer_count> layers;
    rad_weight token_weight = 0, final_weight = 0;
    rad_buf state = 0, norm = 0, scaled = 0, qkv = 0, attended = 0, residual = 0;
    rad_buf gate_up = 0, activated = 0, gated = 0;
    rad_op embed = 0;
    Norm final_norm;

    static rad_weight mapped(RadBuilder* b, const char* name, std::initializer_list<const char*> sources,
                             std::initializer_list<int64_t> shape, RadWeightGroup group) {
        using namespace rad::arch;
        const int rc = sources.size() == 1 ? map_copy(b, name, *sources.begin()) : map_concat(b, name, sources);
        if (rc < 0) return 0;
        return decl_w(b, name, RAD_BF16, shape, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
    }

    static rad_op linear(RadBuilder* b, rad_buf in, rad_buf out, int64_t n_in, int64_t n_out, rad_weight w) {
        return rad::arch::rw(b, RAD_OP(b, "gemm_nt",
                                       RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("N", n_out),
                                                  RAD_INT("K", n_in), RAD_STR("dtype", "bf16")),
                                       RAD_WEIGHTS(w)), {in}, {out});
    }

    static rad_op declare_rows(RadBuilder* b, const char* op, int64_t n,
                               std::initializer_list<rad_buf> in, rad_buf out) {
        return rad::arch::rw(b, RAD_OP(b, op,
                                       RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", n),
                                                  RAD_STR("dtype", "bf16")),
                                       RAD_NOWEIGHTS), in, {out});
    }

    Norm declare_norm(RadBuilder* b, rad_weight weight, rad_buf out) const {
        // UMT5 rounds the normalized activation to BF16 before applying the gain.
        Norm op;
        op.rms = rad::arch::rw(b, RAD_OP(b, "rmsnorm_p",
                                         RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", width),
                                                    RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                                         RAD_NOWEIGHTS), {state}, {norm});
        op.gain = rad::arch::rw(b, RAD_OP(b, "mul_weight",
                                          RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", width),
                                                     RAD_STR("dtype", "bf16")),
                                          RAD_WEIGHTS(weight)), {norm}, {out});
        return op;
    }

    void step_norm(RadCtx* c, Norm op, rad_weight weight, rad_buf out, int64_t rows) const {
        using rad::arch::brows;
        RAD_ISSUE_N(c, op.rms, rows, brows(state, rows), brows(norm, rows));
        RAD_ISSUE_N(c, op.gain, rows, brows(norm, rows), RAD_W(weight), brows(out, rows));
    }
};

}  // namespace wan
