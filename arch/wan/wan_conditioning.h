/* Wan's padded prompt projection into the denoiser's cross-attention width. */
#pragma once

#include "wan_umt5.h"

namespace wan {

struct TextConditioning {
    static constexpr int64_t prompt_rows = rad_umt5::max_tokens;
    static constexpr int64_t width = 3072;

    int declare(RadBuilder* b, rad::arch::Names& names, int lanes) {
        using namespace rad::arch;
        if (lanes < 1) return RAD_E_SHAPE;
        lane_count = lanes;
        RAD_ARCH_TRY(encoder.declare(b, names));

        padded = decl_b(b, names.f("text.padded"), RAD_BF16, {2 * prompt_rows, Umt5Encoder::width});
        hidden = decl_b(b, names.f("text.hidden"), RAD_BF16, {2 * prompt_rows, width});
        projected = decl_b(b, names.f("text.projected"), RAD_BF16,
                           {2 * prompt_rows * lanes, width}, RAD_BUF_PERSIST);
        if (!padded || !hidden || !projected) return RAD_E_INVAL;

        zero = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", prompt_rows, 2 * prompt_rows),
                                        RAD_INT("n", Umt5Encoder::width), RAD_STR("dtype", "bf16"),
                                        RAD_F64("value", 0.0)), RAD_NOWEIGHTS), {}, {padded});
        copy = rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_RANGE("M", 1, prompt_rows), RAD_INT("n", Umt5Encoder::width),
                                        RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                             RAD_NOWEIGHTS), {encoder.encoded}, {padded});
        RAD_ARCH_TRY(declare_linear(b, names, 1, Umt5Encoder::width, padded, hidden, "gelu_tanh", first));
        RAD_ARCH_TRY(declare_linear(b, names, 2, width, hidden, projected, "none", second));
        return zero && copy ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, const RadBatch& batch) const {
        using namespace rad::arch;
        if (!batch.vgen_b) {
            rad_step_fail(c, "wan: missing video conditioning batch");
            return;
        }
        const int64_t lane = batch.vgen_b->lane, negative = batch.vgen_b->n_neg;
        if (lane < 0 || lane >= lane_count || negative < 0 || negative > prompt_rows ||
            batch.n_seq != (negative ? 2 : 1) || batch.n_tok <= negative ||
            batch.n_tok > negative + prompt_rows || !batch.token_ids || !batch.cu_seqlens) {
            rad_step_fail(c, "wan: invalid video conditioning prompts or lane");
            return;
        }
        const int64_t positive = batch.n_tok - negative;
        const int64_t rows = batch.n_seq * prompt_rows;
        encoder.step(c, batch);

        // Wan pads encoder output to 512 with zeros before the biased projections.
        // The projected padding contributes to cross-attention like the prompt rows.
        RAD_ISSUE_N(c, zero, rows, brows(padded, rows));
        RAD_ISSUE_N(c, copy, positive, brows(encoder.encoded, positive), brows(padded, positive));
        if (negative)
            RAD_ISSUE_N(c, copy, negative,
                        brow_slice(encoder.encoded, positive, negative, Umt5Encoder::width),
                        brow_slice(padded, prompt_rows, negative, Umt5Encoder::width));
        RAD_ISSUE_N(c, first.op, rows, brows(padded, rows), RAD_W(first.weight), RAD_W(first.bias),
                    RAD_NONE, brows(hidden, rows));
        RAD_ISSUE_N(c, second.op, rows, brows(hidden, rows), RAD_W(second.weight), RAD_W(second.bias),
                    RAD_NONE, brow_slice(projected, lane * 2 * prompt_rows, rows, width));
    }

    rad_buf buffer() const { return projected; }

    // Caller supplies a validated video lane and branch (0 positive, 1 negative).
    RadOperand context(int lane, int branch) const {
        return rad::arch::brow_slice(projected, (2 * (int64_t)lane + branch) * prompt_rows,
                                     prompt_rows, width);
    }

private:
    struct Linear {
        rad_weight weight = 0, bias = 0;
        rad_op op = 0;
    };

    Umt5Encoder encoder;
    Linear first, second;
    int lane_count = 0;
    rad_buf padded = 0, hidden = 0, projected = 0;
    rad_op zero = 0, copy = 0;

    static int declare_linear(RadBuilder* b, rad::arch::Names& names, int index, int64_t input_width,
                              rad_buf input, rad_buf output, const char* activation, Linear& linear) {
        using namespace rad::arch;
        const char* weight = names.f("text.proj%d.weight", index);
        const char* bias = names.f("text.proj%d.bias", index);
        const char* source = names.ckpt("transformer.condition_embedder.text_embedder.linear_%d", index);
        RAD_ARCH_TRY(map_copy(b, weight, names.ckpt("%s.weight", source)));
        RAD_ARCH_TRY(map_copy(b, bias, names.ckpt("%s.bias", source)));
        linear.weight = decl_w(b, weight, RAD_BF16, {width, input_width}, RAD_ACCESS_PER_REQUEST,
                               RAD_SHARD_NONE, grp_model());
        linear.bias = decl_w(b, bias, RAD_BF16, {width}, RAD_ACCESS_PER_REQUEST,
                             RAD_SHARD_NONE, grp_model());
        if (!linear.weight || !linear.bias) return RAD_E_INVAL;
        linear.op = rw(b, RAD_OP(b, "gemm_nt_bias",
                                 RAD_PARAMS(RAD_RANGE("M", prompt_rows, 2 * prompt_rows),
                                            RAD_INT("N", width), RAD_INT("K", input_width),
                                            RAD_STR("dtype", "bf16"), RAD_STR("act", activation)),
                                 RAD_WEIGHTS(linear.weight, linear.bias)), {input}, {output});
        return linear.op ? RAD_OK : RAD_E_NOKERNEL;
    }
};

}  // namespace wan
