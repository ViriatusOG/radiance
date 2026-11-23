/* Wan DiT's timestep embedding and shared six-way block modulation. */
#pragma once

#include <arch/rad_arch.h>

#include <array>

namespace wan {

struct TimeConditioning {
    static constexpr int64_t width = 3072;
    static constexpr int64_t frequency_width = 256;
    static constexpr int64_t modulation_width = 6 * width;

    rad_buf embedding = 0, modulation = 0;

    int declare(RadBuilder* b, rad::arch::Names& names) {
        using namespace rad::arch;
        timestep = decl_b(b, names.f("time.timestep"), RAD_F32, {2, 1});
        sinusoid = decl_b(b, names.f("time.sinusoid"), RAD_F32, {2, frequency_width});
        hidden = decl_b(b, names.f("time.hidden"), RAD_F32, {2, width});
        activated = decl_b(b, names.f("time.activated"), RAD_F32, {2, width});
        embedding = decl_b(b, names.f("time.embedding"), RAD_F32, {2, width});
        modulation = decl_b(b, names.f("time.modulation"), RAD_F32, {2, modulation_width});
        if (!timestep || !sinusoid || !hidden || !activated || !embedding || !modulation)
            return RAD_E_INVAL;

        zero = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1),
                                        RAD_STR("dtype", "f32"), RAD_F64("value", 0.0)),
                             RAD_NOWEIGHTS), {}, {timestep});
        copy = rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1),
                                        RAD_STR("from", "f32"), RAD_STR("to", "f32")),
                             RAD_NOWEIGHTS), {}, {timestep});
        sinus = rw(b, RAD_OP(b, "timestep_embed_f64",
                              RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("n", frequency_width),
                                         RAD_F64("max_period", 10000.0),
                                         RAD_STR("dtype", "f32")), RAD_NOWEIGHTS), {timestep}, {sinusoid});
        RAD_ARCH_TRY(declare_linear(b, names, "first", "time_embedder.linear_1",
                                    frequency_width, width, sinusoid, hidden, linear[0]));
        activate_hidden = declare_activation(b, hidden, activated);
        RAD_ARCH_TRY(declare_linear(b, names, "second", "time_embedder.linear_2",
                                    width, width, activated, embedding, linear[1]));
        activate_embedding = declare_activation(b, embedding, activated);
        RAD_ARCH_TRY(declare_linear(b, names, "project", "time_proj",
                                    width, modulation_width, activated, modulation, linear[2]));
        return zero && copy && sinus && activate_hidden && activate_embedding ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, const RadVideoGenBatch& video) const {
        using namespace rad::arch;
        if (!video.sched) {
            rad_step_fail(c, "wan: missing video timestep");
            return;
        }
        const int64_t denoise_row = video.has_image ? 1 : 0;
        const int64_t rows = denoise_row + 1;
        // The image's first temporal patch plane reads row 0 at timestep zero.
        // With no reference, every plane reads the one denoising row.
        if (denoise_row) RAD_ISSUE_N(c, zero, 1, brows(timestep, 1));
        RAD_ISSUE_N(c, copy, 1, praw2(video.sched, RAD_F32, 1, 1),
                    brow_slice(timestep, denoise_row, 1, 1));
        RAD_ISSUE_N(c, sinus, rows, brows(timestep, rows), brows(sinusoid, rows));
        issue_linear(c, linear[0], sinusoid, hidden, rows);
        RAD_ISSUE_N(c, activate_hidden, rows, brows(hidden, rows), brows(activated, rows));
        issue_linear(c, linear[1], activated, embedding, rows);
        RAD_ISSUE_N(c, activate_embedding, rows, brows(embedding, rows), brows(activated, rows));
        issue_linear(c, linear[2], activated, modulation, rows);
    }

private:
    struct Linear {
        rad_weight weight = 0, bias = 0;
        rad_op op = 0;
    };

    rad_buf timestep = 0, sinusoid = 0, hidden = 0, activated = 0;
    rad_op zero = 0, copy = 0, sinus = 0, activate_hidden = 0, activate_embedding = 0;
    std::array<Linear, 3> linear;

    static int declare_linear(RadBuilder* b, rad::arch::Names& names, const char* name,
                              const char* checkpoint, int64_t n_in, int64_t n_out,
                              rad_buf input, rad_buf output, Linear& op) {
        using namespace rad::arch;
        const char* weight = names.f("time.%s.weight", name);
        const char* bias = names.f("time.%s.bias", name);
        RAD_ARCH_TRY(map_copy(b, weight, names.ckpt("transformer.condition_embedder.%s.weight", checkpoint)));
        RAD_ARCH_TRY(map_copy(b, bias, names.ckpt("transformer.condition_embedder.%s.bias", checkpoint)));
        op.weight = decl_w(b, weight, RAD_F32, {n_out, n_in}, RAD_ACCESS_PER_TOKEN,
                           RAD_SHARD_NONE, grp_model());
        op.bias = decl_w(b, bias, RAD_F32, {n_out}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        if (!op.weight || !op.bias) return RAD_E_INVAL;
        op.op = rw(b, RAD_OP(b, "gemm_nt_bias",
                             RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("N", n_out),
                                        RAD_INT("K", n_in), RAD_STR("dtype", "f32")),
                             RAD_WEIGHTS(op.weight, op.bias)), {input}, {output});
        return op.op ? RAD_OK : RAD_E_NOKERNEL;
    }

    static rad_op declare_activation(RadBuilder* b, rad_buf input, rad_buf output) {
        return rad::arch::rw(b, RAD_OP(b, "silu",
                                       RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("n", width),
                                                  RAD_STR("dtype", "f32")), RAD_NOWEIGHTS), {input}, {output});
    }

    static void issue_linear(RadCtx* c, const Linear& op, rad_buf input, rad_buf output, int64_t rows) {
        using rad::arch::brows;
        RAD_ISSUE_N(c, op.op, rows, brows(input, rows), RAD_W(op.weight), RAD_W(op.bias),
                    RAD_NONE, brows(output, rows));
    }
};

}  // namespace wan
