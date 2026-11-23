/* Wan TI2V's diffusion transformer. The residual stream, learned
 * modulation, and output head stay FP32. The projection and attention
 * activations use BF16 at the same boundaries as the reference's autocast. */
#pragma once

#include "wan_conditioning.h"
#include "wan_dit_io.h"
#include "wan_time.h"

#include <algorithm>
#include <array>

namespace wan {

struct Denoiser {
    static constexpr int64_t width = DitPatchIO::hidden_width;
    static constexpr int64_t heads = 24;
    static constexpr int64_t ffn_width = 14336;
    static constexpr int layers = 30;

    int declare(RadBuilder* b, rad::arch::Names& names, rad_buf latent, rad_buf text,
                rad_buf velocity, int64_t max_patch_rows) {
        using namespace rad::arch;
        if (!text || !velocity || max_patch_rows < 1 || max_patch_rows > 65535)
            return RAD_E_SHAPE;
        max_rows = max_patch_rows;
        const int64_t scratch_rows = std::max(max_rows, text_rows);
        state = decl_b(b, names.f("dit.state"), RAD_F32, {max_rows, width});
        normalized = decl_b(b, names.f("dit.normalized"), RAD_BF16, {max_rows, width});
        norm_f32 = decl_b(b, names.f("dit.norm_f32"), RAD_F32, {max_rows, width});
        qkv = decl_b(b, names.f("dit.qkv"), RAD_BF16, {max_rows, 3 * width});
        qk_f32 = decl_b(b, names.f("dit.qk_f32"), RAD_F32, {max_rows, 2 * width});
        kv = decl_b(b, names.f("dit.context_kv"), RAD_BF16, {text_rows, 2 * width});
        qk_norm = decl_b(b, names.f("dit.qk_norm"), RAD_BF16, {scratch_rows, width});
        attended = decl_b(b, names.f("dit.attended"), RAD_BF16, {max_rows, width});
        update = decl_b(b, names.f("dit.update"), RAD_BF16, {max_rows, width});
        ffn = decl_b(b, names.f("dit.ffn"), RAD_BF16, {max_rows, ffn_width});
        head_output = decl_b(b, names.f("dit.head_output"), RAD_F32,
                             {max_rows, DitPatchIO::patch_width});
        text_cu = decl_b(b, names.f("dit.text_bounds"), RAD_I32, {2});
        if (!state || !normalized || !norm_f32 || !qkv || !qk_f32 || !kv || !qk_norm ||
            !attended || !update || !ffn || !head_output || !text_cu) return RAD_E_INVAL;

        RAD_ARCH_TRY(time.declare(b, names));
        RAD_ARCH_TRY(io.declare_input(b, names, latent, max_rows));
        to_state = cast(b, io.hidden, state, "bf16", "f32");
        text_bounds = rw(b, RAD_OP(b, "iota", RAD_PARAMS(RAD_RANGE("M", 2, 2),
                                     RAD_INT("start", 0), RAD_INT("step", text_rows)),
                                  RAD_NOWEIGHTS), {}, {text_cu});
        if (!to_state || !text_bounds) return RAD_E_NOKERNEL;

        for (int i = 0; i < layers; ++i) RAD_ARCH_TRY(declare_block(b, names, text, i));

        const auto group = grp_model();
        head_table = mapped_weight(b, names.f("dit.head.table"), "transformer.scale_shift_table",
                                   RAD_F32, {1, 2, width}, group);
        if (!head_table) return RAD_E_INVAL;
        head_norm = mod_norm(b, -1, false, head_table);
        if (!head_norm) return RAD_E_NOKERNEL;
        RAD_ARCH_TRY(declare_linear(b, names, names.f("dit.head"), {"transformer.proj_out"}, group,
                                    "f32", width, DitPatchIO::patch_width, max_rows,
                                    norm_f32, head_output, head));
        return io.declare_output(b, head_output, velocity, RAD_F32);
    }

    // One guidance branch. The caller supplies the lane's latent, this
    // branch's padded text, and a distinct F32 velocity destination.
    int step(RadCtx* c, const RadVideoGenBatch& video, RadOperand latent,
             RadOperand text, RadOperand velocity) const {
        using namespace rad::arch;
        if (!video.sched || !video.lat_cu || !video.geom_d) {
            rad_step_fail(c, "wan: incomplete denoiser batch");
            return RAD_E_INVAL;
        }
        time.step(c, video);
        const int64_t rows = io.step_input(c, video, latent);
        if (rows < 0) return static_cast<int>(rows);
        const int64_t steps = video.has_image ? 2 : 1;
        const auto xa = brows(state, rows);
        const auto branch_time = brows(time.modulation, steps);
        const auto split = video.has_image
            ? praw(video.geom_d + RAD_VGEOM_FIRST_CU + 1, RAD_I32, 1) : RAD_NONE;
        const auto video_cu = praw(video.lat_cu, RAD_I32, 2);
        const auto context_cu = brows(text_cu, 2);
        const auto h = brows(normalized, rows);
        const auto a = brows(attended, rows);
        const auto d = brows(update, rows);
        const auto tmp = brows(norm_f32, rows);
        const auto q = bcol(qkv, 0, width, rows);
        const auto k = bcol(qkv, width, width, rows);
        const auto v = bcol(qkv, 2 * width, width, rows);
        const auto q32 = bcol(qk_f32, 0, width, rows);
        const auto k32 = bcol(qk_f32, width, width, rows);
        const auto ck = bcol(kv, 0, width, text_rows);
        const auto cv = bcol(kv, width, width, text_rows);
        RAD_ISSUE_N(c, to_state, rows, brows(io.hidden, rows), xa);
        RAD_ISSUE_N(c, text_bounds, 2, context_cu);

        for (const Block& block : blocks) {
            RAD_ISSUE_N(c, block.self_norm, rows,
                        xa, branch_time, RAD_W(block.table), split, h);
            issue_linear(c, block.self_qkv, rows, h, brows(qkv, rows));
            step_qk_norm(c, block.self_q, rows, q, q32);
            step_qk_norm(c, block.self_k, rows, k, k32);
            RAD_ISSUE_N(c, block.rope, rows, q32, k32, brows(io.rotary, rows));
            RAD_ISSUE_N(c, block.cast_q, rows, q32, q);
            RAD_ISSUE_N(c, block.cast_k, rows, k32, k);
            RAD_ISSUE_N(c, block.self_attention, rows, q, k, v, video_cu, a);
            issue_linear(c, block.self_out, rows, a, d);
            RAD_ISSUE_N(c, block.self_add, rows, xa, d, branch_time, RAD_W(block.table), split);

            RAD_ISSUE_N(c, block.cross_norm, rows,
                        xa, RAD_W(block.cross_gain), RAD_W(block.cross_bias), tmp);
            RAD_ISSUE_N(c, block.cross_cast, rows, tmp, h);
            issue_linear(c, block.cross_q, rows, h, q);
            issue_linear(c, block.cross_kv, text_rows, text, brows(kv, text_rows));
            step_qk_norm(c, block.cross_q_norm, rows, q, q);
            step_qk_norm(c, block.cross_k_norm, text_rows, ck, ck);
            RAD_ISSUE_N(c, block.cross_attention, rows, q, ck, cv, video_cu, context_cu, a);
            issue_linear(c, block.cross_out, rows, a, d);
            RAD_ISSUE_N(c, block.cross_add, rows, xa, d, RAD_NONE, RAD_NONE, RAD_NONE);

            RAD_ISSUE_N(c, block.ffn_norm, rows,
                        xa, branch_time, RAD_W(block.table), split, h);
            issue_linear(c, block.ffn_up, rows, h, brows(ffn, rows));
            issue_linear(c, block.ffn_down, rows, brows(ffn, rows), d);
            RAD_ISSUE_N(c, block.ffn_add, rows, xa, d, branch_time, RAD_W(block.table), split);
        }
        RAD_ISSUE_N(c, head_norm, rows, xa, brows(time.embedding, steps), RAD_W(head_table), split, tmp);
        issue_linear(c, head, rows, tmp, brows(head_output, rows));
        io.step_output(c, video, rows, RAD_B(head_output), velocity);
        return RAD_OK;
    }

private:
    static constexpr int64_t text_rows = TextConditioning::prompt_rows;
    static_assert(width == heads * rad_wan_dit::head_width);

    struct Linear {
        rad_weight w = 0, b = 0;
        rad_op op = 0;
    };
    struct QkNorm {
        rad_weight weight = 0;
        rad_op norm = 0, gain = 0;
    };
    struct Block {
        rad_weight table = 0, cross_gain = 0, cross_bias = 0;
        Linear self_qkv, self_out, cross_q, cross_kv, cross_out, ffn_up, ffn_down;
        QkNorm self_q, self_k, cross_q_norm, cross_k_norm;
        rad_op self_norm = 0, rope = 0, cast_q = 0, cast_k = 0, self_attention = 0, self_add = 0;
        rad_op cross_norm = 0, cross_cast = 0, cross_attention = 0, cross_add = 0;
        rad_op ffn_norm = 0, ffn_add = 0;
    };

    int64_t max_rows = 0;
    TimeConditioning time;
    DitPatchIO io;
    std::array<Block, layers> blocks;
    rad_buf state = 0, normalized = 0, norm_f32 = 0, qkv = 0, qk_f32 = 0, kv = 0;
    rad_buf qk_norm = 0, attended = 0, update = 0, ffn = 0, text_cu = 0, head_output = 0;
    rad_op to_state = 0, text_bounds = 0, head_norm = 0;
    rad_weight head_table = 0;
    Linear head;

    int declare_block(RadBuilder* b, rad::arch::Names& names, rad_buf text, int index) {
        using namespace rad::arch;
        Block& block = blocks[index];
        const auto group = grp_layer(index);
        auto source = [&](const char* suffix) {
            return names.ckpt("transformer.blocks.%d.%s", index, suffix);
        };
        auto target = [&](const char* suffix) {
            return names.f("dit.block.%d.%s", index, suffix);
        };
        block.table = mapped_weight(b, target("table"), source("scale_shift_table"),
                                    RAD_F32, {1, 6, width}, group);
        block.cross_gain = mapped_weight(b, target("cross_norm.weight"), source("norm2.weight"),
                                         RAD_F32, {width}, group);
        block.cross_bias = mapped_weight(b, target("cross_norm.bias"), source("norm2.bias"),
                                         RAD_F32, {width}, group);
        if (!block.table || !block.cross_gain || !block.cross_bias) return RAD_E_INVAL;

        block.self_norm = mod_norm(b, 0, index == 0, block.table);
        RAD_ARCH_TRY(declare_linear(b, names, target("self.qkv"),
                                    {source("attn1.to_q"), source("attn1.to_k"), source("attn1.to_v")},
                                    group, "bf16", width, 3 * width, max_rows, normalized, qkv,
                                    block.self_qkv));
        RAD_ARCH_TRY(declare_qk_norm(b, target("self.norm_q.weight"), source("attn1.norm_q.weight"),
                                     group, max_rows, qkv, qk_f32, "f32", block.self_q));
        RAD_ARCH_TRY(declare_qk_norm(b, target("self.norm_k.weight"), source("attn1.norm_k.weight"),
                                     group, max_rows, qkv, qk_f32, "f32", block.self_k));
        block.rope = rw(b, RAD_OP(b, "wan_rope",
                                  RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("heads", heads),
                                             RAD_STR("dtype", "f32")), RAD_NOWEIGHTS),
                        {qk_f32, io.rotary}, {qk_f32});
        block.cast_q = cast(b, qk_f32, qkv, "f32", "bf16");
        block.cast_k = cast(b, qk_f32, qkv, "f32", "bf16");
        block.self_attention = attention(b, false);
        RAD_ARCH_TRY(declare_linear(b, names, target("self.out"), {source("attn1.to_out.0")},
                                    group, "bf16", width, width, max_rows, attended, update, block.self_out));
        block.self_add = residual(b, 2, block.table);

        block.cross_norm = rw(b, RAD_OP(b, "layernorm",
                                        RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", width),
                                                   RAD_F64("eps", 1e-6), RAD_STR("dtype", "f32")),
                                        RAD_WEIGHTS(block.cross_gain, block.cross_bias)),
                              {state}, {norm_f32});
        block.cross_cast = cast(b, norm_f32, normalized, "f32", "bf16");
        RAD_ARCH_TRY(declare_linear(b, names, target("cross.q"), {source("attn2.to_q")},
                                    group, "bf16", width, width, max_rows, normalized, qkv, block.cross_q));
        RAD_ARCH_TRY(declare_linear(b, names, target("cross.kv"),
                                    {source("attn2.to_k"), source("attn2.to_v")}, group, "bf16",
                                    width, 2 * width, text_rows, text, kv, block.cross_kv));
        RAD_ARCH_TRY(declare_qk_norm(b, target("cross.norm_q.weight"), source("attn2.norm_q.weight"),
                                     group, max_rows, qkv, qkv, "bf16", block.cross_q_norm));
        RAD_ARCH_TRY(declare_qk_norm(b, target("cross.norm_k.weight"), source("attn2.norm_k.weight"),
                                     group, text_rows, kv, kv, "bf16", block.cross_k_norm));
        block.cross_attention = attention(b, true);
        RAD_ARCH_TRY(declare_linear(b, names, target("cross.out"), {source("attn2.to_out.0")},
                                    group, "bf16", width, width, max_rows, attended, update, block.cross_out));
        block.cross_add = residual(b, -1, 0);

        block.ffn_norm = mod_norm(b, 3, false, block.table);
        RAD_ARCH_TRY(declare_linear(b, names, target("ffn.up"), {source("ffn.net.0.proj")},
                                    group, "bf16", width, ffn_width, max_rows, normalized, ffn,
                                    block.ffn_up, "gelu_tanh"));
        RAD_ARCH_TRY(declare_linear(b, names, target("ffn.down"), {source("ffn.net.2")},
                                    group, "bf16", ffn_width, width, max_rows, ffn, update, block.ffn_down));
        block.ffn_add = residual(b, 5, block.table);
        return block.self_norm && block.rope && block.cast_q && block.cast_k && block.self_attention &&
               block.self_add && block.cross_norm && block.cross_cast && block.cross_attention &&
               block.cross_add && block.ffn_norm && block.ffn_add ? RAD_OK : RAD_E_NOKERNEL;
    }

    static rad_weight mapped_weight(RadBuilder* b, const char* name, const char* source, uint32_t dtype,
                                    std::initializer_list<int64_t> shape, RadWeightGroup group) {
        using namespace rad::arch;
        if (map_copy(b, name, source) < 0) return 0;
        return decl_w(b, name, dtype, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, group);
    }

    static int declare_linear(RadBuilder* b, rad::arch::Names& names, const char* name,
                              std::initializer_list<const char*> sources, RadWeightGroup group,
                              const char* type, int64_t n_in, int64_t n_out, int64_t rows,
                              rad_buf input, rad_buf output, Linear& linear, const char* act = "none") {
        using namespace rad::arch;
        const char* weight_name = names.ckpt("%s.weight", name);
        const char* bias_name = names.ckpt("%s.bias", name);
        const int mode = sources.size() > 1 ? RAD_MAP_CONCAT : RAD_MAP_COPY;
        RadNameMap weight = map_base(weight_name, mode), bias = map_base(bias_name, mode);
        for (const char* source : sources) {
            weight.src[weight.n_src++] = names.ckpt("%s.weight", source);
            bias.src[bias.n_src++] = names.ckpt("%s.bias", source);
        }
        RAD_ARCH_TRY(rad_decl_name_map(b, &weight));
        RAD_ARCH_TRY(rad_decl_name_map(b, &bias));
        const uint32_t dtype = rad_dtype_parse(type);
        linear.w = decl_w(b, weight_name, dtype, {n_out, n_in}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, group);
        linear.b = decl_w(b, bias_name, dtype, {n_out}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, group);
        if (!linear.w || !linear.b) return RAD_E_INVAL;
        linear.op = rw(b, RAD_OP(b, "gemm_nt_bias",
                                 RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", n_out),
                                            RAD_INT("K", n_in), RAD_STR("dtype", type), RAD_STR("act", act)),
                                 RAD_WEIGHTS(linear.w, linear.b)), {input}, {output});
        return linear.op ? RAD_OK : RAD_E_NOKERNEL;
    }

    static void issue_linear(RadCtx* c, const Linear& linear, int64_t rows, RadOperand x, RadOperand y) {
        RAD_ISSUE_N(c, linear.op, rows, x, RAD_W(linear.w), RAD_W(linear.b), RAD_NONE, y);
    }

    int declare_qk_norm(RadBuilder* b, const char* name, const char* source, RadWeightGroup group,
                        int64_t rows, rad_buf in, rad_buf out, const char* type, QkNorm& norm) const {
        using namespace rad::arch;
        norm.weight = mapped_weight(b, name, source, RAD_F32, {width}, group);
        if (!norm.weight) return RAD_E_INVAL;
        norm.norm = rw(b, RAD_OP(b, "rmsnorm_p",
                                 RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                            RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS), {in}, {qk_norm});
        norm.gain = rw(b, RAD_OP(b, "mul_weight",
                                 RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                            RAD_STR("dtype", type)), RAD_WEIGHTS(norm.weight)),
                       {qk_norm}, {out});
        return norm.norm && norm.gain ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step_qk_norm(RadCtx* c, const QkNorm& norm, int64_t rows, RadOperand x, RadOperand y) const {
        const auto tmp = rad::arch::brows(qk_norm, rows);
        RAD_ISSUE_N(c, norm.norm, rows, x, tmp);
        RAD_ISSUE_N(c, norm.gain, rows, tmp, RAD_W(norm.weight), y);
    }

    rad_op cast(RadBuilder* b, rad_buf input, rad_buf output, const char* from, const char* to) const {
        return rad::arch::rw(b, RAD_OP(b, "cast",
                                       RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", width),
                                                  RAD_STR("from", from), RAD_STR("to", to)),
                                       RAD_NOWEIGHTS), {input}, {output});
    }

    rad_op mod_norm(RadBuilder* b, int group, bool round, rad_weight w) const {
        using namespace rad::arch;
        const bool is_head = group == -1;
        return rw(b, RAD_OP(b, "wan_mod_norm",
                            RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", width),
                                       RAD_INT("group", group), RAD_INT("round_norm", round),
                                       RAD_STR("dtype", is_head ? "f32" : "bf16")),
                            RAD_WEIGHTS(w)), {state, is_head ? time.embedding : time.modulation},
                  {is_head ? norm_f32 : normalized});
    }

    rad_op residual(RadBuilder* b, int gate, rad_weight weight) const {
        using namespace rad::arch;
        const RadParam params[] = {RAD_RANGE("M", 1, max_rows), RAD_INT("n", width),
                                   RAD_INT("gate", gate), RAD_STR("dtype", "f32")};
        return weight ? rw(b, rad_decl_op(b, "wan_gated_add", params, 4, &weight, 1),
                           {state, update, time.modulation}, {state})
                      : rw(b, rad_decl_op(b, "wan_gated_add", params, 4, nullptr, 0),
                           {state, update}, {state});
    }

    rad_op attention(RadBuilder* b, bool cross) const {
        using namespace rad::arch;
        const auto op = RAD_OP(b, cross ? "attn_cross" : "attn_dense",
                               RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("head_dim", rad_wan_dit::head_width),
                                          RAD_INT("gqa", 1), RAD_INT("causal", 0), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS);
        return cross ? rw(b, op, {qkv, kv, text_cu}, {attended}) : rw(b, op, {qkv}, {attended});
    }
};

}  // namespace wan
