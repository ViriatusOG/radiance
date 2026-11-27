/* Check the int8 activations supplied to the routed experts. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/fishomni/fishomni.cpp"

extern "C" int rad_declare_audio_generator(RadBuilder*, const RadAudioGenDecl*) { return RAD_OK; }

static int declare_int8_trunk(RadBuilder& b, fishomni::Model& model, int world) {
    RadModelMeta m{};
    m.n_layers = 4;  /* one full-attention layer and three delta-net layers */
    m.n_embd = 2048;
    m.n_head = 16;
    m.n_head_kv = 2;
    m.head_dim = 256;
    m.n_ff = 512;
    m.n_vocab = 252288;
    m.rms_eps = 1e-6f;
    m.rope_theta = 1e7;
    m.n_expert = 8;
    m.n_expert_used = 8;
    MetaKV config(m);
    config.set("full_attention_interval", "4").set("partial_rotary_factor", "0.25")
          .set("linear_num_key_heads", "16").set("linear_num_value_heads", "32")
          .set("linear_key_head_dim", "128").set("linear_value_head_dim", "128")
          .set("linear_conv_kernel_dim", "4").set("shared_expert_intermediate_size", "512");
    const auto* meta = config.meta();
    RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(i8.transform, "fwht128");
    b.encs = {{"ffn_gate_up_exps.", i8}, {"ffn_down_exps.", i8},
              {"weight", rad_enc_plain(RAD_BF16)}};
    auto ctx = build_ctx(world - 1, world);
    RAD_ARCH_TRY(rad::arch::geom_from(model.g, meta, &ctx, "bf16", RAD_BF16));
    return fishomni::declare_trunk(&b, meta, &ctx, model);
}

TEST(fishomni_uses_rotated_int8_for_both_routed_projections) {
    for (int world : {1, 2}) for (bool fused : {false, true}) {
        RadBuilder b;
        if (!fused) b.undeclared.insert("gated_had_quant_i8g");
        fishomni::Model model{};
        REQUIRE_EQ(declare_int8_trunk(b, model, world), RAD_OK);
        CHECK(b.asked_undeclared.empty());
        for (const auto& layer : model.layers) {
            const auto& moe = layer.mlp;
            const auto* gate = b.op_of(moe.op_gu);
            const auto* down = b.op_of(moe.op_dn);
            const auto* norm_quant = b.op_of(moe.op_hq);
            const auto* product = b.op_of(moe.gq_exp.op);
            REQUIRE(gate && down && norm_quant && product);
            CHECK_EQ(pstr(gate, "dtype"), std::string("i8a8h"));
            CHECK_EQ(pstr(down, "dtype"), std::string("i8a8h"));
            CHECK_EQ(norm_quant->op, std::string("had_quant_act_i8g"));
            CHECK_EQ(product->op, std::string(fused ? "gated_had_quant_i8g" : "silu_mul"));
            const auto* down_quant = fused ? product : b.op_of(moe.gq_exp.op_q);
            REQUIRE(down_quant);
            if (!fused) {
                CHECK_EQ(down_quant->op, std::string("had_quant_act_i8g"));
                CHECK(product->writes == down_quant->reads);
            }
            REQUIRE(norm_quant->writes.size() >= 2 && down_quant->writes.size() >= 2);
            REQUIRE(gate->reads.size() >= 2 && down->reads.size() >= 2);
            for (size_t plane = 0; plane < 2; ++plane) {
                CHECK_EQ(gate->reads[plane], norm_quant->writes[plane]);
                CHECK_EQ(down->reads[plane], down_quant->writes[plane]);
            }
        }
    }
}

TEST(fishomni_requires_a_quantiser_for_the_routed_experts_input) {
    RadBuilder b;
    b.refuse.insert("had_quant_act_i8g");
    fishomni::Model model{};
    CHECK_EQ(declare_int8_trunk(b, model, 1), RAD_E_UNSUPPORTED);
}

RAD_TEST_MAIN()
