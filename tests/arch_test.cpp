/* arch_test.cpp -- the declared graph of the architecture plugins, asserted statically.
 *
 * No GPU, no model file, no core. With two-phase construction (spec §3) the whole model is known
 * between declare and plan, so a unit test can check what a plugin declares. A fake builder
 * records every decl_weight, decl_buffer, decl_op, buffer use and name-map call, and a fake RadCtx
 * records every issue, so the run phase's order and operands are checked too.
 *
 * The fake is deliberately not core/build/'s RadBuilder: a defect there must not fail this test,
 * and a plugin that only works against one builder has an undeclared dependency.
 */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include <array>
#include <dlfcn.h>
#include <sstream>
#include <thread>

/* ==================================================================== the plugin under test */
#include "../arch/qwen35_bf16/qwen35_bf16.cpp"
/* The production architecture, for its sizing declare (RadBuildCtx::shape_probe). */
#include "../arch/qwen4exp_fp8/qwen4exp_fp8.cpp"
/* The LLaMA family, for how it reads its rotary scaling. */
#include "../arch/llama_fp8/llama_fp8.cpp"
/* LFM2, for its own layer schedule and short convolution. */
#include "../arch/lfm2_bf16/lfm2_bf16.cpp"
/* LFM3: LFM2's schedule with sliding-window layers, sandwich norms and gated attention. */
#include "../arch/lfm3_bf16/lfm3_bf16.cpp"
/* Gemma 4 dense: two attention classes (sliding-window + proportional-RoPE global) on one stream. */
#include "../arch/gemma4_bf16/gemma4_bf16.cpp"
/* The bf16 / fp8-expert Gemma 4 plugin: the 26B-A4B's dense and routed arms side by side. */
#include "../arch/gemma3/gemma3.cpp"
#include "../arch/gemma4/gemma4.cpp"
/* Qwen3-ASR: a Whisper-style audio tower over a dense Qwen3 decoder. */
#include "../arch/qwen3asr_bf16/qwen3asr_bf16.cpp"
/* Unlimited-OCR: the Llama-MoE decoder and its DeepEncoder tower (SAM + CLIP). */
#include "../arch/unlimitedocr_bf16/unlimitedocr_bf16.cpp"
/* The PLE block alone: no architecture here declares it, and what its step() hands the two
 * windowed ops is its own contract (rad_block_ple.h). */
#include <arch/rad_block_ple.h>
/* Qwen3.5 / 3.6 / 3.8 with block-fp8 weights, dense and routed: one source built twice under
 * three macros, as arch/qwen35moe_fp8 builds it. */
#include "../arch/qwen35_fp8/qwen35_fp8.cpp"
#undef QWEN35_MOE
#undef QWEN35_NS
#undef QWEN35_ARCH_ID
#include "../arch/qwen35moe_fp8/qwen35moe_fp8.cpp"
/* The same source a third time, as qwen35_w8a8: the int8 trunk. */
#undef QWEN35_MOE
#undef QWEN35_W8A8
#undef QWEN35_NS
#undef QWEN35_ARCH_ID
#include "../arch/qwen35_w8a8/qwen35_w8a8.cpp"
/* And a fourth, as qwen35moe_w8a8: the routed build with the int8 trunk. */
#undef QWEN35_MOE
#undef QWEN35_W8A8
#undef QWEN35_NS
#undef QWEN35_ARCH_ID
#include "../arch/qwen35moe_w8a8/qwen35moe_w8a8.cpp"
/* Tencent Hy-Memory: FP8 full attention with Q/K norm and sigmoid-correction MoE. */
#include "../arch/hyv3_fp8/hyv3_fp8.cpp"
#include "../arch/hyv3_bf16/hyv3_bf16.cpp"
#include "../arch/hyv3_gguf/hyv3_gguf.cpp"
#include "hyv3_bf16_fixture.h"
/* The LLaMA family from a dense checkpoint: llama_fp8 under its other descriptor. */
#undef LLAMA_DENSE_SRC
#undef LLAMA_NS
#undef LLAMA_QUANT
#include "../arch/llama_dense_fp8/llama_dense_fp8.cpp"
/* EmbeddingGemma 2: a pooled encoder with no logits. */
#include "../arch/embgemma2/embgemma2.cpp"
/* Nomic BERT: a pooled encoder with dense and routed post-norm layers. */
#include "../arch/nomicbert/nomicbert.cpp"
/* Qwen-Image 2.1: a generator conditioned by Qwen3-VL. */
#include "../arch/qwenimage21/qwenimage21.cpp"
/* Anima: a cross-attention generator conditioned through two tokenizers. */
#include "../arch/anima/anima.cpp"
/* Krea 2: a generator conditioned on twelve tapped layers of Qwen3-VL. */
#include "../arch/krea2/krea2.cpp"

/* ==================================================================== fixtures and helpers */
namespace {

/* Qwen3.8-27B's shape cut to eight layers, so failures stay readable. Eight is the smallest count
 * that exercises the alternation twice: with full_attention_interval 4, layers 3 and 7 are full
 * attention and the other six gated delta net, the 64-layer model's 1:3 ratio.
 *
 * head_dim is 256 while n_embd / n_head is 213, so using the latter would be caught. */
const char* k_keys[] = {
    "qwen35.full_attention_interval",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.conv_kernel",
};
const char* k_vals[] = { "4", "64", "128", "16", "48", "6144", "4" };

RadModelMeta qwen35_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen35";
    m.name        = "test-qwen35-27b";
    m.quant       = "bf16";
    m.n_layers    = 8;
    m.n_embd      = 5120;
    m.n_head      = 24;
    m.n_head_kv   = 4;
    m.head_dim    = 256;
    m.n_ff        = 17408;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_keys) / sizeof(k_keys[0]));
    m.kv_key      = k_keys;
    m.kv_val      = k_vals;
    return m;
}

/* Gemma 4 31B's text geometry cut to twelve layers, the smallest count with two global layers at
 * (l + 1) % 6 == 0 (layers 5 and 11) over ten sliding ones, the model's 5:1 ratio. head_dim 256 is
 * the sliding width; the global class's 512 width and 4 heads are header keys the plugin must
 * read, not RadModelMeta fields. */
const char* k_gem4_keys[] = {
    "sliding_window",
    "global_head_dim",
    "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta",
    "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor",
    "num_kv_shared_layers",
    "final_logit_softcapping",
    "full_attention_interval",
    "attention_k_eq_v",
};
const char* k_gem4_vals[] = { "1024", "512", "4", "10000", "1000000", "0.25", "0", "30", "6", "1" };

RadModelMeta gemma4_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma4";
    m.name        = "test-gemma4-31b";
    m.quant       = "bf16";
    m.n_layers    = 12;
    m.n_embd      = 5376;
    m.n_head      = 32;
    m.n_head_kv   = 16;
    m.head_dim    = 256;
    m.n_ff        = 21504;
    m.n_vocab     = 262144;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_gem4_keys) / sizeof(k_gem4_keys[0]));
    m.kv_key      = k_gem4_keys;
    m.kv_val      = k_gem4_vals;
    return m;
}

/* The same twelve layers with Gemma's assistant head merged under `draft.` as
 * `rad-convert --draft-model` writes it: the head's config flattened, its text config under
 * `draft.text_config.`. The 31B's head: width 1024, four layers (three sliding, one full), the
 * trunk's head geometry and window, and its own 262144-row tied table. */
const char* k_gem4d_keys[] = {
    "sliding_window", "global_head_dim", "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "full_attention_interval", "attention_k_eq_v",
    "draft.architectures", "draft.backbone_hidden_size", "draft.use_ordered_embeddings",
    "draft.text_config.hidden_size", "draft.text_config.num_hidden_layers",
    "draft.text_config.layer_types", "draft.text_config.num_attention_heads",
    "draft.text_config.num_key_value_heads", "draft.text_config.num_global_key_value_heads",
    "draft.text_config.head_dim", "draft.text_config.global_head_dim",
    "draft.text_config.intermediate_size", "draft.text_config.sliding_window",
    "draft.text_config.vocab_size", "draft.text_config.rms_norm_eps",
    "draft.text_config.rope_parameters.sliding_attention.rope_theta",
    "draft.text_config.rope_parameters.full_attention.rope_theta",
    "draft.text_config.rope_parameters.full_attention.partial_rotary_factor",
};
const char* k_gem4d_vals[] = {
    "1024", "512", "4", "10000", "1000000", "0.25", "0", "30", "6", "1",
    "Gemma4AssistantForCausalLM", "5376", "0",
    "1024", "4",
    "sliding_attention sliding_attention sliding_attention full_attention", "32",
    "16", "4",
    "256", "512",
    "8192", "1024",
    "262144", "1e-06",
    "10000", "1000000", "0.25",
};

RadModelMeta gemma4_draft_meta() {
    RadModelMeta m = gemma4_meta();
    m.n_kv   = (int)(sizeof(k_gem4d_keys) / sizeof(k_gem4d_keys[0]));
    m.kv_key = k_gem4d_keys;
    m.kv_val = k_gem4d_vals;
    return m;
}

/* The gated attention layer's ops, listed once for every test.
 *
 * The fused qk_norm_rope_gate is not declared: its schema requires a precomputed cos/sin table
 * rather than theta, and this plugin has none (see rad_block_attn_gated.h). Declaring it with
 * parameters its schema lacks would fail declare outright rather than fall back to the four ops it
 * stands for. */
/* A bf16 block's input norm: `rmsnorm`, or with a fold (rad_norm_bf16.h, two ranks) the
 * previous block's `add` before it and the fused `ar_rmsnorm` after it; a step issues one form. */
void norm_ops(std::vector<std::string>& v, bool fold) {
    if (fold) v.push_back("add");
    v.push_back("rmsnorm");
    if (fold) v.push_back("ar_rmsnorm");
}

/* `fold`: the block's norm takes the previous block's delta. `add_out`: the block declares its
 * own closing add (without it the next block's norm folds it). */
std::vector<std::string> attn_layer_ops(bool tp, bool fold = false, bool add_out = true) {
    std::vector<std::string> v;
    norm_ops(v, fold);
    v.push_back("gemm_nt"); v.push_back("gemm_nt"); v.push_back("gemm_nt");
    v.push_back("rmsnorm"); v.push_back("rmsnorm");
    v.push_back("rope");    v.push_back("rope");
    v.push_back("attn_paged");
    v.push_back("kv_store");
    v.push_back("sigmoid");
    v.push_back("mul");
    v.push_back("gemm_nt");
    if (tp) v.push_back("all_reduce");
    if (add_out) v.push_back("add");
    return v;
}

std::vector<std::string> gdn_layer_ops(bool tp, bool fold = false, bool add_out = true) {
    std::vector<std::string> v;
    norm_ops(v, fold);
    for (const char* o : { "gemm_nt", "gemm_nt",
                           "gdn_conv_prep", "gdn_kkt_solve", "gdn_chunk_scan",
                           "gdn_gated_rmsnorm",
                           "gdn_conv_update", "gdn_recurrent_update",
                           "gemm_nt" })
        v.push_back(o);
    if (tp) v.push_back("all_reduce");
    if (add_out) v.push_back("add");
    return v;
}

std::vector<std::string> mlp_ops(bool tp, bool fold = false, bool add_out = true) {
    std::vector<std::string> v;
    norm_ops(v, fold);
    for (const char* o : { "gemm_nt", "silu_mul", "gemm_nt" }) v.push_back(o);
    if (tp) v.push_back("all_reduce");
    if (add_out) v.push_back("add");
    return v;
}

void append(std::vector<std::string>& v, const std::vector<std::string>& t) {
    v.insert(v.end(), t.begin(), t.end());
}

/* The whole declared program for the 8-layer fixture: layers 3 and 7 are full attention. */
std::vector<std::string> program_ops(bool tp) {
    std::vector<std::string> v = { "embed_lookup" };
    /* The embedding table is vocab-sharded: a rank gathers only ids in its slice and zero-fills
     * the rest, and this collective sums the ranks into x. Hence 2 all_reduces per layer plus
     * one. */
    if (tp) v.push_back("all_reduce");
    /* At two ranks every boundary folds (qwen35_bf16's layer_fold): a mixer past layer 0 takes
     * the previous feed-forward's delta and a feed-forward its mixer's, and only the last
     * feed-forward keeps its add, for the final norm. */
    for (int l = 0; l < 8; ++l) {
        const bool mix_fold = tp && l > 0;
        if ((l + 1) % 4 == 0) append(v, attn_layer_ops(tp, mix_fold, !tp));
        else                  append(v, gdn_layer_ops(tp, mix_fold, !tp));
        append(v, mlp_ops(tp, tp, !tp || l == 7));
    }
    /* At two ranks, after the layers, the weight reads beside the decode reductions (qwen35_bf16's
     * kPfGateUpBytes and the rest): each layer's gate_up and the next GEMM after its feed-forward,
     * and a delta net's out. */
    if (tp)
        for (int l = 0; l < 8; ++l)
            for (int i = 0; i < ((l + 1) % 4 == 0 ? 2 : 3); ++i) v.push_back("prefetch");
    v.push_back("rmsnorm");
    /* gather_rows before the lm_head: the sampler needs one row per sequence, and the widest GEMM
     * over every prefill token would be n_tok/n_seq times the work and an 8 GiB logits buffer
     * (docs/OPS.md). */
    v.push_back("gather_rows");
    v.push_back("logits_gemm");
    return v;
}

}  /* namespace */

/* ==================================================================== tests */

TEST(declare_is_a_qwen35_trunk) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));

    /* Six gated delta net layers to two full attention ones, the interval's 1:3 ratio. */
    CHECK_EQ(count_op(b, "gdn_chunk_scan"), 6);
    CHECK_EQ(count_op(b, "attn_paged"), 2);
    /* The fused gate op is not declared (see attn_layer_ops), so each attention layer runs two
     * per-head rmsnorms and two ropes instead. */
    CHECK_EQ(count_op(b, "qk_norm_rope_gate"), 0);
    CHECK_EQ(count_op(b, "rope"), 4);
    /* One MLP per layer, the same block for both layer kinds. */
    CHECK_EQ(count_op(b, "silu_mul"), 8);
}

TEST(qwen35_bf16_yarn_reaches_every_text_rotary_op) {
    RadModelMeta meta = qwen35_meta();
    std::vector<const char*> keys(meta.kv_key, meta.kv_key + meta.n_kv);
    std::vector<const char*> vals(meta.kv_val, meta.kv_val + meta.n_kv);
    keys.insert(keys.end(), {"rope_parameters.rope_type", "rope_parameters.factor",
                            "rope_parameters.original_max_position_embeddings"});
    vals.insert(vals.end(), {"yarn", "2", "262144"});
    meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
    RadBuildCtx ctx = build_ctx();
    RadBuilder b;
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));
    int rotary = 0;
    for (const auto& op : b.ops) if (op.op == "rope") {
        ++rotary;
        int found = 0;
        for (const auto& p : op.p) {
            if (p.key == "yarn_factor") { CHECK_EQ(p.dval, 2.0); ++found; }
            if (p.key == "yarn_magnitude") { CHECK_NEAR(p.dval, 1.069314718, 1e-7); ++found; }
            if (p.key == "scale") CHECK_EQ(p.dval, 1.0);
        }
        CHECK_EQ(found, 2);
    }
    CHECK_EQ(rotary, 4);
    vals[vals.size()-2] = "nan";
    RadBuilder invalid;
    CHECK_EQ(qwen35_bf16::declare(&invalid, &meta, &ctx), RAD_E_INVAL);
}

/* The LLaMA family reads rotary scaling through the same reader. Llama 3.1's `rope_type: llama3`
 * with factor 8, read as a bare factor, would be a linear position divide by 8, so it is refused
 * by name at declare. Linear stays linear and YaRN reaches the rope op. */
TEST(llama_reads_its_rotary_scaling_by_type) {
    auto declare = [](const char* type, const char* factor, RadBuilder* b) {
        const char* keys[] = {"rope_scaling.rope_type", "rope_scaling.factor",
                              "rope_scaling.original_max_position_embeddings"};
        const char* vals[] = {type, factor, "8192"};
        RadModelMeta m{};
        m.arch_id = "llama"; m.name = "test-llama"; m.quant = "fp8_e4m3";
        m.n_layers = 2; m.n_embd = 1024; m.n_head = 8; m.n_head_kv = 2; m.head_dim = 128;
        m.n_ff = 2048; m.n_vocab = 32000; m.n_ctx_train = 8192;
        m.rms_eps = 1e-5f; m.rope_theta = 5e5f; m.rope_scale = 1.0f;
        m.n_kv = type ? 3 : 0; m.kv_key = keys; m.kv_val = vals;
        RadBuildCtx c = build_ctx();
        return llama_fp8::declare(b, &m, &c);
    };
    for (const char* refused : {"llama3", "longrope", "dynamic"}) {
        RadBuilder b;
        CHECK_EQ(declare(refused, "8", &b), RAD_E_UNSUPPORTED);
    }
    {
        RadBuilder b;
        CHECK_OK(declare("linear", "2", &b));
        CHECK(count_op(b, "rope") > 0);
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 2.0);
            CHECK(!param(&op, "yarn_factor"));
        }
    }
    {
        RadBuilder b;
        CHECK_OK(declare("yarn", "4", &b));
        CHECK(count_op(b, "rope") > 0);
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 1.0);
            CHECK_EQ(pflt(&op, "yarn_factor"), 4.0);
        }
    }
    {
        RadBuilder b;
        CHECK_OK(declare(nullptr, "", &b));
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 1.0);
            CHECK(!param(&op, "yarn_factor"));
        }
    }
}

TEST(fp8_qk_attention_normalizes_before_neox_rope) {
    using namespace rad::arch;
    RadModelMeta meta{};
    meta.arch_id = "hyv3"; meta.name = "test-hyv3"; meta.quant = "fp8";
    meta.n_layers = 1; meta.n_embd = 2048; meta.n_head = 32; meta.n_head_kv = 4;
    meta.head_dim = 128; meta.n_ff = 6912; meta.n_vocab = 120832;
    meta.n_ctx_train = 262144; meta.rms_eps = 1e-5f; meta.rope_theta = 11158840.0f;
    RadBuildCtx ctx = build_ctx();
    Geom g{};
    CHECK_OK(geom_from(g, &meta, &ctx, "bf16", RAD_BF16));

    RadBuilder b;
    b.refuse.insert("qkv_norm_rope_kv_store"); // exercise the original unfused order
    Names nm("");
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL; kd.dtype = RAD_BF16; kd.n_head_kv = g.n_head_kv;
    kd.head_dim = g.head_dim;
    const rad_kvgroup kv = rad_decl_kv_group(&b, "kv_attn", &kd);
    CHECK_OK(rad_bind_layer_kv(&b, 0, kv));

    AttnQKFP8::Wire w{};
    w.x = decl_b(&b, nm.f("x"), RAD_BF16, {g.max_tok, g.n_embd});
    w.h.x = decl_b(&b, nm.f("h"), RAD_BF16, {g.max_tok, g.n_embd});
    CHECK_OK(w.h.declare_qs(&b, nm, g, "h", g.n_embd));
    w.q = decl_b(&b, nm.f("q"), RAD_BF16, {g.max_tok, g.q_dim()});
    w.k = decl_b(&b, nm.f("k"), RAD_BF16, {g.max_tok, g.kv_dim()});
    w.v = decl_b(&b, nm.f("v"), RAD_BF16, {g.max_tok, g.kv_dim()});
    w.attn.x = decl_b(&b, nm.f("attn"), RAD_BF16, {g.max_tok, g.q_dim()});
    CHECK_OK(w.attn.declare_qs(&b, nm, g, "attn", g.q_dim()));
    const rad_op init = rw(&b, RAD_OP(&b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_STR("dtype", g.dtype), RAD_F64("value", 0.0)),
                             RAD_NOWEIGHTS),
                         {}, {w.x});

    AttnQKFP8::Src src{};
    src.norm = nm.ckpt("model.layers.0.input_layernorm.weight");
    src.q = nm.ckpt("model.layers.0.self_attn.q_proj.weight");
    src.k = nm.ckpt("model.layers.0.self_attn.k_proj.weight");
    src.v = nm.ckpt("model.layers.0.self_attn.v_proj.weight");
    src.q_norm = nm.ckpt("model.layers.0.self_attn.q_norm.weight");
    src.k_norm = nm.ckpt("model.layers.0.self_attn.k_norm.weight");
    src.o = nm.ckpt("model.layers.0.self_attn.o_proj.weight");
    AttnQKFP8 attn;
    CHECK_OK(attn.declare(&b, nm, g, 0, kv, w, src));

    const RecWeight* qw = find_w(b, "blk.0.attn_q.weight");
    const RecWeight* kw = find_w(b, "blk.0.attn_k.weight");
    const RecWeight* qn = find_w(b, "blk.0.attn_q_norm.weight");
    CHECK(qw && kw && qn);
    CHECK_EQ(qw->d.shape[0], 4096); CHECK_EQ(qw->d.shape[1], 2048);
    CHECK_EQ(kw->d.shape[0], 512); CHECK_EQ(kw->d.shape[1], 2048);
    CHECK_EQ(qn->d.shape[0], 128); CHECK_EQ(qn->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(qn->d.shard, RAD_SHARD_NONE);
    int q_norms = 0, k_norms = 0;
    for (const auto& op : b.ops) if (op.op == "rmsnorm" && pint(&op, "n") == 128) {
        if (op.w.size() == 1 && op.w[0] == attn.w_q_norm) ++q_norms;
        if (op.w.size() == 1 && op.w[0] == attn.w_k_norm) ++k_norms;
    }
    CHECK_EQ(q_norms, 1); CHECK_EQ(k_norms, 1);

    RadBatch batch = decode_batch();
    RadCtx run{}; run.b = &b; run.batch = &batch;
    RAD_ISSUE(&run, init, brows(w.x, batch.n_tok));
    attn.step(&run, &batch);
    auto issue_of = [&](rad_op h) {
        for (size_t i = 0; i < run.issues.size(); ++i)
            if (run.issues[i].op == h) return (int)i;
        return -1;
    };
    const int q_norm_issue = issue_of(attn.op_q_norm), k_norm_issue = issue_of(attn.op_k_norm);
    const int q_rope_issue = issue_of(attn.op_rope_q), k_rope_issue = issue_of(attn.op_rope_k);
    const int attn_issue = issue_of(attn.op_attn);
    CHECK(q_norm_issue >= 0); CHECK(k_norm_issue >= 0); CHECK(attn_issue >= 0);
    CHECK(q_rope_issue >= 0); CHECK(k_rope_issue >= 0);
    CHECK(q_norm_issue < q_rope_issue); CHECK(k_norm_issue < k_rope_issue);
    CHECK(q_rope_issue < attn_issue); CHECK(k_rope_issue < attn_issue);
    const auto lint = lint_step(b, {&run});
    CHECK(lint.empty());
}

static RadModelMeta hyv3_fp8_test_meta() {
    static const char* keys[] = {
        "qk_norm", "moe_router_use_sigmoid", "moe_router_enable_expert_bias", "route_norm",
        "enable_moe_fp32_combine", "attention_bias", "mlp_bias", "first_k_dense_replace",
        "num_experts", "num_experts_per_tok", "num_shared_experts", "expert_hidden_dim",
        "moe_intermediate_size", "router_scaling_factor", "mlp_layer_types",
        "hyv3.kv_cache.0.k_scale", "hyv3.kv_cache.0.v_scale",
        "hyv3.kv_cache.1.k_scale", "hyv3.kv_cache.1.v_scale",
    };
    static const char* vals[] = {
        "1", "1", "1", "1", "1", "0", "0", "1", "128", "8", "1", "768", "1536",
        "2.826", "dense sparse", "0.0376674123", "0.0004359654",
        "0.0224609375", "0.0005318778",
    };
    RadModelMeta meta{};
    meta.arch_id = "hyv3"; meta.name = "test-hyv3"; meta.quant = "fp8";
    meta.n_layers = 2; meta.n_embd = 2048; meta.n_head = 32; meta.n_head_kv = 4;
    meta.head_dim = 128; meta.n_ff = 6912; meta.n_vocab = 120832;
    meta.n_ctx_train = 262144; meta.rms_eps = 1e-5f; meta.rope_theta = 11158840.0f;
    meta.n_expert = 128; meta.n_expert_used = 8;
    meta.n_kv = (int)(sizeof(keys) / sizeof(keys[0])); meta.kv_key = keys; meta.kv_val = vals;
    return meta;
}

TEST(hyv3_bf16_declares_checkpoint_geometry_and_plain_moe) {
    RadModelMeta meta = hyv3_fp8_test_meta();
    meta.quant = "";
    RadArchProbe probe{};
    CHECK_OK(hyv3_bf16::probe(&meta, &probe));
    RadBuilder b;
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    b.encs = {
        { "attn_q.weight", plain }, { "attn_k.weight", plain },
        { "attn_v.weight", plain }, { "attn_output.weight", plain },
        { "ffn_gate_up.weight", plain }, { "ffn_down.weight", plain },
        { "ffn_gate_up_exps", plain }, { "ffn_down_exps", plain },
        { "ffn_gate_up_shexp.weight", plain }, { "ffn_down_shexp.weight", plain },
        { "output.weight", plain },
    };
    RadBuildCtx ctx = build_ctx();
    const int rc = hyv3_bf16::declare(&b, &meta, &ctx);
    CHECK_EQ(rc, RAD_OK);
    if (rc < 0) return;
    const RecWeight* q_norm = find_w(b, "blk.0.attn_q_norm.weight");
    const RecWeight* k_norm = find_w(b, "blk.0.attn_k_norm.weight");
    const RecWeight* q = find_w(b, "blk.0.attn_q.weight");
    const RecWeight* k = find_w(b, "blk.0.attn_k.weight");
    const RecWeight* v = find_w(b, "blk.0.attn_v.weight");
    const RecWeight* o = find_w(b, "blk.0.attn_output.weight");
    const RecWeight* dense = find_w(b, "blk.0.ffn_gate_up.weight");
    const RecWeight* dense_down = find_w(b, "blk.0.ffn_down.weight");
    const RecWeight* expert = find_w(b, "blk.1.ffn_gate_up_exps.0.weight");
    const RecWeight* expert_down = find_w(b, "blk.1.ffn_down_exps.0.weight");
    const RecWeight* shared = find_w(b, "blk.1.ffn_gate_up_shexp.weight");
    const RecWeight* shared_down = find_w(b, "blk.1.ffn_down_shexp.weight");
    const RecWeight* router = find_w(b, "blk.1.glm_router.weight");
    const RecWeight* correction = find_w(b, "blk.1.glm_router.correction");
    REQUIRE(q_norm && k_norm && q && k && v && o && dense && dense_down && expert &&
            expert_down && shared && shared_down && router && correction);
    CHECK_EQ(q_norm->d.shape[0], hyv3_bf16_fixture::q_norm[0]);
    CHECK_EQ(k_norm->d.shape[0], hyv3_bf16_fixture::k_norm[0]);
    CHECK_EQ(router->d.shape[0], hyv3_bf16_fixture::router[0]);
    CHECK_EQ(router->d.shape[1], hyv3_bf16_fixture::router[1]);
    CHECK_EQ(correction->d.shape[0], hyv3_bf16_fixture::correction[0]);
    CHECK_EQ(q->d.shape[0], hyv3_bf16_fixture::q_proj[0]);
    CHECK_EQ(q->d.shape[1], hyv3_bf16_fixture::q_proj[1]);
    CHECK_EQ(k->d.shape[0], hyv3_bf16_fixture::k_proj[0]);
    CHECK_EQ(k->d.shape[1], hyv3_bf16_fixture::k_proj[1]);
    CHECK_EQ(v->d.shape[0], hyv3_bf16_fixture::v_proj[0]);
    CHECK_EQ(v->d.shape[1], hyv3_bf16_fixture::v_proj[1]);
    CHECK_EQ(o->d.shape[0], hyv3_bf16_fixture::o_proj[0]);
    CHECK_EQ(o->d.shape[1], hyv3_bf16_fixture::o_proj[1]);
    CHECK_EQ(dense->d.shape[0], hyv3_bf16_fixture::dense_gate[0] +
                                  hyv3_bf16_fixture::dense_up[0]);
    CHECK_EQ(dense->d.shape[1], hyv3_bf16_fixture::dense_gate[1]);
    CHECK_EQ(dense_down->d.shape[0], hyv3_bf16_fixture::dense_down[0]);
    CHECK_EQ(dense_down->d.shape[1], hyv3_bf16_fixture::dense_down[1]);
    CHECK_EQ(expert->d.shape[0], hyv3_bf16_fixture::expert_gate[0] +
                                  hyv3_bf16_fixture::expert_up[0]);
    CHECK_EQ(expert->d.shape[1], hyv3_bf16_fixture::expert_gate[1]);
    CHECK_EQ(expert_down->d.shape[0], hyv3_bf16_fixture::expert_down[0]);
    CHECK_EQ(expert_down->d.shape[1], hyv3_bf16_fixture::expert_down[1]);
    CHECK_EQ(shared->d.shape[0], hyv3_bf16_fixture::shared_gate[0] +
                                  hyv3_bf16_fixture::shared_up[0]);
    CHECK_EQ(shared->d.shape[1], hyv3_bf16_fixture::shared_gate[1]);
    CHECK_EQ(shared_down->d.shape[0], hyv3_bf16_fixture::shared_down[0]);
    CHECK_EQ(shared_down->d.shape[1], hyv3_bf16_fixture::shared_down[1]);
    const auto& moe = hyv3_bf16::models[0].layers[1].moe;
    CHECK_EQ(moe.c.n_ff_exp, 768);
    CHECK_EQ(moe.c.n_ff_shared, 768);
    CHECK_EQ(moe.c.expert_bf16, 1);
    RadBatch batch = decode_batch();
    RadCtx run{}; run.b = &b; run.batch = &batch;
    hyv3_bf16::step(&run, &batch);
    CHECK(lint_step(b, {&run}).empty());
}

TEST(hyv3_gguf_maps_llamacpp_moe_tensors_and_lints_tp2) {
    RadModelMeta meta = hyv3_fp8_test_meta();
    meta.quant = "gguf_ftype_15";
    RadArchProbe probe{};
    CHECK_OK(hyv3_gguf::probe(&meta, &probe));
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    for (int rank = 0; rank < 2; ++rank) {
        RadBuilder b;
        b.encs = {
            { "attn_q.weight", plain }, { "attn_k.weight", plain },
            { "attn_v.weight", plain }, { "attn_output.weight", plain },
            { "ffn_gate_up.weight", plain }, { "ffn_down.weight", plain },
            { "ffn_gate_up_exps", plain }, { "ffn_down_exps", plain },
            { "ffn_gate_up_shexp.weight", plain }, { "ffn_down_shexp.weight", plain },
            { "output.weight", plain },
        };
        RadBuildCtx ctx = build_ctx(rank, 2);
        const int rc = hyv3_gguf::declare(&b, &meta, &ctx);
        CHECK_EQ(rc, RAD_OK);
        if (rc < 0) continue;
        auto mapped = [&](const char* target, const char* source) {
            for (const RecMap& m : b.maps)
                if (m.declared == target && m.src.size() == 1 && m.src[0] == source) return true;
            return false;
        };
        CHECK(mapped("blk.0.attn_q.weight", "blk.0.attn_q.weight"));
        CHECK(mapped("blk.1.glm_router.weight", "blk.1.ffn_gate_inp.weight"));
        CHECK(mapped("blk.1.glm_router.correction", "blk.1.exp_probs_b"));
        CHECK(mapped("token_embd.weight", "token_embd.weight"));
        const std::string expert_name =
            "blk.1.ffn_gate_up_exps." + std::to_string(rank * 64) + ".weight";
        const std::string down_name =
            "blk.1.ffn_down_exps." + std::to_string(rank * 64) + ".weight";
        bool expert_gate_up = false, expert_down = false, shared_gate_up = false;
        for (const RecMap& m : b.maps) {
            if (m.declared == expert_name && m.src.size() == 2 &&
                m.src[0] == "blk.1.ffn_gate_exps.weight" &&
                m.src[1] == "blk.1.ffn_up_exps.weight") expert_gate_up = true;
            if (m.declared == down_name && m.src.size() == 1 &&
                m.src[0] == "blk.1.ffn_down_exps.weight") expert_down = true;
            if (m.declared == "blk.1.ffn_gate_up_shexp.weight" && m.src.size() == 2 &&
                m.src[0] == "blk.1.ffn_gate_shexp.weight" &&
                m.src[1] == "blk.1.ffn_up_shexp.weight") shared_gate_up = true;
        }
        CHECK(expert_gate_up); CHECK(expert_down); CHECK(shared_gate_up);
        const auto& moe = hyv3_gguf::models[rank].layers[1].moe;
        CHECK_EQ(moe.c.expert_bf16, 1);
        RadBatch batch = decode_batch();
        RadCtx run{}; run.b = &b; run.batch = &batch;
        hyv3_gguf::step(&run, &batch);
        CHECK(lint_step(b, {&run}).empty());
    }
}

TEST(hyv3_bf16_selects_native_int4_expert_kernels_from_recipe_encodings) {
    RadModelMeta meta = hyv3_fp8_test_meta();
    meta.quant = "";
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    RadEncoding w4a = rad_enc_affine(RAD_U4, RAD_F16, 1, 128);
    rad_enc_add_plane(&w4a, "zero", RAD_U8, 1, 128);
    for (int rank = 0; rank < 2; ++rank) {
        RadBuilder b;
        b.encs = {
            { "attn_q.weight", w4a }, { "attn_k.weight", w4a },
            { "attn_v.weight", w4a }, { "attn_output.weight", w4a },
            { "ffn_gate_up.weight", w4a }, { "ffn_down.weight", w4a },
            { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 },
            { "ffn_gate_up_shexp.weight", plain }, { "ffn_down_shexp.weight", plain },
            { "output.weight", plain },
        };
        RadBuildCtx ctx = build_ctx(rank, 2);
        const int rc = hyv3_bf16::declare(&b, &meta, &ctx);
        CHECK_EQ(rc, RAD_OK);
        if (rc < 0) continue;
        const auto& moe = hyv3_bf16::models[rank].layers[1].moe;
        CHECK_EQ(moe.c.expert_w4, 1);
        CHECK_EQ(moe.c.expert_bf16, 0);
        CHECK_EQ(moe.c.n_ff_exp, 768);
        CHECK_EQ(moe.c.expert_base, rank * 64);
        int asym_w4_gemms = 0;
        for (const RecOp& op : b.ops) {
            const RecParam* dtype = op.op == "gemm_nt_q" ? param(&op, "dtype") : nullptr;
            if (dtype && dtype->sval == "w4a8i8_asym") ++asym_w4_gemms;
        }
        CHECK(asym_w4_gemms > 0);
        RadBatch batch = decode_batch();
        RadCtx run{}; run.b = &b; run.batch = &batch;
        hyv3_bf16::step(&run, &batch);
        CHECK(lint_step(b, {&run}).empty());
    }
}

TEST(hyv3_bf16_supports_int8_linear_recipes_at_tp2) {
    RadModelMeta meta = hyv3_fp8_test_meta();
    meta.quant = "";
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    const RadEncoding w8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    for (int rank = 0; rank < 2; ++rank) {
        RadBuilder b;
        b.encs = {
            { "attn_q.weight", w8 }, { "attn_k.weight", w8 },
            { "attn_v.weight", w8 }, { "attn_output.weight", w8 },
            { "ffn_gate_up.weight", w8 }, { "ffn_down.weight", w8 },
            { "ffn_gate_up_exps", plain }, { "ffn_down_exps", plain },
            { "ffn_gate_up_shexp.weight", w8 }, { "ffn_down_shexp.weight", w8 },
            { "output.weight", w8 },
        };
        RadBuildCtx ctx = build_ctx(rank, 2);
        const int rc = hyv3_bf16::declare(&b, &meta, &ctx);
        CHECK_EQ(rc, RAD_OK);
        if (rc < 0) continue;
        int i8_gemms = 0;
        for (const RecOp& op : b.ops) {
            const RecParam* dtype = op.op == "gemm_nt_q" ? param(&op, "dtype") : nullptr;
            if (dtype && dtype->sval == "i8a8") ++i8_gemms;
        }
        CHECK(i8_gemms > 0);
        CHECK_EQ(find_w(b, "output.weight")->d.dtype, (uint32_t)RAD_I8);
        const auto& moe = hyv3_bf16::models[rank].layers[1].moe;
        CHECK_EQ(moe.c.expert_bf16, 1);
        RadBatch batch = decode_batch();
        RadCtx run{}; run.b = &b; run.batch = &batch;
        hyv3_bf16::step(&run, &batch);
        CHECK(lint_step(b, {&run}).empty());
    }
}

TEST(hyv3_folds_residual_adds_into_the_next_norm_but_materializes_the_final_output) {
    RadModelMeta meta = hyv3_fp8_test_meta();
    meta.quant = "";
    meta.n_layers = 3; // dense, a non-final MoE, and the final MoE
    std::vector<const char*> vals(meta.kv_val, meta.kv_val + meta.n_kv);
    for (int i = 0; i < meta.n_kv; ++i)
        if (std::string(meta.kv_key[i]) == "mlp_layer_types") vals[(size_t)i] = "dense sparse sparse";
    meta.kv_val = vals.data();
    RadEncoding w4a = rad_enc_affine(RAD_U4, RAD_F16, 1, 128);
    rad_enc_add_plane(&w4a, "zero", RAD_U8, 1, 128);
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    for (int world : { 1, 2 }) for (int rank = 0; rank < world; ++rank) {
        RadBuilder b;
        b.encs = {
            { "attn_q.weight", w4a }, { "attn_k.weight", w4a },
            { "attn_v.weight", w4a }, { "attn_output.weight", w4a },
            { "ffn_gate_up.weight", w4a }, { "ffn_down.weight", w4a },
            { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 },
            { "ffn_gate_up_shexp.weight", plain }, { "ffn_down_shexp.weight", plain },
            { "output.weight", plain },
        };
        RadBuildCtx ctx = build_ctx(rank, world);
        ctx.max_tok = 4096;
        REQUIRE_EQ(hyv3_bf16::declare(&b, &meta, &ctx), RAD_OK);
        const auto& m = hyv3_bf16::models[rank];
        CHECK_EQ(m.layers[0].attn.nq_h.res, (rad_buf)0);
        CHECK_EQ(m.layers[0].attn.nq_h.src, m.x);
        CHECK_EQ(m.layers[0].dense_mlp.nq_h.res, m.x);
        CHECK_EQ(m.layers[0].dense_mlp.op_add, (rad_op)0);
        CHECK_EQ(m.shared_one != 0, world == 1);
        for (size_t l = 0; l < m.layers.size(); ++l) {
            CHECK_EQ(m.layers[l].attn.op_add, (rad_op)0);
            if (l == 0) continue;
            CHECK_EQ(m.layers[l].attn.nq_h.src, m.h.x);
            CHECK_EQ(m.layers[l].attn.nq_h.res, m.x);
            CHECK_EQ(m.layers[l].moe.nq_h.src, m.h.x);
            CHECK_EQ(m.layers[l].moe.nq_h.res, m.x);
            CHECK_EQ(m.layers[l].moe.op_add != 0, l + 1 == m.layers.size());
            CHECK_EQ(m.layers[l].moe.op_combine != 0, world > 1);
            CHECK_EQ(m.layers[l].moe.gfold, world == 1); // both arms still round before adding
            const RecOp* norm = b.op_of(m.layers[l].moe.nq_h.op);
            REQUIRE(norm);
            CHECK(std::find(norm->reads.begin(), norm->reads.end(), m.x) != norm->reads.end());
            CHECK(std::find(norm->writes.begin(), norm->writes.end(), m.x) != norm->writes.end());
        }
        CHECK_EQ(count_op(b, "add"), world > 1 ? 3 : 1); // TP2 combines each reduced arm separately
        for (int rows : { 1, 8, 65, 2049, 4096 }) {
            RadBatch batch = decode_batch(); batch.n_tok = rows; batch.max_q_len = rows;
            RadCtx run{}; run.b = &b; run.batch = &batch; run.rank = rank;
            hyv3_bf16::step(&run, &batch);
            CHECK(lint_step(b, {&run}).empty());
            int final_adds = 0, ones = 0;
            for (const auto& issue : run.issues) {
                if (issue.op == m.layers.back().moe.op_add) ++final_adds;
                if (issue.op == m.shared_one) ++ones;
                for (size_t l = 1; l < m.layers.size(); ++l)
                    CHECK(issue.op != m.layers[l].moe.op_sgate); // no per-layer ones fills
            }
            CHECK_EQ(final_adds, (rows + 2047) / 2048);
            CHECK_EQ(ones, world == 1 ? 1 : 0);
        }
    }
}

TEST(hyv3_decode_fuses_qk_norm_rope_and_store_with_unfused_fallbacks) {
    for (int form = 0; form < 3; ++form) {
        RadModelMeta meta = hyv3_fp8_test_meta();
        RadBuildCtx ctx = build_ctx();
        RadBuilder b;
        if (form == 1) b.refuse.insert("qkv_norm_rope_kv_store");
        if (form == 2) ctx.kv_dtype = RAD_I8;
        CHECK_OK(hyv3_fp8::declare(&b, &meta, &ctx));
        const auto& a = hyv3_fp8::models[0].layers[0].attn;
        CHECK_EQ(a.op_qkr_store != 0, form == 0);
        for (const int rows : {1, 64, 65}) {
            RadBatch batch = decode_batch();
            batch.n_tok = rows; batch.max_q_len = rows;
            RadCtx run{}; run.b = &b; run.batch = &batch;
            hyv3_fp8::step(&run, &batch);
            auto issued = [&](rad_op h) {
                return std::any_of(run.issues.begin(), run.issues.end(),
                                   [&](const auto& i) { return i.op == h; });
            };
            const bool folded = form == 0 && rows <= 64;
            CHECK_EQ(issued(a.op_qkr_store), folded);
            CHECK_EQ(issued(a.op_q_norm), !folded);
            CHECK_EQ(issued(a.op_k_norm), !folded);
            CHECK_EQ(issued(a.op_rope_q), !folded);
            CHECK_EQ(issued(a.op_rope_k), !folded);
            CHECK_EQ(issued(a.op_kv_store), !folded);
            CHECK(lint_step(b, {&run}).empty());
        }
    }
}

TEST(hyv3_fp8_declares_dense_then_sigmoid_moe_with_fp32_combine) {
    RadModelMeta meta = hyv3_fp8_test_meta();
    RadBuildCtx ctx = build_ctx();
    RadBuilder b;
    RadArchProbe probe{};
    CHECK_OK(hyv3_fp8::probe(&meta, &probe));
    const int rc = hyv3_fp8::declare(&b, &meta, &ctx);
    CHECK_EQ(rc, RAD_OK);
    if (rc < 0) return;

    const RecWeight* q_norm = find_w(b, "blk.0.attn_q_norm.weight");
    const RecWeight* dense = find_w(b, "blk.0.ffn_gate_up.weight");
    const RecWeight* expert = find_w(b, "blk.1.ffn_gate_up_exps.0.weight");
    const RecMap* expert_map = find_map(b, "blk.1.ffn_gate_up_exps.0.weight");
    const RecWeight* shared = find_w(b, "blk.1.ffn_gate_up_shexp.weight");
    const RecWeight* router = find_w(b, "blk.1.glm_router.weight");
    const RecWeight* bias = find_w(b, "blk.1.glm_router.correction");
    CHECK(q_norm && dense && expert && expert_map && shared && router && bias);
    CHECK_EQ(expert_map->src.size(), 2u);
    CHECK_EQ(expert_map->src[0], std::string("model.layers.1.mlp.experts.0.gate_proj.weight"));
    CHECK_EQ(expert_map->src[1], std::string("model.layers.1.mlp.experts.0.up_proj.weight"));
    CHECK_EQ(q_norm->d.shape[0], 128);
    CHECK_EQ(dense->d.shape[0], 13824); CHECK_EQ(dense->d.shape[1], 2048);
    CHECK_EQ(expert->d.shape[0], 1536); CHECK_EQ(expert->d.shape[1], 2048);
    CHECK_EQ(shared->d.shape[0], 1536); CHECK_EQ(shared->d.shape[1], 2048);
    CHECK_EQ(router->d.shape[0], 128); CHECK_EQ(router->d.shape[1], 2048);
    CHECK_EQ(bias->d.shape[0], 128); CHECK_EQ(bias->d.dtype, (uint32_t)RAD_F32);

    auto& model = hyv3_fp8::models[0];
    auto& moe = model.layers[1].moe;
    CHECK_EQ(moe.c.n_ff_exp, 768); CHECK_EQ(moe.c.n_ff_shared, 768);
    CHECK_EQ(moe.c.expert_f32_scale, 1);
    CHECK_EQ(moe.c.route_scale, 2.826); CHECK_EQ(moe.c.sigmoid_router, 1);
    CHECK_EQ(moe.c.shared_fp32_combine, 1); CHECK_EQ(moe.gfold, true);
    const RecOp* gather = b.op_of(moe.op_gather);
    CHECK(gather != nullptr);
    CHECK_EQ(gather->writes.size(), 1u);
    CHECK_EQ(gather->writes[0], moe.w.h.x);
    CHECK_EQ(moe.op_combine, (rad_op)0);
    CHECK(model.shared_one != 0);
    CHECK(gather->reads == (std::vector<rad_buf>{moe.w.edn, moe.w.ew,
                                               moe.gather_inv ? moe.w.inv : moe.w.sorted,
                                               moe.w.sout, moe.w.sgate}));

    RadBatch batch = decode_batch();
    RadCtx run{}; run.b = &b; run.batch = &batch;
    hyv3_fp8::step(&run, &batch);
    auto issue_of = [&](rad_op h) {
        for (size_t i = 0; i < run.issues.size(); ++i)
            if (run.issues[i].op == h) return (int)i;
        return -1;
    };
    const int gather_i = issue_of(moe.op_gather), add_i = issue_of(moe.op_add);
    CHECK(gather_i >= 0 && add_i >= 0 && gather_i < add_i);
    REQUIRE(moe.op_sigmoid_scatter);
    CHECK(issue_of(moe.op_sigmoid_scatter) >= 0);
    CHECK_EQ(issue_of(moe.op_scatter), -1);
    CHECK_EQ(issue_of(moe.op_topk), -1);
    const auto lint = lint_step(b, {&run});
    CHECK(lint.empty());
}

TEST(hyv3_fp8_tensor_parallel_partitions_experts_and_shared_width) {
    RadModelMeta meta = hyv3_fp8_test_meta();
    for (int rank = 0; rank < 2; ++rank) {
        RadBuildCtx ctx = build_ctx(rank, 2);
        ctx.kv_dtype = RAD_I8;
        RadBuilder b;
        const int rc = hyv3_fp8::declare(&b, &meta, &ctx);
        CHECK_EQ(rc, RAD_OK);
        if (rc < 0) continue;
        auto& model = hyv3_fp8::models[rank];
        const auto& moe = model.layers[1].moe;
        CHECK_EQ(moe.c.n_expert, 64);
        CHECK_EQ(moe.c.expert_base, rank * 64);
        CHECK_EQ(moe.c.n_ff_shared, 384);
        CHECK_EQ(moe.c.shared_lo, rank * 384);
        CHECK_EQ(moe.c.shared_hi, (rank + 1) * 384);
        CHECK(moe.op_moe_ar && moe.op_shared_ar && !moe.op_ar);
        CHECK(moe.op_combine && !moe.gfold && !model.shared_one);
        CHECK(find_w(b, rank == 0 ? "blk.1.ffn_gate_up_exps.0.weight"
                                   : "blk.1.ffn_gate_up_exps.64.weight") != nullptr);
        CHECK_EQ(model.g.n_ff, 3456);
        RadBatch batch = decode_batch();
        RadCtx run{}; run.b = &b; run.batch = &batch; run.rank = rank;
        hyv3_fp8::step(&run, &batch);
        auto issue_of = [&](rad_op h) {
            for (size_t i = 0; i < run.issues.size(); ++i)
                if (run.issues[i].op == h) return (int)i;
            return -1;
        };
        CHECK(issue_of(moe.op_gather) < issue_of(moe.op_moe_ar));
        CHECK(issue_of(moe.op_moe_ar) < issue_of(moe.op_combine));
        CHECK(issue_of(moe.op_shared_ar) < issue_of(moe.op_combine));
        CHECK(lint_step(b, {&run}).empty());
    }
}

/* --kv-cache-dtype i8 on HY-Memory: the cache group is int8 at the model's head_dim, the store and
 * the attention band on it, and no descale is declared (an int8 row carries its own scales). */
TEST(hyv3_serves_the_int8_cache_without_descales) {
    RadModelMeta meta = hyv3_fp8_test_meta();
    RadBuildCtx ctx = build_ctx();
    ctx.kv_dtype = RAD_I8;
    RadBuilder b;
    const int rc = hyv3_fp8::declare(&b, &meta, &ctx);
    CHECK_EQ(rc, RAD_OK);
    if (rc < 0) return;
    CHECK_EQ(b.kv_groups[0].d.dtype, (uint32_t)RAD_I8);
    auto& attn = hyv3_fp8::models[0].layers[0].attn;
    const RecWeight* q = find_w(b, "blk.0.attn_q.weight");
    CHECK(q != nullptr);
    if (q) CHECK_EQ(q->d.shape[0], 4096);
    const RecOp* store = b.op_of(attn.op_kv_store);
    CHECK(store != nullptr);
    if (store) {
        CHECK_EQ(std::string(pstr(store, "kv_dtype")), std::string("i8"));
        CHECK(param(store, "k_scale") == nullptr);
    }
    RadBatch batch = decode_batch();
    RadCtx run{}; run.b = &b; run.batch = &batch;
    hyv3_fp8::step(&run, &batch);
    CHECK(lint_step(b, {&run}).empty());
}

TEST(the_alternation_puts_full_attention_at_3_and_7) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The weight manifest shows a layer's kind: full attention has a q|gate projection, linear
     * attention an in-projection, never both. */
    for (int l = 0; l < 8; ++l) {
        const bool full = ((l + 1) % 4 == 0);
        const std::string qg  = "blk." + std::to_string(l) + ".attn_qg.weight";
        const std::string in  = "blk." + std::to_string(l) + ".ssm_in.weight";
        CHECK_EQ(find_w(b, qg) != nullptr, full);
        CHECK_EQ(find_w(b, in) != nullptr, !full);
        /* Both kinds share the pre-norm name and feed-forward. */
        CHECK(find_w(b, "blk." + std::to_string(l) + ".attn_norm.weight") != nullptr);
        CHECK(find_w(b, "blk." + std::to_string(l) + ".ffn_gate_up.weight") != nullptr);
    }
}

TEST(attention_shapes_at_the_27b_dimensions) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* q_dim = 24 * 256 = 6144; the projection is twice that, query and gate interleaved. */
    const RecWeight* qg = find_w(b, "blk.3.attn_qg.weight");
    CHECK(qg != nullptr);
    CHECK_EQ(qg->d.rank, 2u);
    CHECK_EQ(qg->d.shape[0], 12288);
    CHECK_EQ(qg->d.shape[1], 5120);
    CHECK_EQ(qg->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(qg->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(qg->d.access, RAD_ACCESS_PER_TOKEN);

    /* k and v are separate projections, 4 kv heads of 256. */
    const RecWeight* kw = find_w(b, "blk.3.attn_k.weight");
    const RecWeight* vw = find_w(b, "blk.3.attn_v.weight");
    CHECK(kw != nullptr && vw != nullptr);
    CHECK_EQ(kw->d.shape[0], 1024);
    CHECK_EQ(vw->d.shape[0], 1024);
    CHECK_EQ(kw->d.shard, RAD_SHARD_ROW);

    const RecWeight* o = find_w(b, "blk.3.attn_output.weight");
    CHECK(o != nullptr);
    CHECK_EQ(o->d.shape[0], 5120);
    CHECK_EQ(o->d.shape[1], 6144);
    CHECK_EQ(o->d.shard, RAD_SHARD_COL);

    /* Per-head gains are head_dim wide and replicated, one vector shared by all heads. */
    const RecWeight* qn = find_w(b, "blk.3.attn_q_norm.weight");
    CHECK(qn != nullptr);
    CHECK_EQ(qn->d.shape[0], 256);
    CHECK_EQ(qn->d.shard, RAD_SHARD_NONE);
    CHECK_EQ(qn->d.dtype, (uint32_t)RAD_F32);

    /* The interleaved projection buffer is rank 3, [tokens, heads, 2*head_dim], so query and gate
     * are addressable as column slices. */
    const RecBuf* qgb = find_b(b, "attn_qg");
    CHECK(qgb != nullptr);
    CHECK_EQ(qgb->d.rank, 3u);
    CHECK_EQ(qgb->d.shape[0], 256);
    CHECK_EQ(qgb->d.shape[1], 24);
    CHECK_EQ(qgb->d.shape[2], 512);

    /* qk_norm_rope_gate must stay undeclared until something builds its cos/sin table (see
     * attn_layer_ops); the four ops it stands for carry the same numbers. */
    CHECK(find_op(b, "qk_norm_rope_gate") == nullptr);

    const RecOp* qn2 = find_op(b, "rope");
    CHECK(qn2 != nullptr);
    if (qn2) {
        CHECK_EQ(pint(qn2, "head_dim"), 256);
        CHECK_EQ(pint(qn2, "rotary_dim"), 64);   /* partial_rotary_factor 0.25 of 256 */
        CHECK_EQ(pstr(qn2, "mode"), std::string("mrope"));
        CHECK_NEAR(pflt(qn2, "theta"), 1e7, 1.0);
    }
    const RecOp* at = find_op(b, "attn_paged");
    CHECK_EQ(pint(at, "head_dim"), 256);
    CHECK_EQ(pint(at, "gqa"), 6);
    CHECK_EQ(pint(at, "causal"), 1);
    CHECK_EQ(pint(at, "window"), 0);
    /* block_size is declared deliberately. It is RAD_DERIVED (kernel-supplied) and every libr4d
     * attention row pins C_EQ("block_size", 16), so 16 changes nothing on the accelerated path.
     * But a geometry no accelerated row serves (e.g. head_dim 256 at gqa 4) resolves to libref,
     * which accepts any block size and names none, leaving the KV group unsizable. The block size
     * belongs to the cache the model asks for, so the model declares it. */
    CHECK_EQ(pint(at, "block_size"), 16);

    /* ...and it reaches kv_store, which needs it as a parameter. */
    const RecOp* ks = find_op(b, "kv_store");
    CHECK_EQ(pint(ks, "block_size"), 16);
    CHECK_EQ(pint(ks, "n_head_kv"), 4);

    /* The gate: a per-head sigmoid over head_dim, then an elementwise multiply over the attention
     * output, `out = attn * sigmoid(gate)`. */
    const RecOp* sg = find_op(b, "sigmoid");
    CHECK_EQ(pint(sg, "n"), 256);
    CHECK_EQ(param(sg, "M")->ihi, 256LL * 24);
    const RecOp* mu = find_op(b, "mul");
    CHECK_EQ(pint(mu, "n"), 6144);
}

TEST(gdn_shapes_and_both_paths) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* conv_dim = 2*16*128 + 48*128 = 10240, followed by one `a` and one `b` per value head:
     * 10240 + 96 = 10336. The fusion lets gdn_conv_prep read the decay and beta inputs from `x`. */
    const RecWeight* in = find_w(b, "blk.0.ssm_in.weight");
    CHECK(in != nullptr);
    CHECK_EQ(in->d.shape[0], 10336);
    CHECK_EQ(in->d.shape[1], 5120);
    CHECK_EQ(in->d.shard, RAD_SHARD_ROW);

    const RecWeight* z = find_w(b, "blk.0.ssm_z.weight");
    CHECK(z != nullptr);
    CHECK_EQ(z->d.shape[0], 6144);            /* 48 value heads of 128 */

    const RecWeight* cv = find_w(b, "blk.0.ssm_conv1d.weight");
    CHECK(cv != nullptr);
    CHECK_EQ(cv->d.rank, 2u);
    CHECK_EQ(cv->d.shape[0], 10240);          /* depthwise: one filter per convolved channel */
    CHECK_EQ(cv->d.shape[1], 4);
    CHECK_EQ(cv->d.shard, RAD_SHARD_ROW);
    /* Must be BF16, and nothing else checks it: libr4d's conv reads this pointer as bf16, so an
     * f32 plane would be misread on the device path only (libref reads the operand's own dtype). */
    CHECK_EQ(cv->d.dtype, (uint32_t)RAD_BF16);
    /* Qwen3.5's conv1d has no bias, so none is declared; the operand is passed absent. */
    CHECK(find_w(b, "blk.0.ssm_conv1d.bias") == nullptr);

    /* The decay weight is named for its convention: it is a log and the kernels take exp(). */
    const RecWeight* al = find_w(b, "blk.0.ssm_a_log");
    CHECK(al != nullptr);
    CHECK_EQ(al->d.shape[0], 48);
    CHECK_EQ(al->d.dtype, (uint32_t)RAD_F32);
    CHECK(find_w(b, "blk.0.ssm_a") == nullptr);

    const RecWeight* on = find_w(b, "blk.0.ssm_norm.weight");
    CHECK(on != nullptr);
    CHECK_EQ(on->d.shape[0], 128);            /* over head_v, shared by every head */
    CHECK_EQ(on->d.shard, RAD_SHARD_NONE);

    const RecWeight* ow = find_w(b, "blk.0.ssm_out.weight");
    CHECK(ow != nullptr);
    CHECK_EQ(ow->d.shape[0], 5120);
    CHECK_EQ(ow->d.shape[1], 6144);
    CHECK_EQ(ow->d.shard, RAD_SHARD_COL);

    /* Both the chunked and recurrent paths are declared, so both appear in the graph dump and
     * resolve at declare. */
    CHECK_EQ(count_op(b, "gdn_conv_prep"), 6);
    CHECK_EQ(count_op(b, "gdn_kkt_solve"), 6);
    CHECK_EQ(count_op(b, "gdn_chunk_scan"), 6);
    CHECK_EQ(count_op(b, "gdn_gated_rmsnorm"), 6);
    CHECK_EQ(count_op(b, "gdn_conv_update"), 6);
    CHECK_EQ(count_op(b, "gdn_recurrent_update"), 6);

    const RecOp* cp = find_op(b, "gdn_conv_prep");
    CHECK_EQ(pint(cp, "head_k"), 128);
    CHECK_EQ(pint(cp, "head_v"), 128);
    CHECK_EQ(pint(cp, "conv_width"), 4);
    CHECK_EQ(pint(cp, "chunk"), 64);
    CHECK_EQ(param(cp, "M")->kind, RAD_P_RANGE);
    /* Weight operands: the conv weight and the two per-head vectors; no bias. */
    CHECK_EQ((int)cp->w.size(), 3);

    /* chunk is RAD_DERIVED (kernel-chosen) on the two ops that tile on it. */
    CHECK(param(find_op(b, "gdn_kkt_solve"), "chunk") == nullptr);
    CHECK(param(find_op(b, "gdn_chunk_scan"), "chunk") == nullptr);

    /* The recurrent path is ranged on q_len, since a speculative verify carries n_spec+1 rows per
     * sequence, and states n_head_v explicitly: with 48 value heads against 16 key heads, inferring
     * it from a flattened view gives a plausible wrong answer. */
    const RecOp* ru = find_op(b, "gdn_recurrent_update");
    CHECK_EQ(param(ru, "q_len")->kind, RAD_P_RANGE);
    CHECK_EQ(pint(ru, "n_head_v"), 48);
    CHECK_EQ(pstr(ru, "act"), std::string("silu"));
    CHECK_EQ(pint(find_op(b, "gdn_conv_update"), "conv_width"), 4);

    /* q and k have 16 heads and v 48: the 3:1 sharing is in the buffer shapes. */
    const RecBuf* gq = find_b(b, "gdn_q");
    const RecBuf* gv = find_b(b, "gdn_v");
    CHECK(gq != nullptr && gv != nullptr);
    CHECK_EQ(gq->d.shape[1], 16);
    CHECK_EQ(gq->d.shape[2], 128);
    CHECK_EQ(gv->d.shape[1], 48);
    CHECK_EQ(gv->d.shape[2], 128);
    /* Decay and beta are f32: the gate is a cumulative sum spanning hundreds of nats, and every
     * decay factor is an exponent of it.
     *
     * The chunk inverse is bf16 by contract: libr4d's gdn_kkt_solve writes it and gdn_chunk_scan
     * reads it as bf16, so f32 would double its size and misread every element. libref reads each
     * operand's own dtype, so only the device path would show it. */
    CHECK_EQ(find_b(b, "gdn_g")->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(find_b(b, "gdn_beta")->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(find_b(b, "gdn_kkt")->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(find_b(b, "gdn_kkt")->d.shape[2], 64);
}

TEST(three_kv_groups_with_every_layer_bound) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.kv_groups.size(), 3);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_attn"));
    CHECK_EQ(b.kv_groups[1].name, std::string("kv_gdn_state"));
    CHECK_EQ(b.kv_groups[2].name, std::string("kv_gdn_conv"));

    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.kind, RAD_KV_FULL);
    CHECK_EQ(f.n_head_kv, 4);
    CHECK_EQ(f.head_dim, 256);
    CHECK_EQ(f.dtype, (uint32_t)RAD_BF16);

    /* 48 heads of 128x128 fp32: 3 MiB per layer, 144 MiB per sequence across the real model's 48
     * linear layers. fp32 because that is the state width libr4d has a kernel for. */
    const RadKVGroupDecl& s = b.kv_groups[1].d;
    CHECK_EQ(s.kind, RAD_KV_LINEAR);
    CHECK_EQ(s.n_head_kv, 48);
    CHECK_EQ(s.state_dim[0], 128);
    CHECK_EQ(s.state_dim[1], 128);
    CHECK_EQ(s.dtype, (uint32_t)RAD_F32);

    /* The conv window is counted in heads (2*16 + 48 of width 128 = 10240 channels), so a
     * tensor-parallel split never straddles a q/k/v boundary. */
    const RadKVGroupDecl& c = b.kv_groups[2].d;
    CHECK_EQ(c.kind, RAD_KV_CONV);
    CHECK_EQ(c.n_head_kv, 80);
    CHECK_EQ(c.head_dim, 128);
    CHECK_EQ(c.conv_width, 4);

    /* Every layer is bound to the groups it owns state in; a linear layer owns two. */
    std::map<rad_kvgroup, std::set<int>> bound;
    for (const auto& p : b.binds) bound[p.second].insert(p.first);
    CHECK_EQ((int)bound[1].size(), 2);          /* full attention: layers 3 and 7 */
    CHECK_EQ((int)bound[2].size(), 6);
    CHECK_EQ((int)bound[3].size(), 6);
    CHECK(bound[1].count(3) && bound[1].count(7));
    CHECK(bound[2].count(0) && bound[2].count(6));
    CHECK(bound[2] == bound[3]);                /* the two GDN groups cover the same layers */
    for (int l = 0; l < 8; ++l)
        CHECK_EQ(bound[1].count(l) + bound[2].count(l), (size_t)1);
}

TEST(the_name_map_owns_both_sides) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The interleaved query-and-gate projection is one checkpoint tensor and one declared weight. */
    const RecMap* qg = find_map(b, "blk.3.attn_qg.weight");
    CHECK(qg != nullptr);
    CHECK_EQ(qg->mode, RAD_MAP_COPY);
    CHECK_EQ((int)qg->src.size(), 1);
    CHECK_EQ(qg->src[0],
             std::string("model.language_model.layers.3.self_attn.q_proj.weight"));

    /* The GDN in-projection is a convert-time fusion of qkv, a and b, in gdn_conv_prep's row
     * order. */
    const RecMap* in = find_map(b, "blk.0.ssm_in.weight");
    CHECK(in != nullptr);
    CHECK_EQ(in->mode, RAD_MAP_CONCAT);
    CHECK_EQ((int)in->src.size(), 3);
    CHECK_EQ(in->src[0],
             std::string("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"));
    CHECK_EQ(in->src[1],
             std::string("model.language_model.layers.0.linear_attn.in_proj_a.weight"));
    CHECK_EQ(in->src[2],
             std::string("model.language_model.layers.0.linear_attn.in_proj_b.weight"));
    CHECK_EQ(in->concat_dim, 0);

    /* A_log maps straight through; the checkpoint tensor is already a log. */
    const RecMap* al = find_map(b, "blk.0.ssm_a_log");
    CHECK(al != nullptr);
    CHECK_EQ(al->src[0], std::string("model.language_model.layers.0.linear_attn.A_log"));

    /* gate and up are one weight and one GEMM. */
    const RecMap* gu = find_map(b, "blk.1.ffn_gate_up.weight");
    CHECK(gu != nullptr);
    CHECK_EQ(gu->mode, RAD_MAP_CONCAT);
    CHECK_EQ((int)gu->src.size(), 2);
    CHECK_EQ(gu->src[0], std::string("model.language_model.layers.1.mlp.gate_proj.weight"));
    CHECK_EQ(gu->src[1], std::string("model.language_model.layers.1.mlp.up_proj.weight"));

    /* The feed-forward's norm is the post-attention norm, not the input norm. */
    const RecMap* fn = find_map(b, "blk.1.ffn_norm.weight");
    CHECK(fn != nullptr);
    CHECK_EQ(fn->src[0],
             std::string("model.language_model.layers.1.post_attention_layernorm.weight"));

    /* This model does not tie its lm_head; the second source serves family members that do. */
    const RecMap* lm = find_map(b, "output.weight");
    CHECK(lm != nullptr);
    CHECK_EQ(lm->mode, RAD_MAP_COPY);
    CHECK_EQ((int)lm->src.size(), 2);
    CHECK_EQ(lm->src[0], std::string("lm_head.weight"));
    CHECK_EQ(lm->src[1], std::string("model.language_model.embed_tokens.weight"));

    /* Every declared weight has a map entry, so rad-convert can fill all of them. */
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(every_op_declares_the_buffers_it_touches) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* A transient without declared uses has no liveness, so the planner must keep it live for the
     * whole program. A GDN layer has eleven intermediates, so this decides the arena size. */
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);

    const RecOp* cp = find_op(b, "gdn_conv_prep");
    CHECK_EQ((int)cp->reads.size(), 1);      /* the fused projection row */
    CHECK_EQ((int)cp->writes.size(), 5);     /* q, k, v, g, beta */

    /* kv_store writes the group's pool, not a declared buffer, so its write set is empty by
     * design; it reads the two projections it stores. */
    const RecOp* ks = find_op(b, "kv_store");
    CHECK_EQ((int)ks->reads.size(), 2);
    CHECK_EQ((int)ks->writes.size(), 0);

    /* The residual add reads the stream and the block output and writes the stream. */
    const RecOp* ad = find_op(b, "add");
    CHECK_EQ((int)ad->reads.size(), 2);
    CHECK_EQ((int)ad->writes.size(), 1);
    CHECK_EQ(ad->writes[0], ad->reads[0]);
}

/* The gated attention layer runs the four ops the undeclared fused op stands for; no refusal in
 * the fake hierarchy is needed to reach this sequence. */
TEST(the_gated_attention_layer_runs_the_unfused_sequence) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));

    /* The per-head norms are rmsnorm over head_dim with M ranged over tokens x heads. The q norm
     * reads the strided interleaved projection and writes the contiguous q buffer, de-interleaving
     * and normalising in one pass. */
    const RecOp* an = nullptr;      /* the first attention layer's q norm */
    for (size_t i = 0; i + 1 < b.ops.size(); ++i)
        if (b.ops[i].op == "rope") { an = &b.ops[i - 2]; break; }
    CHECK(an != nullptr);
    if (!an) return;
    CHECK_EQ(an->op, std::string("rmsnorm"));
    CHECK_EQ(pint(an, "n"), 256);
    CHECK_EQ(param(an, "M")->ihi, 256LL * 24);
    CHECK_EQ((an + 1)->op, std::string("rmsnorm"));
    CHECK_EQ(param(an + 1, "M")->ihi, 256LL * 4);

    const RecOp* rq = an + 2;
    CHECK_EQ(rq->op, std::string("rope"));
    CHECK_EQ(pint(rq, "head_dim"), 256);
    CHECK_EQ(pint(rq, "n_head"), 24);
    CHECK_EQ(pint(rq, "n_head_kv"), 0);      /* q only: no k or v heads */
    CHECK_EQ(pint(rq, "rotary_dim"), 64);
    CHECK_EQ(pstr(rq, "mode"), std::string("mrope"));
    CHECK_NEAR(pflt(rq, "theta"), 1e7, 1.0);

    const RecOp* rk = an + 3;
    CHECK_EQ(rk->op, std::string("rope"));
    CHECK_EQ(pint(rk, "n_head"), 4);         /* the k row, 4 heads */
    CHECK_EQ(pint(rk, "n_head_kv"), 0);

    /* The step issues them, once per full-attention layer. */
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    int n_rope = 0;
    for (const auto& i : c.issues)
        if (b.op_of(i.op) && b.op_of(i.op)->op == "rope") ++n_rope;
    CHECK_EQ(n_rope, 4);                     /* two per attention layer, two layers */

    /* The sigmoid reads the interleaved projection: a column slice at offset head_dim, width
     * head_dim, of the buffer whose declared width is the pitch. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "sigmoid") continue;
        CHECK_EQ(i.opd[0].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(i.opd[0].offset, 256);      /* the gate half of the head's pair */
        CHECK_EQ(i.opd[0].cols, 256);
        CHECK_EQ(i.opd[0].rows, batch.n_tok);
    }
}

TEST(tensor_parallel_divides_dimensions_and_places_the_collectives) {
    RadBuilder b0, b1;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  c0 = build_ctx(0, 2), c1 = build_ctx(1, 2);
    CHECK_OK(qwen35_bf16::declare(&b0, &meta, &c0));
    CHECK_OK(qwen35_bf16::declare(&b1, &meta, &c1));

    /* Declare runs once per rank, and the plugin declares already-divided dimensions. */
    const RecWeight* qg = find_w(b0, "blk.3.attn_qg.weight");
    CHECK_EQ(qg->d.shape[0], 6144);           /* 12 heads x 256 x 2 */
    CHECK_EQ(qg->d.shape[1], 5120);           /* the residual stream is not sharded */
    CHECK_EQ(find_w(b0, "blk.3.attn_k.weight")->d.shape[0], 512);      /* 2 kv heads */
    const RecWeight* o = find_w(b0, "blk.3.attn_output.weight");
    CHECK_EQ(o->d.shape[0], 5120);
    CHECK_EQ(o->d.shape[1], 3072);
    CHECK_EQ(o->d.shard, RAD_SHARD_COL);

    /* The GDN row halves by head: 8 key and 24 value heads give 2*8*128 + 24*128 = 5120 convolved
     * channels, plus 48 for a and b. */
    const RecWeight* in = find_w(b0, "blk.0.ssm_in.weight");
    CHECK_EQ(in->d.shape[0], 5168);
    CHECK_EQ(find_w(b0, "blk.0.ssm_z.weight")->d.shape[0], 3072);
    CHECK_EQ(find_w(b0, "blk.0.ssm_conv1d.weight")->d.shape[0], 5120);
    CHECK_EQ(find_w(b0, "blk.0.ssm_a_log")->d.shape[0], 24);
    CHECK_EQ(find_w(b0, "blk.0.ssm_out.weight")->d.shape[1], 3072);
    /* The gain vectors are head-width and stay whole. */
    CHECK_EQ(find_w(b0, "blk.0.ssm_norm.weight")->d.shape[0], 128);
    CHECK_EQ(find_w(b0, "blk.3.attn_q_norm.weight")->d.shape[0], 256);

    const RecWeight* gu = find_w(b0, "blk.0.ffn_gate_up.weight");
    CHECK_EQ(gu->d.shape[0], 17408);          /* 2 * (17408 / 2) */
    CHECK_EQ(find_w(b0, "blk.0.ffn_down.weight")->d.shape[1], 8704);

    /* gqa and head width are rank-invariant. */
    CHECK_EQ(pint(find_op(b0, "attn_paged"), "gqa"), 6);
    CHECK_EQ(pint(find_op(b0, "gdn_conv_prep"), "head_k"), 128);
    CHECK_EQ(pint(find_op(b0, "gdn_recurrent_update"), "n_head_v"), 24);

    /* The KV groups halve with the heads they hold. */
    CHECK_EQ(b0.kv_groups[0].d.n_head_kv, 2);
    CHECK_EQ(b0.kv_groups[1].d.n_head_kv, 24);
    CHECK_EQ(b0.kv_groups[2].d.n_head_kv, 40);   /* 2*8 + 24 */

    /* Two collectives per layer (the block's and the feed-forward's column-parallel projections)
     * plus one for the vocab-sharded embedding, and no others. */
    CHECK_EQ(count_op(b0, "all_reduce"), 17);
    CHECK_EQ(join(op_names(b0)), join(program_ops(/*tp=*/true)));

    const RecOp* ar = find_op(b0, "all_reduce");
    CHECK_EQ(pint(ar, "world_size"), 2);
    CHECK_EQ(pint(ar, "exact"), 1);           /* must be stated, or no kernel matches */
    CHECK_EQ(param(ar, "numel")->kind, RAD_P_RANGE);
    CHECK_EQ(param(ar, "numel")->ihi, 256LL * 5120);

    /* The vocabulary split is ragged: ceil(total/world) rows on every rank but the last, which
     * takes the rest. This must match the rule core/engine.cpp restates and checks; since 248320
     * divides by two, the sum and shard are asserted rather than a value "floor" would also give. */
    const RecWeight* lm0 = find_w(b0, "output.weight");
    const RecWeight* lm1 = find_w(b1, "output.weight");
    CHECK_EQ(lm0->d.shape[0] + lm1->d.shape[0], 248320);
    CHECK_EQ(lm0->d.shape[0], (248320 + 1) / 2);
    CHECK_EQ(lm0->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(lm1->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(pint(find_op(b0, "logits_gemm"), "n_vocab"), lm0->d.shape[0]);
    /* The draft head stays whole on every rank: its argmax is a row_topk over one rank's plane
     * that bypasses the sampler's merge, so a sharded plane would only propose tokens from that
     * rank's half. */
    if (const RecWeight* dh = find_w(b0, "mtp.draft_head.weight")) {
        CHECK_EQ(dh->d.shape[0], 248320);
        CHECK_EQ(dh->d.shard, RAD_SHARD_NONE);
    }
    /* The embedding uses the same ragged split as the lm_head (whole, it would be gigabytes of
     * replicated bf16 per card). The shards sum to the vocabulary and each rank's embed_lookup gets
     * its slice's vocab_offset. */
    const RecWeight* te0 = find_w(b0, "token_embd.weight");
    const RecWeight* te1 = find_w(b1, "token_embd.weight");
    CHECK_EQ(te0->d.shape[0] + te1->d.shape[0], 248320);
    CHECK_EQ(te0->d.shape[0], (248320 + 1) / 2);
    CHECK_EQ(te0->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(te1->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(te0->d.access, RAD_ACCESS_VOCAB);
    CHECK_EQ(pint(find_op(b0, "embed_lookup"), "n_vocab"), te0->d.shape[0]);
    CHECK_EQ(pint(find_op(b0, "embed_lookup"), "vocab_offset"), 0);
    CHECK_EQ(pint(find_op(b1, "embed_lookup"), "vocab_offset"), te0->d.shape[0]);

    /* A rank count that does not divide the heads is refused, not rounded. */
    RadBuilder b5;
    RadBuildCtx c5 = build_ctx(0, 5);
    CHECK_EQ(qwen35_bf16::declare(&b5, &meta, &c5), RAD_E_INVAL);

    /* With one rank there are no collectives. */
    RadBuilder bs;
    RadBuildCtx cs = build_ctx(0, 1);
    CHECK_OK(qwen35_bf16::declare(&bs, &meta, &cs));
    CHECK_EQ(count_op(bs, "all_reduce"), 0);
}

TEST(a_decode_step_issues_both_layer_types_in_order) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 8; ++l) {
        if ((l + 1) % 4 == 0) {
            /* Issue order differs from declare order in one place: kv_store writes the cache
             * before attn_paged reads it. */
            const char* seq[] = { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt",
                                  "rmsnorm", "rmsnorm", "rope", "rope",
                                  "kv_store", "attn_paged",
                                  "sigmoid", "mul", "gemm_nt", "add" };
            for (const char* s : seq) want.push_back(s);
        } else {
            /* Decode takes the recurrent path, which folds in the gated norm. In a pure decode
             * step z (the second gemm_nt) follows the convolution, so it runs beside that kernel
             * rather than beside in (rad_block_gdn.h). */
            const char* seq[] = { "rmsnorm", "gemm_nt", "gdn_conv_update", "gemm_nt",
                                  "gdn_recurrent_update", "gemm_nt", "add" };
            for (const char* s : seq) want.push_back(s);
        }
        for (const char* s : { "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt", "add" })
            want.push_back(s);
    }
    want.push_back("rmsnorm");
    want.push_back("gather_rows");
    want.push_back("logits_gemm");
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The ranged value: RAD_N_BATCH for token-ranged ops, the query length for q_len-banded ops,
     * an explicit row count where rows are tokens x heads, and n_out for the two vocabulary ops,
     * whose band is the number of sampled rows, not n_tok. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "attn_paged" || o->op == "gdn_conv_update" ||
            o->op == "gdn_recurrent_update")
            CHECK_EQ(i.n, (int64_t)batch.max_q_len);
        else if (o->op == "sigmoid")
            CHECK_EQ(i.n, batch.n_tok * 24);
        else if (o->op == "rmsnorm")
            /* Three rmsnorm row counts: layer pre-norms over tokens (RAD_N_BATCH), and the
             * per-head q and k norms at tokens x n_head and tokens x n_head_kv. */
            CHECK(i.n == RAD_N_BATCH || i.n == batch.n_tok * 24 || i.n == batch.n_tok * 4);
        else if (o->op == "gather_rows" || o->op == "logits_gemm")
            CHECK_EQ(i.n, batch.n_out);
        else
            CHECK_EQ(i.n, RAD_N_BATCH);
    }

    /* Every operand is a declared handle or a batch pointer, and every buffer operand carries the
     * step's row count, not the declared extent: kernels read extents from operands, so an
     * un-narrowed operand would compute a one-row decode over the whole arena. */
    for (const auto& i : c.issues)
        for (const auto& od : i.opd) {
            if (od.kind == RAD_OPK_BUF) {
                CHECK(od.handle >= 1 && od.handle <= b.bufs.size());
                CHECK_EQ(od.rows, batch.n_tok);
            }
            if (od.kind == RAD_OPK_WEIGHT) CHECK(od.handle >= 1 && od.handle <= b.weights.size());
        }
}

TEST(a_prefill_step_takes_the_chunked_gdn_path) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = prefill_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> got = issued_ops(b, c);
    int n_prep = 0, n_scan = 0, n_recur = 0, n_gnorm = 0;
    for (const auto& s : got) {
        if (s == "gdn_conv_prep")        ++n_prep;
        if (s == "gdn_chunk_scan")       ++n_scan;
        if (s == "gdn_recurrent_update") ++n_recur;
        if (s == "gdn_gated_rmsnorm")    ++n_gnorm;
    }
    CHECK_EQ(n_prep, 6);
    CHECK_EQ(n_scan, 6);
    CHECK_EQ(n_gnorm, 6);
    CHECK_EQ(n_recur, 0);      /* declared but not issued in prefill */

    /* The decay and beta inputs are the last 2*n_head_v columns of the projection row: read
     * inside gdn_conv_prep on the chunked path, as strided views on the recurrent one. */
    const RecOp* cp = nullptr;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gdn_conv_prep") {
            cp = o;
            /* Sixteen operands, inputs before outputs (docs/OPS.md, "Linear attention").
             * `a` and `b_gate` (the recurrent path's strided views) are passed although optional:
             * libr4d dereferences them and `cache_idx` unconditionally, so omitting them would
             * silently resolve the layer to libref. */
            CHECK_EQ((int)i.opd.size(), 16);
            CHECK_EQ(i.opd[2].kind, (uint8_t)RAD_OPK_NONE);   /* no conv bias on this model */
            CHECK_EQ(i.opd[5].kind, (uint8_t)RAD_OPK_KV);     /* the conv state */
            CHECK_EQ(i.opd[6].rows, batch.n_seq + 1);         /* cu_seqlens */
            CHECK_EQ(i.opd[7].offset, 10240);                 /* a */
            CHECK_EQ(i.opd[7].cols, 48);
            CHECK_EQ(i.opd[8].offset, 10288);                 /* b_gate */
            CHECK_EQ(i.opd[10].rows, batch.n_seq);            /* has_init == ctx_lens */
            break;
        }
    }
    CHECK(cp != nullptr);

    /* The scan reads and writes the recurrent state group, as h0 and ht. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gdn_chunk_scan") {
            /* Eleven: the state index after the two KV operands makes the scan write the
             * sequence's manager-assigned slot, not its batch row. */
            CHECK_EQ((int)i.opd.size(), 11);
            CHECK_EQ(i.opd[6].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(i.opd[9].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(i.opd[10].rows, batch.n_seq);
            break;
        }
    }
}

TEST(the_recurrent_path_slices_a_and_b_out_of_the_projection_row) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    bool seen = false;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "gdn_recurrent_update") continue;
        seen = true;
        /* Sixteen, as both kernel registries declare for gdn_recurrent_update and
         * core/runtime/issue.cpp requires. The count comes from the schema, since this harness
         * only records operands. The last two (the folded out-projection quantiser) are unused by
         * the bf16 block; operands are positional, so an absent optional operand is a null
         * operand, not a shorter list. */
        CHECK_EQ((int)i.opd.size(), 16);
        CHECK_EQ(i.opd[14].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(i.opd[15].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(i.opd[3].offset, 10240);      /* a: first column after the conv row */
        CHECK_EQ(i.opd[3].cols, 48);
        CHECK_EQ(i.opd[4].offset, 10288);      /* b: one per value head after a */
        CHECK_EQ(i.opd[4].cols, 48);
        CHECK_EQ(i.opd[7].kind, (uint8_t)RAD_OPK_KV);
        break;
    }
    CHECK(seen);
}

TEST(what_the_plugin_refuses) {
    RadModelMeta meta = qwen35_meta();

    /* A drafter would need a second set of per-rank handles and name-map namespace; nothing
     * unimplemented returns success (spec §17). */
    RadBuilder bd;
    RadBuildCtx cd = build_ctx();
    cd.is_draft = 1;
    CHECK_EQ(qwen35_bf16::declare(&bd, &meta, &cd), RAD_E_UNSUPPORTED);

    /* A routed (MoE) checkpoint is a different graph. */
    RadBuilder bm;
    RadModelMeta moe = meta;
    moe.n_expert = 128;
    moe.n_expert_used = 8;
    RadBuildCtx cm = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&bm, &moe, &cm), RAD_E_UNSUPPORTED);

    /* A container that does not describe the linear layers is refused, not defaulted; a guessed
     * head count would load and produce wrong text. */
    RadBuilder bn;
    RadModelMeta bare = meta;
    bare.n_kv = 2;                 /* interval and rope width only; no ssm keys */
    RadBuildCtx cn = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&bn, &bare, &cn), RAD_E_INVAL);

    /* So is one that omits the rotary width. */
    RadBuilder br;
    RadModelMeta norot = meta;
    static const char* rk[] = { "qwen35.full_attention_interval", "qwen35.ssm.state_size",
                                "qwen35.ssm.group_count", "qwen35.ssm.time_step_rank",
                                "qwen35.ssm.inner_size", "qwen35.ssm.conv_kernel" };
    static const char* rv[] = { "4", "128", "16", "48", "6144", "4" };
    norot.n_kv = 6; norot.kv_key = rk; norot.kv_val = rv;
    RadBuildCtx cr = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&br, &norot, &cr), RAD_E_INVAL);
}

TEST(the_mtp_layer_sits_beside_the_trunk_and_not_inside_it) {
    /* `nextn_predict_layers` is not part of the layer count in this family: Qwen3.8-27B-FP8 sets
     * num_hidden_layers = 64, lists 64 layer_types, ships layers 0..63 under
     * `model.language_model.layers`, and keeps its MTP block in a separate file under `mtp.`.
     * Subtracting it would drop the last (full-attention) trunk layer and silently run one layer
     * short. So every declared layer is a trunk layer and `nextn` does not change the graph. */
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    static const char* mk[] = { "qwen35.full_attention_interval", "qwen35.rope.dimension_count",
                                "qwen35.ssm.state_size", "qwen35.ssm.group_count",
                                "qwen35.ssm.time_step_rank", "qwen35.ssm.inner_size",
                                "qwen35.ssm.conv_kernel", "qwen35.nextn_predict_layers" };
    static const char* mv[] = { "4", "64", "128", "16", "48", "6144", "4", "1" };
    meta.n_kv = 8; meta.kv_key = mk; meta.kv_val = mv;
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The same graph as without nextn. */
    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));
    /* The last trunk layer, blk.7, is declared; (7 + 1) % 4 == 0 makes it full attention, the
     * layer a subtraction would drop. */
    CHECK(find_w(b, "blk.7.attn_norm.weight") != nullptr);
    CHECK(find_w(b, "blk.8.attn_norm.weight") == nullptr);
}

/* The PLE windowed ops are issued once per half of the step, and the halves differ in whether the
 * ops see `num_accepted`, which picks their write-back layout. Decode rows may reject drafts, so
 * `ngram_ids` and `ple_conv` must see it; a prefill chunk commits everything, so they must not, and
 * the chunk's tail stays at offset zero for the scheduler's `num_accepted` of 1 next step. A mixed
 * step therefore needs two issues; a pure step issues one. */
TEST(the_ple_windows_are_issued_once_per_half_of_a_step) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_layer = 1; g.n_embd = 64; g.max_tok = 64; g.max_seqs = 4; g.max_spec = 3;
    RadBufDecl hd{};
    hd.dtype = RAD_BF16; hd.rank = 2; hd.shape[0] = g.max_tok; hd.shape[1] = 4 * g.n_embd;
    PleLayer::Wire w{};
    w.h = rad_decl_buffer(&b, "h", &hd);
    PleLayer::Config pc{};
    pc.hc = 4; pc.embed_dim = 32; pc.heads = 4; pc.ngram = 3; pc.eos = 7; pc.rows = 1000;
    pc.shards = 2; pc.conv_width = 4; pc.dilation = 3;
    PleLayer::Src src{};
    src.shard_prefix = "ple.shard_"; src.shard_suffix = ".weight"; src.mult = "ple.mult";
    src.vocab_sizes = "ple.vsz"; src.offsets = "ple.off"; src.key_proj = "ple.k";
    src.value_proj = "ple.v"; src.norm_key = "ple.nk"; src.norm_query = "ple.nq";
    src.norm_conv = "ple.nc"; src.conv1d = "ple.conv";
    PleLayer ple;
    CHECK_OK(ple.declare(&b, nm, g, pc, 0, w, src));

    /* Two decode rows (a verify of 4 and a plain decode of 1) and one prefill chunk of 6. */
    static const int32_t tok[11] = { 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15 };
    static const int32_t cu[4]   = { 0, 4, 5, 11 };
    static const int32_t ctxl[3] = { 40, 30, 12 };
    static const int32_t nacc[3] = { 2, 1, 1 };
    static int32_t tk_idx[3] = { 5, 6, 7 }, cv_idx[3] = { 1, 2, 3 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = ple.kv_tok;  kv[0].state_index = tk_idx;
    kv[1].group = ple.kv_conv; kv[1].state_index = cv_idx;

    RadBatch mixed{};
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 11; mixed.n_seq = 3;
    mixed.n_seq_decode = 2; mixed.n_tok_decode = 5;
    mixed.token_ids = tok; mixed.cu_seqlens = cu; mixed.ctx_lens = ctxl;
    mixed.num_accepted = nacc;
    mixed.n_kv_groups = 2; mixed.kv = kv;

    RadCtx c{ .b = &b, .batch = &mixed };
    /* The hash is issued at the top of the step and the rest at the layer. */
    ple.ids(&c, &mixed);
    ple.step(&c, &mixed);

    const char* want[] = { "ngram_ids", "ngram_ids", "embed_lookup_q", "cast", "gemm_nt",
                           "gemm_nt", "ple_gate", "ple_conv", "ple_conv", "add" };
    std::vector<std::string> wv(want, want + 10);
    CHECK_EQ(join(issued_ops(b, c)), join(wv));
    REQUIRE(c.issues.size() == 10);

    /* The projections read the gather's rows from the card: the copy reads the gather's output
     * and both GEMMs read the copy, never the host buffer. */
    {
        const RecIssue& gth = c.issues[2];
        const RecIssue& cp  = c.issues[3];
        REQUIRE(gth.opd.size() == 5 && cp.opd.size() == 2);
        CHECK_EQ(cp.opd[0].handle, gth.opd[3].handle);
        CHECK_EQ(cp.opd[0].rows, (int64_t)11);
        CHECK(cp.opd[1].handle != gth.opd[3].handle);
        CHECK_EQ(c.issues[4].opd[0].handle, cp.opd[1].handle);
        CHECK_EQ(c.issues[5].opd[0].handle, cp.opd[1].handle);
    }

    /* ngram_ids: tok, state, cu, 3 weights, cache_idx, has_init, num_accepted, ids, ahead_ids. */
    {
        const RecIssue& d = c.issues[0];
        const RecIssue& p = c.issues[1];
        REQUIRE(d.opd.size() == 11 && p.opd.size() == 11);
        /* No following prompt (n_ahead 0); a decode half never has one. */
        CHECK_EQ(d.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(d.n, (int64_t)5);
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK(d.opd[2].raw == (void*)cu);             CHECK_EQ(d.opd[2].rows, (int64_t)3);
        CHECK(d.opd[6].raw == (void*)tk_idx);         CHECK_EQ(d.opd[6].rows, (int64_t)2);
        CHECK(d.opd[7].raw == (void*)ctxl);
        CHECK_EQ(d.opd[8].kind, (uint8_t)RAD_OPK_RAW);
        CHECK(d.opd[8].raw == (void*)nacc);           CHECK_EQ(d.opd[8].rows, (int64_t)2);
        CHECK_EQ(d.opd[9].rows, (int64_t)5);

        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)11);         /* token operands stay at their base */
        CHECK(p.opd[2].raw == (void*)(cu + 2));       CHECK_EQ(p.opd[2].rows, (int64_t)2);
        CHECK(p.opd[6].raw == (void*)(tk_idx + 2));   CHECK_EQ(p.opd[6].rows, (int64_t)1);
        CHECK(p.opd[7].raw == (void*)(ctxl + 2));
        CHECK_EQ(p.opd[8].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[9].rows, (int64_t)11);
    }
    /* ple_conv: x, w, state, cu, resid, cache_idx, has_init, num_accepted, y. */
    {
        const RecIssue& d = c.issues[7];
        const RecIssue& p = c.issues[8];
        REQUIRE(d.opd.size() == 9 && p.opd.size() == 9);
        CHECK_EQ(d.n, (int64_t)5);
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK(d.opd[3].raw == (void*)cu);
        CHECK(d.opd[5].raw == (void*)cv_idx);
        CHECK(d.opd[7].raw == (void*)nacc);
        CHECK_EQ(d.opd[8].rows, (int64_t)5);

        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)11);
        CHECK(p.opd[3].raw == (void*)(cu + 2));
        CHECK(p.opd[5].raw == (void*)(cv_idx + 2));
        CHECK(p.opd[6].raw == (void*)(ctxl + 2));
        CHECK_EQ(p.opd[7].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[8].rows, (int64_t)11);
    }

    /* A pure step issues one half: a decode step passes `num_accepted`, a prefill step does not. */
    for (int phase : { (int)RAD_PHASE_DECODE, (int)RAD_PHASE_PREFILL }) {
        RadBatch pure = mixed;
        pure.phase = phase;
        pure.n_seq_decode = 0; pure.n_tok_decode = 0;   /* the phase decides, not these counts */
        RadCtx cp{ .b = &b, .batch = &pure };
        ple.ids(&cp, &pure);
        ple.step(&cp, &pure);
        int n_ids = 0, n_conv = 0;
        for (const auto& i : cp.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o) continue;
            const bool ids = o->op == "ngram_ids", conv = o->op == "ple_conv";
            if (!ids && !conv) continue;
            n_ids += ids; n_conv += conv;
            CHECK_EQ(i.n, (int64_t)11);
            const RadOperand& na = i.opd[ids ? 8 : 7];
            if (phase == RAD_PHASE_DECODE) CHECK(na.kind == RAD_OPK_RAW && na.raw == (void*)nacc);
            else                           CHECK_EQ(na.kind, (uint8_t)RAD_OPK_NONE);
        }
        CHECK_EQ(n_ids, 1);
        CHECK_EQ(n_conv, 1);
    }

    /* A following prompt rides the prefill half only: its ngram_ids reads `tok` that many tokens
     * further and writes their ids into the `ple_ids` rows after the step's, which the gather also
     * gets. The decode half never sees them. */
    {
        RadBatch ahead = mixed;
        ahead.n_ahead = 7;
        RadCtx ca{ .b = &b, .batch = &ahead };
        ple.ids(&ca, &ahead);
        ple.step(&ca, &ahead);
        REQUIRE(ca.issues.size() == 10);
        const RecIssue& d = ca.issues[0];
        const RecIssue& p = ca.issues[1];
        const RecIssue& gth = ca.issues[2];
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK_EQ(d.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)(11 + 7));
        CHECK_EQ(p.opd[9].rows, (int64_t)11);
        CHECK_EQ(p.opd[10].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(p.opd[10].rows, (int64_t)7);
        CHECK_EQ(p.opd[10].offset, (int64_t)(11 * pc.heads));
        REQUIRE(gth.opd.size() == 5);
        CHECK_EQ(gth.opd[4].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(gth.opd[4].handle, p.opd[10].handle);
        CHECK_EQ(gth.opd[4].rows, (int64_t)7);
        CHECK_EQ(gth.opd[4].offset, (int64_t)(11 * pc.heads));
    }
}

/* ==================================================================== the sizing declare
 *
 * The activation arena is lent out by level, and a level is the plugin declaring again at a
 * smaller max_tok. The plugin must keep this safe: the sizing declare must not replace the state
 * its step reads (written at the real size), and must describe the same buffers with only their
 * rows smaller, since every other extent is a stride the real kernels were chosen for.
 *
 * Qwen3.8-Flash-Next's metadata, cut to eight layers. */
namespace {
const char* q4_keys[] = {
    "full_attention_interval", "hc_count", "hc_lowrank", "heads_per_ngram",
    "hidden_act", "indexer_budget", "indexer_compress_ratio", "indexer_head_dim",
    "indexer_n_heads", "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
    "make_ngram_vocab_size_divisible_by",
    "moe_intermediate_size", "mtp_num_hidden_layers", "ngram_size", "ngram_vocab_size_base",
    "num_experts", "num_experts_per_tok", "output_gate_type", "partial_rotary_factor",
    "ple_conv_kernel_size", "ple_embed_dim", "ple_layer_ids", "shared_expert_intermediate_size",
    "split_ngram_parts", "rope_parameters.partial_rotary_factor", "eos_token_id",
};
const char* q4_vals[] = {
    "4", "4", "320", "8",
    "silu", "2048", "4", "128",
    "4",
    "linear_attention linear_attention linear_attention full_attention "
    "linear_attention linear_attention linear_attention full_attention",
    "4", "128",
    "16", "48", "128", "128",
    "640", "1", "3", "20000000",
    "512", "10", "sigmoid", "0.25",
    "4", "2560", "2", "640",
    "128", "0.25", "248044",
};

RadModelMeta qwen4exp_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen4exp";
    m.name        = "test-q38-flashnext";
    m.quant       = "";
    m.n_layers    = 8;
    m.n_embd      = 2560;
    m.n_head      = 24;
    m.n_head_kv   = 2;
    m.head_dim    = 256;
    m.n_ff        = 640;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(q4_keys) / sizeof(q4_keys[0]));
    m.kv_key      = q4_keys;
    m.kv_val      = q4_vals;
    return m;
}
}  /* namespace */

/* On a draft head's pass the n-gram block hashes each sequence's newest token (`ngram_hint`, the
 * history form on the history pass, the round form after it) and gathers those rows, so a table on
 * a drive delivers them while the card drafts. A history pass that drafts nothing and a trunk step
 * get neither; without a drafter, or where no library serves the hash, nothing is declared. */
TEST(the_ple_hint_runs_on_the_drafting_passes_only) {
    using namespace rad::arch;
    for (const int mode : { 0, 1, 2 }) {       /* served; no library hashes the hint; no drafter */
        RadBuilder b;
        if (mode == 1) b.undeclared.insert("ngram_hint");
        Names nm("");
        Geom g{};
        g.n_layer = 1; g.n_embd = 64; g.max_tok = 64; g.max_seqs = 4; g.max_spec = mode == 2 ? 0 : 3;
        RadBufDecl hd{};
        hd.dtype = RAD_BF16; hd.rank = 2; hd.shape[0] = g.max_tok; hd.shape[1] = 4 * g.n_embd;
        PleLayer::Wire w{};
        w.h = rad_decl_buffer(&b, "h", &hd);
        PleLayer::Config pc{};
        pc.hc = 4; pc.embed_dim = 32; pc.heads = 4; pc.ngram = 3; pc.eos = 7; pc.rows = 1000;
        pc.shards = 2; pc.conv_width = 4; pc.dilation = 3;
        PleLayer::Src src{};
        src.shard_prefix = "ple.shard_"; src.shard_suffix = ".weight"; src.mult = "ple.mult";
        src.vocab_sizes = "ple.vsz"; src.offsets = "ple.off"; src.key_proj = "ple.k";
        src.value_proj = "ple.v"; src.norm_key = "ple.nk"; src.norm_query = "ple.nq";
        src.norm_conv = "ple.nc"; src.conv1d = "ple.conv";
        PleLayer ple;
        CHECK_OK(ple.declare(&b, nm, g, pc, 0, w, src));
        const bool on = mode == 0;
        CHECK_EQ(count_op(b, "ngram_hint"), on ? 2 : 0);
        CHECK_EQ(count_op(b, "embed_lookup_q"), on ? 2 : 1);
        CHECK_EQ(find_b(b, "ple_hint_tail") != nullptr, on);

        /* Two sequences: a history pass of 3 and 2 rows, then a round of one row each. */
        static const int32_t tok[5] = { 1, 2, 2, 3, 3 }, pos[5] = { 10, 11, 11, 20, 20 };
        static const int32_t cu[3] = { 0, 3, 5 }, cu1[3] = { 0, 1, 2 };
        static int32_t tk_idx[2] = { 5, 6 };
        RadKVGroupBatch kv[1]{};
        kv[0].group = ple.kv_tok; kv[0].state_index = tk_idx;
        RadBatch hist{};
        hist.phase = RAD_PHASE_DECODE;
        hist.n_tok = 5; hist.n_seq = 2; hist.token_ids = tok; hist.positions = pos;
        hist.cu_seqlens = cu; hist.n_kv_groups = 1; hist.kv = kv;
        hist.draft_pass = -1; hist.n_draft_out = 2;
        RadBatch quiet = hist;
        quiet.n_draft_out = 0;
        RadBatch round = hist;
        round.n_tok = 2; round.cu_seqlens = cu1; round.draft_pass = 2; round.n_draft_out = 0;
        RadBatch trunk = hist;
        trunk.draft_pass = 0; trunk.n_draft_out = 0;

        for (const RadBatch* bt : { &hist, &quiet, &round, &trunk }) {
            RadCtx c{ .b = &b, .batch = bt };
            ple.hint_ids(&c, bt);
            ple.hint_fetch(&c, bt);
            const bool runs = on && (bt == &hist || bt == &round);
            REQUIRE(c.issues.size() == (runs ? 2u : 0u));
            if (!runs) continue;
            const RecOp* h = b.op_of(c.issues[0].op);
            const RecOp* f = b.op_of(c.issues[1].op);
            REQUIRE(h != nullptr && f != nullptr);
            CHECK_EQ(h->op, std::string("ngram_hint"));
            CHECK_EQ(pint(h, "hist"), bt == &hist ? 1 : 0);
            CHECK_EQ(f->op, std::string("embed_lookup_q"));
            /* A row of ids a sequence, gathered a head a row. */
            CHECK_EQ(c.issues[0].n, (int64_t)2);
            CHECK_EQ(c.issues[1].n, (int64_t)(2 * pc.heads));
            CHECK_EQ(c.issues[0].opd[8].handle, c.issues[1].opd[0].handle);
            CHECK_EQ(c.issues[0].opd[8].rows, (int64_t)2);
        }
    }
}

/* The n-gram block's projections run in the encoding the container stores. Rotated int8 ones are
 * the int8 GEMM over the embedding's rotated codes: the first quantises the bf16 plane the copy
 * wrote, the second reads the same codes and has no quantiser. An int8 key beside a bf16 value
 * leaves the value on the plain GEMM over the bf16 plane. */
TEST(the_ple_projections_take_their_containers_encoding) {
    using namespace rad::arch;
    RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(i8.transform, "fwht128");
    for (const bool both : { true, false }) {
        RadBuilder b;
        b.encs = { { "ple_key", i8 } };
        if (both) b.encs.push_back({ "ple_value", i8 });
        Names nm("");
        Geom g{};
        g.n_layer = 1; g.n_embd = 64; g.max_tok = 64; g.max_seqs = 4; g.max_spec = 3;
        RadBufDecl hd{};
        hd.dtype = RAD_BF16; hd.rank = 2; hd.shape[0] = g.max_tok; hd.shape[1] = 4 * g.n_embd;
        PleLayer::Wire w{};
        w.h = rad_decl_buffer(&b, "h", &hd);
        PleLayer::Config pc{};
        pc.hc = 4; pc.embed_dim = 128; pc.heads = 4; pc.ngram = 3; pc.eos = 7; pc.rows = 1000;
        pc.shards = 2; pc.conv_width = 4; pc.dilation = 3;
        PleLayer::Src src{};
        src.shard_prefix = "ple.shard_"; src.shard_suffix = ".weight"; src.mult = "ple.mult";
        src.vocab_sizes = "ple.vsz"; src.offsets = "ple.off"; src.key_proj = "ple.k";
        src.value_proj = "ple.v"; src.norm_key = "ple.nk"; src.norm_query = "ple.nq";
        src.norm_conv = "ple.nc"; src.conv1d = "ple.conv";
        PleLayer ple;
        CHECK_OK(ple.declare(&b, nm, g, pc, 0, w, src));

        CHECK(ple.key.i8 && ple.key.i8_rot && ple.key.op_q != 0);
        CHECK_EQ(ple.value.i8, both);
        CHECK_EQ(ple.value.bf16, !both);
        CHECK(ple.value.op_q == 0);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), 1);
        CHECK_EQ(count_op(b, "gemm_nt_q"), both ? 2 : 1);
        CHECK_EQ(count_op(b, "gemm_nt"), both ? 0 : 1);
        for (const auto& o : b.ops)
            if (o.op == "gemm_nt_q") CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
        const RecBuf* q8 = find_b(b, "ple_emb_dev.q8");
        REQUIRE(q8 != nullptr);
        CHECK_EQ(q8->d.dtype, (uint32_t)RAD_I8);
        CHECK_EQ(q8->d.shape[1], (int64_t)128);

        /* A decode step: the quantiser once, then both projections. */
        static const int32_t tok[4] = { 1, 2, 3, 4 };
        static const int32_t cu[2]  = { 0, 4 };
        static const int32_t ctxl[1] = { 40 };
        static const int32_t nacc[1] = { 2 };
        static int32_t tk_idx[1] = { 5 }, cv_idx[1] = { 1 };
        RadKVGroupBatch kv[2]{};
        kv[0].group = ple.kv_tok;  kv[0].state_index = tk_idx;
        kv[1].group = ple.kv_conv; kv[1].state_index = cv_idx;
        RadBatch dec{};
        dec.phase = RAD_PHASE_DECODE;
        dec.n_tok = 4; dec.n_seq = 1; dec.n_seq_decode = 1; dec.n_tok_decode = 4;
        dec.token_ids = tok; dec.cu_seqlens = cu; dec.ctx_lens = ctxl; dec.num_accepted = nacc;
        dec.n_kv_groups = 2; dec.kv = kv;
        RadCtx c{ .b = &b, .batch = &dec };
        ple.ids(&c, &dec);
        ple.step(&c, &dec);
        const std::vector<std::string> ops = issued_ops(b, c);
        CHECK_EQ(count_issued(ops, "had_quant_act_i8g"), 1);
        CHECK_EQ(count_issued(ops, "gemm_nt_q"), both ? 2 : 1);
        CHECK_EQ(count_issued(ops, "gemm_nt"), both ? 0 : 1);
    }
}

TEST(qwen4exp_pass_key_reuses_only_proven_uniform_prefill_grids) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx build = build_ctx(0, 1);
    build.max_tok = 64; build.max_seqs = 2; build.max_spec = 0; build.max_ctx = 65536;
    RadBuilder graph;
    REQUIRE_EQ(qwen4exp_fp8::declare(&graph, &meta, &build), RAD_OK);
    rad_kvgroup kv_full = 0;
    for (size_t i = 0; i < graph.kv_groups.size(); ++i)
        if (graph.kv_groups[i].name.find("kv_attn") != std::string::npos) {
            kv_full = (rad_kvgroup)(i + 1);
            break;
        }
    REQUIRE(kv_full != 0);

    RadCtx ctx{};
    RadKVGroupBatch group{};
    group.group = kv_full;
    RadBatch batch{};
    batch.phase = RAD_PHASE_PREFILL;
    batch.n_tok = 64; batch.n_seq = 1; batch.max_q_len = 64;
    /* The split law uses the declared launch context, not this rounded live-batch bound. */
    batch.max_ctx_len = 1024; batch.n_kv_groups = 1; batch.kv = &group;
    int32_t used[1] = {2304};
    const uint64_t key6 = qwen4exp_fp8::pass_key(&ctx, &batch, used);
    CHECK(key6 != 0);
    used[0] = 2305;
    CHECK_EQ(qwen4exp_fp8::pass_key(&ctx, &batch, used), key6);
    used[0] = 3072;
    CHECK(qwen4exp_fp8::pass_key(&ctx, &batch, used) != key6);

    /* Same rounded live bound, different exact grids under the launcher's model context. */
    used[0] = 896;
    const uint64_t key_896 = qwen4exp_fp8::pass_key(&ctx, &batch, used);
    used[0] = 1152;
    CHECK(qwen4exp_fp8::pass_key(&ctx, &batch, used) != key_896);

    /* A shorter one-sequence chunk still proves q_len = n_tok; do not require the model max_tok. */
    used[0] = 2304;
    batch.n_tok = 32; batch.max_q_len = 32;
    const uint64_t key_chunk = qwen4exp_fp8::pass_key(&ctx, &batch, used);
    CHECK(key_chunk != 0);
    CHECK(key_chunk != key6);
    batch.n_tok = 64; batch.max_q_len = 64;

    batch.phase = RAD_PHASE_MIXED;
    CHECK_EQ(qwen4exp_fp8::pass_key(&ctx, &batch, used), (uint64_t)0);
    /* Decode, verify and draft passes issue at the keyed context bound, so they record. */
    batch.phase = RAD_PHASE_DECODE; batch.n_spec = 1;
    CHECK(qwen4exp_fp8::pass_key(&ctx, &batch, used) != 0);
    batch.n_spec = 0; batch.draft_pass = -1;
    CHECK(qwen4exp_fp8::pass_key(&ctx, &batch, used) != 0);
    /* A drafter's or a verify's rows in a prefill pass read the live maximum: not keyed. */
    batch.phase = RAD_PHASE_PREFILL;
    CHECK_EQ(qwen4exp_fp8::pass_key(&ctx, &batch, used), (uint64_t)0);
    batch.draft_pass = 0; batch.n_spec = 1;
    CHECK_EQ(qwen4exp_fp8::pass_key(&ctx, &batch, used), (uint64_t)0);
    batch.n_spec = 0; batch.n_seq = 2;
    CHECK_EQ(qwen4exp_fp8::pass_key(&ctx, &batch, used), (uint64_t)0);
    batch.n_seq = 1; used[0] = 0;
    CHECK_EQ(qwen4exp_fp8::pass_key(&ctx, &batch, used), (uint64_t)0);
}

TEST(qwen4exp_yarn_reaches_attention_indexer_and_compressed_keys) {
    RadModelMeta meta = qwen4exp_meta();
    std::vector<const char*> keys(meta.kv_key, meta.kv_key + meta.n_kv);
    std::vector<const char*> vals(meta.kv_val, meta.kv_val + meta.n_kv);
    keys.insert(keys.end(), {"text_config.rope_parameters.rope_type", "text_config.rope_parameters.factor",
                            "text_config.rope_parameters.original_max_position_embeddings"});
    vals.insert(vals.end(), {"yarn", "2", "262144"});
    meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 524288;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int direct = 0, table = 0, compressed = 0;
    for (const auto& op : b.ops) {
        if (op.op == "rope") ++direct;
        else if (op.op == "rope_table") ++table;
        else if (op.op == "qsa_block_key") ++compressed;
        else continue;
        int found = 0;
        for (const auto& p : op.p) {
            if (p.key == "yarn_factor") { CHECK_EQ(p.dval, 2.0); ++found; }
            if (p.key == "yarn_magnitude") { CHECK_NEAR(p.dval, 1.069314718, 1e-7); ++found; }
            if (p.key == "scale") CHECK_EQ(p.dval, 1.0);
        }
        CHECK_EQ(found, 2);
    }
    CHECK(direct > 0); CHECK(table > 0); CHECK(compressed > 0);
}

/* A streamed attention group's selection must be made resident and rewritten between
 * `qsa_select` and the sparse attention, and the layer's KV store must come first: a block this
 * step writes may not be in the ring, and the resolve copies such a block whole out of the host
 * mirror, so the mirror write precedes the resolve. The order per layer is therefore
 * store -> resolve -> attention, with the same `sel` buffer in the resolve and the attention, and
 * exactly one resolve per attention launch -- a second one for one launch would read slots as block
 * ids, and one for an earlier launch would let a later launch's evictions move the blocks it names.
 *
 * The indexer issues none: it produces `sel`, and a resolve between `qsa_select` and the store
 * would copy the mirror before this step's tokens reached it. */
TEST(a_streamed_qwen4exp_group_stores_then_resolves_then_attends) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);

    /* The parity configuration's declaration: blocks == slots and the block's own byte pitch. */
    const RecOp* rop = nullptr;
    for (const RecOp& o : b.ops) if (o.op == "kv_resolve") { rop = &o; break; }
    REQUIRE(rop != nullptr);
    int64_t blocks = 0, bbytes = 0, pitch = 0;
    for (const RecParam& p : rop->p) {
        if (p.key == "blocks")      blocks = p.ival;
        if (p.key == "block_bytes") bbytes = p.ival;
        if (p.key == "sel_pitch")   pitch  = p.ival;
    }
    REQUIRE(blocks > 0);
    REQUIRE(bbytes > 0);
    REQUIRE(pitch > 0);

    std::unique_ptr<Kit> k = kit(b, { 1 }, {});
    std::vector<int32_t> res((size_t)blocks, -1), sblk((size_t)blocks, -1),
                         sst((size_t)blocks * 2, 0), ctl(16, 0);
    RadKVStream st{};
    st.host              = (uint8_t*)res.data();
    st.res               = res.data();
    st.slot_block        = sblk.data();
    st.slot_state        = sst.data();
    st.ctl               = ctl.data();
    st.block_bytes       = bbytes;
    st.host_layer_stride = blocks * bbytes;
    st.slots             = blocks;
    st.host_bytes        = blocks * bbytes * (int64_t)b.kv_groups.size();
    int streamed = 0;
    for (size_t gi = 0; gi < b.kv_groups.size(); ++gi)
        if (b.kv_groups[gi].name.find("kv_attn") != std::string::npos) {
            k->kv[gi].stream = &st;
            ++streamed;
        }
    CHECK_EQ(streamed, 1);

    RadCtx ctx;
    ctx.b = &b;
    ctx.batch = &k->b;
    qwen4exp_fp8::step(&ctx, &k->b);

    int selects = 0, stores = 0, resolves = 0, attn = 0;
    size_t last_select = 0, last_store = 0;
    for (size_t i = 0; i < ctx.issues.size(); ++i) {
        const RecOp* o = b.op_of(ctx.issues[i].op);
        if (!o) continue;
        if (o->op == "qsa_select") { ++selects; last_select = i; }
        else if (o->op == "kv_store_streamed" ||
                 o->op == "qk_norm_rope_gate_kv_store_streamed") {
            ++stores;
            last_store = i;
            /* The mirror and the residency map are the stream's raw planes, and they are the
             * operands the layer's KV reaches storage through. A streamed group's store is folded
             * into the prologue whenever the fused pair resolves and the step fits its rows, so
             * both forms are accepted and each is checked at its own operand positions. */
            const bool folded = o->op == "qk_norm_rope_gate_kv_store_streamed";
            const int ro = folded ? 12 : 4, ho = folded ? 13 : 5;
            CHECK_EQ(ctx.issues[i].opd[ro].kind, (int)RAD_OPK_RAW);   /* res */
            CHECK_EQ(ctx.issues[i].opd[ro].dtype, (uint32_t)RAD_I32);
            CHECK_EQ(ctx.issues[i].opd[ro].rows, blocks);
            CHECK_EQ(ctx.issues[i].opd[ho].kind, (int)RAD_OPK_RAW);   /* host, this layer's row */
            CHECK_EQ(ctx.issues[i].opd[ho].dtype, (uint32_t)RAD_U8);
            CHECK_EQ(ctx.issues[i].opd[ho].rows, blocks);
            CHECK_EQ(ctx.issues[i].opd[ho].cols, bbytes);
        } else if (o->op == "kv_resolve") {
            ++resolves;
            CHECK(i > last_store);                  /* the mirror write precedes the copy */
            CHECK(i > last_select);                 /* and the selection precedes the rewrite */
            CHECK_EQ(ctx.issues[i].opd[0].handle, ctx.issues[last_select].opd[5].handle);
            CHECK_EQ(ctx.issues[i].opd[0].offset, ctx.issues[last_select].opd[5].offset);
            CHECK_EQ(ctx.issues[i].opd[1].kind, (int)RAD_OPK_BUF);   /* nsel, its row count */
            /* One resolve per attention launch: the launch that follows reads exactly its rows. */
            size_t j = i + 1;
            for (; j < ctx.issues.size(); ++j) {
                const RecOp* n = b.op_of(ctx.issues[j].op);
                if (!n) continue;
                if (n->op == "attn_paged" || n->op == "attn_paged_gate_quant") break;
            }
            REQUIRE(j < ctx.issues.size());
            CHECK_EQ(ctx.issues[j].opd[2].handle, ctx.issues[i].opd[0].handle);
            CHECK_EQ(ctx.issues[j].opd[2].offset, ctx.issues[i].opd[0].offset);
            CHECK_EQ(ctx.issues[j].opd[2].rows,   ctx.issues[i].opd[0].rows);
        } else if (o->op == "attn_paged" || o->op == "attn_paged_gate_quant") {
            ++attn;
            CHECK(ctx.issues[i].opd[2].handle == ctx.issues[last_select].opd[5].handle);
            CHECK_EQ(ctx.issues[i].opd[2].offset, ctx.issues[last_select].opd[5].offset);
        }
    }
    CHECK(selects > 0);
    CHECK_EQ(stores, 2);           /* one store a full-attention layer, before its attention */
    CHECK_EQ(attn, stores);
    CHECK_EQ(resolves, attn);      /* exactly one resolve per attention launch, never two */
}

/* The parity configuration's stream: a ring as wide as the group's context and an identity map, so
 * a resolve may rewrite a selection but can never copy. Only the rewrite is under test here, so
 * the maps alias one allocation and no kernel runs. */
struct StreamRig {
    std::vector<int32_t> res, slot_block, slot_state, ctl;
    RadKVStream st{};
};

/* `blocks` and `block_bytes` come from the declaration's own `kv_resolve`, so the rig is exactly
 * what the plugin planned for. The group is reached by name because `kv_attn` is the only full
 * group a stream flows through (the indexer's own block keys stay resident). */
void attach_stream(const RadBuilder& b, Kit& k, StreamRig& r, int64_t blocks, int64_t bbytes) {
    r.res.assign((size_t)blocks, -1);
    r.slot_block.assign((size_t)blocks, -1);
    r.slot_state.assign((size_t)blocks * 2, 0);
    r.ctl.assign(16, 0);
    r.st.host              = (uint8_t*)r.res.data();
    r.st.res               = r.res.data();
    r.st.slot_block        = r.slot_block.data();
    r.st.slot_state        = r.slot_state.data();
    r.st.ctl               = r.ctl.data();
    r.st.block_bytes       = bbytes;
    r.st.host_layer_stride = blocks * bbytes;
    r.st.slots             = blocks;
    r.st.host_bytes        = blocks * bbytes * (int64_t)b.kv_groups.size();
    int streamed = 0;
    for (size_t gi = 0; gi < b.kv_groups.size(); ++gi)
        if (b.kv_groups[gi].name.find("kv_attn") != std::string::npos) {
            k.kv[gi].stream = &r.st;
            ++streamed;
        }
    CHECK_EQ(streamed, 1);
}

/* The `blocks`, `block_bytes` and `sel_pitch` the plugin declared for `kv_resolve`. */
void resolve_geom(const RadBuilder& b, int64_t* blocks, int64_t* bbytes, int64_t* pitch) {
    for (const RecOp& o : b.ops)
        if (o.op == "kv_resolve")
            for (const RecParam& p : o.p) {
                if (p.key == "blocks")      *blocks = p.ival;
                if (p.key == "block_bytes") *bbytes = p.ival;
                if (p.key == "sel_pitch")   *pitch  = p.ival;
            }
}

/* The `slots` the plugin declared for `kv_resolve`: the ceiling a ring may not pass. */
int64_t resolve_slots(const RadBuilder& b) {
    int64_t slots = 0;
    for (const RecOp& o : b.ops)
        if (o.op == "kv_resolve")
            for (const RecParam& p : o.p)
                if (p.key == "slots") slots = p.ival;
    return slots;
}

/* The declared slot domain covers a ring the engine raised, not just the flag's own number. The
 * engine raises --kv-resident to the smallest ring one decode step can name when the operator asks
 * for less (kvstream_ring, kvstream/plan.h), and the resolve refuses a pool deeper than its declared
 * `slots` (libref/ref_kvstream.cpp: the parameter is the ceiling the operand may not pass). So the
 * declaration states that step bound: max_seqs x (1 + n_spec) x sel_pitch, the same product
 * kvstream_ceiling_blocks states on the engine's side. With nothing streaming it stays the context,
 * which is what an undeclared ring has always meant. */
TEST(the_declared_slot_ceiling_covers_a_ring_raised_to_one_decode_step) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    c.kv_resident_blocks = 8192;              /* 2048 blocks of 4: under one step's selection */
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int64_t blocks = 0, bbytes = 0, pitch = 0;
    resolve_geom(b, &blocks, &bbytes, &pitch);
    REQUIRE(blocks > 0);
    REQUIRE(pitch > 0);
    const int64_t slots = resolve_slots(b);
    CHECK_EQ(slots, c.max_seqs * (1 + c.max_spec) * pitch);
    CHECK(slots > c.kv_resident_blocks / 4);  /* the raised depth, not the flag's */
    CHECK(slots < blocks);

    /* Nothing streaming: the ring covers the context, so the declaration is the context. */
    c.kv_resident_blocks = 0;
    RadBuilder b2;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b2, &meta, &c), RAD_OK);
    CHECK_EQ(resolve_slots(b2), blocks);
}

/* A streamed group's prefill stage: every layer's mirror row is copied into the one-layer stage
 * view after that layer's store and before its attention, in layer order. The stage is the whole
 * context at identity addressing, so the attention reads block ids straight out of it and issues
 * no resolve. Both the sparse path (above qsa_exact_to) and the dense path (below it, which has no
 * resolve at all) must stage; a skipped stage would read the previous layer's bytes.
 *
 * A staged store is two writers: `kv_store_streamed` (the mirror, and the ring only where the
 * residency map says the block is resident) and then `kv_store` at the same slots (the identity
 * copy the stage view reads). Without the second, a prefill before the group's first decode
 * advances the mirror and not the pool, and the attention reads the previous layer's bytes. This
 * test counts both and pins their order. */
TEST(a_staged_prefill_copies_each_layer_before_its_attention) {
    for (int dense = 0; dense < 2; ++dense) {
        RadModelMeta meta = qwen4exp_meta();
        RadBuildCtx c = build_ctx(0, 1);
        c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
        RadBuilder b;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
        int64_t blocks = 0, bbytes = 0, pitch = 0;
        resolve_geom(b, &blocks, &bbytes, &pitch);
        REQUIRE(blocks > 0);
        REQUIRE(bbytes > 0);

        /* 100 tokens of history is below qsa_exact_to (dense); 5000 is above it (sparse). */
        auto k = kit(b, {}, { 32 }, dense ? 100 : 5000);
        StreamRig rig;
        attach_stream(b, *k, rig, blocks, bbytes);

        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &k->b;
        ctx.stage_on = true;
        qwen4exp_fp8::step(&ctx, &k->b);

        int stores = 0, identity = 0, attn = 0;
        size_t last_store = 0;
        int last_slot_opd = -1;
        for (size_t i = 0; i < ctx.issues.size(); ++i) {
            const RecOp* o = b.op_of(ctx.issues[i].op);
            if (!o) continue;
            if (o->op == "kv_store_streamed" ||
                o->op == "qk_norm_rope_gate_kv_store_streamed") {
                ++stores;
                last_store = i;
                /* The streamed store's slot operand: 2 for the separate store, 10 for the folded
                 * prologue (qg, k, cos/sin, pos, two norm weights, q, k, the v gap, v, slot,
                 * cache, res, host). */
                last_slot_opd = o->op == "kv_store_streamed" ? 2 : 10;
            } else if (o->op == "kv_store") {
                /* A staged pass writes the pool at block-id identity, and kv_store_streamed writes
                 * its device copy only where the residency map says the block is resident. Before
                 * a decode the map is empty, so the stage would hold the previous layer's bytes
                 * unless the identity store follows the mirror writer with the same slots. */
                ++identity;
                CHECK(last_slot_opd >= 0);                     /* a streamed store came first */
                CHECK(i > last_store);
                CHECK_EQ(ctx.issues[i].opd[2].kind,
                         ctx.issues[last_store].opd[last_slot_opd].kind);
                CHECK_EQ(ctx.issues[i].opd[2].handle,
                         ctx.issues[last_store].opd[last_slot_opd].handle);
                CHECK_EQ(ctx.issues[i].opd[2].offset,
                         ctx.issues[last_store].opd[last_slot_opd].offset);
            } else if (o->op == "attn_paged" || o->op == "attn_paged_gate_quant") {
                ++attn;
            }
        }
        CHECK(stores > 0);
        CHECK_EQ(identity, stores);                    /* one identity store a staged store */
        CHECK_EQ(attn, stores);
        CHECK_EQ(ctx.stages.size(), (size_t)stores);   /* one stage a layer, no repeats */

        /* The stage copy's wait is immediately before the layer's store, not after it: the copy
         * reads the mirror, which the store has not written yet (core/kvstream/stage.h). */
        int prev_layer = -1;
        for (size_t s = 0; s < ctx.stages.size(); ++s) {
            const StageRec& r = ctx.stages[s];
            CHECK(r.layer > prev_layer);               /* layers in order */
            prev_layer = r.layer;
            REQUIRE(r.issue < ctx.issues.size());
            const RecOp* next = b.op_of(ctx.issues[r.issue].op);
            REQUIRE(next != nullptr);
            /* The wait sits immediately before the layer's store, which is either the separate
             * streamed store or the prologue that has it folded in (a step narrow enough for the
             * fused prologue). */
            CHECK(next->op == "kv_store_streamed" ||
                  next->op == "qk_norm_rope_gate_kv_store_streamed");
            if (s + 1 < ctx.stages.size())
                CHECK(ctx.stages[s + 1].issue > r.issue);
        }
        /* And the next layer's copy is handed over after every attention launch of this one, so the
         * ride it waits on is behind all of them. */
        CHECK_EQ(ctx.stages_next.size(), (size_t)stores);
        for (size_t s = 0; s < ctx.stages_next.size(); ++s) {
            const StageRec& r = ctx.stages_next[s];
            CHECK_EQ(r.layer, ctx.stages[s].layer);
            int seen = 0;
            for (size_t i = 0; i < r.issue && i < ctx.issues.size(); ++i) {
                const RecOp* o = b.op_of(ctx.issues[i].op);
                if (o && (o->op == "attn_paged" || o->op == "attn_paged_gate_quant")) ++seen;
            }
            /* This layer's own attention (at least) is behind the hand-over; a layer wider than a
             * launch has more than one. */
            CHECK(seen >= (int)s + 1);
        }
    }
}

/* Below `qsa_exact_to` a prefill takes the dense attention, which reads the block table and never
 * `sel`. The selection is still produced, but a resolve there would be read by nothing: work on the
 * step path whose only other effect is on the residency map's epoch, in a configuration the design
 * says never sees the op. So the dense step must issue none. */
TEST(qsa_below_the_exact_bound_takes_the_dense_path_without_a_resolve) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int64_t blocks = 0, bbytes = 0, pitch = 0;
    resolve_geom(b, &blocks, &bbytes, &pitch);
    REQUIRE(blocks > 0);
    REQUIRE(bbytes > 0);
    REQUIRE(pitch > 0);

    /* reach = max_ctx_len + max_q_len = (100 + 32) + 32: far below the 2051 bound, so dense. */
    auto k = kit(b, {}, { 32 }, 100);
    StreamRig rig;
    attach_stream(b, *k, rig, blocks, bbytes);

    RadCtx ctx;
    ctx.b = &b;
    ctx.batch = &k->b;
    qwen4exp_fp8::step(&ctx, &k->b);

    int selects = 0, resolves = 0, dense = 0;
    rad_buf sel = 0;
    for (const RecIssue& i : ctx.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "qsa_select") { ++selects; sel = i.opd[5].handle; }
        else if (o->op == "kv_resolve") ++resolves;
        else if (o->op == "attn_paged" || o->op == "attn_paged_gate_quant") {
            if (i.opd[2].handle != sel) ++dense;   /* the block table, not the selection */
        }
    }
    CHECK(selects > 0);   /* the score/select half still runs */
    CHECK_EQ(resolves, 0);
    CHECK(dense > 0);     /* and the launch that ran is the dense one, not a sparse one */
}

/* Above `qsa_exact_to` the prefill is sparse: the attention takes the step in one launch, and the
 * resolve right before it rewrites exactly its rows, in the same `sel` buffer the selection
 * produced. */
TEST(qsa_above_the_exact_bound_resolves_once_per_sparse_launch) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int64_t blocks = 0, bbytes = 0, pitch = 0;
    resolve_geom(b, &blocks, &bbytes, &pitch);
    REQUIRE(blocks > 0);
    REQUIRE(bbytes > 0);
    REQUIRE(pitch > 0);

    /* reach = (3000 + 32) + 32: above the bound, so every layer's launch is sparse. */
    auto k = kit(b, {}, { 32 }, 3000);
    StreamRig rig;
    attach_stream(b, *k, rig, blocks, bbytes);

    RadCtx ctx;
    ctx.b = &b;
    ctx.batch = &k->b;
    qwen4exp_fp8::step(&ctx, &k->b);

    int selects = 0, resolves = 0, attn = 0;
    size_t last_select = 0;
    for (size_t i = 0; i < ctx.issues.size(); ++i) {
        const RecOp* o = b.op_of(ctx.issues[i].op);
        if (!o) continue;
        if (o->op == "qsa_select") { ++selects; last_select = i; }
        else if (o->op == "kv_resolve") {
            ++resolves;
            const RadOperand& rsel  = ctx.issues[i].opd[0];
            const RadOperand& rnsel = ctx.issues[i].opd[1];
            const RadOperand& ssel  = ctx.issues[last_select].opd[5];
            const RadOperand& snsel = ctx.issues[last_select].opd[6];
            CHECK(i > last_select);              /* after the selection it rewrites */
            CHECK_EQ(rsel.handle, ssel.handle);
            CHECK_EQ(rsel.offset, ssel.offset);
            CHECK_EQ(rsel.rows,   ssel.rows);    /* the launch's rows, not the step's */
            CHECK_EQ(rnsel.offset, snsel.offset);
            CHECK_EQ(rnsel.rows,   snsel.rows);
            CHECK_EQ(rsel.rows, (int64_t)32);    /* one select launch of the whole step */
        } else if (o->op == "attn_paged" || o->op == "attn_paged_gate_quant") {
            ++attn;
            CHECK_EQ(ctx.issues[i].opd[2].handle,
                     ctx.issues[last_select].opd[5].handle);
            CHECK_EQ(ctx.issues[i].opd[2].offset,
                     ctx.issues[last_select].opd[5].offset);
            CHECK_EQ(ctx.issues[i].opd[2].rows, ctx.issues[last_select].opd[5].rows);
        }
    }
    CHECK_EQ(selects, 2);      /* one indexer per full-attention layer of the eight */
    CHECK_EQ(resolves, selects);
    CHECK_EQ(attn, selects);
}

/* A step wider than the score plane selects in windows, and the gated sparse attention takes the
 * whole step in one launch. The resolve is the attention's, not the indexer's: exactly one per
 * launch, over exactly that launch's rows, after the layer's store. So a launch may span several
 * select windows, and the resolve that precedes it rewrites all of their rows. */
TEST(qsa_a_widened_step_resolves_the_launch_that_reads_each_window) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 4096; c.max_mixed_tok = 2048;
    c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    const int64_t PR = m.qsacfg.plane_rows;
    REQUIRE(PR > 0);
    int64_t blocks = 0, bbytes = 0, pitch = 0;
    resolve_geom(b, &blocks, &bbytes, &pitch);
    REQUIRE(blocks > 0);
    REQUIRE(bbytes > 0);
    REQUIRE(pitch > 0);

    auto k = kit(b, {}, { 4096 }, 4096);   /* T = 2 windows, one sequence: reach is above */
    StreamRig rig;
    attach_stream(b, *k, rig, blocks, bbytes);

    RadCtx ctx;
    ctx.b = &b;
    ctx.batch = &k->b;
    qwen4exp_fp8::step(&ctx, &k->b);

    struct Span { int64_t off, rows; };
    std::vector<Span> resolved;
    int selects = 0, stores = 0, resolves = 0, attn = 0;
    size_t last_select = 0, last_store = 0;
    for (size_t i = 0; i < ctx.issues.size(); ++i) {
        const RecOp* o = b.op_of(ctx.issues[i].op);
        if (!o) continue;
        if (o->op == "qsa_select") {
            ++selects;
            last_select = i;
        } else if (o->op == "kv_store_streamed" ||
                   o->op == "qk_norm_rope_gate_kv_store_streamed") {
            ++stores;
            last_store = i;
        } else if (o->op == "kv_resolve") {
            ++resolves;
            const RadOperand& rsel = ctx.issues[i].opd[0];
            CHECK(i > last_store);              /* the mirror write precedes the copy */
            CHECK(i > last_select);
            CHECK_EQ(rsel.handle, ctx.issues[last_select].opd[5].handle);
            /* `sel` offsets are element offsets into a `pitch`-wide row buffer. */
            CHECK_EQ(rsel.offset % pitch, 0);
            resolved.push_back({ rsel.offset / pitch, rsel.rows });
        } else if (o->op == "attn_paged" || o->op == "attn_paged_gate_quant") {
            ++attn;
            const RadOperand& asel = ctx.issues[i].opd[2];
            /* The resolve that reads this launch's rows is the issue right before it. */
            REQUIRE(i > 0);
            const RecOp* p = b.op_of(ctx.issues[i - 1].op);
            REQUIRE(p != nullptr);
            CHECK_EQ(std::string(p->op), std::string("kv_resolve"));
            CHECK_EQ(ctx.issues[i - 1].opd[0].handle, asel.handle);
            CHECK_EQ(ctx.issues[i - 1].opd[0].offset, asel.offset);
            CHECK_EQ(ctx.issues[i - 1].opd[0].rows,   asel.rows);
            CHECK_EQ(asel.offset % pitch, 0);
        }
    }
    CHECK_EQ(selects, 4);      /* two full layers, two windows each */
    CHECK_EQ(stores, 2);
    CHECK_EQ(resolves, attn);  /* one resolve per attention launch, never two */
    CHECK(attn > 0);
    /* Every row of the step is resolved once a layer, by the one launch that reads it. */
    int64_t covered_to = 0;
    for (const Span& s : resolved) {
        CHECK_EQ(s.off, covered_to % 4096);
        covered_to += s.rows;
    }
    CHECK_EQ(covered_to, (int64_t)4096 * stores);   /* one launch, the whole step, a layer */
}

/* With no stream on the attention group's batch there is nothing to resolve: the selection is a
 * block table the pool already addresses. Issuing the op anyway would make a non-streaming
 * configuration pay for it. */
TEST(qsa_without_a_stream_the_selection_is_never_resolved) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);

    auto k = kit(b, {}, { 32 }, 3000);   /* the sparse path, but no stream to make resident */
    RadCtx ctx;
    ctx.b = &b;
    ctx.batch = &k->b;
    qwen4exp_fp8::step(&ctx, &k->b);

    int selects = 0, resolves = 0, attn = 0;
    rad_buf sel = 0;
    for (const RecIssue& i : ctx.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "qsa_select") { ++selects; sel = i.opd[5].handle; }
        else if (o->op == "kv_resolve") ++resolves;
        else if (o->op == "attn_paged" || o->op == "attn_paged_gate_quant") {
            ++attn;
            CHECK_EQ(i.opd[2].handle, sel);   /* the sparse path still read the selection */
        }
    }
    CHECK_EQ(selects, 2);   /* the two full-attention layers of the eight */
    CHECK_EQ(resolves, 0);
    CHECK_EQ(attn, selects);
}

/* At or below the exactness bound nothing reads the selection, so the indexer issues none of its
 * query side (docs/QSA.md, "Below the bound"; AttnGatedFP8::qsa_use, QsaIndexer::step): a step of
 * several rows a sequence -- a chain verify -- attends densely over the block table, and a step of
 * one row a sequence issues the sparse row's launch over the sequence's own table and length. The
 * stateful half (work list, block keys, tail) runs on every step either way. Above the bound, by
 * the rounded context bound, both score and select per layer and attend over the selection. */
TEST(qsa_below_the_exact_bound_the_indexer_issues_no_query_score_or_select) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    size_t full = 0;
    for (const auto& l : m.layers) full += l.full ? 1 : 0;
    REQUIRE(full > 0);

    struct Seen { size_t work = 0, bkey = 0, tail = 0, score = 0, select = 0, qprep = 0;
                  size_t attn = 0, by_table = 0, by_sel = 0, dense = 0; };
    auto look = [&](const Kit& k) {
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &k.b;
        qwen4exp_fp8::step(&ctx, &k.b);
        Seen s;
        for (const RecIssue& i : ctx.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o) continue;
            if (o->op == "qsa_work") ++s.work;
            else if (o->op == "qsa_block_key") ++s.bkey;
            else if (o->op == "qsa_tail_store") ++s.tail;
            else if (o->op == "qsa_score") ++s.score;
            else if (o->op == "qsa_select") ++s.select;
            else if (o->op == "attn_paged" || o->op == "attn_paged_gate_quant") {
                ++s.attn;
                const RadOperand& t = i.opd[2];
                const RadOperand& u = i.opd[3];
                if (t.kind == RAD_OPK_BUF && t.handle == m.qsaw.sel) {
                    ++s.by_sel;
                    CHECK(u.kind == RAD_OPK_BUF && u.handle == m.qsaw.sequ);
                } else {
                    CHECK(t.kind == RAD_OPK_RAW && t.raw == (void*)k.table.data());
                    CHECK(u.kind == RAD_OPK_RAW && u.raw == (void*)k.used.data());
                    /* the sparse row's launch (q_len 1, a row a sequence) or the dense one */
                    if (o->op == "attn_paged_gate_quant" || (i.opd.size() > 6 && i.opd[6].kind == RAD_OPK_NONE))
                        ++s.by_table;
                    else
                        ++s.dense;
                }
            }
            for (const auto& l : m.layers)
                if (l.full && l.qsa.op_qprep && i.op == l.qsa.op_qprep) ++s.qprep;
        }
        return s;
    };

    /* A chain verify of two sequences, 4 rows each, at contexts 40 and 47. */
    {
        const auto k = kit(b, { 4, 4 }, {});
        const Seen s = look(*k);
        CHECK_EQ(s.work, full); CHECK_EQ(s.bkey, full); CHECK_EQ(s.tail, full);
        CHECK_EQ(s.score, (size_t)0); CHECK_EQ(s.select, (size_t)0); CHECK_EQ(s.qprep, (size_t)0);
        CHECK_EQ(s.attn, full); CHECK_EQ(s.dense, full);
        CHECK_EQ(s.by_sel, (size_t)0);
    }
    /* A plain decode step: one row a sequence, the sparse row over the block table. */
    {
        const auto k = kit(b, { 1, 1, 1 }, {});
        const Seen s = look(*k);
        CHECK_EQ(s.work, full); CHECK_EQ(s.bkey, full); CHECK_EQ(s.tail, full);
        CHECK_EQ(s.score, (size_t)0); CHECK_EQ(s.select, (size_t)0); CHECK_EQ(s.qprep, (size_t)0);
        CHECK_EQ(s.attn, full); CHECK_EQ(s.by_table, full);
        CHECK_EQ(s.by_sel, (size_t)0); CHECK_EQ(s.dense, (size_t)0);
    }
    /* The same two steps past the bound by the rounded context bound: everything, per layer, and
     * the attention over the selection. */
    for (int rows : { 4, 1 }) {
        const auto k = kit(b, { rows, rows }, {});
        k->b.max_ctx_len = 3072;
        const Seen s = look(*k);
        CHECK_EQ(s.work, full); CHECK_EQ(s.bkey, full); CHECK_EQ(s.tail, full);
        CHECK_EQ(s.score, full); CHECK_EQ(s.select, full); CHECK_EQ(s.qprep, full);
        CHECK(s.attn >= full);
        CHECK_EQ(s.by_sel, s.attn); CHECK_EQ(s.dense, (size_t)0); CHECK_EQ(s.by_table, (size_t)0);
    }
    /* One row a sequence is the table's only where every sequence has one: a sequence with no
     * row beside one with two is the same T == n_seq and attends over a selection, which at or
     * below the bound is the first full-attention layer's for the whole pass. */
    {
        const auto k = kit(b, { 1, 1 }, {});
        k->b.max_q_len = 2;
        const Seen s = look(*k);
        CHECK_EQ(s.select, (size_t)1); CHECK_EQ(s.score, (size_t)1); CHECK_EQ(s.qprep, (size_t)1);
        CHECK_EQ(s.work, full); CHECK_EQ(s.bkey, full); CHECK_EQ(s.tail, full);
        CHECK_EQ(s.by_sel, s.attn); CHECK_EQ(s.by_table, (size_t)0);
    }
}

/* At decode heights the plugin issues a layer's small launches where a recorded pass lets them
 * ride (qwen4exp_fp8.cpp: kVerifyRows, QsaSide): the k and v projections in front of the q|gate
 * projection; the indexer's keys behind the attention's own launches where nothing of the layer
 * reads them; the router's weight read beside the mixer's write; and from kVerifyRows rows the
 * shared down projection in front of the routed one and the reads at the verify step's workgroup
 * count. A prompt step keeps the declared order. */
TEST(a_flash_next_decode_step_orders_its_small_launches) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 2);
    c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    REQUIRE(m.qsa_defer);
    auto run = [&](const Kit& k) {
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &k.b;
        qwen4exp_fp8::step(&ctx, &k.b);
        return ctx.issues;
    };
    /* Where an op was issued, when it was issued exactly once; -1 never, -2 more than once. */
    auto pos = [](const std::vector<RecIssue>& is, rad_op op) {
        int64_t at = -1;
        for (size_t i = 0; i < is.size(); ++i)
            if (is[i].op == op) at = at == -1 ? (int64_t)i : -2;
        return at;
    };
    const auto ver = kit(b, { 4 }, {});          /* a chain-3 verify */
    const auto two = kit(b, { 2 }, {});          /* a chain-1 verify, the fewest rows of one */
    const auto one = kit(b, { 1 }, {});          /* a plain decode step */
    const auto pre = kit(b, {}, { 32 });         /* a prompt chunk */
    const auto iv = run(*ver), i2 = run(*two), i1 = run(*one), ip = run(*pre);

    size_t full = 0;
    for (const auto& l : m.layers) {
        const rad_op wr = l.hc_mix.op_ar_write ? l.hc_mix.op_ar_write : l.hc_mix.op_write;
        const rad_op out = l.full ? l.attn.o.op : l.gdn.out.op;
        /* The router's read: behind the mixer's out projection and in front of its write, at the
         * step's workgroup count; a prompt step reads nothing ahead. */
        REQUIRE(l.op_pf_router != 0);
        REQUIRE(l.op_pf_router_v != 0);
        for (const auto* is : { &iv, &i2 }) {
            CHECK(pos(*is, l.op_pf_router_v) > pos(*is, out));
            CHECK(pos(*is, l.op_pf_router_v) < pos(*is, wr));
            CHECK_EQ(pos(*is, l.op_pf_router), (int64_t)-1);
        }
        CHECK(pos(i1, l.op_pf_router) > pos(i1, out));
        CHECK(pos(i1, l.op_pf_router) < pos(i1, wr));
        CHECK_EQ(pos(i1, l.op_pf_router_v), (int64_t)-1);
        CHECK_EQ(pos(ip, l.op_pf_router), (int64_t)-1);
        CHECK_EQ(pos(ip, l.op_pf_router_v), (int64_t)-1);
        /* The shared down projection: in front of the routed down GEMM from kVerifyRows rows,
         * behind it at one row and on a prompt step. */
        REQUIRE(pos(iv, l.mlp.op_dn) >= 0);
        REQUIRE(pos(iv, l.mlp.sh_down.op) >= 0);
        CHECK(pos(iv, l.mlp.sh_down.op) < pos(iv, l.mlp.op_dn));
        CHECK(pos(i2, l.mlp.sh_down.op) < pos(i2, l.mlp.op_dn));
        CHECK(pos(i1, l.mlp.sh_down.op) > pos(i1, l.mlp.op_dn));
        CHECK(pos(ip, l.mlp.sh_down.op) > pos(ip, l.mlp.op_dn));
        if (!l.full) {
            /* The out projection's read at the step's workgroup count. */
            REQUIRE(l.gdn.op_pf_out != 0);
            REQUIRE(l.gdn.op_pf_out_v != 0);
            for (const auto* is : { &iv, &i2 }) {
                CHECK(pos(*is, l.gdn.op_pf_out_v) >= 0);
                CHECK_EQ(pos(*is, l.gdn.op_pf_out), (int64_t)-1);
            }
            CHECK(pos(i1, l.gdn.op_pf_out) >= 0);
            CHECK_EQ(pos(i1, l.gdn.op_pf_out_v), (int64_t)-1);
            continue;
        }
        ++full;
        REQUIRE(l.attn.op_pf_o != 0);
        REQUIRE(l.attn.op_pf_o_v != 0);
        for (const auto* is : { &iv, &i2 }) {
            CHECK(pos(*is, l.attn.op_pf_o_v) >= 0);
            CHECK_EQ(pos(*is, l.attn.op_pf_o), (int64_t)-1);
        }
        CHECK(pos(i1, l.attn.op_pf_o) >= 0);
        CHECK_EQ(pos(i1, l.attn.op_pf_o_v), (int64_t)-1);
        for (const auto* is : { &iv, &i2, &i1 }) {
            /* k and v in front of q|gate, the indexer's projection in front of all three. */
            REQUIRE(pos(*is, l.attn.qg.op) >= 0);
            CHECK(pos(*is, l.qsa.proj.op) >= 0);
            CHECK(pos(*is, l.qsa.proj.op) < pos(*is, l.attn.kp.op));
            CHECK(pos(*is, l.attn.kp.op) < pos(*is, l.attn.vp.op));
            CHECK(pos(*is, l.attn.vp.op) < pos(*is, l.attn.qg.op));
            /* The keys behind the attention's projections, in their own order, and all of them
             * in front of the write that ends the block. */
            CHECK(pos(*is, l.qsa.op_work) > pos(*is, l.attn.qg.op));
            CHECK(pos(*is, l.qsa.op_bkey) > pos(*is, l.qsa.op_work));
            CHECK(pos(*is, l.qsa.op_tail) > pos(*is, l.qsa.op_bkey));
            CHECK(pos(*is, l.qsa.op_tail) < pos(*is, wr));
        }
        /* A prompt step: the indexer in front of the attention, q|gate in front of k and v. */
        CHECK(pos(ip, l.qsa.op_tail) > pos(ip, l.qsa.op_work));
        CHECK(pos(ip, l.qsa.op_tail) < pos(ip, l.attn.qg.op));
        CHECK(pos(ip, l.attn.qg.op) < pos(ip, l.attn.kp.op));
        CHECK(pos(ip, l.attn.kp.op) < pos(ip, l.attn.vp.op));
    }
    REQUIRE(full > 0);

    /* Past the exactness bound the attention reads this layer's selection, so the indexer stays
     * in front of it at every height. */
    {
        const auto k = kit(b, { 4 }, {});
        k->b.max_ctx_len = 3072;
        const auto is = run(*k);
        for (const auto& l : m.layers) {
            if (!l.full) continue;
            CHECK(pos(is, l.qsa.op_work) >= 0);
            CHECK(pos(is, l.qsa.op_tail) < pos(is, l.attn.kp.op));
        }
    }
}

/* Whether a step is within the indexer's budget is read from the attention group's exact host
 * ceiling where the batch carries it (RadBatch::kv_used_max_h), not from the context bound rounded
 * up to 1024 tokens: a 4-row verify at 1500 tokens of context is within it though
 * max_ctx_len + max_q_len (2048 + 4) is past it. The kernel does not move with that -- such a step
 * keeps the sparse row a query, as before, because the dense and the sparse kernel agree on the
 * keys and not on the last bit -- but its selection is every block, the same bytes at every layer,
 * so the pass's first full-attention layer issues it as always and the others issue none
 * (kQsaOnce). The pass key carries the side, so the two are two recordings
 * (qwen4exp_fp8::pass_key). */
TEST(qsa_reads_the_exact_host_bound_and_keys_its_side_of_the_exact_bound) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 64; c.max_seqs = 4; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    const int64_t exact_to = m.qsacfg.budget + m.qsacfg.ratio - 1;
    REQUIRE(exact_to > 1024);
    size_t full = 0;
    for (const auto& l : m.layers) full += l.full ? 1 : 0;
    REQUIRE(full > 1);

    struct Seen { size_t score = 0, select = 0, work = 0, attn = 0, by_sel = 0, first_sel = 0, first_attn = 0; };
    auto look = [&](const Kit& k) {
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &k.b;
        qwen4exp_fp8::step(&ctx, &k.b);
        CHECK(ctx.step_fail.empty());
        Seen s;
        size_t at = 0;
        for (const RecIssue& i : ctx.issues) {
            ++at;
            const RecOp* o = b.op_of(i.op);
            if (!o) continue;
            if (o->op == "qsa_score") ++s.score;
            else if (o->op == "qsa_work") ++s.work;
            else if (o->op == "qsa_select") {
                ++s.select;
                if (!s.first_sel) s.first_sel = at;
            } else if (o->op == "attn_paged" || o->op == "attn_paged_gate_quant") {
                ++s.attn;
                if (!s.first_attn) s.first_attn = at;
                if (i.opd[2].kind == RAD_OPK_BUF && i.opd[2].handle == m.qsaw.sel) ++s.by_sel;
            }
        }
        return s;
    };
    const auto k = kit(b, { 4 }, {});
    k->b.max_ctx_len = 2048;                                  /* 1500 rounded up */
    std::vector<int32_t> umax((size_t)k->b.n_kv_groups, 0);
    /* No host ceiling: the rounded bound decides, and 2048 + 4 is past the bound. */
    {
        const Seen s = look(*k);
        CHECK_EQ(s.score, full); CHECK_EQ(s.select, full); CHECK_EQ(s.work, full);
        CHECK(s.attn >= full); CHECK_EQ(s.by_sel, s.attn);
    }
    RadCtx kc;
    kc.b = &b;
    kc.batch = &k->b;
    const uint64_t key_none = qwen4exp_fp8::pass_key(&kc, &k->b, nullptr);
    /* 1504 rows of context at most: within the bound. The same sparse launches, every layer's,
     * over the first layer's selection; the stateful half still at every layer. */
    for (auto& u : umax) u = 1504;
    k->b.kv_used_max_h = umax.data();
    {
        const Seen s = look(*k);
        CHECK_EQ(s.score, (size_t)1); CHECK_EQ(s.select, (size_t)1); CHECK_EQ(s.work, full);
        CHECK(s.attn >= full); CHECK_EQ(s.by_sel, s.attn);
        CHECK(s.first_sel > 0 && s.first_sel < s.first_attn);
    }
    const uint64_t key_in = qwen4exp_fp8::pass_key(&kc, &k->b, umax.data());
    /* The last step within it, and the first past it. */
    for (auto& u : umax) u = (int32_t)exact_to;
    CHECK_EQ(look(*k).score, (size_t)1);
    CHECK_EQ(qwen4exp_fp8::pass_key(&kc, &k->b, umax.data()), key_in);
    for (auto& u : umax) u = (int32_t)exact_to + 1;
    CHECK_EQ(look(*k).score, full);
    const uint64_t key_out = qwen4exp_fp8::pass_key(&kc, &k->b, umax.data());
    CHECK(key_in != 0 && key_out != 0);
    CHECK(key_in != key_out);
    CHECK_EQ(key_out, key_none);
    /* One row a sequence at the same context is the block table's whatever the rounded bound
     * says: no selection at all. */
    {
        const auto k1 = kit(b, { 1 }, {});
        k1->b.max_ctx_len = 2048;
        std::vector<int32_t> u1((size_t)k1->b.n_kv_groups, 1501);
        k1->b.kv_used_max_h = u1.data();
        const Seen s = look(*k1);
        CHECK_EQ(s.score, (size_t)0); CHECK_EQ(s.select, (size_t)0); CHECK_EQ(s.work, full);
        CHECK_EQ(s.attn, full); CHECK_EQ(s.by_sel, (size_t)0);
    }
}

/* The plugin declares by each weight's stored encoding. Flash-Next's experts, gated-residual
 * mixing matrices, MTP fc and 2-bit draft head each have a recipe-chosen format the plugin reads
 * from rad_weight_encoding, at convert and at load alike. With int4-rotated, E4M3-row and u2
 * encodings it declares the rotated expert kernels, the E4M3 mix and the 2-bit head; with no model
 * behind it, the fp8 experts and the bf16 mix. */
TEST(the_encodings_the_model_holds_choose_the_flash_next_kernels) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding row8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&row8) && rad_enc_valid(&u2));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 },
               { "_hc_down.weight", row8 }, { "_hc_up.weight", row8 },
               { "fc_hidden", row8 }, { "fc_embedding", row8 }, { "draft_head", u2 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    CHECK_EQ(m.moecfg.expert_w4, 1);
    CHECK_EQ(m.moecfg.expert_rot, 1);
    CHECK(m.layers[0].hc_mix.mix_e4m3);
    int rotated = 0, fp8_experts = 0, head_views = 0, scale_views = 0;
    for (const RecOp& o : b.ops)
        if (o.op.rfind("moe_gemm", 0) == 0)
            for (const auto& p : o.p) {
                if (p.key == "dtype" && p.sval == "w4a8h") ++rotated;
                if (p.key == "dtype" && p.sval == "fp8a8") ++fp8_experts;
            }
    for (const RecWeight& w : b.weights) {
        if (w.d.planes && std::string(w.d.planes) == "scale,zero") ++head_views;
        /* every expert's scale is a view of its logical weight, not a separate weight */
        if (w.name.find("_exps.") != std::string::npos &&
            w.name.size() > 6 && w.name.compare(w.name.size() - 6, 6, ".scale") == 0) {
            ++scale_views;
            CHECK(w.d.source != nullptr && w.d.planes && std::string(w.d.planes) == "scale");
        }
    }
    CHECK(rotated > 0);
    CHECK_EQ(fp8_experts, 0);
    CHECK_EQ(head_views, 1);
    CHECK(scale_views > 0);

    RadBuilder plain;
    REQUIRE_EQ(qwen4exp_fp8::declare(&plain, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& p = qwen4exp_fp8::g_model[0];
    CHECK_EQ(p.moecfg.expert_w4, 0);
    CHECK(!p.layers[0].hc_mix.mix_e4m3);
}

/* The input to gate_up and the gated input to down are distinct activations: a trellis
 * gate may read rotated int8 while a codebook (W5NL, W6NL) down reads a rotated E4M3 gated
 * product. */
TEST(flashnext_trellis_gate_and_codebook_down_use_separate_activation_formats) {
    RadEncoding tq{};
    rad_enc_clear(&tq);
    rad_enc_copy_str(tq.scheme, "trellis");
    rad_enc_add_plane(&tq, "codes", RAD_U5, 1, 1);
    rad_enc_add_plane(&tq, "scale", RAD_F32, 1, 0);
    const int t = rad_enc_add_table(&tq, "trellis", RAD_I8, 1 << 16, 1);
    tq.plane[t].block[0] = 128;
    rad_enc_add_table(&tq, "trellis.mad", RAD_U32, 2, 1);
    rad_enc_add_plane(&tq, "scale.col", RAD_F32, 0, 128);
    rad_enc_copy_str(tq.transform, "fwht128");

    RadEncoding w5{};
    rad_enc_clear(&w5);
    rad_enc_copy_str(w5.scheme, "affine");
    rad_enc_add_plane(&w5, "codes", RAD_U5, 1, 1);
    rad_enc_add_plane(&w5, "scale", RAD_F8E4M3, 1, 64);
    rad_enc_add_plane(&w5, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&w5, "table", RAD_F32, 1, 32);
    rad_enc_copy_str(w5.transform, "fwht128");
    REQUIRE(rad_enc_valid(&tq));
    REQUIRE(rad_enc_valid(&w5));
    RadEncoding r8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(r8.transform, "fwht128");
    REQUIRE(rad_enc_valid(&r8));

    for (int gate_bits : { 4, 5 }) for (int down_bits : { 5, 6 }) {
        RadEncoding gateq = tq, downq = w5;
        gateq.plane[0].dtype = gate_bits == 4 ? RAD_U4 : RAD_U5;
        downq.plane[0].dtype = down_bits == 5 ? RAD_U5 : RAD_U6;
        downq.plane[3].extent[1] = 1 << down_bits;
        REQUIRE(rad_enc_valid(&gateq));
        REQUIRE(rad_enc_valid(&downq));
        for (int rank = 0; rank < 2; ++rank) {
            RadModelMeta meta = qwen4exp_meta();
            RadBuildCtx c = build_ctx(rank, 2);
            c.max_tok = 64; c.max_seqs = 2; c.max_spec = 0; c.max_ctx = 4096;
            RadBuilder b;
            const RadEncoding plain = rad_enc_plain(RAD_BF16);
            b.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain },
                       { "blk.1.ffn_down_exps.3.weight", plain },
                       { "blk.1.ffn_gate_up_exps.10.weight", plain },
                       { "blk.1.ffn_down_exps.10.weight", plain },
                       { "ffn_gate_up_exps", gateq }, { "ffn_down_exps", downq },
                       { "ssm_inz.weight", tq }, { "attn_qg.weight", tq } };
            for (const char* name : { "qsa_qk.weight", "attn_k.weight", "attn_v.weight",
                                      "attn_output.weight", "ssm_out.weight",
                                      "ffn_gate_up_shexp", "ffn_down_shexp" })
                b.encs.push_back({ name, r8 });
            REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
            const auto& m = qwen4exp_fp8::g_model[rank];
            const auto& moe = m.layers[0].mlp;
            CHECK_EQ(moe.fmt[0], 11);
            CHECK_EQ(moe.fmt[1], down_bits == 5 ? 7 : 12);
            CHECK_EQ(m.layers[1].mlp.prot.size(), (size_t)2);
            CHECK_EQ(m.layers[1].mlp.fmt[0], 11);
            CHECK_EQ(m.layers[1].mlp.fmt[1], down_bits == 5 ? 7 : 12);
            CHECK(m.a_eff.q && m.a_eff.s && m.a_eff.q8 && m.a_eff.s8);
            CHECK_EQ(moe.gq_exp.a.cq(), m.a_eff.q);
            CHECK_EQ(moe.gq_exp.a.cs(), m.a_eff.s);
            const RecOp* gate = b.op_of(moe.op_gu);
            const RecOp* down = b.op_of(moe.op_dn);
            const RecOp* quant = b.op_of(moe.gq_exp.op);
            REQUIRE(gate && down && quant);
            CHECK_EQ(quant->op, std::string("gated_had_quant_fp8"));
            REQUIRE(!down->reads.empty());
            CHECK_EQ(down->reads[0], m.a_eff.q);
            bool gate_i8 = false, down_w5 = false;
            for (const auto& p : gate->p) gate_i8 |= p.key == "dtype" && p.sval == "i8a8h";
            for (const auto& p : down->p)
                down_w5 |= p.key == "dtype" &&
                           p.sval == (down_bits == 5 ? "w5nl64a8h" : "w6nl64a8h");
            CHECK(gate_i8);
            CHECK(down_w5);

            /* Declaration alone cannot verify the step's actual wiring. Inspect the issued
             * gate -> rotated E4M3 producer -> down chain on each TP rank as well. */
            std::unique_ptr<Kit> k = kit(b, { 1 }, {});
            RadCtx ctx;
            ctx.b = &b;
            ctx.batch = &k->b;
            qwen4exp_fp8::step(&ctx, &k->b);
            const RecIssue *gu_issue = nullptr, *quant_issue = nullptr, *dn_issue = nullptr;
            size_t igu = 0, iq = 0, idn = 0;
            for (size_t i = 0; i < ctx.issues.size(); ++i) {
                const RecIssue& issue = ctx.issues[i];
                if (issue.op == moe.op_gu) { gu_issue = &issue; igu = i; }
                if (issue.op == moe.gq_exp.op) { quant_issue = &issue; iq = i; }
                if (issue.op == moe.op_dn) { dn_issue = &issue; idn = i; }
            }
            REQUIRE(gu_issue && quant_issue && dn_issue);
            CHECK(igu < iq && iq < idn);
            REQUIRE(gu_issue->opd.size() >= 9);
            REQUIRE(quant_issue->opd.size() >= 3);
            REQUIRE(dn_issue->opd.size() >= 2);
            CHECK_EQ(gu_issue->opd.back().handle, quant_issue->opd[0].handle);
            CHECK_EQ(quant_issue->opd[1].handle, m.a_eff.q);
            CHECK_EQ(quant_issue->opd[2].handle, m.a_eff.s);
            CHECK_EQ(dn_issue->opd[0].handle, quant_issue->opd[1].handle);
            CHECK_EQ(dn_issue->opd[1].handle, quant_issue->opd[2].handle);
            CHECK_EQ(dn_issue->opd[0].offset, quant_issue->opd[1].offset);
            CHECK_EQ(dn_issue->opd[1].offset, quant_issue->opd[2].offset);
            CHECK(gu_issue->opd[0].handle != quant_issue->opd[1].handle);
        }
    }
}

/* The MTP-only expert tap must see history rows even in chunks that would ordinarily
 * store only KV. When round 1 is needed it selects after, not before, the routed block. */
TEST(flashnext_mtp_calibration_runs_the_history_rows_through_the_experts) {
    for (int rank = 0; rank < 2; ++rank) {
    RadModelMeta meta = qwen4exp_meta();
    RadBuildCtx build = build_ctx(rank, 2);
    build.max_tok = 64; build.max_seqs = 2; build.max_spec = 1; build.max_ctx = 4096;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &build), RAD_OK);
    auto& model = qwen4exp_fp8::g_model[rank];
    REQUIRE(model.have_mtp);
    auto k = kit(b, {}, { 8 }, 10);
    k->b.draft_pass = -1;
    k->b.phase = RAD_PHASE_DECODE;
    k->out = { 0, 1, 2, 3, 4, 5, 6, 7 };
    k->b.out_ids = k->out.data();
    k->b.n_out = 8;
    k->dout = { 7 };
    k->b.draft_out_ids = k->dout.data();
    auto issue = [](const RadCtx& ctx, rad_op op) {
        for (size_t i = 0; i < ctx.issues.size(); ++i)
            if (ctx.issues[i].op == op) return (int)i;
        return -1;
    };

    /* A normal history-only chunk skips the expensive draft feed-forward. */
    k->b.n_draft_out = 0;
    model.mtp.mlp.c.calib = 0;
    RadCtx normal{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
    model.mtp.step(&normal, &k->b);
    CHECK_EQ(issue(normal, model.mtp.mlp.op_gu), -1);

    model.mtp.mlp.c.calib = 1;
    for (int incomplete : { 0, 1, 7 }) {
        k->b.n_out = incomplete;
        RadCtx invalid{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
        model.mtp.step(&invalid, &k->b);
        CHECK(!invalid.step_fail.empty());
        CHECK_EQ(issue(invalid, model.mtp.mlp.op_gu), -1);
    }
    k->b.n_out = 8;
    RadCtx tap{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
    model.mtp.step(&tap, &k->b);
    CHECK(tap.step_fail.empty());
    CHECK(issue(tap, model.mtp.mlp.op_gu) >= 0);
    CHECK(issue(tap, model.mtp.mlp.op_dn) >= 0);
    CHECK_EQ(issue(tap, model.mtp.op_dsel), -1);
    CHECK_EQ(issue(tap, model.mtp.op_logits), -1);

    k->b.n_draft_out = 1;
    RadCtx draft{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
    model.mtp.step(&draft, &k->b);
    const int dn = issue(draft, model.mtp.mlp.op_dn);
    const int select = issue(draft, model.mtp.op_dsel);
    const int back = issue(draft, model.mtp.op_dback);
    const int logits = issue(draft, model.mtp.op_logits);
    CHECK(dn >= 0 && dn < select && select < back && back < logits);

    /* Once the target was written the ordinary history fast path returns. */
    model.mtp.mlp.calib_done = true;
    k->b.n_draft_out = 0;
    RadCtx done{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
    model.mtp.step(&done, &k->b);
    CHECK_EQ(issue(done, model.mtp.mlp.op_gu), -1);
    model.mtp.mlp.c.calib = 0;
    model.mtp.mlp.calib_done = false;
    }
}

/* Run this case separately with RADIANCE_CALIB_DIR set and RADIANCE_CALIB_LAYERS=8:
 * the synthetic model has eight trunk layers, so 8 is its MTP layer. */
TEST(flashnext_mtp_calibration_selects_only_the_requested_layer) {
    const char* dir = std::getenv("RADIANCE_CALIB_DIR");
    const char* layers = std::getenv("RADIANCE_CALIB_LAYERS");
    if (!dir || !*dir || !layers || std::strcmp(layers, "8") != 0) {
        CHECK(true);             /* the ordinary suite does not enable a calibration */
        return;
    }
    for (int rank = 0; rank < 2; ++rank) {
        RadModelMeta meta = qwen4exp_meta();
        RadBuildCtx c = build_ctx(rank, 2);
        c.max_tok = 64; c.max_seqs = 2; c.max_spec = 1; c.max_ctx = 4096;
        RadBuilder b;
        const RadEncoding bf16 = rad_enc_plain(RAD_BF16);
        b.encs = { { "ffn_gate_up_exps", bf16 }, { "ffn_down_exps", bf16 } };
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
        auto& m = qwen4exp_fp8::g_model[rank];
        CHECK(m.moecfg.calib);
        for (const auto& layer : m.layers) {
            CHECK_EQ(layer.mlp.c.calib, 0);
            CHECK_EQ(layer.mlp.op_gram_gu, 0u);
        }
        CHECK_EQ(m.mtp.mlp.c.calib, 1);
        CHECK(m.mtp.op_attn_gram);
        CHECK(m.mtp.b_attn_gram);
        CHECK(m.mtp.mlp.op_gram_gu);
        CHECK(m.mtp.mlp.op_gram_dn);
        CHECK(m.mtp.mlp.op_gram_dne);
        CHECK_EQ(m.mtp.mlp.c.n_ff_exp, 640);
        CHECK_EQ(m.mtp.mlp.c.n_expert, 256);

        /* The fake stream cannot populate per-expert offsets; retain the pooled tap
         * and lint the full trunk -> calibrated history issue order instead. */
        m.mtp.mlp.op_gram_dne = 0;
        auto k = kit(b, {}, { 8 }, 10);
        RadCtx trunk{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
        qwen4exp_fp8::step(&trunk, &k->b);
        /* The trunk samples only its last prompt row. Serial history separately names
         * all eight of the trunk's wide hidden rows as its own out_ids. */
        k->out = { 0, 1, 2, 3, 4, 5, 6, 7 };
        k->b.out_ids = k->out.data();
        k->b.n_out = 8;
        k->b.draft_pass = -1;
        k->b.phase = RAD_PHASE_DECODE;
        k->b.n_draft_out = 1;
        k->dout = { 7 };
        k->b.draft_out_ids = k->dout.data();
        RadCtx history{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
        qwen4exp_fp8::step(&history, &k->b);
        CHECK(history.step_fail.empty());
        CHECK_EQ(m.mtp.attn_gram_rows, 8);
        CHECK(issue_of(b, history, "gram_accum") != nullptr);
        const auto lint = lint_step(b, { &trunk, &history });
        for (const std::string& problem : lint) std::fprintf(stderr, "MTP tap lint: %s\n", problem.c_str());
        CHECK(lint.empty());
        k->b.n_draft_out = 0;   /* an intermediate, history-only prefill chunk */
        RadCtx middle{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
        qwen4exp_fp8::step(&middle, &k->b);
        CHECK(middle.step_fail.empty());
        const auto mlint = lint_step(b, { &trunk, &middle });
        for (const std::string& problem : mlint)
            std::fprintf(stderr, "MTP history-only lint: %s\n", problem.c_str());
        CHECK(mlint.empty());
    }
}

/* A quantised container may keep one linear as the checkpoint holds it. Flash-Next's recipe leaves
 * the QSA indexer's projection plain bf16 (its top-k is a hard choice between blocks), so in a
 * four-bit model with fp8 linears it is declared bf16 and read by `gemm_nt` over its input's bf16
 * plane, while the attention projections beside it stay block fp8. */
TEST(a_quantised_flash_next_reads_its_plain_indexer_as_bf16) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&plain));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 }, { "qsa_qk", plain } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int qk = 0, qk_bf16 = 0, attn_k_fp8 = 0;
    for (const RecWeight& w : b.weights) {
        if (w.name.find(".qsa_qk.weight") != std::string::npos) {
            ++qk;
            qk_bf16 += w.d.dtype == RAD_BF16 && !w.d.planes;
        }
        if (w.name.find(".attn_k.weight") != std::string::npos) attn_k_fp8 += w.d.dtype == RAD_F8E4M3;
    }
    CHECK(qk > 0);
    CHECK_EQ(qk_bf16, qk);
    CHECK(attn_k_fp8 > 0);
    /* every op that reads an indexer weight is the plain GEMM */
    int qk_ops = 0, qk_plain_ops = 0;
    for (const RecOp& o : b.ops)
        for (rad_weight wh : o.w)
            if (wh && b.weights[wh - 1].name.find(".qsa_qk.weight") != std::string::npos) {
                ++qk_ops;
                qk_plain_ops += o.op == "gemm_nt";
            }
    CHECK(qk_ops > 0);
    CHECK_EQ(qk_plain_ops, qk_ops);
}

/* A quantised layer may keep a few experts plain bf16. Layer 1 of a four-bit Flash-Next keeps
 * experts 3 and 10: that layer's sort places them last, its four-bit GEMMs serve the others in
 * slots 0..n-3 with no gap (a layer's experts are consecutive fixed-size units and routing counts
 * are in sort order), and the two are static bf16 weights behind a bf16 grouped GEMM pair, one of
 * which writes the same routed output as the four-bit down GEMM. Other layers are ordinary. */
TEST(a_four_bit_layer_serves_its_plain_experts_beside_the_rest) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&plain));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain }, { "blk.1.ffn_down_exps.3.weight", plain },
               { "blk.1.ffn_gate_up_exps.10.weight", plain }, { "blk.1.ffn_down_exps.10.weight", plain },
               { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    CHECK_EQ(m.moecfg.expert_w4, 1);
    const int64_t ne = m.moecfg.n_expert;

    auto param = [](const RecOp& o, const char* k) -> const RecParam* {
        for (const RecParam& p : o.p) if (p.key == k) return &p;
        return nullptr;
    };
    auto layer_of = [&](const RecOp& o) -> int {
        for (rad_weight wh : o.w)
            if (wh) {
                const std::string& n = b.weights[wh - 1].name;
                if (n.rfind("blk.", 0) == 0) return std::atoi(n.c_str() + 4);
            }
        return -1;
    };
    /* the sorts: only layer 1's names the two experts */
    int prot_sorts = 0, plain_sorts = 0;
    for (const RecOp& o : b.ops)
        if (o.op == "moe_scatter" || o.op == "router_topk_scatter") {
            const RecParam* p = param(o, "protect");
            if (p) { ++prot_sorts; CHECK_EQ(p->sval, std::string("3,10")); }
            else ++plain_sorts;
        }
    CHECK(prot_sorts >= 1);
    CHECK(plain_sorts > 0);
    /* the grouped GEMMs of layer 1 */
    int q_ops = 0, bf_ops = 0;
    for (const RecOp& o : b.ops) {
        if (layer_of(o) != 1) continue;
        if (o.op == "moe_gemm_q") {
            ++q_ops;
            CHECK_EQ(param(o, "n_expert")->ival, ne - 2);
            for (rad_weight wh : o.w) {
                const std::string& n = b.weights[wh - 1].name;
                CHECK(n.find("exps.3.") == std::string::npos && n.find("exps.10.") == std::string::npos);
            }
        }
        if (o.op == "moe_gemm") {
            ++bf_ops;
            CHECK_EQ(param(o, "n_expert")->ival, 2);
            CHECK_EQ(param(o, "dtype")->sval, std::string("bf16"));
            REQUIRE_EQ(o.w.size(), (size_t)2);
            const std::string& n0 = b.weights[o.w[0] - 1].name;
            const std::string& n1 = b.weights[o.w[1] - 1].name;
            CHECK(n0.find("exps.3.weight") != std::string::npos);
            CHECK(n1.find("exps.10.weight") != std::string::npos);
        }
    }
    CHECK_EQ(q_ops, 2);
    CHECK_EQ(bf_ops, 2);
    /* layer 1's four-bit experts fill slots 0..ne-3 with none missing; the plain ones are static
     * bf16 weights */
    std::vector<int> slot_seen((size_t)ne, 0);
    int plain_w = 0;
    for (const RecWeight& w : b.weights) {
        if (w.name.rfind("blk.1.ffn_", 0) != 0 || w.name.find("_exps.") == std::string::npos) continue;
        const bool prot_name = w.name.find("exps.3.") != std::string::npos ||
                               w.name.find("exps.10.") != std::string::npos;
        if (prot_name) {
            ++plain_w;
            CHECK_EQ(w.d.dtype, (uint32_t)RAD_BF16);
            CHECK_EQ(w.d.group.expert, -1);
            CHECK_EQ(w.d.access, RAD_ACCESS_PER_TOKEN);
        } else if (w.d.group.expert >= 0 && w.d.group.expert < ne) {
            slot_seen[(size_t)w.d.group.expert] = 1;
        }
    }
    CHECK_EQ(plain_w, 4);
    int filled = 0;
    for (int64_t s = 0; s < ne; ++s) filled += slot_seen[(size_t)s];
    CHECK_EQ(filled, (int)(ne - 2));
    CHECK_EQ(slot_seen[(size_t)(ne - 1)] + slot_seen[(size_t)(ne - 2)], 0);

    /* An odd plain count leaves the rest unsplittable into two equal parity tables: refused. */
    RadBuilder odd;
    odd.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain }, { "blk.1.ffn_down_exps.3.weight", plain },
                 { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    CHECK(qwen4exp_fp8::declare(&odd, &meta, &c) != RAD_OK);

    /* With two ranks each holding a slice of every expert, a plain expert is sliced evenly too
     * (gate_up 2 x 320 rows, down 320 columns) and its GEMMs are that wide. */
    {
        RadBuildCtx c2 = build_ctx(0, 2);
        c2.max_tok = 2048; c2.max_seqs = 8; c2.max_spec = 3; c2.max_ctx = 200000;
        RadBuilder sb;
        sb.encs = b.encs;
        REQUIRE_EQ(qwen4exp_fp8::declare(&sb, &meta, &c2), RAD_OK);
        int sliced = 0;
        for (const RecWeight& w : sb.weights) {
            if (w.name != "blk.1.ffn_gate_up_exps.3.weight" && w.name != "blk.1.ffn_down_exps.3.weight")
                continue;
            ++sliced;
            if (w.name.find("gate_up") != std::string::npos) {
                CHECK_EQ(w.d.shape[0], 640);
                CHECK_EQ(w.d.shard, RAD_SHARD_ROW);
                CHECK_EQ(w.d.n_row_parts, 2);
            } else {
                CHECK_EQ(w.d.shape[1], 320);
                CHECK_EQ(w.d.shard, RAD_SHARD_COL);
            }
        }
        CHECK_EQ(sliced, 2);
        int widths_ok = 0;
        for (const RecOp& o : sb.ops)
            if (o.op == "moe_gemm" && param(o, "n_expert")->ival == 2)
                widths_ok += (param(o, "N")->ival == 640 && param(o, "K")->ival == 2560) ||
                             (param(o, "N")->ival == 2560 && param(o, "K")->ival == 320);
        CHECK_EQ(widths_ok, 2);
    }

    /* The expert the format is probed from may be a plain one; the probe reads on to the first
     * four-bit expert, so the model is four-bit, not bf16. */
    RadBuilder first;
    first.encs = { { "blk.0.ffn_gate_up_exps.0.weight", plain }, { "blk.0.ffn_down_exps.0.weight", plain },
                   { "blk.0.ffn_gate_up_exps.1.weight", plain }, { "blk.0.ffn_down_exps.1.weight", plain },
                   { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&first, &meta, &c), RAD_OK);
    CHECK_EQ(qwen4exp_fp8::g_model[0].moecfg.expert_w4, 1);
    CHECK_EQ(qwen4exp_fp8::g_model[0].moecfg.expert_bf16, 0);
}

TEST(a_sizing_declare_leaves_the_step_state_alone_and_shrinks_only_rows) {
    (void)&qwen4exp_fp8::step;     /* included for its declare */
    RadModelMeta meta = qwen4exp_meta();
    RadArchProbe pb{};
    CHECK_OK(qwen4exp_fp8::probe(&meta, &pb));
    CHECK(pb.shape_probe_ok);
    CHECK(pb.control_vector_ok);

    /* A 4096 budget under a span (RadBuildCtx::max_mixed_tok 2048: a step wider than 2048 is one
     * sequence) reads in pass-wide pieces and scores a pass-tall plane in windows, and at four
     * ranks pipelines its prefill (step_pipelined), declaring an all-reduce at `overlap` 1 a
     * block; the sizing declare, at a max_tok no wider than a pass, has to declare the same ones.
     * At two ranks the gated residual's fused write takes the all-reduce, so nothing pipelines.
     * Without the span a 4096 budget declares none of it. */
    struct Shape { int64_t tok, mixed; };
    for (const Shape sh : { Shape{ 2048, 0 }, Shape{ 4096, 0 }, Shape{ 4096, 2048 } })
    for (int world : { 1, 2, 4 }) {
        for (int rank = 0; rank < world; ++rank) {
            const int64_t real_tok = sh.tok;
            const bool mixed = sh.mixed > 0;
            RadBuildCtx c = build_ctx(rank, world);
            c.max_tok  = real_tok;
            c.max_mixed_tok = sh.mixed;
            c.max_seqs = 8;
            c.max_spec = 3;
            c.max_ctx  = 200000;
            RadBuilder full;
            REQUIRE_EQ(qwen4exp_fp8::declare(&full, &meta, &c), RAD_OK);
            const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
            const int64_t max_tok = m.g.max_tok;
            const int64_t max_work = m.qsacfg.max_work;
            /* The score plane: a pass tall where a wider step is one sequence, max_tok rows
             * otherwise. */
            if (const RecBuf* sc = find_b(full, "qsa_score")) {
                const int64_t want = mixed ? 2048 : c.max_tok;
                CHECK_EQ(sc->d.shape[0], want);
                CHECK_EQ(m.qsacfg.plane_rows, want < c.max_tok ? want : 0);
            }
            /* And under a span every connection read is declared no wider than a pass: a wider
             * step reads in pieces, and the read's scratch follows its band. */
            {
                const int64_t want = (mixed && c.max_tok > 2048) ? 2048 : c.max_tok;
                int reads = 0, wide = 0;
                for (const auto& o : full.ops) {
                    if (o.op != "hc_read") continue;
                    ++reads;
                    const RecParam* pm = param(&o, "M");
                    if (!pm || pm->ihi != want) ++wide;
                }
                CHECK(reads > 0);
                CHECK_EQ(wide, 0);
            }
            /* The pipelined step's all-reduces at `overlap` 1, a block each, only where it runs,
             * with the buffer its second lane touches named concurrent. */
            {
                int lane = 0;
                for (const auto& o : full.ops)
                    if (o.op == "all_reduce" && pint(&o, "overlap") == 1) ++lane;
                const bool pipe = mixed && world > 2;
                CHECK_EQ(lane > 0, pipe);
                const RecBuf* x = find_b(full, "x");
                REQUIRE(x != nullptr);
                const rad_buf xh = (rad_buf)(x - full.bufs.data()) + 1;
                CHECK_EQ(std::count(full.concurrent.begin(), full.concurrent.end(), xh) > 0, pipe);
                CHECK_EQ(find_b(full, "pipe_cu") != nullptr, mixed);
            }
            const rad_op last_op = (rad_op)full.ops.size();
            CHECK_EQ(max_tok, real_tok);

            RadBuildCtx p = c;
            p.max_tok = 256;
            p.shape_probe = 1;
            RadBuilder small;
            REQUIRE_EQ(qwen4exp_fp8::declare(&small, &meta, &p), RAD_OK);

            /* The step's state is still the real declare's. */
            CHECK_EQ(m.g.max_tok, max_tok);
            CHECK_EQ(m.qsacfg.max_work, max_work);
            CHECK_EQ((rad_op)full.ops.size(), last_op);

            /* The same graph, and the same buffers with only their rows smaller. */
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            int fixed_differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                CHECK_EQ(small.ops[i].op, full.ops[i].op);
                /* Only the ranged parameter may differ; a fixed one following max_tok would run
                 * the kernel at the real size against a level's buffers. */
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    /* Capacities in libr4d's schema (RAD_PROLE_CAPACITY) may shrink: qsa_work
                     * launches as wide as its `page` operand regardless of `work`, and
                     * attn_paged_gate_quant by its block table's rows regardless of `max_seqs`. */
                    const bool cap = (full.ops[i].op == "qsa_work" && a.key == "work") ||
                                     (full.ops[i].op == "attn_paged_gate_quant" &&
                                      a.key == "max_seqs");
                    if (cap && q.ival <= a.ival) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    op %zu '%s' param '%s': %lld at 2048, %lld at 256\n",
                                i, full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival);
                        ++fixed_differ;
                    }
                }
            }
            CHECK_EQ(fixed_differ, 0);
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            int shrank = 0;
            for (size_t i = 0; i < full.bufs.size(); ++i) {
                const RadBufDecl& f = full.bufs[i].d;
                const RadBufDecl& q = small.bufs[i].d;
                CHECK_EQ(small.bufs[i].name, full.bufs[i].name);
                CHECK_EQ(q.rank, f.rank);
                CHECK_EQ(q.kind, f.kind);
                for (uint32_t d = 1; d < f.rank; ++d) {
                    if (q.shape[d] != f.shape[d])
                        fprintf(stderr, "    '%s' extent %u: %lld at 2048, %lld at 256\n",
                                full.bufs[i].name.c_str(), d, (long long)f.shape[d],
                                (long long)q.shape[d]);
                    CHECK_EQ(q.shape[d], f.shape[d]);
                }
                CHECK(q.shape[0] <= f.shape[0]);
                shrank += q.shape[0] < f.shape[0];
            }
            CHECK(shrank > 10);
        }
    }
}

/* Four ranks on Flash-Next: KV heads replicate, expert widths split, most fused collectives drop.
 *
 *   - KV heads cannot be cut four ways, so they are replicated in pairs: one KV head per rank,
 *     ranks {0,1} reading rows [0,256) of the checkpoint's k/v projection and {2,3} [256,512).
 *     This is a span, not a shard index, and the only place a rank names the container rows it
 *     wants, so it is asserted as agreeing spans whose union is the whole projection.
 *   - An expert is 640 columns, five 128-column scale blocks, which four ranks cannot cut evenly.
 *     So the width divides, unevenly by class, not the expert set: every rank routes all 512
 *     experts and computes [lo, hi) of each, the odd block rotating with expert number mod 4, five
 *     blocks per four experts on every rank. Sharding the experts would not work: protection is
 *     per-rank, so a rank could own one plain bf16 expert and an odd count of quantised ones, while
 *     the grouped GEMM takes two equal parity tables (the recipe protects experts in pairs).
 *   - Fused all-reduces are world-2 kernels (`ar_wire_is_exact` and `ar_fused_ok` return false
 *     otherwise), except the gated residual's write: `ar_hc_write` and `ar_gather_hc_write` also
 *     exist at four, on the exact wire only (`wire` 0) and over a decode or verify step's rows,
 *     max_seqs * (1 + max_spec). Every block still declares its standalone `all_reduce`, and every
 *     MoE block its `moe_gather`, for the steps the fused write declines. The world-2 contrast
 *     below catches a fused declare gated on the wrong condition.
 *
 * Expected numbers derive from this file's meta (n_head 24, n_head_kv 2, head_dim 256,
 * num_experts 512); the production container's unsharded k_proj height (512 rows) is asserted as a
 * relation, not a literal. */
TEST(flashnext_tp4_replicates_kv_heads_and_splits_expert_widths) {
    RadModelMeta meta = qwen4exp_meta();
    const int      world    = 4;
    const int64_t  n_head   = meta.n_head;        /* 24 query heads */
    const int64_t  n_head_kv = meta.n_head_kv;    /* 2 KV heads, fewer than the ranks */
    const int64_t  head_dim = meta.head_dim;      /* 256 */
    const int64_t  ne_all   = atoll(rad_meta_gets(&meta, "num_experts", "0"));
    REQUIRE(n_head % world == 0);
    REQUIRE_EQ(world % n_head_kv, (int64_t)0);   /* replication needs the multiple */
    REQUIRE(ne_all % world == 0);

    /* Full-attention layers of the eight-layer fixture: 3 and 7 (interval 4). */
    const int lfull = 3;
    const std::string kq = "blk." + std::to_string(lfull) + ".attn_k.weight";
    const std::string kv = "blk." + std::to_string(lfull) + ".attn_v.weight";

    RadBuilder b[world];
    RadBuildCtx  c[world];
    for (int r = 0; r < world; ++r) {
        c[r] = build_ctx(r, world);
        c[r].max_tok  = 2048;
        c[r].max_seqs = 8;
        c[r].max_spec = 3;
        c[r].max_ctx  = 200000;
        /* geom_from must accept a world larger than the KV head count. */
        REQUIRE_EQ(qwen4exp_fp8::declare(&b[r], &meta, &c[r]), RAD_OK);
    }

    for (int r = 0; r < world; ++r) {
        /* (1) A quarter of the query heads and one KV head, so gqa is 6, and the page is the QSA
         * compress block of 4 (qwen4exp_fp8.cpp's lay.attn.kv_block must agree with the row's
         * C_EQ("block_size", 4) or nothing resolves). */
        const RecOp* at = find_op(b[r], "attn_paged");
        REQUIRE(at != nullptr);
        CHECK_EQ(pint(at, "n_head"), n_head / world);
        CHECK_EQ(pint(at, "gqa"), n_head / world);   /* one KV head, so gqa == the rank's heads */
        CHECK_EQ(pint(at, "block_size"), 4);
        /* The full-attention KV group (group 0, kv_attn, declared first) holds this rank's one
         * head. */
        REQUIRE(b[r].kv_groups.size() > 0);
        CHECK_EQ(b[r].kv_groups[0].name, std::string("kv_attn"));
        CHECK_EQ(b[r].kv_groups[0].d.n_head_kv, 1);
        CHECK_EQ(b[r].kv_groups[0].d.head_dim, head_dim);

        /* (2) One KV head's rows (256 per rank) and the span locating them in the entry. Each pair
         * of ranks declares the same span, and the two spans cover the checkpoint's whole k_proj
         * (asserted as a relation below). */
        for (const std::string& wn : {kq, kv}) {
            const RecWeight* w = find_w(b[r], wn);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], head_dim);          /* one head */
            CHECK_EQ(w->d.shape[1], meta.n_embd);       /* K is not sharded */
            CHECK_EQ(w->d.shard, RAD_SHARD_ROW);
            CHECK(w->span_set);
            CHECK_EQ(w->span_hi - w->span_lo, w->d.shape[0]);
            CHECK_EQ(w->span_lo, (int64_t)(r / (world / n_head_kv)) * head_dim);
            CHECK_EQ(w->span_hi, w->span_lo + head_dim);
        }
        /* (3) Every rank routes every expert and computes a slice of each. The router and expert
         * set are replicated (n_expert 512, base 0); ranks divide each expert's width in whole
         * 128-column scale blocks. */
        const RecOp* sc = find_op(b[r], "moe_scatter");
        REQUIRE(sc != nullptr);
        CHECK_EQ(pint(sc, "n_expert"), ne_all);
        CHECK_EQ(pint(sc, "expert_base"), (int64_t)0);
        const RecOp* tk = find_op(b[r], "router_topk");
        REQUIRE(tk != nullptr);
        CHECK_EQ(pint(tk, "n_expert"), ne_all);         /* replicated routing */
        if (const RecOp* ts = find_op(b[r], "router_topk_scatter")) {
            CHECK_EQ(pint(ts, "n_expert"), ne_all);
            CHECK_EQ(pint(ts, "expert_base"), (int64_t)0);
        }
        /* Each rank's width per class, as a span on every expert's down projection (no expert is
         * protected here, so class = expert number mod 4). These literals are the TP4 MoE contract:
         * five 128-blocks over four ranks, the odd block rotating with the class. They are written
         * out rather than recomputed from the plugin's formula on purpose. */
        const int64_t blk = 128;                    /* RAD_FP8_BLOCK, the scale block width */
        const int64_t nblk = meta.n_ff / blk;        /* five at the fixture's 640 */
        const int64_t cls_lo[4][4] = { { 0, 512, 384, 256 }, { 256, 0, 512, 384 },
                                       { 384, 256, 0, 512 }, { 512, 384, 256, 0 } };
        const int64_t cls_hi[4][4] = { { 256, 640, 512, 384 }, { 384, 256, 640, 512 },
                                       { 512, 384, 256, 640 }, { 640, 512, 384, 256 } };
        REQUIRE_EQ(nblk, (int64_t)5);
        int spanned[4] = { 0, 0, 0, 0 };
        for (const RecWeight& w : b[r].weights) {
            if (w.name.rfind("blk.0.ffn_down_exps.", 0) != 0) continue;
            if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
            const int q = atoi(w.name.c_str() + 20) % 4;      /* the expert's class */
            CHECK(w.span_set);
            CHECK_EQ(w.span_lo, cls_lo[r][q]);
            CHECK_EQ(w.span_hi, cls_hi[r][q]);
            ++spanned[q];
        }
        for (int q = 0; q < 4; ++q)
            CHECK_EQ(spanned[q], ne_all / 4);            /* every expert of each class declared */
        /* The four ranks' slices of one class tile its width exactly: no block unserved and none
         * claimed twice (the all-reduce would double it). */
        for (int q = 0; q < 4; ++q) {
            int64_t covered = 0;
            for (int r2 = 0; r2 < world; ++r2)
                for (int64_t i = 0; i < nblk; ++i) {
                    const int64_t lo = cls_lo[r2][q], hi = cls_hi[r2][q];
                    if (lo <= i * blk && i * blk < hi) ++covered;
                }
            CHECK_EQ(covered, nblk);   /* per class, exactly once each */
        }
        /* Every rank takes the same width, five blocks per four experts; that is why there are
         * four classes rather than two. */
        {
            int64_t sum = 0;
            for (int q = 0; q < 4; ++q) sum += cls_hi[r][q] - cls_lo[r][q];
            CHECK_EQ(sum, nblk * blk);
        }
        /* The gate/up GEMM's four classes follow the same split, doubled since the operand stacks
         * gate and up; buffers take the widest class. No parity op remains. */
        const RecOp* gu = find_op(b[r], "moe_gemm_q_mod4");
        REQUIRE(gu != nullptr);
        CHECK_EQ(pint(gu, "N"), 2 * (cls_hi[r][0] - cls_lo[r][0]));
        CHECK_EQ(pint(gu, "N_1"), 2 * (cls_hi[r][1] - cls_lo[r][1]));
        CHECK_EQ(pint(gu, "N_2"), 2 * (cls_hi[r][2] - cls_lo[r][2]));
        CHECK_EQ(pint(gu, "N_3"), 2 * (cls_hi[r][3] - cls_lo[r][3]));
        CHECK_EQ(pint(gu, "n_expert"), ne_all);
        CHECK_EQ(gu->w.size(), (size_t)(2 * ne_all));   /* codes and scales, each expert once */
        CHECK_EQ(count_op(b[r], "moe_gemm_q"), 0);

        /* (4) A standalone collective per reduction, plus the exact four-rank ar_hc_write and
         * ar_gather_hc_write (the gather form on each MoE connection), but no fused norm. Both
         * fused forms state the exact wire and range over a decode or verify step's rows,
         * 8 * (1 + 3), the architecture's own bound. */
        CHECK(count_op(b[r], "all_reduce") > 0);
        CHECK_EQ(pint(find_op(b[r], "all_reduce"), "world_size"), world);
        CHECK(count_op(b[r], "ar_hc_write") > 0);
        CHECK(count_op(b[r], "moe_gather") > 0);
        CHECK(count_op(b[r], "ar_gather_hc_write") > 0);
        CHECK(count_op(b[r], "ar_gather_hc_write") <= count_op(b[r], "moe_gather"));
        for (const RecOp& o : b[r].ops) {
            if (o.op != "ar_hc_write" && o.op != "ar_gather_hc_write") continue;
            CHECK_EQ(pint(&o, "world_size"), world);
            CHECK_EQ(pint(&o, "wire"), 0);                 /* exact only, stated */
            const RecParam* m = param(&o, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->kind, (int)RAD_P_RANGE);
            CHECK_EQ(m->ihi, c[r].max_seqs * (1 + c[r].max_spec));
        }
        CHECK_EQ(count_op(b[r], "ar_rmsnorm_quant_fp8"), 0);
    }

    /* The pairs agree and their union is the whole projection, n_head_kv * head_dim rows (a
     * container fact, so asserted as a relation). */
    for (const std::string& wn : {kq, kv}) {
        CHECK_EQ(find_w(b[0], wn)->span_hi, find_w(b[1], wn)->span_hi);
        CHECK_EQ(find_w(b[2], wn)->span_lo, find_w(b[3], wn)->span_lo);
        CHECK_EQ(find_w(b[0], wn)->span_lo, find_w(b[2], wn)->span_hi - n_head_kv * head_dim);
        const int64_t lo = find_w(b[0], wn)->span_lo;
        const int64_t hi = find_w(b[2], wn)->span_hi;
        CHECK_EQ(hi - lo, n_head_kv * head_dim);
        CHECK_EQ(hi - lo, world * find_w(b[0], wn)->d.shape[0] / (world / n_head_kv));
    }

    /* (5) All four ranks declare the same ops; shards differ only in parameters and spans, since
     * the plan must not vary by rank. */
    for (int r = 1; r < world; ++r) {
        if (op_names(b[r]) != op_names(b[0])) {
            fprintf(stderr, "    rank %d differs from rank 0 in its op list (%zu vs %zu)\n", r,
                    b[r].ops.size(), b[0].ops.size());
            for (size_t i = 0; i < b[0].ops.size() && i < b[r].ops.size(); ++i)
                if (b[0].ops[i].op != b[r].ops[i].op) {
                    fprintf(stderr, "    first at %zu: '%s' vs '%s'\n", i,
                            b[0].ops[i].op.c_str(), b[r].ops[i].op.c_str());
                    break;
                }
        }
        CHECK_EQ(join(op_names(b[r])), join(op_names(b[0])));
    }

    /* The shared expert is split by whole blocks, unevenly at four: 640 is five 128-blocks, and
     * rank 2 takes the extra one, after the plain experts' extra tiles on ranks 0 and 1
     * (qwen4exp_fp8.cpp's shared_place). Ranks 2, 3, 0, 1 cut [0,256) [256,384) [384,512)
     * [512,640). At two ranks, rank 0 takes three of five. */
    const int64_t sh_lo[] = { 384, 512, 0,   256 };
    const int64_t sh_hi[] = { 512, 640, 256, 384 };
    for (int r = 0; r < world; ++r) {
        const RecWeight* sg = find_w(b[r], "blk.0.ffn_gate_up_shexp.weight");
        REQUIRE(sg != nullptr);
        CHECK(sg->span_set);
        CHECK_EQ(sg->span_lo, sh_lo[r]);
        CHECK_EQ(sg->span_hi, sh_hi[r]);
        CHECK_EQ(sg->d.shape[0], 2 * (sh_hi[r] - sh_lo[r]));   /* gate's and up's rows */
    }
    /* Every block of the shared expert has exactly one owner. */
    for (int64_t i = 0; i < 640; i += 128) {
        int owners = 0;
        for (int r = 0; r < world; ++r) {
            const RecWeight* sg = find_w(b[r], "blk.0.ffn_gate_up_shexp.weight");
            owners += sg->span_lo <= i && i < sg->span_hi;
        }
        CHECK_EQ(owners, 1);
    }

    /* At two ranks the KV heads split evenly, so the even shard selects the rows and no span is
     * declared. The fused forms resolve at world 2 over every step. */
    {
        RadBuilder h[2];
        for (int r = 0; r < 2; ++r) {
            RadBuildCtx c2 = build_ctx(r, 2);
            c2.max_tok  = 2048;
            c2.max_seqs = 8;
            c2.max_spec = 3;
            c2.max_ctx  = 200000;
            REQUIRE_EQ(qwen4exp_fp8::declare(&h[r], &meta, &c2), RAD_OK);
            const RecWeight* w = find_w(h[r], kq);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], n_head_kv / 2 * head_dim);   /* one of two heads */
            CHECK(!w->span_set);
            CHECK_EQ(h[r].kv_groups[0].d.n_head_kv, n_head_kv / 2);
            CHECK_EQ(pint(find_op(h[r], "attn_paged"), "gqa"), (n_head / 2) / (n_head_kv / 2));
            CHECK_EQ(pint(find_op(h[r], "moe_scatter"), "n_expert"), ne_all);
            CHECK_EQ(pint(find_op(h[r], "moe_scatter"), "expert_base"), 0);
            CHECK(count_op(h[r], "ar_hc_write") > 0);
            CHECK(count_op(h[r], "ar_gather_hc_write") > 0);
            /* Two ranks keep the exact write over every step, prefill chunks included. */
            for (const RecOp& o : h[r].ops) {
                if (o.op != "ar_hc_write") continue;
                CHECK_EQ(pint(&o, "wire"), 0);
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK_EQ(m->ihi, c2.max_tok);
            }
            /* No ar_rmsnorm_quant_fp8 at either world: qwen4exp hands every block's norm
             * `kArNone`, since the reduction rides in the gated residual's write instead
             * (qwen4exp_fp8.cpp's block declares). */
            CHECK_EQ(count_op(h[r], "ar_rmsnorm_quant_fp8"), 0);
        }
    }

    /* Which steps the four-rank fused write takes: decode and verify steps, up to
     * max_seqs * (1 + max_spec) rows and never more than max_tok, whose message goes out exact (on
     * the lossy wire: under its floor or off its 128-element tile). The bound is the architecture's,
     * not a message size: 64 sequences at depth 3 are 256 rows (1.25 MiB) and still take it, while
     * a prefill chunk never does. The preceding block asks the same predicate. At two ranks it
     * equals ar_wire_is_exact, which itself stays false at four. */
    {
        rad::arch::Geom g4{};
        g4.world = 4; g4.n_embd = 2560; g4.wire_exact = 1; g4.wire_min_bytes = 128 * 1024;
        g4.max_tok = 2048; g4.max_seqs = 8; g4.max_spec = 3;
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)32);
        CHECK_EQ(rad::arch::ar_exact_write_rows(g4), (int64_t)32);
        CHECK(!rad::arch::ar_exact_write_takes(g4, 0));
        CHECK(rad::arch::ar_exact_write_takes(g4, 1));
        CHECK(rad::arch::ar_exact_write_takes(g4, 8));      /* a decode step */
        CHECK(rad::arch::ar_exact_write_takes(g4, 32));     /* every sequence verifying 1 + 3 */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 33));    /* more: a prefill chunk */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 2048));
        CHECK(!rad::arch::ar_wire_is_exact(g4, 8));
        g4.wire_exact = 0;
        CHECK(rad::arch::ar_exact_write_takes(g4, 25));     /* 125 KiB, under the floor */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 26));    /* 130 KiB, goes on the ti8 wire */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 32));
        g4.wire_exact = 1;
        g4.max_spec = 0;                                    /* one row per sequence */
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)8);
        CHECK(rad::arch::ar_exact_write_takes(g4, 8));
        CHECK(!rad::arch::ar_exact_write_takes(g4, 9));
        g4.max_seqs = 64; g4.max_spec = 3;
        CHECK(rad::arch::ar_exact_write_takes(g4, 256));    /* 1.25 MiB */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 257));
        g4.max_tok = 128;                                   /* caps a step's rows */
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)128);
        CHECK(rad::arch::ar_exact_write_takes(g4, 128));
        CHECK(!rad::arch::ar_exact_write_takes(g4, 129));
        rad::arch::Geom g8 = g4;
        g8.world = 8;
        CHECK_EQ(rad::arch::ar_exact_write_rows(g8), (int64_t)0);
        CHECK(!rad::arch::ar_exact_write_takes(g8, 8));
        rad::arch::Geom g2 = g4;
        g2.world = 2; g2.max_tok = 4096; g2.max_seqs = 8;
        CHECK_EQ(rad::arch::ar_exact_write_rows(g2), g2.max_tok);
        for (int exact : { 0, 1 }) {
            g2.wire_exact = exact;
            for (int64_t T : { 1, 8, 25, 26, 33, 205, 4096 })
                CHECK(rad::arch::ar_exact_write_takes(g2, T) == rad::arch::ar_wire_is_exact(g2, T));
        }
    }
    /* The four-rank fused gather takes the fused write's steps, within one MoE pass. Only the
     * exact bound exists at four (the rotated form is never declared, so `rows6` is 0), and a step
     * the write declines keeps its plain gather. At two ranks ar_wire_is_exact picks the bound. */
    {
        rad::arch::Geom g4{};
        g4.world = 4; g4.n_embd = 2560; g4.wire_exact = 1; g4.wire_min_bytes = 128 * 1024;
        g4.max_tok = 2048; g4.max_seqs = 8; g4.max_spec = 3;
        CHECK(rad::arch::gather_taken(g4, 8, 32, 0));        /* a decode step */
        CHECK(rad::arch::gather_taken(g4, 32, 32, 0));       /* a verify step, one pass */
        CHECK(!rad::arch::gather_taken(g4, 33, 2048, 0));    /* the write declines it */
        CHECK(!rad::arch::gather_taken(g4, 8, 0, 0));        /* not declared */
        CHECK(!rad::arch::gather_taken(g4, 20, 16, 0));      /* two passes */
        g4.wire_exact = 0;
        CHECK(rad::arch::gather_taken(g4, 25, 32, 0));       /* under the floor */
        CHECK(!rad::arch::gather_taken(g4, 26, 32, 0));      /* goes on the ti8 wire */
        rad::arch::Geom g2 = g4;
        g2.world = 2;
        for (int64_t T : { 1, 8, 25, 26, 205, 4096 })
            for (int64_t r6 : { 0, 2048 }) {
                const int64_t r = rad::arch::ar_wire_is_exact(g2, T) ? 2048 : r6;
                CHECK(rad::arch::gather_taken(g2, T, 2048, r6) == (r > 0 && T <= r));
            }
    }
    /* At four, the lossy wire declares no rotated write or gather: only the exact forms, stating
     * `wire` 0, for decode and verify steps under the floor. */
    {
        RadBuilder b4;
        RadBuildCtx c4 = build_ctx(0, world);
        c4.max_tok = 2048; c4.max_seqs = 8; c4.max_spec = 3; c4.max_ctx = 200000;
        c4.tp_wire_lossy = 1;
        c4.tp_wire_min_bytes = 128 * 1024;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b4, &meta, &c4), RAD_OK);
        CHECK(count_op(b4, "ar_hc_write") > 0);
        CHECK(count_op(b4, "ar_gather_hc_write") > 0);
        for (const RecOp& o : b4.ops) {
            if (o.op != "ar_hc_write" && o.op != "ar_gather_hc_write") continue;
            CHECK_EQ(pint(&o, "wire"), 0);
            const RecParam* m = param(&o, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->ihi, c4.max_seqs * (1 + c4.max_spec));
        }
    }

    /* A world that is not a multiple of the KV head count still declares: qwen4exp opts in via
     * geom_from's `attn_world_ok`, so attention runs on the largest multiple below the world and the
     * remaining ranks compute none (see the tp3 test). Nothing is rounded into a half head. A
     * plugin that does not opt in is refused at this world by geom_from. */
    {
        RadBuilder b3;
        RadBuildCtx c3 = build_ctx(0, 3);
        c3.max_tok = 2048; c3.max_seqs = 8; c3.max_spec = 3; c3.max_ctx = 200000;
        CHECK_EQ(qwen4exp_fp8::declare(&b3, &meta, &c3), RAD_OK);
    }
    {
        rad::arch::Geom g3{};
        RadBuildCtx c3 = build_ctx(0, 3);
        c3.max_tok = 2048; c3.max_seqs = 8; c3.max_spec = 3; c3.max_ctx = 200000;
        CHECK_EQ(rad::arch::geom_from(g3, &meta, &c3, "bf16", RAD_BF16, /*arch_splits_ff=*/true), RAD_E_INVAL);
        CHECK_EQ(rad::arch::geom_from(g3, &meta, &c3, "bf16", RAD_BF16, /*arch_splits_ff=*/true,
                           /*attn_world_ok=*/true), RAD_OK);
        CHECK(g3.attn_zero == false);
        CHECK_EQ(g3.attn_world, 2);
    }

    /* ---- A layer that keeps experts plain stays at four classes. Two plain experts leave 510
     * quantised ones, four leave 508; either way `moe_gemm_q_mod4` serves the layer, the class being
     * the expert's place in the sort mod 4 (plain ones sorted last). At 510, classes 0 and 1 hold
     * 128 experts and 2 and 3 hold 127 (the first tables take the remainder). The plain experts'
     * extra tiles go to ranks 0 and 1, so every rank holds the same routed width; the shared
     * expert's extra block is rank 2's. */
    {
        const RadEncoding w4p = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                     rad_enc_copy_str(e.transform, "fwht128"); return e; }();
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        /* The four-class slices, [lo, hi) by rank and place mod 4. */
        const int64_t c4_lo[4][4] = { { 0, 512, 384, 256 }, { 256, 0, 512, 384 },
                                      { 384, 256, 0, 512 }, { 512, 384, 256, 0 } };
        const int64_t c4_hi[4][4] = { { 256, 640, 512, 384 }, { 384, 256, 640, 512 },
                                      { 512, 384, 256, 640 }, { 640, 512, 384, 256 } };
        for (int nprot : { 2, 4 }) {
            const std::vector<int> pe = nprot == 2 ? std::vector<int>{ 3, 10 }
                                                   : std::vector<int>{ 3, 10, 21, 40 };
            for (int r = 0; r < world; ++r) {
                RadBuilder bp;
                for (int e : pe) {
                    bp.encs.push_back({ "blk.1.ffn_gate_up_exps." + std::to_string(e) + ".weight",
                                        plain });
                    bp.encs.push_back({ "blk.1.ffn_down_exps." + std::to_string(e) + ".weight",
                                        plain });
                }
                bp.encs.push_back({ "ffn_gate_up_exps", w4p });
                bp.encs.push_back({ "ffn_down_exps", w4p });
                RadBuildCtx cp = build_ctx(r, world);
                cp.max_tok = 2048; cp.max_seqs = 8; cp.max_spec = 3; cp.max_ctx = 200000;
                REQUIRE_EQ(qwen4exp_fp8::declare(&bp, &meta, &cp), RAD_OK);
                const int64_t n_reg = ne_all - nprot;
                /* layer 1's quantised GEMMs */
                int l1_ops = 0, other_ops = 0;
                for (const RecOp& o : bp.ops) {
                    if (o.op != "moe_gemm_q" && o.op != "moe_gemm_q_mod4") continue;
                    bool l1 = false;
                    for (rad_weight wh : o.w)
                        l1 = l1 || bp.weights[wh - 1].name.rfind("blk.1.", 0) == 0;
                    if (!l1) { CHECK_EQ(o.op, std::string("moe_gemm_q_mod4")); ++other_ops; continue; }
                    ++l1_ops;
                    CHECK_EQ(o.op, std::string("moe_gemm_q_mod4"));
                    CHECK_EQ(pint(&o, "n_expert"), n_reg);
                    CHECK_EQ(o.w.size(), (size_t)(2 * n_reg));
                    /* four runs, codes then scales, class by class; class q holds places q,
                     * q + 4, ... (128, 128, 127, 127 at 510) */
                    size_t at = 0;
                    for (int q = 0; q < 4; ++q) {
                        const int64_t nq = (n_reg + 3 - q) / 4;
                        for (int half = 0; half < 2; ++half)
                            for (int64_t j = 0; j < nq && at < o.w.size(); ++j, ++at) {
                                const std::string& wn = bp.weights[o.w[at] - 1].name;
                                const int e = atoi(wn.c_str() + wn.find("_exps.") + 6);
                                int place = e;
                                for (int x : pe) place -= x < e ? 1 : 0;
                                CHECK_EQ(place, (int)(q + 4 * j));
                                CHECK((wn.find(".scale") != std::string::npos) == (half == 1));
                            }
                    }
                    CHECK_EQ(at, o.w.size());
                }
                CHECK_EQ(l1_ops, 2);
                CHECK(other_ops > 0);
                /* each quantised expert of layer 1 gets its class's slice */
                int seen = 0;
                for (const RecWeight& w : bp.weights) {
                    if (w.name.rfind("blk.1.ffn_down_exps.", 0) != 0) continue;
                    if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                    if (w.d.dtype == RAD_BF16) continue;          /* plain ones checked below */
                    const int e = atoi(w.name.c_str() + 20);
                    int place = e;
                    for (int x : pe) place -= x < e ? 1 : 0;
                    CHECK(w.span_set);
                    CHECK_EQ(w.span_lo, c4_lo[r][place % 4]);
                    CHECK_EQ(w.span_hi, c4_hi[r][place % 4]);
                    ++seen;
                }
                CHECK_EQ(seen, (int)n_reg);
                /* the plain experts' tiles by rank: 3, 3, 2, 2 */
                const int64_t p4_lo[4] = { 0, 192, 384, 512 },   p4_hi[4] = { 192, 384, 512, 640 };
                int pseen = 0;
                for (const RecWeight& w : bp.weights) {
                    if (w.name.rfind("blk.1.ffn_", 0) != 0 || w.d.dtype != RAD_BF16) continue;
                    if (w.name.find("_exps.") == std::string::npos) continue;
                    if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                    CHECK(w.span_set);
                    CHECK_EQ(w.span_lo, p4_lo[r]);
                    CHECK_EQ(w.span_hi, p4_hi[r]);
                    ++pseen;
                }
                CHECK_EQ(pseen, 2 * nprot);
            }
        }
    }

    /* ---- The plain bf16 experts at four ranks.
     *
     * A second grouped GEMM serves them over their own slice of the same width: whole 64-wide
     * output tiles, one range per rank, uniform across a layer's protected experts (their GEMM
     * takes one table, so every entry has one geometry; they do not alternate by parity). The
     * width must be the expert's own, not derived from this rank's parity slices, which only match
     * at two ranks.
     *
     * Ten tiles over four ranks is 3, 3, 2, 2: uneven, whole tiles, each owned once. Ranks 0 and 1
     * compute 192 columns of every protected expert, ranks 2 and 3 compute 128; every rank holds
     * the same routed width, and rank 2 the shared expert's extra block (shared_place). Two
     * protected experts, as the production recipe pairs them. */
    {
        const RadEncoding w4p = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                     rad_enc_copy_str(e.transform, "fwht128"); return e; }();
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        /* [lo, hi) of the plain experts' columns per rank, written out: 3, 3, 2, 2 tiles. */
        const int64_t p_lo[4] = { 0,   192, 384, 512 };
        const int64_t p_hi[4] = { 192, 384, 512, 640 };
        for (int r = 0; r < world; ++r) {
            RadBuilder bp;
            bp.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain },
                        { "blk.1.ffn_down_exps.3.weight", plain },
                        { "blk.1.ffn_gate_up_exps.10.weight", plain },
                        { "blk.1.ffn_down_exps.10.weight", plain },
                        { "ffn_gate_up_exps", w4p }, { "ffn_down_exps", w4p } };
            RadBuildCtx cp = build_ctx(r, world);
            cp.max_tok = 2048; cp.max_seqs = 8; cp.max_spec = 3; cp.max_ctx = 200000;
            REQUIRE_EQ(qwen4exp_fp8::declare(&bp, &meta, &cp), RAD_OK);
            const int64_t wp = p_hi[r] - p_lo[r];
            int seen = 0;
            for (const RecWeight& w : bp.weights) {
                if (w.name.rfind("blk.1.ffn_gate_up_exps.", 0) != 0 &&
                    w.name.rfind("blk.1.ffn_down_exps.", 0) != 0) continue;
                if (w.d.dtype != RAD_BF16) continue;      /* plain ones only */
                if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                CHECK(w.span_set);
                CHECK_EQ(w.span_lo, p_lo[r]);
                CHECK_EQ(w.span_hi, p_hi[r]);
                ++seen;
            }
            /* Both projections of both protected experts. */
            CHECK_EQ(seen, 4);
            /* The gate/up GEMM's N is twice this rank's share (stacked pair) and the down GEMM's
             * K is the share. */
            int gu_ok = 0, dn_ok = 0;
            for (const RecOp& o : bp.ops) {
                if (o.op != "moe_gemm") continue;
                if (pint(&o, "N") == 2 * wp && pint(&o, "n_expert") == 2) ++gu_ok;
                if (pint(&o, "K") == wp && pint(&o, "n_expert") == 2) ++dn_ok;
            }
            CHECK_EQ(gu_ok, 1);
            CHECK_EQ(dn_ok, 1);
        }
    }
}

/* ==================================================================== media */
namespace {

/* The fixture above plus what a Qwen3.8 container adds for pictures: the text config's
 * interleaved M-RoPE and a vision tower cut to two blocks at the real widths, enough to see a
 * block repeat. */
const char* k_vl_keys[] = {
    "qwen35.full_attention_interval",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.conv_kernel",
    "rope_parameters.mrope_section",
    "rope_parameters.mrope_interleaved",
    "vision_config.depth",
    "vision_config.hidden_size",
    "vision_config.num_heads",
    "vision_config.intermediate_size",
    "vision_config.patch_size",
    "vision_config.temporal_patch_size",
    "vision_config.in_channels",
    "vision_config.spatial_merge_size",
    "vision_config.out_hidden_size",
    "vision_config.num_position_embeddings",
    "vision_config.deepstack_visual_indexes",
    "vision_config.hidden_act",
};
const char* k_vl_vals[] = { "4", "64", "128", "16", "48", "6144", "4",
                            "11 11 10", "1",
                            "2", "1152", "16", "4304", "16", "2", "3", "2", "5120", "2304", "",
                            "gelu_pytorch_tanh" };

RadModelMeta qwen35_vl_meta() {
    RadModelMeta m = qwen35_meta();
    m.n_kv   = (int)(sizeof(k_vl_keys) / sizeof(k_vl_keys[0]));
    m.kv_key = k_vl_keys;
    m.kv_val = k_vl_vals;
    return m;
}

RadBuildCtx vl_ctx(int64_t patches) {
    RadBuildCtx c = build_ctx();
    c.max_enc_patches = patches;
    return c;
}

const RecOp* issued_op(const RadBuilder& b, const RadCtx& c, const std::string& name, int nth = 0) {
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == name && nth-- == 0) return o;
    }
    return nullptr;
}

}  /* namespace */

/* Three ranks over two KV heads: the attention world is 2 (rad_arch.h), so rank 2 computes no
 * attention (a `fill` of its output wire and a zero KV group). The delta net's 16 K heads go 5, 5,
 * 6 with the V heads at the same multiple, the extra head on the rank without attention, declared
 * as container spans with no padding. */
TEST(flashnext_tp3_zeroes_the_third_ranks_attention_and_splits_gdn_heads_unevenly) {
    RadModelMeta meta = qwen4exp_meta();
    const int      world     = 3;
    const int64_t  n_head    = meta.n_head;       /* 24 */
    const int64_t  n_head_kv = meta.n_head_kv;    /* 2: does not divide 3 */
    const int64_t  head_dim  = meta.head_dim;     /* 256 */
    const int64_t  nk = atoll(rad_meta_gets(&meta, "linear_num_key_heads", "0"));     /* 16 */
    const int64_t  nv = atoll(rad_meta_gets(&meta, "linear_num_value_heads", "0"));   /* 48 */
    const int64_t  hk = atoll(rad_meta_gets(&meta, "linear_key_head_dim", "0"));      /* 128 */
    const int64_t  hv = atoll(rad_meta_gets(&meta, "linear_value_head_dim", "0"));    /* 128 */
    REQUIRE(n_head_kv < world && world % n_head_kv != 0);
    REQUIRE(nk % world != 0);
    const int attn_world = (int)(n_head_kv * (world / n_head_kv));   /* 2 */

    const int lfull = 3;
    const std::string kq = "blk." + std::to_string(lfull) + ".attn_k.weight";
    const std::string kv = "blk." + std::to_string(lfull) + ".attn_v.weight";
    const std::string qg = "blk." + std::to_string(lfull) + ".attn_qg.weight";
    const std::string oo = "blk." + std::to_string(lfull) + ".attn_output.weight";

    RadBuilder b[world];
    RadBuildCtx  c[world];
    for (int r = 0; r < world; ++r) {
        c[r] = build_ctx(r, world);
        c[r].max_tok  = 2048;
        c[r].max_seqs = 8;
        c[r].max_spec = 3;
        c[r].max_ctx  = 200000;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b[r], &meta, &c[r]), RAD_OK);
    }

    /* Every rank declares the same KV groups in the same order, since the scheduler indexes groups
     * by position and one block table serves all ranks. Ops may differ per rank (issue walks each
     * rank's own program), but the attention ranks declare the same ops. */
    for (int r = 1; r < world; ++r) {
        REQUIRE_EQ(b[r].kv_groups.size(), b[0].kv_groups.size());
        for (size_t g = 0; g < b[0].kv_groups.size(); ++g) {
            CHECK_EQ(b[r].kv_groups[g].name, b[0].kv_groups[g].name);
            CHECK_EQ(b[r].kv_groups[g].d.kind, b[0].kv_groups[g].d.kind);
        }
        if (r < attn_world) {
            REQUIRE_EQ(b[r].ops.size(), b[0].ops.size());
            for (size_t i = 0; i < b[0].ops.size(); ++i) CHECK_EQ(b[r].ops[i].op, b[0].ops[i].op);
        }
    }

    std::vector<std::pair<int64_t, int64_t>> k_span((size_t)world);   /* each rank's K heads [lo, hi) */
    for (int r = 0; r < world; ++r) {
        const bool zero = r >= attn_world;
        CHECK_EQ(b[r].kv_groups[0].d.n_head_kv, 1);
        /* The attention-zero rank has no attention op or projection weight, fills its output
         * wire, and carves no cache for the attention group. */
        if (zero) {
            CHECK(find_op(b[r], "attn_paged") == nullptr);
            CHECK(find_op(b[r], "kv_store") == nullptr);
            CHECK(find_op(b[r], "fill") != nullptr);
            CHECK(find_w(b[r], kq) == nullptr);
            CHECK(find_w(b[r], qg) == nullptr);
            CHECK(find_w(b[r], oo) == nullptr);
            CHECK(b[r].kv_groups[0].zero);
            /* It still joins the all-reduce and runs the indexer. */
            CHECK(find_op(b[r], "qsa_select") != nullptr);
        } else {
            CHECK(!b[r].kv_groups[0].zero);
            CHECK(find_op(b[r], "fill") == nullptr);
        }
        /* (1) Each attention rank has the two-rank geometry: 12 heads, one KV head, gqa 12. */
        const RecOp* at = find_op(b[r], "attn_paged");
        if (!zero) {
            REQUIRE(at != nullptr);
            CHECK_EQ(pint(at, "n_head"), n_head / attn_world);
            CHECK_EQ(pint(at, "gqa"), n_head / attn_world);
        }
        /* (2) The projections: the span of this rank's heads in the container. */
        for (const std::string& wn : {kq, kv}) {
            if (zero) break;
            const RecWeight* w = find_w(b[r], wn);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], head_dim);
            CHECK(w->span_set);
            const int ar = zero ? attn_world - 1 : r;
            CHECK_EQ(w->span_lo, (int64_t)ar * head_dim);
            CHECK_EQ(w->span_hi, w->span_lo + head_dim);
        }
        if (!zero) {
            const RecWeight* w = find_w(b[r], qg);
            REQUIRE(w != nullptr);
            const int ar = zero ? attn_world - 1 : r;
            const int64_t qg_rows = 2 * (n_head / attn_world) * head_dim;
            CHECK_EQ(w->d.shape[0], qg_rows);
            CHECK(w->span_set);
            CHECK_EQ(w->span_lo, (int64_t)ar * qg_rows);
            const RecWeight* o = find_w(b[r], oo);
            REQUIRE(o != nullptr);
            CHECK_EQ(o->d.shard, RAD_SHARD_COL);
            CHECK(o->span_set);
            CHECK_EQ(o->span_lo, (int64_t)ar * (n_head / attn_world) * head_dim);
        }
        /* (3) The delta net: 5, 5, 6 K heads and 15, 15, 18 V heads on every rank, as per-part
         * spans of [q|k|v|z] and of the conv channels, and a single span of A_log. */
        const int64_t base = nk / world, rem = nk % world;
        /* The extra head goes to the rank without attention first (rad_arch.h remainder_place):
         * rank 2 takes 6, ranks 0 and 1 take 5. */
        const int64_t pl   = r >= attn_world ? r - attn_world : (world - attn_world) + r % attn_world;
        const int64_t my_k = base + (pl < rem ? 1 : 0);
        const int64_t k0   = pl * base + (pl < rem ? pl : rem);
        const int64_t my_v = (nv / nk) * my_k, v0 = (nv / nk) * k0;
        const RecWeight* in = find_w(b[r], "blk.0.ssm_inz.weight");
        REQUIRE(in != nullptr);
        REQUIRE_EQ(in->d.n_row_parts, 4);
        CHECK_EQ(in->d.row_parts[0], my_k * hk);
        CHECK_EQ(in->d.row_parts[2], my_v * hv);
        REQUIRE_EQ(in->n_part_spans, 4);
        CHECK_EQ(in->part_lo[0], k0 * hk);
        CHECK_EQ(in->part_lo[1], k0 * hk);
        CHECK_EQ(in->part_lo[2], v0 * hv);
        CHECK_EQ(in->part_lo[3], v0 * hv);
        CHECK_EQ(in->part_total[0], nk * hk);
        CHECK_EQ(in->part_total[2], nv * hv);
        k_span[(size_t)r] = { in->part_lo[0] / hk, in->part_hi[0] / hk };
        const RecWeight* cv = find_w(b[r], "blk.0.ssm_conv1d.weight");
        REQUIRE(cv != nullptr);
        REQUIRE_EQ(cv->n_part_spans, 3);
        CHECK_EQ(cv->part_lo[2], v0 * hv);
        CHECK_EQ(cv->part_hi[2], (v0 + my_v) * hv);
        const RecWeight* al = find_w(b[r], "blk.0.ssm_a_log");
        REQUIRE(al != nullptr);
        CHECK_EQ(al->d.shape[0], my_v);
        CHECK(al->span_set);
        CHECK_EQ(al->span_lo, v0);
        CHECK_EQ(al->span_hi, v0 + my_v);
        const RecWeight* ow = find_w(b[r], "blk.0.ssm_out.weight");
        REQUIRE(ow != nullptr);
        CHECK_EQ(ow->d.shard, RAD_SHARD_COL);
        CHECK(ow->span_set);
        CHECK_EQ(ow->span_lo, v0 * hv);
        CHECK_EQ(ow->span_hi, (v0 + my_v) * hv);
        /* The GDN state group holds this rank's V heads. */
        bool seen = false;
        for (const auto& g : b[r].kv_groups)
            if (g.name == "kv_gdn_state") { seen = true; CHECK_EQ(g.d.n_head_kv, (int)my_v); }
        CHECK(seen);
    }
    /* The three ranks' K head spans cover the container's 16 heads exactly once. */
    {
        std::vector<int> owner((size_t)nk, -1);
        for (int r = 0; r < world; ++r)
            for (int64_t h = k_span[(size_t)r].first; h < k_span[(size_t)r].second; ++h) {
                REQUIRE(h >= 0 && h < nk);
                CHECK_EQ(owner[(size_t)h], -1);
                owner[(size_t)h] = r;
            }
        for (int64_t h = 0; h < nk; ++h) CHECK(owner[(size_t)h] >= 0);
    }
}

TEST(a_vision_container_declares_the_tower_when_the_engine_asks_for_it) {
    /* With no encoder budget the tower is not declared, whatever the container carries. */
    {
        RadBuilder b;
        RadModelMeta meta = qwen35_vl_meta();
        RadBuildCtx  ctx  = vl_ctx(0);
        CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));
        CHECK(b.encoder_name.empty());
        CHECK(find_w(b, "v.patch.weight") == nullptr);
        CHECK_EQ(count_op(b, "scatter_rows"), 0);
    }
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO));
    CHECK_EQ(b.encoder.patch_dim, 3 * 2 * 16 * 16);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 1024);
    CHECK_EQ(b.encoder.n_embd, 5120);
    const RecBuf* out = find_b(b, "v.out");
    REQUIRE(out != nullptr);
    CHECK_EQ(out->d.shape[0], 256);
    CHECK_EQ(out->d.shape[1], 5120);

    /* 3 stem + 12 per block + 6 merger, each mapped from the checkpoint's model.visual.*. */
    int nv = 0;
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            ++nv;
            CHECK(find_map(b, w.name) != nullptr);
            CHECK_EQ(w.d.shard, (int)RAD_SHARD_NONE);
        }
    CHECK_EQ(nv, 3 + 12 * 2 + 6);
    const RecMap* qkv = find_map(b, "v.blk.1.qkv.weight");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)1);
    CHECK_EQ(qkv->src[0], std::string("model.visual.blocks.1.attn.qkv.weight"));

    /* The tower's rotary is 2-D, half of the 72-wide head per axis. */
    int axial = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope" && pstr(&o, "mode") == "axial") {
            ++axial;
            CHECK_EQ(pstr(&o, "sections"), std::string("18 18"));
            CHECK_EQ(pint(&o, "head_dim"), 72);
        }
    CHECK_EQ(axial, 2);
    /* The language model's is the interleaved M-RoPE with the checkpoint's sections. */
    int im = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope" && pstr(&o, "mode") == "imrope") {
            ++im;
            CHECK_EQ(pstr(&o, "sections"), std::string("11 11 10"));
            CHECK_EQ(pint(&o, "rotary_dim"), 64);
        }
    CHECK_EQ(im, 4);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* A sizing declare sizes the tower without declaring it to the core. */
    RadBuilder bp;
    RadBuildCtx cp = ctx;
    cp.shape_probe = 1;
    CHECK_OK(qwen35_bf16::declare(&bp, &meta, &cp));
    CHECK(bp.encoder_name.empty());
    CHECK(find_w(bp, "v.patch.weight") != nullptr);
}

TEST(an_encoder_pass_issues_the_tower_and_nothing_else) {
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* Two segments of 8 x 8 patches: two pictures or two temporal groups of one video. */
    static uint16_t pix[128 * 1536];
    static int32_t  coord[4 * 128];
    static const int32_t cu[] = { 0, 64, 128 };
    RadBatch batch{};
    batch.enc         = 1;
    batch.enc_n_patch = 128;
    batch.enc_pixels  = pix;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 2;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 64;
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    std::vector<std::string> want = { "gemm_nt_bias", "grid_embed" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "rope", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The merger runs over merged rows, a quarter of the patches. */
    const RecIssue* last = &c.issues.back();
    CHECK_EQ(last->n, 32);
    /* The attention reads the segment boundaries, and q, k, v as column slices. */
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    REQUIRE_EQ(at->opd.size(), (size_t)5);
    CHECK_EQ(at->opd[0].cols, 1152);
    CHECK_EQ(at->opd[1].offset, 1152);
    CHECK_EQ(at->opd[2].offset, 2304);
    CHECK(at->opd[3].raw == cu);
    CHECK_EQ(at->opd[3].rows, 3);
    /* The rotary reads the row and column planes; the grid embedding all four. */
    const RecIssue* ro = issue_of(b, c, "rope");
    REQUIRE(ro != nullptr);
    CHECK(ro->opd[1].raw == coord);
    CHECK_EQ(ro->opd[1].rows, 2);
    CHECK_EQ(ro->opd[1].cols, 128);
    const RecIssue* ge = issue_of(b, c, "grid_embed");
    REQUIRE(ge != nullptr);
    CHECK_EQ(ge->opd[2].rows, 4);

    /* A pass larger than the declared tower fails the step rather than overrunning. */
    RadBatch big = batch;
    big.enc_n_patch = 2048;
    RadCtx c2{ .b = &b, .batch = &big };
    qwen35_bf16::step(&c2, &big);
    CHECK(!c2.step_fail.empty());
    CHECK(c2.issues.empty());
}

TEST(a_step_with_pictures_scatters_their_rows_and_rotates_by_three_components) {
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* A prefill chunk whose tokens 4..11 are one picture's rows. */
    static int32_t rope[3 * 64];
    static const int32_t rows[] = { 4, 5, 6, 7, 8, 9, 10, 11 };
    static uint16_t embd[8 * 5120];
    RadBatch batch = prefill_batch();
    batch.rope_pos   = rope;
    batch.rope_mixed = 1;
    batch.n_mm_rows  = 8;
    batch.mm_rows    = rows;
    batch.mm_embd    = embd;
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    const std::vector<std::string> ops = issued_ops(b, c);
    REQUIRE(ops.size() > 2);
    CHECK_EQ(ops[0], std::string("embed_lookup"));
    CHECK_EQ(ops[1], std::string("scatter_rows"));
    const RecIssue* sc = issue_of(b, c, "scatter_rows");
    REQUIRE(sc != nullptr);
    CHECK_EQ(sc->n, 8);
    CHECK(sc->opd[0].raw == embd);
    CHECK_EQ(sc->opd[0].rows, 8);
    CHECK_EQ(sc->opd[0].cols, 5120);
    CHECK(sc->opd[1].raw == rows);
    CHECK_EQ(sc->opd[2].rows, 64);

    /* Every language-model rotary takes the [3, T] planes. */
    int n = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rope" || pstr(o, "mode") != "imrope") continue;
        ++n;
        CHECK(i.opd[1].raw == rope);
        CHECK_EQ(i.opd[1].rows, 3);
        CHECK_EQ(i.opd[1].cols, 64);
    }
    CHECK_EQ(n, 4);

    /* A text-only step on the same program: no scatter, one position per token. */
    static const int32_t pos[] = { 90, 91, 92, 93 };
    RadBatch text = decode_batch();
    text.positions = pos;
    RadCtx ct{ .b = &b, .batch = &text };
    qwen35_bf16::step(&ct, &text);
    CHECK_LINT(b, &ct);
    CHECK(ct.step_fail.empty());
    CHECK(issued_op(b, ct, "scatter_rows") == nullptr);
    for (const auto& i : ct.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rope") continue;
        CHECK(i.opd[1].raw == pos);
        CHECK_EQ(i.opd[1].cols, 0);
    }
}

namespace {
/* The Flash-Next fixture with its vision tower and its interleaved M-RoPE. */
std::vector<const char*> q4vl_keys, q4vl_vals;
RadModelMeta qwen4exp_vl_meta() {
    RadModelMeta m = qwen4exp_meta();
    if (q4vl_keys.empty()) {
        q4vl_keys.assign(m.kv_key, m.kv_key + m.n_kv);
        q4vl_vals.assign(m.kv_val, m.kv_val + m.n_kv);
        for (int i = 7; i < (int)(sizeof(k_vl_keys) / sizeof(k_vl_keys[0])); ++i) {
            q4vl_keys.push_back(k_vl_keys[i]);
            /* Flash-Next's tower emits rows at its width 2560, not the 27B's 5120. */
            q4vl_vals.push_back(std::strcmp(k_vl_keys[i], "vision_config.out_hidden_size") ? k_vl_vals[i]
                                                                                           : "2560");
        }
    }
    m.n_kv   = (int)q4vl_keys.size();
    m.kv_key = q4vl_keys.data();
    m.kv_val = q4vl_vals.data();
    return m;
}
}  /* namespace */

TEST(a_sizing_declare_with_a_vision_tower_is_the_same_graph) {
    RadModelMeta meta = qwen4exp_vl_meta();
    /* A serving configuration (fp8 cache, lossy wire, three MTP rounds) at levels below a
     * 2048-token chunk. */
    for (int world : { 1, 2 }) for (int lossy : { 0, 1 }) for (int64_t R : { 256, 1024 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuildCtx c = build_ctx(rank, world);
            c.max_tok  = 2048;
            c.max_seqs = 8;
            c.max_spec = 3;
            c.max_ctx  = 200000;
            c.max_enc_patches = 16384;
            c.kv_dtype = RAD_I8;
            c.tp_wire_lossy = lossy;
            c.tp_wire_min_bytes = 64 * 1024;
            RadBuilder full;
            REQUIRE_EQ(qwen4exp_fp8::declare(&full, &meta, &c), RAD_OK);
            CHECK_EQ(full.encoder_name, std::string("vision"));
            RadBuildCtx p = c;
            p.max_tok = R;
            p.shape_probe = 1;
            RadBuilder small;
            REQUIRE_EQ(qwen4exp_fp8::declare(&small, &meta, &p), RAD_OK);
            CHECK(small.encoder_name.empty());
            size_t first = 0;
            while (first < full.ops.size() && first < small.ops.size() &&
                   full.ops[first].op == small.ops[first].op)
                ++first;
            if (first < full.ops.size() || first < small.ops.size())
                fprintf(stderr, "    world %d rank %d lossy %d R %lld: %zu ops vs %zu, first "
                                "difference at %zu ('%s' vs '%s')\n", world, rank, lossy,
                        (long long)R, full.ops.size(), small.ops.size(),
                        first, first < full.ops.size() ? full.ops[first].op.c_str() : "-",
                        first < small.ops.size() ? small.ops[first].op.c_str() : "-");
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            CHECK_EQ(first, full.ops.size());
            /* Every fixed parameter matches the real declare's, the tower's included: a level runs
             * the real declare's kernels, so only ranged row counts may differ. */
            int fixed_differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    const bool cap = (full.ops[i].op == "qsa_work" && a.key == "work") ||
                                     (full.ops[i].op == "attn_paged_gate_quant" &&
                                      a.key == "max_seqs");
                    if (cap && q.ival <= a.ival) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    op %zu '%s' param '%s': %lld real, %lld at %lld\n", i,
                                full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival,
                                (long long)R);
                        ++fixed_differ;
                    }
                }
            }
            CHECK_EQ(fixed_differ, 0);
            /* A level shrinks only a buffer's rows; other extents stay the real declare's. */
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            for (size_t i = 0; i < full.bufs.size(); ++i) {
                const RadBufDecl& f = full.bufs[i].d;
                const RadBufDecl& q = small.bufs[i].d;
                CHECK_EQ(q.rank, f.rank);
                for (uint32_t d = 1; d < f.rank && d < q.rank; ++d) {
                    if (q.shape[d] != f.shape[d])
                        fprintf(stderr, "    '%s' extent %u: %lld real, %lld at %lld\n",
                                full.bufs[i].name.c_str(), d, (long long)f.shape[d],
                                (long long)q.shape[d], (long long)R);
                    CHECK_EQ(q.shape[d], f.shape[d]);
                }
                CHECK(q.shape[0] <= f.shape[0]);
            }
        }
    }
}

/* CalibExperts stores each expert's Gram as a packed upper triangle; a packing slip could move
 * entries between experts or transpose them while every result stays k x k and symmetric. Two
 * passes of distinct symmetric Grams go in, and each expert must come back as the sum of its own
 * two, entry for entry, at an awkward width. */
TEST(the_per_expert_tap_keeps_each_experts_gram_whole) {
    const int64_t ne = 5, k = 7;
    rad::arch::CalibExperts ce;
    ce.init(ne, k);
    std::vector<float> want((size_t)(ne * k * k), 0.0f);
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> full((size_t)(ne * k * k));
        for (int64_t e = 0; e < ne; ++e)
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t lo = std::min(i, j), hi = std::max(i, j);
                    const float v = (float)(1000 * e + 31 * lo + hi + 7 * pass);
                    full[(size_t)((e * k + i) * k + j)] = v;
                    want[(size_t)((e * k + i) * k + j)] += v;
                }
        ce.add(full.data());
    }
    std::vector<float> got((size_t)(k * k));
    int bad = 0;
    for (int64_t e = 0; e < ne; ++e) {
        ce.unpack(e, got.data());
        for (int64_t x = 0; x < k * k; ++x) bad += got[(size_t)x] != want[(size_t)(e * k * k + x)];
    }
    CHECK_EQ(bad, 0);
}

/* ==================================================================== LFM2 */
namespace {

/* d1-3B's graph cut to four layers: conv, conv, attention, conv, an irregular schedule the plugin
 * must read, not derive. A VL checkpoint (it has a text_config) with a tied head. */
const char* k_lfm2_keys[] = { "text_config.num_hidden_layers", "layer_types", "conv_L_cache",
                              "tie_word_embeddings" };
const char* k_lfm2_vals[] = { "4", "conv conv full_attention conv", "3", "true" };

RadModelMeta lfm2_meta(int n_kv = 4) {
    RadModelMeta m{};
    m.arch_id = "lfm2"; m.name = "test-lfm2"; m.quant = "";
    m.n_layers = 4; m.n_embd = 64; m.n_head = 4; m.n_head_kv = 1; m.head_dim = 16;
    m.n_ff = 128; m.n_vocab = 512; m.n_ctx_train = 32768;
    m.rms_eps = 1e-5f; m.rope_theta = 1e6f; m.rope_scale = 1.0f;
    m.n_kv = n_kv; m.kv_key = k_lfm2_keys; m.kv_val = k_lfm2_vals;
    return m;
}

std::vector<std::string> lfm2_layer_ops(bool attention) {
    std::vector<std::string> v = { "rmsnorm" };
    if (attention) {
        for (const char* s : { "gemm_nt", "gemm_nt", "gemm_nt", "rmsnorm_rows", "rmsnorm_rows",
                               "rope_rows", "rope_rows", "attn_paged", "kv_store", "gemm_nt" })
            v.push_back(s);
    } else {
        for (const char* s : { "gemm_nt", "mul", "causal_conv1d", "mul", "gemm_nt" })
            v.push_back(s);
    }
    v.push_back("add");
    append(v, mlp_ops(false));
    return v;
}

}  /* namespace */

TEST(lfm2_declares_the_checkpoints_own_schedule) {
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    std::vector<std::string> want = { "embed_lookup" };
    for (bool att : { false, false, true, false }) append(want, lfm2_layer_ops(att));
    for (const char* s : { "gather_rows", "rmsnorm", "logits_gemm" }) want.push_back(s);
    CHECK_EQ(join(op_names(b)), join(want));

    /* Two state groups: paged KV bound to the attention layer, and a conv window (n_embd channels,
     * the checkpoint's width) bound to the other three. */
    REQUIRE(b.kv_groups.size() == 2);
    CHECK_EQ(b.kv_groups[0].d.kind, (int)RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[1].d.kind, (int)RAD_KV_CONV);
    CHECK_EQ(b.kv_groups[1].d.conv_width, (int64_t)3);
    CHECK_EQ(b.kv_groups[1].d.n_head_kv * b.kv_groups[1].d.head_dim, (int64_t)64);
    REQUIRE(b.binds.size() == 4);
    for (const auto& bd : b.binds)
        CHECK_EQ(bd.second, bd.first == 2 ? (rad_kvgroup)1 : (rad_kvgroup)2);

    /* A plain RMSNorm at the checkpoint's epsilon, NeoX over the whole head; the head norms and
     * rope ask for the rows forms (their prefill row counts are what those forms are for). */
    const RecOp* norm = find_op(b, "rmsnorm");
    CHECK_NEAR(pflt(norm, "eps"), 1e-5, 1e-9);
    CHECK_EQ(param(norm, "wadd") ? pflt(norm, "wadd") : 0.0, 0.0);
    REQUIRE(find_op(b, "rmsnorm_rows") != nullptr);
    CHECK_EQ(pstr(find_op(b, "rope_rows"), "mode"), "neox");
    CHECK_EQ(pint(find_op(b, "rope_rows"), "rotary_dim"), 16);
    CHECK_EQ(pint(find_op(b, "attn_paged"), "gqa"), 4);
    CHECK_EQ(pint(find_op(b, "causal_conv1d"), "width"), 3);

    /* The VL checkpoint's names, and a tied head read from the embedding when no lm_head
     * exists. */
    const RecMap* in = find_map(b, "blk.0.conv_in.weight");
    REQUIRE(in != nullptr);
    CHECK_EQ(in->src[0], "model.language_model.layers.0.conv.in_proj.weight");
    const RecMap* head = find_map(b, "output.weight");
    REQUIRE(head != nullptr && head->src.size() == 2);
    CHECK_EQ(head->src[0], "lm_head.weight");
    CHECK_EQ(head->src[1], "model.language_model.embed_tokens.weight");
}

TEST(lfm2_text_checkpoint_keeps_its_names_under_model) {
    /* No text_config: a text-only LFM2, with layers under `model.` */
    static const char* keys[] = { "layer_types", "conv_L_cache" };
    static const char* vals[] = { "conv conv full_attention conv", "3" };
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    meta.n_kv = 2; meta.kv_key = keys; meta.kv_val = vals;
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
    const RecMap* q = find_map(b, "blk.2.attn_q.weight");
    REQUIRE(q != nullptr);
    CHECK_EQ(q->src[0], "model.layers.2.self_attn.q_proj.weight");
}

TEST(lfm2_refuses_what_it_does_not_serve) {
    {
        RadBuilder b;
        RadModelMeta meta = lfm2_meta(1);   /* no layer_types */
        RadBuildCtx ctx = build_ctx();
        CHECK(lfm2_bf16::declare(&b, &meta, &ctx) < 0);
    }
    {
        RadBuilder b;
        RadModelMeta meta = lfm2_meta();
        RadBuildCtx ctx = build_ctx(0, 2);
        CHECK_EQ(lfm2_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    }
    {
        static const char* keys[] = { "layer_types", "conv_L_cache" };
        static const char* vals[] = { "conv sliding_attention conv conv", "3" };
        RadBuilder b;
        RadModelMeta meta = lfm2_meta();
        meta.n_kv = 2; meta.kv_key = keys; meta.kv_val = vals;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(lfm2_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    }
}

TEST(lfm2_issues_the_convolution_once_per_half_of_a_mixed_step) {
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    /* Two decode rows (a verify of 3 and a plain decode) and one prefill chunk of 5. */
    static const int32_t tok[9]  = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    static const int32_t pos[9]  = { 40, 41, 42, 30, 0, 1, 2, 3, 4 };
    static const int32_t cu[4]   = { 0, 3, 4, 9 };
    static const int32_t ctxl[3] = { 40, 30, 0 };
    static const int32_t nacc[3] = { 2, 1, 1 };
    static const int32_t out[3]  = { 2, 3, 8 };
    static int32_t cv_idx[3] = { 4, 5, 6 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = lfm2_bf16::g_model[0].kv_attn;
    kv[1].group = lfm2_bf16::g_model[0].kv_conv; kv[1].state_index = cv_idx;
    RadBatch mixed{};
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 9; mixed.n_seq = 3; mixed.max_q_len = 5;
    mixed.n_seq_decode = 2; mixed.n_tok_decode = 4;
    mixed.token_ids = tok; mixed.positions = pos; mixed.cu_seqlens = cu; mixed.ctx_lens = ctxl;
    mixed.num_accepted = nacc; mixed.n_out = 3; mixed.out_ids = out;
    mixed.n_kv_groups = 2; mixed.kv = kv;

    RadCtx c{ .b = &b, .batch = &mixed };
    lfm2_bf16::step(&c, &mixed);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    int convs = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "mul" && i.opd.size() == 3 && i.opd[0].cols > 0) {
            /* conv(B * x): the B and x thirds of the [B | C | x] row as column slices. */
            if (i.opd[1].cols > 0) {
                CHECK_EQ(i.opd[0].offset, (int64_t)0);
                CHECK_EQ(i.opd[1].offset, (int64_t)128);
            } else {
                CHECK_EQ(i.opd[0].offset, (int64_t)64);    /* C * conv(...) */
            }
            CHECK_EQ(i.opd[0].cols, (int64_t)64);
        }
        if (o->op != "causal_conv1d") continue;
        REQUIRE(i.opd.size() == 8);
        if (convs++ % 2 == 0) {      /* the decode half */
            CHECK_EQ(i.n, (int64_t)4);
            CHECK_EQ(i.opd[0].rows, (int64_t)4);
            CHECK(i.opd[3].raw == (void*)cu);         CHECK_EQ(i.opd[3].rows, (int64_t)3);
            CHECK(i.opd[4].raw == (void*)cv_idx);
            CHECK(i.opd[6].raw == (void*)nacc);
        } else {                     /* prefill half: per-seq arrays past the decode seqs */
            CHECK_EQ(i.n, (int64_t)9);
            CHECK_EQ(i.opd[0].rows, (int64_t)9);
            CHECK(i.opd[3].raw == (void*)(cu + 2));   CHECK_EQ(i.opd[3].rows, (int64_t)2);
            CHECK(i.opd[4].raw == (void*)(cv_idx + 2));
            CHECK(i.opd[5].raw == (void*)(ctxl + 2));
            CHECK_EQ(i.opd[6].kind, (uint8_t)RAD_OPK_NONE);
        }
    }
    CHECK_EQ(convs, 6);   /* three conv layers, two halves each */
}

/* ==================================================================== LFM3 */
namespace {

/* 1.7B's graph cut to four layers, one of each kind after a leading conv: conv, full, sliding,
 * conv. The rotary lives under rope_parameters.<layer type> only. */
const char* k_lfm3_keys[] = { "layer_types", "conv_L_cache", "tie_word_embeddings",
                              "sliding_window",
                              "rope_parameters.full_attention.rope_theta",
                              "rope_parameters.full_attention.partial_rotary_factor",
                              "rope_parameters.sliding_attention.rope_theta" };
const char* k_lfm3_vals[] = { "conv full_attention sliding_attention conv", "7", "true", "256",
                              "1000000.0", "0.5", "10000" };

RadModelMeta lfm3_meta() {
    RadModelMeta m{};
    m.arch_id = "lfm3"; m.name = "test-lfm3"; m.quant = "";
    m.n_layers = 4; m.n_embd = 64; m.n_head = 4; m.n_head_kv = 1; m.head_dim = 16;
    m.n_ff = 128; m.n_vocab = 512; m.n_ctx_train = 32768;
    /* the typed theta is the converter's fallback for a config without a top-level one: not read */
    m.rms_eps = 1e-5f; m.rope_theta = 1e4f; m.rope_scale = 1.0f;
    m.n_kv = 7; m.kv_key = k_lfm3_keys; m.kv_val = k_lfm3_vals;
    return m;
}

/* A layer's ops: the mixer between its two norms, then the feed-forward between its two. */
std::vector<std::string> lfm3_layer_ops(bool attention) {
    std::vector<std::string> v = { "rmsnorm" };
    if (attention) {
        for (const char* s : { "gemm_nt", "gemm_nt", "gemm_nt", "gemm_nt", "rmsnorm_rows",
                               "rmsnorm_rows", "rope_rows", "rope_rows", "attn_paged", "kv_store",
                               "sigmoid", "mul", "gemm_nt" })
            v.push_back(s);
    } else {
        for (const char* s : { "gemm_nt", "mul", "causal_conv1d", "mul", "gemm_nt" })
            v.push_back(s);
    }
    for (const char* s : { "rmsnorm", "add", "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt", "rmsnorm",
                           "add" })
        v.push_back(s);
    return v;
}

/* The mixed step of the four-layer model: two decode rows (a verify of 3 and a plain decode) and
 * one prefill chunk of 5. `kv` is the caller's, three groups. */
RadBatch lfm3_mixed_batch(RadKVGroupBatch* kv) {
    static const int32_t tok[9]  = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    static const int32_t pos[9]  = { 40, 41, 42, 30, 0, 1, 2, 3, 4 };
    static const int32_t cu[4]   = { 0, 3, 4, 9 };
    static const int32_t ctxl[3] = { 40, 30, 0 };
    static const int32_t nacc[3] = { 2, 1, 1 };
    static const int32_t out[3]  = { 2, 3, 8 };
    static int32_t cv_idx[3] = { 4, 5, 6 };
    const lfm3_bf16::Model& m = lfm3_bf16::g_model[0];
    kv[0] = RadKVGroupBatch{}; kv[1] = RadKVGroupBatch{}; kv[2] = RadKVGroupBatch{};
    kv[0].group = m.full.kv;
    kv[1].group = m.sliding.kv;
    kv[2].group = m.kv_conv; kv[2].state_index = cv_idx;
    RadBatch mixed{};
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 9; mixed.n_seq = 3; mixed.max_q_len = 5;
    mixed.n_seq_decode = 2; mixed.n_tok_decode = 4;
    mixed.token_ids = tok; mixed.positions = pos; mixed.cu_seqlens = cu; mixed.ctx_lens = ctxl;
    mixed.num_accepted = nacc; mixed.n_out = 3; mixed.out_ids = out;
    mixed.n_kv_groups = 3; mixed.kv = kv;
    return mixed;
}

}  /* namespace */

TEST(lfm3_declares_three_layer_kinds_with_their_own_rotary_and_window) {
    RadBuilder b;
    b.undeclared.insert("norm_add_norm");   /* a host engine's graph; the joins have their own test */
    RadModelMeta meta = lfm3_meta();
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm3_bf16::declare(&b, &meta, &ctx));

    std::vector<std::string> want = { "embed_lookup" };
    for (bool att : { false, true, true, false }) append(want, lfm3_layer_ops(att));
    for (const char* s : { "gather_rows", "rmsnorm", "logits_gemm" }) want.push_back(s);
    CHECK_EQ(join(op_names(b)), join(want));

    /* Three state groups: paged KV for the full layer, the same bounded by the window for the
     * sliding one, and the conv window at the checkpoint's width. */
    REQUIRE(b.kv_groups.size() == 3);
    CHECK_EQ(b.kv_groups[0].d.kind, (int)RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[1].d.kind, (int)RAD_KV_WINDOW);
    CHECK_EQ(b.kv_groups[1].d.window, (int64_t)256);
    CHECK_EQ(b.kv_groups[2].d.kind, (int)RAD_KV_CONV);
    CHECK_EQ(b.kv_groups[2].d.conv_width, (int64_t)7);
    REQUIRE(b.binds.size() == 4);
    for (const auto& bd : b.binds)
        CHECK_EQ(bd.second, bd.first == 1 ? (rad_kvgroup)1 : bd.first == 2 ? (rad_kvgroup)2
                                                                           : (rad_kvgroup)3);

    /* The full layer: half the head rotates, on the head ladder, at its theta, and attention has
     * no window. q and k are rotated by one op each. */
    for (int i : { 0, 1 }) {
        const RecOp* r = find_op(b, "rope_rows", i);
        REQUIRE(r != nullptr);
        CHECK_EQ(pint(r, "rotary_dim"), 8);
        CHECK_EQ(pstr(r, "ladder"), "head");
        CHECK_EQ(pstr(r, "mode"), "neox");
        CHECK_NEAR(pflt(r, "theta"), 1e6, 1e-3);
    }
    CHECK_EQ(pint(find_op(b, "attn_paged", 0), "window"), 0);
    /* The sliding layer: the whole head at its own theta (the ladders coincide, so none is
     * named), and the window on the op as on the group. */
    for (int i : { 2, 3 }) {
        const RecOp* r = find_op(b, "rope_rows", i);
        REQUIRE(r != nullptr);
        CHECK_EQ(pint(r, "rotary_dim"), 16);
        CHECK(param(r, "ladder") == nullptr);
        CHECK_NEAR(pflt(r, "theta"), 1e4, 1e-6);
    }
    CHECK_EQ(pint(find_op(b, "attn_paged", 1), "window"), 256);
    for (int i : { 0, 1 }) {
        CHECK_EQ(pint(find_op(b, "attn_paged", i), "gqa"), 4);
        CHECK_EQ(pint(find_op(b, "attn_paged", i), "causal"), 1);
        CHECK(param(find_op(b, "attn_paged", i), "scale") == nullptr);   /* 1/sqrt(head_dim) */
    }

    /* Plain RMSNorms at the checkpoint's epsilon, four a layer and the final one. */
    CHECK_EQ(count_op(b, "rmsnorm"), 4 * 4 + 1);
    const RecOp* norm = find_op(b, "rmsnorm");
    CHECK_NEAR(pflt(norm, "eps"), 1e-5, 1e-9);
    CHECK_EQ(param(norm, "wadd") ? pflt(norm, "wadd") : 0.0, 0.0);
    CHECK_EQ(pint(find_op(b, "causal_conv1d"), "width"), 7);

    /* The names: under `model`, the post norms beside the pre norms, o_proj and the head norms
     * under their LFM3 names, a tied head read from the embedding. */
    for (const auto& [declared, src] : std::vector<std::pair<const char*, const char*>>{
             { "blk.0.attn_norm.weight", "model.layers.0.operator_norm.weight" },
             { "blk.0.attn_post_norm.weight", "model.layers.0.post_operator_norm.weight" },
             { "blk.0.ffn_post_norm.weight", "model.layers.0.post_ffn_norm.weight" },
             { "blk.0.conv_in.weight", "model.layers.0.conv.in_proj.weight" },
             { "blk.1.attn_output.weight", "model.layers.1.self_attn.o_proj.weight" },
             { "blk.1.attn_q_norm.weight", "model.layers.1.self_attn.q_norm.weight" },
             { "blk.2.attn_k_norm.weight", "model.layers.2.self_attn.k_norm.weight" } }) {
        const RecMap* mp = find_map(b, declared);
        REQUIRE(mp != nullptr);
        CHECK_EQ(mp->src[0], src);
    }
    const RecMap* head = find_map(b, "output.weight");
    REQUIRE(head != nullptr && head->src.size() == 2);
    CHECK_EQ(head->src[0], "lm_head.weight");
    CHECK_EQ(head->src[1], "model.embed_tokens.weight");

    /* The gate at the attention output's width: gate_proj's row h, head_dim times, for each head
     * in turn, so the weight is [n_head * head_dim, n_embd] and channel c reads row c / head_dim. */
    std::vector<int> rows;
    for (const auto& mp : b.maps) {
        if (mp.declared != "blk.1.attn_gate.weight") continue;
        CHECK_EQ(mp.mode, (int)RAD_MAP_ROWS);
        for (size_t i = 0; i < mp.src.size(); ++i) {
            CHECK_EQ(mp.src[i], "model.layers.1.self_attn.gate_proj.weight");
            rows.push_back(mp.src_index[i]);
        }
    }
    REQUIRE_EQ(rows.size(), (size_t)64);
    for (size_t c = 0; c < rows.size(); ++c) CHECK_EQ(rows[c], (int)(c / 16));
    const RecWeight* gate = find_w(b, "blk.1.attn_gate.weight");
    REQUIRE(gate != nullptr);
}

TEST(lfm3_refuses_what_it_does_not_serve) {
    auto declare_with = [](std::vector<const char*> keys, std::vector<const char*> vals,
                           int world = 1) {
        RadBuilder b;
        RadModelMeta meta = lfm3_meta();
        meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
        RadBuildCtx ctx = build_ctx(0, world);
        return lfm3_bf16::declare(&b, &meta, &ctx);
    };
    const std::vector<const char*> k(std::begin(k_lfm3_keys), std::end(k_lfm3_keys));
    const std::vector<const char*> v(std::begin(k_lfm3_vals), std::end(k_lfm3_vals));
    auto with = [&](const char* key, const char* val) {
        std::vector<const char*> kk = k, vv = v;
        bool found = false;
        for (size_t i = 0; i < kk.size(); ++i)
            if (!std::strcmp(kk[i], key)) { vv[i] = val; found = true; }
        if (!found) { kk.push_back(key); vv.push_back(val); }
        return declare_with(kk, vv);
    };
    auto without = [&](const char* key) {
        std::vector<const char*> kk, vv;
        for (size_t i = 0; i < k.size(); ++i)
            if (std::strcmp(k[i], key)) { kk.push_back(k[i]); vv.push_back(v[i]); }
        return declare_with(kk, vv);
    };
    CHECK_OK(declare_with(k, v));
    CHECK_EQ(declare_with(k, v, 2), RAD_E_UNSUPPORTED);                 /* two ranks */
    CHECK_EQ(without("layer_types"), RAD_E_FORMAT);
    CHECK_EQ(with("layer_types", "conv full_attention conv"), RAD_E_FORMAT);        /* 3 of 4 */
    CHECK_EQ(with("layer_types", "conv full_attention recurrent conv"), RAD_E_UNSUPPORTED);
    CHECK_EQ(with("conv_bias", "true"), RAD_E_UNSUPPORTED);
    CHECK_EQ(with("use_sandwich_norm", "false"), RAD_E_UNSUPPORTED);
    CHECK_EQ(with("conv_L_cache", "9"), RAD_E_UNSUPPORTED);
    /* A sliding layer needs its window and each attention class its own theta: the typed
     * rope_theta is not a stand-in. */
    CHECK_EQ(without("sliding_window"), RAD_E_INVAL);
    CHECK_EQ(without("rope_parameters.full_attention.rope_theta"), RAD_E_INVAL);
    CHECK_EQ(without("rope_parameters.sliding_attention.rope_theta"), RAD_E_INVAL);
    CHECK_EQ(with("rope_parameters.full_attention.rope_type", "yarn"), RAD_E_UNSUPPORTED);
    CHECK_EQ(with("rope_parameters.full_attention.partial_rotary_factor", "0"), RAD_E_INVAL);
    CHECK_EQ(with("rope_parameters.full_attention.partial_rotary_factor", "1.5"), RAD_E_INVAL);
    /* A schedule without sliding layers needs neither the window nor their theta. */
    {
        std::vector<const char*> kk = { "layer_types", "conv_L_cache",
                                        "rope_parameters.full_attention.rope_theta" };
        std::vector<const char*> vv = { "conv full_attention conv conv", "7", "1000000.0" };
        CHECK_OK(declare_with(kk, vv));
    }
}

TEST(lfm3_issues_a_mixed_step_through_both_attention_groups) {
    RadBuilder b;
    RadModelMeta meta = lfm3_meta();
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm3_bf16::declare(&b, &meta, &ctx));

    RadKVGroupBatch kv[3];
    RadBatch mixed = lfm3_mixed_batch(kv);
    RadCtx c{ .b = &b, .batch = &mixed };
    lfm3_bf16::step(&c, &mixed);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    /* Each conv layer issues its convolution once per half of the step; each attention layer
     * stores and attends through its own group, and gates the output before the projection. */
    int convs = 0, attns = 0, gates = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        convs += o->op == "causal_conv1d";
        attns += o->op == "attn_paged";
        gates += o->op == "sigmoid";
        if (o->op == "attn_paged") CHECK_EQ(i.n, (int64_t)5);
    }
    CHECK_EQ(convs, 2 * 2);
    CHECK_EQ(attns, 2);
    CHECK_EQ(gates, 2);

    /* A draft pass or media rows are refused by name. */
    RadBatch draft = mixed;
    draft.draft_pass = 1;
    RadCtx c2{ .b = &b, .batch = &draft };
    lfm3_bf16::step(&c2, &draft);
    CHECK(!c2.step_fail.empty());
}

namespace {

/* The rotated int8 trunk as data/recipes/lfm3-1.7b-iq8r.recipe states it: int8 codes rotated by
 * fwht128 with a bf16 scale per [1 x 128] of K, the group every kernel library serves. */
RadEncoding lfm3_iq8r() {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

/* The same trunk at IQ8R's 256: a u8 scale per [1 x 256] of K under a bf16 scale a row. */
RadEncoding lfm3_iq8r_256() {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_U8, 1, 256);
    rad_enc_add_plane(&e, "scale.1", RAD_BF16, 1, 0);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

const char* k_lfm3_trunk[] = { "attn_q.weight", "attn_k.weight", "attn_v.weight",
                               "attn_output.weight", "conv_in.weight", "conv_out.weight",
                               "ffn_gate_up.weight", "ffn_down.weight" };

/* The recipe's encodings by declared name: the eight trunk linears as `enc`, every other weight
 * (the gate among them) plain bf16. */
void lfm3_trunk_encs(RadBuilder* b, const RadEncoding& enc) {
    b->encs.clear();
    for (const char* t : k_lfm3_trunk) b->encs.push_back({ t, enc });
    b->encs.push_back({ "", rad_enc_plain(RAD_BF16) });
}

/* The four-layer model at widths a 128- and a 256-column scale group divide. */
RadModelMeta lfm3_wide_meta() {
    RadModelMeta m = lfm3_meta();
    m.n_embd = 256; m.n_head = 16; m.n_head_kv = 4; m.head_dim = 16; m.n_ff = 512;
    return m;
}

}  /* namespace */

/* The IQ8R container: every trunk linear is one gemm_nt_q at i8a8h over its input's int8 codes at
 * the weights' scale group (the recipe's 128, or 256), behind its own rotating quantiser of the
 * bf16 plane; the gate is still gemm_nt over the norm's bf16 output; no E4M3 plane is declared;
 * the rest of the graph is the bf16 model's. */
TEST(lfm3_iq8r_container_declares_the_rotated_int8_trunk) {
  const struct { RadEncoding enc; int64_t group; } forms[] = { { lfm3_iq8r(), 128 },
                                                               { lfm3_iq8r_256(), 256 } };
  for (const auto& form : forms) {
    RadBuilder b;
    b.undeclared.insert("norm_add_norm");
    lfm3_trunk_encs(&b, form.enc);
    RadModelMeta meta = lfm3_wide_meta();
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(lfm3_bf16::declare(&b, &meta, &ctx), RAD_OK);
    const lfm3_bf16::Model& m = lfm3_bf16::g_model[0];
    CHECK(m.q_i8);
    CHECK(!m.g.w_bf16);
    CHECK_EQ(m.g8, form.group);

    /* conv layers: in, out, gate|up, down; attention layers: q, k, v, out, gate|up, down */
    const int linears = 2 * 4 + 2 * 6;
    CHECK_EQ(count_op(b, "gemm_nt_q"), linears);
    CHECK_EQ(count_op(b, "had_quant_act_i8g"), linears);
    CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
    CHECK_EQ(count_op(b, "gemm_nt"), 2);              /* the two gates */
    for (int i = 0; i < linears; ++i) {
        const RecOp* gq = find_op(b, "gemm_nt_q", i);
        const RecOp* hq = find_op(b, "had_quant_act_i8g", i);
        REQUIRE(gq != nullptr && hq != nullptr);
        CHECK_EQ(pstr(gq, "dtype"), "i8a8h");
        CHECK_EQ(pint(gq, "group"), form.group);
        CHECK_EQ(pint(hq, "group"), form.group);
        CHECK_EQ(pint(hq, "n"), pint(gq, "K"));
    }
    for (const auto& bf : b.bufs) {
        CHECK(bf.d.dtype != RAD_F8E4M3);
        if (bf.name == "h.q8" || bf.name == "attn_out.q8" || bf.name == "conv_out.q8" ||
            bf.name == "ffn.q8")
            CHECK_EQ(bf.d.dtype, (uint32_t)RAD_I8);
    }
    /* What stays bf16: the gate, the convolution, the head (tied to the embedding). */
    for (const char* w : { "blk.1.attn_gate.weight", "blk.0.conv.weight", "output.weight",
                           "token_embd.weight" }) {
        const RecWeight* r = find_w(b, w);
        REQUIRE(r != nullptr);
        CHECK_EQ(r->d.dtype, (uint32_t)RAD_BF16);
    }
    CHECK_EQ(count_op(b, "causal_conv1d"), 2);
    CHECK_EQ(count_op(b, "attn_paged"), 2);
    CHECK_EQ(count_op(b, "rmsnorm"), 4 * 4 + 1);

    /* A step issues each linear's quantiser before its GEMM. */
    RadKVGroupBatch kv[3];
    RadBatch mixed = lfm3_mixed_batch(kv);
    RadCtx c{ .b = &b, .batch = &mixed };
    lfm3_bf16::step(&c, &mixed);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    int quants = 0, gemms = 0;
    bool quant_first = true;
    for (size_t i = 0; i < c.issues.size(); ++i) {
        const RecOp* o = b.op_of(c.issues[i].op);
        if (!o) continue;
        quants += o->op == "had_quant_act_i8g";
        if (o->op != "gemm_nt_q") continue;
        ++gemms;
        const RecOp* before = i ? b.op_of(c.issues[i - 1].op) : nullptr;
        quant_first &= before && before->op == "had_quant_act_i8g";
    }
    CHECK_EQ(gemms, linears);
    CHECK_EQ(quants, linears);
    CHECK(quant_first);

    /* Each linear quantises the plane its input is and reads that plane's codes: the stream for
     * the projections that open a mixer or the feed-forward, the mixer's own output for the one
     * that closes it (the gated attention output, the gated convolution output), the SwiGLU
     * product for the down projection. */
    auto name_of = [&](const RadOperand& o) {
        return o.kind == RAD_OPK_BUF && o.handle ? b.bufs[o.handle - 1].name : std::string("?");
    };
    std::vector<std::string> reads;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "had_quant_act_i8g") {
            reads.push_back(name_of(i.opd[0]));
            CHECK_EQ(name_of(i.opd[1]), reads.back() + ".q8");
        }
        if (o->op == "gemm_nt_q") CHECK_EQ(name_of(i.opd[0]), reads.back() + ".q8");
    }
    const std::vector<std::string> conv = { "h", "conv_out", "h", "ffn" },
                                   attn = { "h", "h", "h", "attn_out", "h", "ffn" };
    std::vector<std::string> want;
    for (bool att : { false, true, true, false }) append(want, att ? attn : conv);
    CHECK_EQ(join(reads), join(want));
  }
}

/* Plain int8 at one bf16 scale a 128 of K is the same trunk at i8a8 behind the unrotated
 * quantiser; a bf16 container declares no code plane at all. */
TEST(lfm3_int8_trunk_and_bf16_trunk_take_their_own_gemms) {
    {
        RadBuilder b;
        lfm3_trunk_encs(&b, rad_enc_affine(RAD_I8, RAD_BF16, 1, 128));
        RadModelMeta meta = lfm3_wide_meta();
        RadBuildCtx ctx = build_ctx();
        REQUIRE_EQ(lfm3_bf16::declare(&b, &meta, &ctx), RAD_OK);
        CHECK_EQ(lfm3_bf16::g_model[0].g8, (int64_t)128);
        CHECK_EQ(count_op(b, "gemm_nt_q"), 20);
        CHECK_EQ(count_op(b, "quant_act_i8g"), 20);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), 0);
        CHECK_EQ(pstr(find_op(b, "gemm_nt_q"), "dtype"), "i8a8");
    }
    {
        RadBuilder b;
        b.encs = { { "", rad_enc_plain(RAD_BF16) } };
        RadModelMeta meta = lfm3_wide_meta();
        RadBuildCtx ctx = build_ctx();
        REQUIRE_EQ(lfm3_bf16::declare(&b, &meta, &ctx), RAD_OK);
        CHECK(lfm3_bf16::g_model[0].g.w_bf16);
        CHECK_EQ(count_op(b, "gemm_nt_q"), 0);
        CHECK_EQ(count_op(b, "gemm_nt"), 22);
        for (const auto& bf : b.bufs) CHECK(bf.d.dtype != RAD_I8);
    }
}

/* Rotation is each linear's own: at one scale group a plain int8 linear in a rotated trunk keeps
 * the unrotated quantiser and i8a8, and the others keep theirs. */
TEST(lfm3_plain_int8_linear_in_a_rotated_trunk_keeps_its_own_quantiser) {
    RadBuilder b;
    lfm3_trunk_encs(&b, lfm3_iq8r());
    b.encs.insert(b.encs.begin(), { "attn_output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
    RadModelMeta meta = lfm3_wide_meta();
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(lfm3_bf16::declare(&b, &meta, &ctx), RAD_OK);
    CHECK_EQ(count_op(b, "quant_act_i8g"), 2);            /* the two attention layers' outputs */
    CHECK_EQ(count_op(b, "had_quant_act_i8g"), 18);
    int plain = 0, rotated = 0;
    for (int i = 0; i < 20; ++i) {
        const RecOp* gq = find_op(b, "gemm_nt_q", i);
        REQUIRE(gq != nullptr);
        plain += pstr(gq, "dtype") == "i8a8";
        rotated += pstr(gq, "dtype") == "i8a8h";
    }
    CHECK_EQ(plain, 2);
    CHECK_EQ(rotated, 18);
    RadKVGroupBatch kv[3];
    RadBatch mixed = lfm3_mixed_batch(kv);
    RadCtx c{ .b = &b, .batch = &mixed };
    lfm3_bf16::step(&c, &mixed);
    CHECK(c.step_fail.empty());
    for (size_t i = 1; i < c.issues.size(); ++i) {
        const RecOp* o = b.op_of(c.issues[i].op);
        const RecOp* before = b.op_of(c.issues[i - 1].op);
        if (!o || o->op != "gemm_nt_q") continue;
        REQUIRE(before != nullptr);
        CHECK_EQ(before->op, pstr(o, "dtype") == "i8a8" ? "quant_act_i8g" : "had_quant_act_i8g");
    }
}

/* What an 8-bit container may not be: a trunk of mixed formats, a quantised gate, a block-fp8
 * trunk, a narrower rotated form, or widths the scale group does not divide. */
TEST(lfm3_refuses_trunks_it_does_not_serve) {
    auto declare_with = [](const std::function<void(RadBuilder*)>& encs, RadModelMeta meta) {
        RadBuilder b;
        encs(&b);
        RadBuildCtx ctx = build_ctx();
        return lfm3_bf16::declare(&b, &meta, &ctx);
    };
    const RadEncoding iq = lfm3_iq8r();
    CHECK_OK(declare_with([&](RadBuilder* b) { lfm3_trunk_encs(b, iq); }, lfm3_wide_meta()));
    /* one linear left bf16 in an int8 trunk, and one quantised in a bf16 trunk */
    CHECK_EQ(declare_with([&](RadBuilder* b) {
                 lfm3_trunk_encs(b, iq);
                 b->encs.insert(b->encs.begin(), { "ffn_down.weight", rad_enc_plain(RAD_BF16) });
             }, lfm3_wide_meta()), RAD_E_DTYPE);
    CHECK_EQ(declare_with([&](RadBuilder* b) {
                 b->encs = { { "ffn_down.weight", iq }, { "", rad_enc_plain(RAD_BF16) } };
             }, lfm3_wide_meta()), RAD_E_DTYPE);
    /* the two rotated groups in one trunk */
    CHECK_EQ(declare_with([&](RadBuilder* b) {
                 lfm3_trunk_encs(b, iq);
                 b->encs.insert(b->encs.begin(), { "attn_output.weight", lfm3_iq8r_256() });
             }, lfm3_wide_meta()), RAD_E_DTYPE);
    /* IQ8R's 256 beside plain 128-group int8 */
    CHECK_EQ(declare_with([&](RadBuilder* b) {
                 lfm3_trunk_encs(b, lfm3_iq8r_256());
                 b->encs.insert(b->encs.begin(),
                                { "attn_output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
             }, lfm3_wide_meta()), RAD_E_DTYPE);
    /* the gate */
    CHECK_EQ(declare_with([&](RadBuilder* b) {
                 lfm3_trunk_encs(b, iq);
                 b->encs.insert(b->encs.begin(), { "attn_gate.weight", iq });
             }, lfm3_wide_meta()), RAD_E_DTYPE);
    /* block fp8 */
    CHECK_EQ(declare_with([&](RadBuilder* b) {
                 lfm3_trunk_encs(b, rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128));
             }, lfm3_wide_meta()), RAD_E_DTYPE);
    /* a 64-wide stream under a 128-column group */
    CHECK_EQ(declare_with([&](RadBuilder* b) { lfm3_trunk_encs(b, iq); }, lfm3_meta()),
             RAD_E_UNSUPPORTED);
}

/* The sandwich joins. Where a library serves `norm_add_norm`, a layer's post norm, add and
 * feed-forward norm are one launch, and so are its feed-forward's post norm, add and the next
 * layer's norm, which that layer then neither declares nor issues; the last layer's feed-forward
 * keeps the chain, since the final norm is on the gathered rows. Where no loaded library declares
 * the op (a host engine) it is not asked for, and where it is declared and does not resolve the
 * chain is declared beside it: both issue the chain. */
TEST(lfm3_fuses_its_sandwich_joins_where_a_library_serves_them) {
    /* 0: served; 1: declared, not resolved; 2: not declared */
    auto run = [](int form, std::vector<std::string>* declared, std::string* note) {
        RadBuilder b;
        if (form == 1) b.refuse.insert("norm_add_norm");
        if (form == 2) b.undeclared.insert("norm_add_norm");
        RadModelMeta meta = lfm3_meta();
        RadBuildCtx ctx = build_ctx();
        CHECK_OK(lfm3_bf16::declare(&b, &meta, &ctx));
        CHECK(b.asked_undeclared.empty());
        RadKVGroupBatch kv[3];
        RadBatch mixed = lfm3_mixed_batch(kv);
        RadCtx c{ .b = &b, .batch = &mixed };
        lfm3_bf16::step(&c, &mixed);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        *declared = op_names(b);
        for (const auto& n : b.notes)
            if (n.find("sandwich joins") != std::string::npos) *note = n;
        return issued_ops(b, c);
    };
    /* What a step issues: the convolution once per half of the mixed step. */
    auto want = [](bool fused) {
        std::vector<std::string> v = { "embed_lookup" };
        int l = 0;
        for (bool att : { false, true, true, false }) {
            if (!fused || l == 0) v.push_back("rmsnorm");
            if (att)
                for (const char* s : { "gemm_nt", "gemm_nt", "gemm_nt", "gemm_nt", "rmsnorm_rows",
                                       "rmsnorm_rows", "rope_rows", "rope_rows", "kv_store",
                                       "attn_paged", "sigmoid", "mul", "gemm_nt" })
                    v.push_back(s);
            else
                for (const char* s : { "gemm_nt", "mul", "causal_conv1d", "causal_conv1d", "mul",
                                       "gemm_nt" })
                    v.push_back(s);
            if (fused) v.push_back("norm_add_norm");
            else for (const char* s : { "rmsnorm", "add", "rmsnorm" }) v.push_back(s);
            for (const char* s : { "gemm_nt", "silu_mul", "gemm_nt" }) v.push_back(s);
            if (fused && l < 3) v.push_back("norm_add_norm");
            else for (const char* s : { "rmsnorm", "add" }) v.push_back(s);
            ++l;
        }
        for (const char* s : { "gather_rows", "rmsnorm", "logits_gemm" }) v.push_back(s);
        return v;
    };

    std::vector<std::string> decl;
    std::string note;
    const std::vector<std::string> fused = run(0, &decl, &note);
    CHECK_EQ(join(fused), join(want(true)));
    CHECK_EQ(count_issued(fused, "norm_add_norm"), 2 * 4 - 1);
    CHECK_EQ(count_issued(fused, "add"), 1);
    CHECK_EQ(count_issued(fused, "rmsnorm"), 1 + 1 + 1);   /* layer 0's, the last post norm, final */
    /* nothing of the chain is declared where a join stands in for it */
    CHECK_EQ(count_issued(decl, "rmsnorm"), 3);
    CHECK_EQ(count_issued(decl, "add"), 1);
    CHECK(note.find("7 of 7 fused") != std::string::npos);

    const std::vector<std::string> refused = run(1, &decl, &note);
    CHECK_EQ(join(refused), join(want(false)));
    CHECK_EQ(count_issued(decl, "norm_add_norm"), 7);          /* asked for, each time */
    CHECK(note.find("0 of 7 fused") != std::string::npos);

    const std::vector<std::string> host = run(2, &decl, &note);
    CHECK_EQ(join(host), join(want(false)));
    CHECK_EQ(count_issued(decl, "norm_add_norm"), 0);
    CHECK(note.find("0 of 7 fused") != std::string::npos);

    /* The join's operands: the arm's output and the stream in, the stream and the next norm out,
     * with the post norm's gain and the gain of the norm it writes -- the next layer's between
     * two layers. */
    {
        RadBuilder b;
        RadModelMeta meta = lfm3_meta();
        RadBuildCtx ctx = build_ctx();
        CHECK_OK(lfm3_bf16::declare(&b, &meta, &ctx));
        const lfm3_bf16::Model& m = lfm3_bf16::g_model[0];
        const RecOp* a = find_op(b, "norm_add_norm", 0);
        const RecOp* f = find_op(b, "norm_add_norm", 1);
        REQUIRE(a != nullptr && f != nullptr);
        for (const RecOp* r : { a, f }) {
            CHECK_EQ(pint(r, "n"), 64);
            CHECK_NEAR(pflt(r, "eps"), 1e-5, 1e-9);
            CHECK_EQ(pstr(r, "dtype"), "bf16");
            REQUIRE(r->reads.size() == 2 && r->writes.size() == 2 && r->w.size() == 2);
            CHECK_EQ(r->reads[0], m.b_arm);
            CHECK_EQ(r->reads[1], m.b_x);
            CHECK_EQ(r->writes[0], m.b_x);
            CHECK_EQ(r->writes[1], m.b_h);
        }
        CHECK_EQ(a->w[0], m.layers[0].w_post_norm);
        CHECK_EQ(a->w[1], m.layers[0].w_ffn_norm);
        CHECK_EQ(f->w[0], m.layers[0].w_ffn_post_norm);
        CHECK_EQ(f->w[1], m.layers[1].w_norm);
        CHECK(m.layers[1].op_norm == 0 && m.layers[3].op_ffn_join == 0);
    }
}

/* ==================================================================== LFM2-VL's tower */
namespace {

/* lfm2_meta's text model with a SigLIP2 tower cut to two layers of 64 (four heads of 16), d1's
 * patch, position table and unshuffle, and a 96-wide projector. */
const char* k_lfm2vl_keys[] = { "text_config.num_hidden_layers", "layer_types", "conv_L_cache",
                                "tie_word_embeddings", "vision_config.model_type",
                                "vision_config.num_hidden_layers", "vision_config.hidden_size",
                                "vision_config.num_attention_heads", "vision_config.intermediate_size",
                                "vision_config.patch_size", "vision_config.num_patches",
                                "vision_config.hidden_act", "vision_config.vision_use_head",
                                "downsample_factor", "projector_hidden_size",
                                "projector_hidden_act", "projector_bias",
                                "projector_use_layernorm", "max_num_patches" };
const char* k_lfm2vl_vals[] = { "4", "conv conv full_attention conv", "3", "true",
                                "siglip2_vision_model", "2", "64", "4", "128", "16", "256",
                                "gelu_pytorch_tanh", "false", "2", "96", "gelu", "true", "false",
                                "1024" };
const char* k_lfm2vl_norm_vals[] = { "4", "conv conv full_attention conv", "3", "true",
                                     "siglip2_vision_model", "2", "64", "4", "128", "16", "256",
                                     "gelu_pytorch_tanh", "false", "2", "96", "gelu", "true", "true",
                                     "1024" };

RadModelMeta lfm2vl_meta(bool proj_norm = false) {
    RadModelMeta m = lfm2_meta();
    m.n_kv = (int)(sizeof(k_lfm2vl_keys) / sizeof(k_lfm2vl_keys[0]));
    m.kv_key = k_lfm2vl_keys;
    m.kv_val = proj_norm ? k_lfm2vl_norm_vals : k_lfm2vl_vals;
    return m;
}

}  /* namespace */

TEST(lfm2_vl_declares_the_siglip2_tower_and_its_projector) {
    {   /* No encoder budget: text only, whatever the container carries. */
        RadBuilder b;
        RadModelMeta meta = lfm2vl_meta();
        RadBuildCtx ctx = vl_ctx(0);
        CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
        CHECK(b.encoder_name.empty());
        CHECK_EQ(count_op(b, "scatter_rows"), 0);
    }
    RadBuilder b;
    RadModelMeta meta = lfm2vl_meta();
    RadBuildCtx ctx = vl_ctx(1024);
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)RAD_MM_IMAGE);
    CHECK_EQ(b.encoder.patch_dim, 3 * 16 * 16);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 1024);
    CHECK_EQ(b.encoder.n_embd, 64);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* 3 stem + 12 per layer + 2 post-norm + 4 projector, each mapped from the checkpoint. */
    int nv = 0;
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            ++nv;
            CHECK(find_map(b, w.name) != nullptr);
        }
    CHECK_EQ(nv, 3 + 2 * 12 + 2 + 4);
    /* q, k and v are concatenated at convert into one QKV weight and bias. */
    const RecMap* qkv = find_map(b, "v.blk.1.qkv.weight");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)3);
    CHECK_EQ(qkv->src[0],
             std::string("model.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight"));
    CHECK_EQ(qkv->src[2],
             std::string("model.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.weight"));
    const RecMap* l1 = find_map(b, "v.proj.fc1.weight");
    REQUIRE(l1 != nullptr);
    CHECK_EQ(l1->src[0], std::string("model.multi_modal_projector.linear_1.weight"));
    /* The projector reads four patches as one row; it uses erf GELU, the tower tanh GELU. */
    const RecWeight* w1 = find_w(b, "v.proj.fc1.weight");
    REQUIRE(w1 != nullptr);
    CHECK_EQ(w1->d.shape[0], 96);
    CHECK_EQ(w1->d.shape[1], 4 * 64);
    int tanh = 0, erf = 0;
    for (const auto& o : b.ops)
        if (o.op == "gemm_nt_bias") {
            tanh += pstr(&o, "act") == "gelu_tanh";
            erf += pstr(&o, "act") == "gelu";
        }
    CHECK_EQ(tanh, 2);
    CHECK_EQ(erf, 1);
    /* No rotary in the tower; the position table is resampled as torch's antialiased bilinear,
     * and attention's block height is bounded by the longest segment the processor makes. */
    CHECK_EQ(count_op(b, "grid_embed_aa"), 1);
    CHECK_EQ(count_op(b, "grid_embed"), 0);
    CHECK_EQ(pint(find_op(b, "attn_dense"), "max_seqlen"), 1024);
    CHECK_EQ(pint(find_op(b, "attn_dense"), "head_dim"), 16);

    /* A projector with a layer norm declares it over the merged rows. */
    RadBuilder bn;
    RadModelMeta mn = lfm2vl_meta(true);
    CHECK_OK(lfm2_bf16::declare(&bn, &mn, &ctx));
    CHECK(find_w(bn, "v.proj.norm.weight") != nullptr);
    CHECK_EQ(count_op(bn, "layernorm"), count_op(b, "layernorm") + 1);
}

TEST(lfm2_vl_encoder_pass_and_the_rows_it_scatters) {
    RadBuilder b;
    RadModelMeta meta = lfm2vl_meta();
    RadBuildCtx ctx = vl_ctx(1024);
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    /* A tile of 8 x 8 patches and a thumbnail of 4 x 8. */
    static uint16_t pix[96 * 768];
    static int32_t  coord[4 * 96];
    static const int32_t cu[] = { 0, 64, 96 };
    RadBatch batch{};
    batch.enc         = RAD_MM_IMAGE;
    batch.enc_n_patch = 96;
    batch.enc_pixels  = pix;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 2;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 64;
    RadCtx c{ .b = &b, .batch = &batch };
    lfm2_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    std::vector<std::string> want = { "gemm_nt_bias", "grid_embed_aa" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));
    /* The position resample reads all four coordinate planes; the projector runs on merged rows. */
    const RecIssue* ge = issue_of(b, c, "grid_embed_aa");
    REQUIRE(ge != nullptr);
    CHECK(ge->opd[2].raw == coord);
    CHECK_EQ(ge->opd[2].rows, 4);
    CHECK_EQ(c.issues.back().n, 24);
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    CHECK_EQ(at->opd[1].offset, 64);
    CHECK_EQ(at->opd[2].offset, 128);
    CHECK(at->opd[3].raw == cu);

    /* A text step with media rows scatters them over the lookup before the first layer. */
    static const int32_t tok[4] = { 1, 2, 3, 4 }, pos[4] = { 0, 1, 2, 3 }, cus[2] = { 0, 4 };
    static const int32_t ctxl[1] = { 0 }, out[1] = { 3 }, rows[2] = { 1, 2 };
    static uint16_t embd[2 * 64];
    static int32_t cv_idx[1] = { 0 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = lfm2_bf16::g_model[0].kv_attn;
    kv[1].group = lfm2_bf16::g_model[0].kv_conv; kv[1].state_index = cv_idx;
    RadBatch tb{};
    tb.phase = RAD_PHASE_PREFILL;
    tb.n_tok = 4; tb.n_seq = 1; tb.max_q_len = 4;
    tb.token_ids = tok; tb.positions = pos; tb.cu_seqlens = cus; tb.ctx_lens = ctxl;
    tb.n_out = 1; tb.out_ids = out; tb.n_kv_groups = 2; tb.kv = kv;
    tb.n_mm_rows = 2; tb.mm_rows = rows; tb.mm_embd = embd;
    RadCtx ct{ .b = &b, .batch = &tb };
    lfm2_bf16::step(&ct, &tb);
    CHECK_LINT(b, &ct);
    CHECK(ct.step_fail.empty());
    const std::vector<std::string> ops = issued_ops(b, ct);
    REQUIRE(ops.size() > 2);
    CHECK_EQ(ops[0], std::string("embed_lookup"));
    CHECK_EQ(ops[1], std::string("scatter_rows"));
}

/* =============================== gemma4 dense: two attention classes =========================== */

TEST(declare_is_a_gemma4_trunk) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    /* 12 layers: 10 sliding + 2 global (layers 5 and 11). */
    CHECK_EQ(count_op(b, "attn_paged"),  12);
    /* the cache store rides the fused prologue where it resolves (qkv_norm_rope_kv_store) */
    CHECK_EQ(count_op(b, "qkv_norm_rope_kv_store"), 12);
    CHECK_EQ(count_op(b, "kv_store"),     0);
    CHECK_EQ(count_op(b, "rmsnorm_p"),   12);   /* weightless value norm, one per layer */
    CHECK_EQ(count_op(b, "rope"),        24);   /* q + k */
    CHECK_EQ(count_op(b, "gelu_mul"),    12);   /* Gemma's FFN activation, not SwiGLU */
    CHECK_EQ(count_op(b, "silu_mul"),     0);
    CHECK_EQ(count_op(b, "gemm_nt"),     70);   /* 10x6 sliding (q,k,v,o,gu,down) + 2x5 global (no v) */
    CHECK_EQ(count_op(b, "rmsnorm"),     73);   /* 4 attn + 2 ffn per layer, plus the final norm */
    CHECK_EQ(count_op(b, "add"),         24);   /* the two GemmaRMSNorm residuals per layer */
    CHECK_EQ(count_op(b, "embed_lookup"),  1);
    CHECK_EQ(count_op(b, "gather_rows"),   1);
    CHECK_EQ(count_op(b, "logits_gemm"),   1);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);
    CHECK_EQ(count_op(b, "all_reduce"),    0);  /* world 1 */
}

/* The plugin states the reply format, since template analysis cannot read Gemma 4's template.
 * Calls are the object family (a name and one object in Gemma's dialect) and the thought channel
 * may reopen after the answer. chatparse_gemma4_test covers parsing; this checks the export. */
TEST(gemma4_declares_its_reply_format) {
    RadModelMeta meta = gemma4_meta();
    const RadChatFormat* f = gemma4_bf16::chat_format(&meta);
    REQUIRE(f != nullptr);
    CHECK_EQ(f->struct_size, (uint32_t)sizeof(RadChatFormat));
    CHECK_EQ(f->args_form, (uint32_t)RAD_CHAT_ARGS_OBJECT);
    CHECK_EQ(std::string(f->call_open), std::string("<|tool_call>"));
    CHECK_EQ(std::string(f->call_close), std::string("<tool_call|>"));
    CHECK_EQ(std::string(f->name_prefix), std::string("call:"));
    CHECK_EQ(std::string(f->name_suffix), std::string(""));
    CHECK_EQ(std::string(f->object_string_open), std::string("<|\"|>"));
    CHECK_EQ(std::string(f->object_string_close), std::string("<|\"|>"));
    CHECK(f->object_bare_keys != 0);
    CHECK_EQ(std::string(f->reasoning_open), std::string("<|channel>thought\n"));
    CHECK_EQ(std::string(f->reasoning_close), std::string("<channel|>"));
    CHECK(f->reasoning_anywhere != 0);
    REQUIRE(f->n_triggers == 1);
    CHECK_EQ(std::string(f->triggers[0]), std::string("<|tool_call>"));
    CHECK(f->arg_prefix == nullptr && f->arg_value_suffix == nullptr);
}

TEST(gemma4_declares_two_kv_groups_and_binds_the_schedule) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.kv_groups.size(), 2);
    CHECK_EQ((int)b.kv_groups[0].d.kind, RAD_KV_WINDOW);   /* sliding, declared first */
    CHECK_EQ(b.kv_groups[0].d.window,    1024);
    CHECK_EQ(b.kv_groups[0].d.n_head_kv, 16);
    CHECK_EQ(b.kv_groups[0].d.head_dim,  256);
    CHECK_EQ((int)b.kv_groups[1].d.kind, RAD_KV_FULL);     /* global */
    CHECK_EQ(b.kv_groups[1].d.window,    0);
    CHECK_EQ(b.kv_groups[1].d.n_head_kv, 4);
    CHECK_EQ(b.kv_groups[1].d.head_dim,  512);

    /* (l + 1) % 6 == 0 gives global layers 5 and 11; group 1 is sliding (declared first), 2
     * global. */
    CHECK_EQ((int)b.binds.size(), 12);
    for (int l = 0; l < 12; ++l) {
        const bool want_global = ((l + 1) % 6 == 0);
        CHECK_EQ((int)b.binds[(size_t)l].first, l);
        CHECK_EQ((int)b.binds[(size_t)l].second, want_global ? 2 : 1);
    }
}

/* A step with an image: the rows go into the token lookup before its all-reduce, since
 * RadBatch::mm_embd holds them at rank 0 and zeros at every other rank. */
TEST(gemma4_scatters_image_rows_into_the_lookup_before_its_all_reduce) {
    std::vector<const char*> keys(std::begin(k_gem4_keys), std::end(k_gem4_keys));
    std::vector<const char*> vals(std::begin(k_gem4_vals), std::end(k_gem4_vals));
    const char* vk[] = { "vision_config.model_type", "vision_config.hidden_size",
                         "vision_config.num_attention_heads", "vision_config.head_dim",
                         "vision_config.intermediate_size", "vision_config.pooling_kernel_size",
                         "vision_config.position_embedding_size", "vision_config.num_hidden_layers",
                         "vision_config.patch_size", "vision_config.rope_parameters.rope_theta",
                         "vision_config.rope_parameters.rope_type", "vision_config.hidden_activation" };
    const char* vv[] = { "gemma4_vision", "768", "12", "64", "3072", "3", "10240", "2", "16", "100",
                         "default", "gelu_pytorch_tanh" };
    keys.insert(keys.end(), std::begin(vk), std::end(vk));
    vals.insert(vals.end(), std::begin(vv), std::end(vv));
    static const int32_t rows[] = { 4, 5, 6, 7, 8, 9, 10, 11 };
    static uint16_t embd[8 * 5376];
    for (int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadModelMeta meta = gemma4_meta();
            meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
            RadBuildCtx ctx = build_ctx(rank, world);
            ctx.max_enc_patches = 9 * 64;
            CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
            RadBatch batch = prefill_batch();
            batch.n_mm_rows = 8; batch.mm_rows = rows; batch.mm_embd = embd;
            RadCtx c; c.b = &b; c.rank = rank; c.world = world; c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
            CHECK_LINT(b, &c);
            CHECK(c.step_fail.empty());
            const std::vector<std::string> ops = issued_ops(b, c);
            REQUIRE(ops.size() > 3);
            CHECK_EQ(ops[0], std::string("embed_lookup"));
            CHECK_EQ(ops[1], std::string("scatter_rows"));
            CHECK(c.issues[1].opd[0].raw == embd);
            if (world == 2) CHECK_EQ(ops[2], std::string("all_reduce"));
        }
    }
}

/* The E2B/E4B form on the same twelve layers: the last four share KV (no cache of their own),
 * per-layer embeddings 64 wide, a global class with its own v (attention_k_eq_v 0) and no
 * num_global_key_value_heads, so the global layers take num_key_value_heads. */
const char* k_gem4e_keys[] = {
    "sliding_window", "global_head_dim", "num_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "full_attention_interval", "attention_k_eq_v",
    "hidden_size_per_layer_input",
};
const char* k_gem4e_vals[] = { "1024", "512", "16", "10000", "1000000", "0.25", "4", "30", "6", "0",
                               "64" };

RadModelMeta gemma4_e4b_meta() {
    RadModelMeta m = gemma4_meta();
    m.name   = "test-gemma4-e4b-form";
    m.n_kv   = (int)(sizeof(k_gem4e_keys) / sizeof(k_gem4e_keys[0]));
    m.kv_key = k_gem4e_keys;
    m.kv_val = k_gem4e_vals;
    return m;
}

/* E2B: the same twelve layers with `use_double_wide_mlp`, so the four KV-shared layers (8..11)
 * declare 2 * intermediate_size (transformers' Gemma4TextMLP). */
RadModelMeta gemma4_e2b_meta() {
    static const std::vector<const char*> keys = [] {
        std::vector<const char*> v(std::begin(k_gem4e_keys), std::end(k_gem4e_keys));
        v.push_back("use_double_wide_mlp");
        return v;
    }();
    static const std::vector<const char*> vals = [] {
        std::vector<const char*> v(std::begin(k_gem4e_vals), std::end(k_gem4e_vals));
        v.push_back("true");
        return v;
    }();
    RadModelMeta m = gemma4_e4b_meta();
    m.name   = "test-gemma4-e2b-form";
    m.n_kv   = (int)keys.size();
    m.kv_key = keys.data();
    m.kv_val = vals.data();
    return m;
}

/* Layers 8..11 keep no cache: only 0..7 are bound, and each shared layer's attention reads the last
 * earlier layer of its class that keeps one (sliding 8..10 -> 7, global 11 -> 5). Every layer runs
 * the per-layer branch (gate, gelu times its slice, projection, norm, add), and the step lints
 * clean in bf16 and in w4a8, where the junction closes the branch instead of the feed-forward. */
TEST(a_gemma4_e4b_form_shares_kv_and_runs_the_per_layer_branch) {
    for (bool w4 : { false, true }) {
        RadBuilder b;
        if (w4) {
            const RadEncoding e = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
            b.encs = { { "attn_q.", e }, { "attn_k.", e }, { "attn_v.", e }, { "attn_o.", e },
                       { "ffn_gate_up.", e }, { "ffn_down.", e } };
        }
        RadModelMeta meta = gemma4_e4b_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

        CHECK_EQ((int)b.binds.size(), 8);
        for (const auto& bd : b.binds) CHECK((int)bd.first < 8);
        CHECK_EQ(b.kv_groups[1].d.n_head_kv, 16);   /* num_key_value_heads, the global key absent */
        CHECK_EQ(count_op(b, "attn_paged"), 12);
        CHECK_EQ(count_op(b, "act_mul"), 12);
        CHECK_EQ(count_op(b, "embed_lookup"), 2);   /* the token table and the per-layer table */

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch, .world = 1 };
        gemma4_bf16::step(&c, &batch);
        CHECK_LINT(b, &c);
        std::vector<int64_t> src;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (o && o->op == "attn_paged") {
                CHECK_EQ(i.opd[1].kind, (uint8_t)RAD_OPK_KV);
                src.push_back(i.opd[1].offset);
            }
        }
        const std::vector<int64_t> want = { 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 5 };
        CHECK(src == want);
    }
}

/* At two ranks the per-layer table is vocab-sharded like the token table and its lookup is reduced
 * like the token lookup, before anything reads it; the rest of the branch is replicated. A step
 * reduces the two lookups and, a layer, attention's and the feed-forward's partial sums -- folded
 * into the attention junction where it resolves -- and never the branch's output, which every rank
 * holds whole. The prefill overlap, whose slices skip the branch, stays off. */
TEST(the_gemma4_e4b_form_runs_at_two_ranks) {
    const int64_t L = 12, P = 12 * 64;
    for (bool w4 : { false, true }) {
        for (int rank : { 0, 1 }) {
            RadBuilder b;
            if (w4) {
                const RadEncoding e = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
                b.encs = { { "attn_q.", e }, { "attn_k.", e }, { "attn_v.", e }, { "attn_o.", e },
                           { "ffn_gate_up.", e }, { "ffn_down.", e } };
            }
            RadModelMeta meta = gemma4_e4b_meta();
            RadBuildCtx  ctx  = build_ctx(rank, 2);
            ctx.max_tok = 4096;   /* past two slices of the overlap */
            CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
            CHECK(!gemma4_bf16::g_model[rank].overlap);
            CHECK_EQ(gemma4_bf16::g_model[rank].fuse, w4);

            /* the per-layer table takes the token table's slice of the vocabulary */
            const RecOp* tok = find_op(b, "embed_lookup", 0);
            const RecOp* ple = find_op(b, "embed_lookup", 1);
            REQUIRE(tok != nullptr && ple != nullptr);
            CHECK_EQ(pint(ple, "n_embd"), P);
            CHECK_EQ(pint(ple, "n_vocab"), pint(tok, "n_vocab"));
            CHECK_EQ(pint(ple, "vocab_offset"), pint(tok, "vocab_offset"));

            RadBatch batch = decode_batch();
            RadCtx c; c.b = &b; c.rank = rank; c.world = 2; c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
            CHECK_LINT(b, &c);
            CHECK(c.step_fail.empty());
            const std::vector<std::string> ops = issued_ops(b, c);
            REQUIRE(ops.size() > 4);
            CHECK_EQ(ops[0], std::string("embed_lookup"));
            CHECK_EQ(ops[1], std::string("all_reduce"));
            CHECK_EQ(ops[2], std::string("embed_lookup"));
            CHECK_EQ(ops[3], std::string("all_reduce"));
            CHECK_EQ(c.issues[3].n, batch.n_tok * P);
            const int folded = count_issued(ops, "ar_rmsnorm_quant_i8");
            CHECK_EQ(count_issued(ops, "all_reduce") + folded, (int)(2 + 2 * L));
            CHECK_EQ(folded, w4 ? (int)L : 0);
        }
    }
}

/* With an image at two ranks, the pad rows of the image positions go into the per-layer inputs
 * after their lookup's all-reduce: every rank writes them whole, so before it they would count
 * twice. */
TEST(the_gemma4_e4b_form_writes_image_pad_rows_after_the_per_layer_all_reduce) {
    std::vector<const char*> keys(std::begin(k_gem4e_keys), std::end(k_gem4e_keys));
    std::vector<const char*> vals(std::begin(k_gem4e_vals), std::end(k_gem4e_vals));
    const char* vk[] = { "vision_config.model_type", "vision_config.hidden_size",
                         "vision_config.num_attention_heads", "vision_config.head_dim",
                         "vision_config.intermediate_size", "vision_config.pooling_kernel_size",
                         "vision_config.position_embedding_size", "vision_config.num_hidden_layers",
                         "vision_config.patch_size", "vision_config.rope_parameters.rope_theta",
                         "vision_config.rope_parameters.rope_type", "vision_config.hidden_activation" };
    const char* vv[] = { "gemma4_vision", "768", "12", "64", "3072", "3", "10240", "2", "16", "100",
                         "default", "gelu_pytorch_tanh" };
    keys.insert(keys.end(), std::begin(vk), std::end(vk));
    vals.insert(vals.end(), std::begin(vv), std::end(vv));
    static const int32_t rows[] = { 4, 5, 6, 7, 8, 9, 10, 11 };
    static uint16_t embd[8 * 5376];
    for (int rank : { 0, 1 }) {
        RadBuilder b;
        RadModelMeta meta = gemma4_e4b_meta();
        meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
        RadBuildCtx ctx = build_ctx(rank, 2);
        ctx.max_enc_patches = 9 * 64;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        RadBatch batch = prefill_batch();
        batch.n_mm_rows = 8; batch.mm_rows = rows; batch.mm_embd = embd;
        RadCtx c; c.b = &b; c.rank = rank; c.world = 2; c.batch = &batch;
        gemma4_bf16::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        size_t i = 0;   /* the per-layer lookup: the second embed_lookup */
        for (int seen = 0; i < ops.size(); ++i)
            if (ops[i] == "embed_lookup" && ++seen == 2) break;
        REQUIRE(i + 4 < ops.size());
        const std::vector<std::string> want = { "all_reduce", "fill", "mul_weight", "scatter_rows" };
        CHECK(std::vector<std::string>(ops.begin() + (long)i + 1, ops.begin() + (long)i + 5) == want);
    }
}

/* E4B's attention width: 8 q heads over 2 KV heads in both classes, so four ranks replicate each
 * KV head over two. Each rank's k and v rows are its head's (rows (rank / 2) * head_dim on), as one
 * span or as the k and v parts of a fused q|k|v with the scale plane alongside; q stays an even
 * share. Every attention junction that runs keeps every row, since the per-layer branch follows
 * the feed-forward, and a step still reduces the two lookups and two a layer. */
TEST(the_gemma4_e4b_form_replicates_kv_heads_at_four_ranks) {
    std::vector<const char*> vals(std::begin(k_gem4e_vals), std::end(k_gem4e_vals));
    vals[2] = "2";   /* num_key_value_heads */
    const int64_t L = 12, hs = 256, hg = 512;
    for (bool fused : { false, true }) {
        for (int rank = 0; rank < 4; ++rank) {
            RadBuilder b;
            if (fused) {
                const RadEncoding e = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
                /* attn_q for the KV-shared layers, which have no q|k|v row */
                b.encs = { { "attn_qkv.", e }, { "attn_q.", e }, { "attn_o.", e },
                           { "ffn_gate_up.", e }, { "ffn_down.", e } };
            }
            RadModelMeta meta = gemma4_e4b_meta();
            meta.n_head = 8; meta.n_head_kv = 2; meta.kv_val = vals.data();
            RadBuildCtx ctx = build_ctx(rank, 4);
            CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
            CHECK_EQ(b.kv_groups[0].d.n_head_kv, 1);
            CHECK_EQ(b.kv_groups[1].d.n_head_kv, 1);
            const RecOp* at = find_op(b, "attn_paged", 0);
            REQUIRE(at != nullptr);
            CHECK_EQ(pint(at, "n_head"), 2);
            CHECK_EQ(pint(at, "gqa"), 2);
            /* layer 0 sliding, layer 5 full; both keep a cache of their own */
            const std::initializer_list<std::pair<int, int64_t>> own = { { 0, hs }, { 5, hg } };
            for (const auto& [l, hd] : own) {
                const int64_t kv0 = (rank / 2) * hd;
                if (fused) {
                    for (const char* v : { "weight", "scale" }) {
                        const RecWeight* w =
                            find_w(b, "blk." + std::to_string(l) + ".attn_qkv." + v);
                        REQUIRE(w != nullptr);
                        REQUIRE_EQ(w->n_part_spans, 3);
                        CHECK_EQ(w->part_lo[0], rank * 2 * hd);
                        CHECK_EQ(w->part_total[0], 8 * hd);
                        for (int p : { 1, 2 }) {
                            CHECK_EQ(w->part_lo[p], kv0);
                            CHECK_EQ(w->part_hi[p], kv0 + hd);
                            CHECK_EQ(w->part_total[p], 2 * hd);
                        }
                    }
                } else {
                    for (const char* kv : { "attn_k", "attn_v" }) {
                        const RecWeight* w =
                            find_w(b, "blk." + std::to_string(l) + "." + kv + ".weight");
                        REQUIRE(w != nullptr);
                        CHECK(w->span_set);
                        CHECK_EQ(w->span_lo, kv0);
                        CHECK_EQ(w->span_hi, kv0 + hd);
                    }
                }
            }
            CHECK_EQ(gemma4_bf16::g_model[rank].fuse, fused);

            RadBatch batch = decode_batch();
            RadCtx c; c.b = &b; c.rank = rank; c.world = 4; c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
            CHECK_LINT(b, &c);
            CHECK(c.step_fail.empty());
            /* a 4-row step takes the split form: every one keeps all rows */
            int split = 0;
            for (const auto& i : c.issues) {
                const RecOp* o = b.op_of(i.op);
                if (o && o->op == "ar_rmsnorm_quant_i8" && param(o, "sp")) {
                    CHECK_EQ(pint(o, "sp"), 2);
                    ++split;
                }
            }
            CHECK_EQ(split, fused ? (int)L : 0);
            const std::vector<std::string> ops = issued_ops(b, c);
            CHECK_EQ(count_issued(ops, "all_reduce") + count_issued(ops, "ar_rmsnorm_quant_i8"),
                     (int)(2 + 2 * L));
        }
    }
}

/* A double-wide MLP is per layer: layers 0..7 declare gate_up [2F, E] and down [E, F] with F =
 * intermediate_size, layers 8..11 [4F, E] and [E, 2F], and every activation op carries its layer's
 * width. The narrow layers use the `gate_up` and `ffn` buffers and the wide ones `gate_up_wide`
 * and `ffn_wide`, each at its own row width. */
TEST(a_gemma4_e2b_form_declares_the_double_wide_layers) {
    RadBuilder b;
    RadModelMeta meta = gemma4_e2b_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const int64_t E = meta.n_embd, F = meta.n_ff;
    for (int l : { 0, 7, 8, 11 }) {
        const int64_t w = l >= 8 ? 2 * F : F;
        const RecWeight* gu = find_w(b, "blk." + std::to_string(l) + ".ffn_gate_up.weight");
        const RecWeight* dn = find_w(b, "blk." + std::to_string(l) + ".ffn_down.weight");
        REQUIRE(gu != nullptr && dn != nullptr);
        CHECK_EQ(gu->d.shape[0], 2 * w);
        CHECK_EQ(gu->d.shape[1], E);
        CHECK_EQ(gu->d.n_row_parts, 2);
        CHECK_EQ(gu->d.row_parts[0], w);
        CHECK_EQ(gu->d.row_parts[1], w);
        CHECK_EQ(dn->d.shape[0], E);
        CHECK_EQ(dn->d.shape[1], w);
    }
    for (int l = 0; l < 12; ++l) {
        const RecOp* act = find_op(b, "gelu_mul", l);
        REQUIRE(act != nullptr);
        CHECK_EQ(pint(act, "n"), l >= 8 ? 2 * F : F);
    }
    /* The narrow and the wide layers keep their own pair, since a GEMM reads its activation packed. */
    const RecBuf* gu = find_b(b, "gate_up");
    const RecBuf* ff = find_b(b, "ffn");
    const RecBuf* guw = find_b(b, "gate_up_wide");
    const RecBuf* ffw = find_b(b, "ffn_wide");
    REQUIRE(gu != nullptr && ff != nullptr && guw != nullptr && ffw != nullptr);
    CHECK_EQ(gu->d.shape[0], ctx.max_tok);
    CHECK_EQ(gu->d.shape[1], 2 * F);
    CHECK_EQ(ff->d.shape[1], F);
    CHECK_EQ(guw->d.shape[1], 4 * F);
    CHECK_EQ(ffw->d.shape[1], 2 * F);

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch, .world = 1 };
    gemma4_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
}

/* A world the heads do not divide over, and a model whose every layer shares KV, are refused. */
TEST(the_gemma4_e4b_form_is_refused_where_it_cannot_run) {
    {
        RadBuilder b;
        RadModelMeta meta = gemma4_e4b_meta();
        RadBuildCtx  ctx  = build_ctx(0, 3);
        CHECK(stderr_of_rc([&] { return gemma4_bf16::declare(&b, &meta, &ctx); }) != RAD_OK);
    }
    {
        const char* vals[] = { "1024", "512", "16", "10000", "1000000", "0.25", "12", "30", "6", "0",
                               "64" };
        RadBuilder b;
        RadModelMeta meta = gemma4_e4b_meta();
        meta.kv_val = vals;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_E_INVAL);
    }
}

/* The 31B's linears on the asymmetric grids, u4 (a4) or u5 codes with a u8 zero point and an f16
 * scale per [1 x 128], at one and two ranks: each is two GEMMs over the codes and the joined
 * (scale, zero) view, up to 64 rows the input cast to f16 for w4a16_asym / w5a16, above that
 * quant_act_i8g's int8 codes for w4a8i8_asym / w5a8i8. q|k|v stay one GEMM; o_proj and down_proj
 * are cut along K, in whole scale groups. A decode step issues only the f16 band, a step of 200
 * rows only the int8 one. The residual junctions are fused (the next case covers them). */
TEST(the_gemma4_31b_form_reads_asymmetric_int4_and_int5_linears) {
    static const int32_t tall_out[] = { 99, 199 };
    for (const uint32_t cdt : { (uint32_t)RAD_U4, (uint32_t)RAD_U5 }) {
        const bool five = cdt == RAD_U5;
        const std::string dec = five ? "w5a16" : "w4a16_asym", pre = five ? "w5a8i8" : "w4a8i8_asym";
        RadEncoding e = rad_enc_affine(cdt, RAD_F16, 1, 128);
        rad_enc_add_plane(&e, "zero", RAD_U8, 1, 128);
        for (int world : { 1, 2 }) {
            for (int rank = 0; rank < world; ++rank) {
                RadBuilder b;
                b.encs = { { "attn_qkv.", e }, { "attn_o.", e }, { "ffn_gate_up.", e },
                           { "ffn_down.", e } };
                RadModelMeta meta = gemma4_meta();
                RadBuildCtx  ctx  = build_ctx(rank, world);
                CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
                CHECK(gemma4_bf16::g_model[rank].fuse);
                int n_dec = 0, n_pre = 0, other = 0;
                for (const RecOp& o : b.ops) {
                    if (o.op != "gemm_nt_q") continue;
                    const std::string dt = pstr(&o, "dtype");
                    const RecParam* m = param(&o, "M");
                    REQUIRE(m != nullptr);
                    if (dt == dec) {
                        ++n_dec;
                        CHECK_EQ(m->ihi, 64LL);
                    } else if (dt == pre) {
                        ++n_pre;
                        CHECK_EQ(m->ihi, (long long)ctx.max_tok);
                    } else {
                        ++other;
                    }
                    CHECK_EQ(pint(&o, "group"), 128);
                    CHECK(param(&o, "pb1") == nullptr);
                }
                /* twelve layers of q|k|v, o, gate_up and down */
                CHECK_EQ(n_dec, 48);
                CHECK_EQ(n_pre, 48);
                CHECK_EQ(other, 0);
                CHECK_EQ(count_op(b, "quant_act_i8g"), 48);
                int codes = 0, joined = 0;
                for (const RecWeight& w : b.weights) {
                    if (!w.d.planes) continue;
                    if (std::string(w.d.planes) == "codes") {
                        ++codes;
                        CHECK_EQ(w.d.dtype, cdt);
                    } else if (std::string(w.d.planes) == "scale,zero") {
                        ++joined;
                        CHECK_EQ(w.d.dtype, (uint32_t)RAD_F16);
                    }
                }
                CHECK_EQ(codes, 48);
                CHECK_EQ(joined, 48);
                /* o_proj of a sliding (0) and a global (5) layer, and down_proj: this rank's K */
                const std::initializer_list<std::pair<std::string, int64_t>> cut = {
                    { "blk.0.attn_o", 32 * 256 }, { "blk.5.attn_o", 32 * 512 },
                    { "blk.0.ffn_down", 21504 } };
                for (const auto& [base, k] : cut)
                    for (const char* v : { ".weight", ".scale" }) {
                        const RecWeight* w = find_w(b, base + v);
                        REQUIRE(w != nullptr);
                        CHECK_EQ(w->d.shard, (int)RAD_SHARD_COL);
                        CHECK_EQ(w->d.shape[0], (int64_t)5376);
                        CHECK_EQ(w->d.shape[1], k / world);
                        CHECK_EQ(w->d.shape[1] % 128, (int64_t)0);
                    }

                RadBatch tall = prefill_batch();
                tall.n_tok = 200; tall.max_q_len = 100; tall.max_ctx_len = 100;
                tall.out_ids = tall_out;
                for (const RadBatch& batch : { decode_batch(), tall }) {
                    RadCtx c; c.b = &b; c.rank = rank; c.world = world; c.batch = &batch;
                    gemma4_bf16::step(&c, &batch);
                    CHECK_LINT(b, &c);
                    CHECK(c.step_fail.empty());
                    const std::string want = batch.n_tok <= 64 ? dec : pre;
                    int gemms = 0, casts = 0;
                    for (const auto& i : c.issues) {
                        const RecOp* o = b.op_of(i.op);
                        if (!o) continue;
                        if (o->op == "gemm_nt_q") {
                            ++gemms;
                            CHECK_EQ(pstr(o, "dtype"), want);
                        }
                        if (o->op == "cast" && pstr(o, "to") == "f16") ++casts;
                    }
                    CHECK_EQ(gemms, 48);
                    CHECK_EQ(casts, batch.n_tok <= 64 ? 48 : 0);
                }
            }
        }
    }
}

/* Gemma 4 12B (`gemma4_unified`, served here through the arch id's alias) on the same twelve
 * layers: width 3840, 16 query heads over 8 sliding KV heads and ONE global KV head, its layer
 * schedule from layer_types. */
const char* k_gem12_keys[] = {
    "model_type", "sliding_window", "global_head_dim", "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "attention_k_eq_v", "hidden_size_per_layer_input", "layer_types",
};
const char* k_gem12_vals[] = {
    "gemma4_unified", "1024", "512", "1", "10000", "1000000", "0.25", "0", "30", "1", "0",
    "sliding_attention sliding_attention sliding_attention sliding_attention sliding_attention "
    "full_attention sliding_attention sliding_attention sliding_attention sliding_attention "
    "sliding_attention full_attention",
};

RadModelMeta gemma4_12b_meta() {
    RadModelMeta m = gemma4_meta();
    m.name      = "test-gemma4-12b-form";
    m.n_embd    = 3840;
    m.n_head    = 16;
    m.n_head_kv = 8;
    m.n_ff      = 15360;
    m.n_kv      = (int)(sizeof(k_gem12_keys) / sizeof(k_gem12_keys[0]));
    m.kv_key    = k_gem12_keys;
    m.kv_val    = k_gem12_vals;
    return m;
}

/* At one rank the global class is 16 q heads over one KV head (gqa 16). At two and four the
 * sliding heads split (4 and 2 a rank, gqa 2) and the global head is replicated: each rank keeps
 * a share of the q heads (gqa 8 and 4) and the whole k row (rows 0..511), as one span or as the k
 * part of a fused q|k. Each layer's attention reads its own cache in its class's group, and a
 * step reduces the lookup and two a layer. */
TEST(the_gemma4_12b_form_declares_and_steps_at_one_two_and_four_ranks) {
    const int64_t L = 12, hs = 256, hg = 512;
    for (bool fused : { false, true }) {
        for (int world : { 1, 2, 4 }) {
            for (int rank = 0; rank < world; ++rank) {
                RadBuilder b;
                if (fused) {
                    const RadEncoding e = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
                    b.encs = { { "attn_qkv.", e }, { "attn_o.", e }, { "ffn_gate_up.", e },
                               { "ffn_down.", e } };
                }
                RadModelMeta meta = gemma4_12b_meta();
                RadBuildCtx  ctx  = build_ctx(rank, world);
                CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
                CHECK_EQ(gemma4_bf16::g_model[rank].fuse, fused);
                REQUIRE_EQ((int)b.kv_groups.size(), 2);
                CHECK_EQ(b.kv_groups[0].d.n_head_kv, 8 / world);
                CHECK_EQ(b.kv_groups[0].d.head_dim, hs);
                CHECK_EQ(b.kv_groups[1].d.n_head_kv, 1);
                CHECK_EQ(b.kv_groups[1].d.head_dim, hg);
                const int64_t nq = 16 / world;
                if (!fused) {
                    const RecOp* as = find_op(b, "attn_paged", 0);
                    const RecOp* ag = find_op(b, "attn_paged", 5);
                    REQUIRE(as != nullptr && ag != nullptr);
                    CHECK_EQ(pint(as, "head_dim"), hs);
                    CHECK_EQ(pint(as, "n_head"), nq);
                    CHECK_EQ(pint(as, "gqa"), 2);
                    CHECK_EQ(pint(as, "window"), 1024);
                    CHECK_EQ(pint(ag, "head_dim"), hg);
                    CHECK_EQ(pint(ag, "n_head"), nq);
                    CHECK_EQ(pint(ag, "gqa"), 16 / world);
                    CHECK_EQ(pint(ag, "window"), 0);
                    const RecWeight* q0 = find_w(b, "blk.0.attn_q.weight");
                    const RecWeight* k0 = find_w(b, "blk.0.attn_k.weight");
                    const RecWeight* q5 = find_w(b, "blk.5.attn_q.weight");
                    const RecWeight* k5 = find_w(b, "blk.5.attn_k.weight");
                    REQUIRE(q0 && k0 && q5 && k5);
                    CHECK_EQ(q0->d.shape[0], nq * hs); CHECK_EQ(q0->d.shape[1], 3840);
                    CHECK_EQ(k0->d.shape[0], (8 / world) * hs);
                    CHECK_EQ(q5->d.shape[0], nq * hg);
                    CHECK_EQ(k5->d.shape[0], hg);
                    CHECK(find_w(b, "blk.5.attn_v.weight") == nullptr);   /* k == v */
                    CHECK(!k0->span_set);                                 /* an even share */
                    CHECK_EQ(k5->span_set, world > 1);
                    if (world > 1) { CHECK_EQ(k5->span_lo, 0); CHECK_EQ(k5->span_hi, hg); }
                } else {
                    const RecWeight* s0 = find_w(b, "blk.0.attn_qkv.weight");
                    REQUIRE(s0 != nullptr);
                    CHECK_EQ(s0->n_part_spans, 0);                        /* even shares */
                    for (const char* v : { "weight", "scale" }) {
                        const RecWeight* w = find_w(b, std::string("blk.5.attn_qkv.") + v);
                        REQUIRE(w != nullptr);
                        if (world == 1) { CHECK_EQ(w->n_part_spans, 0); continue; }
                        REQUIRE_EQ(w->n_part_spans, 2);                   /* q and k, no v */
                        CHECK_EQ(w->part_lo[0], rank * nq * hg);
                        CHECK_EQ(w->part_hi[0], (rank + 1) * nq * hg);
                        CHECK_EQ(w->part_total[0], 16 * hg);
                        CHECK_EQ(w->part_lo[1], 0);
                        CHECK_EQ(w->part_hi[1], hg);
                        CHECK_EQ(w->part_total[1], hg);
                    }
                }

                RadBatch batch = decode_batch();
                RadCtx c; c.b = &b; c.rank = rank; c.world = world; c.batch = &batch;
                gemma4_bf16::step(&c, &batch);
                CHECK_LINT(b, &c);
                CHECK(c.step_fail.empty());
                std::vector<std::pair<int, int64_t>> kv;
                for (const auto& i : c.issues) {
                    const RecOp* o = b.op_of(i.op);
                    if (!o || (o->op != "attn_paged" && o->op != "attn_paged_qrow")) continue;
                    REQUIRE(i.opd.size() > 1);
                    CHECK_EQ((int)i.opd[1].kind, (int)RAD_OPK_KV);
                    kv.push_back({ (int)i.opd[1].handle, (int64_t)i.opd[1].offset });
                }
                REQUIRE_EQ((int)kv.size(), (int)L);
                for (int l = 0; l < (int)L; ++l) {
                    CHECK_EQ(kv[(size_t)l].first, (l % 6 == 5) ? 2 : 1);
                    CHECK_EQ(kv[(size_t)l].second, (int64_t)l);
                }
                const std::vector<std::string> ops = issued_ops(b, c);
                CHECK_EQ(count_issued(ops, "all_reduce") + count_issued(ops, "ar_rmsnorm_quant_i8"),
                         world > 1 ? (int)(1 + 2 * L) : 0);
            }
        }
    }
}

TEST(gemma4_attention_shapes_for_both_classes) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    /* layer 0 (sliding): q 32x256, k/v 16x256, windowed. */
    {
        const RecOp* at = find_op(b, "attn_paged", 0);
        CHECK(at != nullptr);
        CHECK_EQ(pint(at, "head_dim"), 256);
        CHECK_EQ(pint(at, "gqa"),      2);
        CHECK_EQ(pint(at, "window"),   1024);
        CHECK_EQ(pint(at, "n_head"),   32);
        CHECK_EQ(pint(at, "block_size"), 16);
        CHECK_EQ(pstr(at, "q_dtype"), std::string("bf16"));
    }
    /* layer 5 (the first global, (5+1)%6==0): q 32x512, k 4x512, windowless. */
    {
        const RecOp* at = find_op(b, "attn_paged", 5);
        CHECK(at != nullptr);
        CHECK_EQ(pint(at, "head_dim"), 512);
        CHECK_EQ(pint(at, "gqa"),      8);
        CHECK_EQ(pint(at, "window"),   0);
        CHECK_EQ(pint(at, "n_head"),   32);
    }

    /* rope: sliding is standard over the full 256 (theta 1e4); global is proportional over 128 of
     * 512 (the "head" ladder, theta 1e6). Ropes are declared q, k per layer, so index =
     * 2*layer + (q ? 0 : 1). */
    {
        const RecOp* r  = find_op(b, "rope", 0);   /* layer 0, q */
        CHECK(r != nullptr);
        CHECK_EQ(pint(r, "head_dim"),   256);
        CHECK_EQ(pint(r, "rotary_dim"), 256);
        CHECK_NEAR(pflt(r, "theta"),    1e4, 1.0);
        CHECK_EQ(pstr(r, "ladder"), std::string("rotary"));
        CHECK_EQ(pint(r, "n_head"),    32);

        const RecOp* rg = find_op(b, "rope", 10);  /* layer 5, q */
        CHECK(rg != nullptr);
        CHECK_EQ(pint(rg, "head_dim"),   512);
        CHECK_EQ(pint(rg, "rotary_dim"), 128);
        CHECK_NEAR(pflt(rg, "theta"),    1e6, 1.0);
        CHECK_EQ(pstr(rg, "ladder"), std::string("head"));
        CHECK_EQ(pint(rg, "n_head"),    32);

        /* the k rope is a separate buffer, run as a q-only form over the k heads: n_head is the
         * count to rotate (n_head_kv = 4) and n_head_kv is 0. */
        const RecOp* rkg = find_op(b, "rope", 11); /* layer 5, k */
        CHECK(rkg != nullptr);
        CHECK_EQ(pint(rkg, "n_head"),    4);
        CHECK_EQ(pint(rkg, "n_head_kv"), 0);
    }

    /* the weightless value norm reads 256 dims on a sliding layer and 512 on a global one, and
     * the global layer has no v_proj (v is k). */
    {
        const RecOp* vn  = find_op(b, "rmsnorm_p", 0);   /* layer 0 */
        CHECK(vn != nullptr);
        CHECK_EQ(pint(vn, "n"), 256);
        const RecOp* vng = find_op(b, "rmsnorm_p", 5);   /* layer 5 */
        CHECK(vng != nullptr);
        CHECK_EQ(pint(vng, "n"), 512);
    }
    CHECK(find_w(b, "blk.0.attn_v.weight") != nullptr);  /* sliding has a v projection */
    CHECK(find_w(b, "blk.5.attn_v.weight") == nullptr);  /* global: k == v */
    CHECK(find_w(b, "blk.5.attn_q.weight") != nullptr);
    CHECK(find_w(b, "blk.5.attn_o.weight") != nullptr);
}

TEST(gemma4_final_norm_softcap_and_vocab) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const RecOp* cap = find_op(b, "logit_softcap", 0);
    CHECK(cap != nullptr);
    CHECK_EQ(pint(cap, "n"), 262144);
    CHECK_NEAR(pflt(cap, "cap"), 30.0, 1e-9);
    CHECK_EQ(pstr(cap, "dtype"), std::string("f32"));

    const RecOp* lg = find_op(b, "logits_gemm", 0);
    CHECK(lg != nullptr);
    CHECK_EQ(pint(lg, "n_vocab"), 262144);
    CHECK_EQ(pint(lg, "n_embd"),  5376);

    const RecWeight* emb = find_w(b, "token_embd.weight");
    CHECK(emb != nullptr);
    CHECK_EQ(emb->d.shape[0], 262144);
    CHECK_EQ(emb->d.shape[1], 5376);
    CHECK_EQ(emb->d.shard, RAD_SHARD_ROW);

    /* lm_head is tied to the embedding. */
    CHECK(find_map(b, "output.weight") != nullptr);

    /* The embedding is scaled by sqrt(n_embd) at lookup (as the reference's build_inp_embd and
     * HF's ScaledWordEmbedding do), not baked into the table, so the table stays tied to the head.
     * HF multiplies by `embed_scale.to(bf16)`, so sqrt(5376) = 73.32 is 73.5. */
    const RecOp* el = find_op(b, "embed_lookup", 0);
    CHECK(el != nullptr);
    CHECK_NEAR(pflt(el, "mult"), 73.5, 1e-9);
}

TEST(gemma4_every_op_declares_the_buffers_it_touches) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
}

/* attention_k_eq_v decides whether global layers have a v_proj: 1 (31B) reads V from K, 0 declares
 * one, and absence is refused. */
TEST(gemma4_global_v_follows_attention_k_eq_v) {
    for (const char* v : { "1", "0", "" }) {
        std::vector<const char*> keys(std::begin(k_gem4_keys), std::end(k_gem4_keys));
        std::vector<const char*> vals(std::begin(k_gem4_vals), std::end(k_gem4_vals));
        if (*v) vals.back() = v;
        else { keys.pop_back(); vals.pop_back(); }
        RadModelMeta meta = gemma4_meta();
        meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        const int rc = gemma4_bf16::declare(&b, &meta, &ctx);
        if (!*v) { CHECK(rc != RAD_OK); continue; }
        CHECK_OK(rc);
        CHECK(find_w(b, "blk.0.attn_v.weight") != nullptr);              /* sliding: always */
        CHECK_EQ(find_w(b, "blk.5.attn_v.weight") != nullptr, *v == '0');  /* global */
    }
}

TEST(gemma4_every_weight_has_a_name_map) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(a_gemma4_decode_step_runs_both_classes_in_order) {
    RadBuilder b;
    b.refuse.insert("qkv_norm_rope");   /* force the unfused prologue; fused is checked below */
    b.refuse.insert("qkv_norm_rope_kv_store");
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 12; ++l) {
        if ((l + 1) % 6 == 0) {
            /* global: k == v, so no v gemm; the weightless v norm derives from k. */
            for (const char* s : { "rmsnorm", "gemm_nt", "gemm_nt",
                                   "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope",
                                   "kv_store", "attn_paged", "gemm_nt", "rmsnorm", "add" })
                want.push_back(s);
        } else {
            /* sliding: a v gemm feeds the weightless v norm. */
            for (const char* s : { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt",
                                   "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope",
                                   "kv_store", "attn_paged", "gemm_nt", "rmsnorm", "add" })
                want.push_back(s);
        }
        for (const char* s : { "rmsnorm", "gemm_nt", "gelu_mul", "gemm_nt", "rmsnorm", "add" })
            want.push_back(s);
        want.push_back("scale");   /* per-layer scalar over the residual stream */
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The sampled-row ops band on n_out; the attention bands on q_len. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "attn_paged")
            CHECK_EQ(i.n, (int64_t)batch.max_q_len);
        else if (o->op == "gather_rows" || o->op == "logits_gemm" || o->op == "logit_softcap")
            CHECK_EQ(i.n, (int64_t)batch.n_out);
    }
}

/* The w4a8 trunk fuses its residual junctions: the first block's entry, each attention ->
 * feed-forward and each feed-forward -> next attention is one `rmsnorm_quant_i8` (post-norm, add,
 * layer scalar, norm and quantiser), and q, k and v read one quantised pair. The last layer goes to
 * the final norm unfused. Without the fused op the whole chain stays unfused, still with the shared
 * q/k/v pair. */
static std::vector<std::string> gemma4_w4_step(const std::set<std::string>& refuse, int world = 1,
                                               bool prefetch_off = false) {
    RadBuilder b;
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
               { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
    for (const auto& r : refuse) b.refuse.insert(r);
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx(0, world);
    ctx.weight_prefetch_off = prefetch_off ? 1 : 0;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch, .world = world };
    gemma4_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    return issued_ops(b, c);
}

/* At two ranks the all-reduce before each junction is folded into it; only the embedding's and
 * the last feed-forward's (which feeds the final norm, not a quantiser) remain standalone. Without
 * the fused collective, junctions keep their preceding all-reduces. Beside each reduction a
 * prefetch of the next GEMM's weight (the lm_head after the last), and down_proj's beside the
 * attention (prefetch_beside). */
TEST(a_gemma4_w4a8_step_at_two_ranks_folds_the_all_reduce_into_the_junction) {
    std::vector<std::string> want = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "prefetch", "qkv_norm_rope_kv_store", "attn_paged", "quant_act_i8",
                               "gemm_nt_q", "prefetch", "ar_rmsnorm_quant_i8", "gemm_nt_q",
                               "gelu_mul_quant_i8", "gemm_nt_q" })
            want.push_back(s);
        want.push_back("prefetch");
        if (l < 11) want.push_back("ar_rmsnorm_quant_i8");
        else for (const char* s : { "all_reduce", "rmsnorm", "add", "scale" }) want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({}, 2)), join(want));

    std::vector<std::string> plain = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) plain.push_back("gemm_nt_q");
        for (const char* s : { "prefetch", "qkv_norm_rope_kv_store", "attn_paged", "quant_act_i8",
                               "gemm_nt_q", "prefetch", "all_reduce", "rmsnorm_quant_i8", "gemm_nt_q",
                               "gelu_mul_quant_i8", "gemm_nt_q" })
            plain.push_back(s);
        plain.push_back("prefetch");
        plain.push_back("all_reduce");
        if (l < 11) plain.push_back("rmsnorm_quant_i8");
        else for (const char* s : { "rmsnorm", "add", "scale" }) plain.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        plain.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({ "ar_rmsnorm_quant_i8" }, 2)), join(plain));
}

/* --weight-prefetch off (RadBuildCtx::weight_prefetch_off) declares none of the reads beside the
 * reductions, and the rest of the step is the step with them. */
TEST(a_gemma4_step_under_weight_prefetch_off_issues_no_prefetch) {
    const std::vector<std::string> on = gemma4_w4_step({}, 2), off = gemma4_w4_step({}, 2, true);
    CHECK_EQ(count_issued(on, "prefetch"), 36);
    CHECK_EQ(count_issued(off, "prefetch"), 0);
    std::vector<std::string> rest;
    for (const std::string& s : on)
        if (s != "prefetch") rest.push_back(s);
    CHECK_EQ(join(off), join(rest));
}

TEST(a_gemma4_w4a8_step_fuses_its_residual_junctions) {
    std::vector<std::string> want = { "embed_lookup", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "qkv_norm_rope_kv_store",
                               "attn_paged", "quant_act_i8", "gemm_nt_q", "rmsnorm_quant_i8",
                               "gemm_nt_q", "gelu_mul_quant_i8", "gemm_nt_q" })
            want.push_back(s);
        if (l < 11) want.push_back("rmsnorm_quant_i8");
        else for (const char* s : { "rmsnorm", "add", "scale" }) want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({})), join(want));
}

TEST(a_gemma4_w4a8_step_without_the_fused_ops_keeps_the_chain_and_one_qkv_quantiser) {
    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        want.push_back("rmsnorm");
        want.push_back("quant_act_i8");
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope", "kv_store",
                               "attn_paged", "quant_act_i8", "gemm_nt_q", "rmsnorm", "add",
                               "rmsnorm", "quant_act_i8", "gemm_nt_q", "gelu_mul", "quant_act_i8",
                               "gemm_nt_q", "rmsnorm", "add", "scale" })
            want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({ "rmsnorm_quant_i8", "qkv_norm_rope", "qkv_norm_rope_kv_store",
                                   "gelu_mul_quant_i8" })), join(want));
}

/* =============================== gemma4: the assistant (MTP) head =============================== */

namespace {
/* The draft passes the core stages for a head with no cache of its own: one row per drafting
 * sequence. Four sequences; the history pass marks every row as a draft row. */
static const int32_t kHeadRows[] = { 0, 1, 2, 3 };
RadBatch head_batch(int pass) {
    RadBatch b{};
    b.phase         = RAD_PHASE_DECODE;
    b.n_tok         = 4;
    b.n_seq         = 4;
    b.max_q_len     = 1;
    b.max_ctx_len   = 100;
    b.draft_pass    = pass;
    if (pass < 0) {
        b.n_out         = 4;
        b.out_ids       = kHeadRows;
        b.n_draft_out   = 4;
        b.draft_out_ids = kHeadRows;
    }
    return b;
}
}  /* namespace */

TEST(gemma4_declares_no_head_without_one_or_without_speculation) {
    {
        RadBuilder b;
        RadModelMeta meta = gemma4_meta();
        RadBuildCtx  ctx  = build_ctx();
        ctx.max_spec = 3;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK(find_w(b, "mtp.pre_proj.weight") == nullptr);
    }
    {
        RadBuilder b;
        RadModelMeta meta = gemma4_draft_meta();
        RadBuildCtx  ctx  = build_ctx();      /* max_spec 0 */
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK(find_w(b, "mtp.pre_proj.weight") == nullptr);
        CHECK_EQ(count_op(b, "attn_paged"), 12);
    }
}

TEST(gemma4_assistant_is_a_serial_head_with_no_cache_of_its_own) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
    CHECK_EQ(b.drafter.depth, 3);
    CHECK_EQ(b.drafter_name, std::string("mtp"));
    CHECK_EQ((int)b.drafter.proposal, 0);         /* proposes through the sampler chain */
    CHECK(b.drafter_shared);
    CHECK_EQ(b.drafter_pos, gemma4_bf16::kHeadPosOffset);

    /* It binds no layer and writes no K or V; only the trunk's twelve bindings exist. */
    CHECK_EQ((int)b.binds.size(), 12);
    CHECK_EQ((int)b.kv_groups.size(), 2);

    /* Four head attention layers beside the trunk's twelve, none storing. */
    CHECK_EQ(count_op(b, "attn_paged"), 16);
    CHECK_EQ(count_op(b, "kv_store") + count_op(b, "qkv_norm_rope_kv_store"), 12);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);    /* trunk only; head logits are not capped */
    CHECK_EQ(count_op(b, "logits_gemm"), 2);

    for (int l = 0; l < 4; ++l) {
        const RecOp* at = find_op(b, "attn_paged", 12 + l);
        REQUIRE(at != nullptr);
        const bool full = (l == 3);
        CHECK_EQ(pint(at, "head_dim"), full ? 512 : 256);
        CHECK_EQ(pint(at, "gqa"), full ? 8 : 2);
        CHECK_EQ(pint(at, "window"), full ? 0 : 1024);
        CHECK_EQ(pint(at, "causal"), 1);
        CHECK_NEAR(pflt(at, "scale"), 1.0, 1e-12);
        const RecParam* ql = param(at, "q_len");
        REQUIRE(ql != nullptr);
        CHECK_EQ(ql->ival, 1);                        /* one row per sequence: decode band only */
        CHECK_EQ(ql->ihi, 1);
    }

    /* Weights: q-only attention, the two projections and the head's own tied table. */
    const RecWeight* wq0 = find_w(b, "mtp.blk.0.attn_q.weight");
    const RecWeight* wq3 = find_w(b, "mtp.blk.3.attn_q.weight");
    REQUIRE(wq0 != nullptr); REQUIRE(wq3 != nullptr);
    CHECK_EQ(wq0->d.shape[0], 32 * 256); CHECK_EQ(wq0->d.shape[1], 1024);
    CHECK_EQ(wq3->d.shape[0], 32 * 512); CHECK_EQ(wq3->d.shape[1], 1024);
    const RecWeight* wo3 = find_w(b, "mtp.blk.3.attn_o.weight");
    REQUIRE(wo3 != nullptr);
    CHECK_EQ(wo3->d.shape[0], 1024); CHECK_EQ(wo3->d.shape[1], 32 * 512);
    CHECK(find_w(b, "mtp.blk.0.attn_k.weight") == nullptr);
    CHECK(find_w(b, "mtp.blk.0.attn_v.weight") == nullptr);
    const RecWeight* gu = find_w(b, "mtp.blk.1.ffn_gate_up.weight");
    REQUIRE(gu != nullptr);
    CHECK_EQ(gu->d.shape[0], 2 * 8192); CHECK_EQ(gu->d.shape[1], 1024);
    const RecWeight* pre  = find_w(b, "mtp.pre_proj.weight");
    const RecWeight* post = find_w(b, "mtp.post_proj.weight");
    REQUIRE(pre != nullptr); REQUIRE(post != nullptr);
    CHECK_EQ(pre->d.shape[0], 1024);  CHECK_EQ(pre->d.shape[1], 2 * 5376);
    CHECK_EQ(post->d.shape[0], 5376); CHECK_EQ(post->d.shape[1], 1024);
    CHECK_EQ(pre->d.shard, RAD_SHARD_NONE);
    const RecWeight* lm = find_w(b, "mtp.output.weight");
    REQUIRE(lm != nullptr);
    CHECK_EQ(lm->d.shape[0], 262144); CHECK_EQ(lm->d.shape[1], 1024);
    CHECK_EQ(lm->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(lm->d.access, RAD_ACCESS_VOCAB);

    /* Checkpoint names merged under `draft.`: the head's table is its own embedding (no
     * fallback), and gate and up are fused like the trunk's. */
    const RecMap* mlm = find_map(b, "mtp.output.weight");
    REQUIRE(mlm != nullptr);
    CHECK_EQ(mlm->src[0], std::string("draft.model.embed_tokens.weight"));
    const RecMap* mgu = find_map(b, "mtp.blk.1.ffn_gate_up.weight");
    REQUIRE(mgu != nullptr);
    CHECK_EQ(mgu->mode, RAD_MAP_CONCAT);
    CHECK_EQ(mgu->src[0], std::string("draft.model.layers.1.mlp.gate_proj.weight"));
    CHECK_EQ(mgu->src[1], std::string("draft.model.layers.1.mlp.up_proj.weight"));
    const RecMap* mpre = find_map(b, "mtp.pre_proj.weight");
    REQUIRE(mpre != nullptr);
    CHECK_EQ(mpre->src[0], std::string("draft.pre_projection.weight"));

    /* The entry embeds through the trunk's table with the trunk's scale, sqrt(5376) rounded to
     * bf16 as the reference's embedding module holds it. */
    const RecOp* el = find_op(b, "embed_lookup", 1);
    REQUIRE(el != nullptr);
    CHECK_EQ(pint(el, "n_embd"), 5376);
    CHECK_NEAR(pflt(el, "mult"), 73.5, 1e-9);
    CHECK(el->w.size() == 1 && find_op(b, "embed_lookup", 0)->w == el->w);

    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(gemma4_head_passes_read_the_last_trunk_layer_of_their_class) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    for (int pass : { -1, 2 }) {
        RadBatch batch = head_batch(pass);
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());

        std::vector<std::string> want;
        want.push_back(pass < 0 ? "gather_rows" : "cast");   /* the trunk's h or the head's h' */
        for (const char* s : { "embed_lookup", "cast", "gemm_nt" }) want.push_back(s);
        for (int l = 0; l < 4; ++l)
            for (const char* s : { "rmsnorm", "gemm_nt", "rmsnorm", "rope", "attn_paged", "gemm_nt",
                                   "rmsnorm", "add", "rmsnorm", "gemm_nt", "gelu_mul", "gemm_nt",
                                   "rmsnorm", "add", "scale" })
                want.push_back(s);
        for (const char* s : { "rmsnorm", "logits_gemm", "gemm_nt" }) want.push_back(s);
        CHECK_EQ(join(issued_ops(b, c)), join(want));

        /* Attention reads the trunk's cache: head layers 0..2 the last sliding trunk layer (10),
         * layer 3 the last full one (11). Groups: 1 sliding, 2 full. */
        int seen = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "attn_paged") continue;
            REQUIRE(i.opd.size() > 1);
            CHECK_EQ((int)i.opd[1].kind, (int)RAD_OPK_KV);
            CHECK_EQ((int)i.opd[1].handle, seen < 3 ? 1 : 2);
            CHECK_EQ((long long)i.opd[1].offset, seen < 3 ? 10LL : 11LL);
            CHECK_EQ(i.n, 1);
            ++seen;
        }
        CHECK_EQ(seen, 4);
    }
}

TEST(gemma4_head_at_two_ranks_reduces_before_each_post_norm) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx(0, 2);
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const RecWeight* wq0 = find_w(b, "mtp.blk.0.attn_q.weight");
    REQUIRE(wq0 != nullptr);
    CHECK_EQ(wq0->d.shape[0], 16 * 256);
    const RecWeight* lm = find_w(b, "mtp.output.weight");
    REQUIRE(lm != nullptr);
    CHECK_EQ(lm->d.shape[0], 131072);

    RadBatch batch = head_batch(-1);
    RadCtx c{ .b = &b, .batch = &batch, .world = 2 };
    gemma4_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());
    const std::vector<std::string> got = issued_ops(b, c);
    std::vector<std::string> want = { "gather_rows", "embed_lookup", "all_reduce", "cast", "gemm_nt" };
    for (int l = 0; l < 4; ++l)
        for (const char* s : { "rmsnorm", "gemm_nt", "rmsnorm", "rope", "attn_paged", "gemm_nt",
                               "all_reduce", "rmsnorm", "add", "rmsnorm", "gemm_nt", "gelu_mul",
                               "gemm_nt", "all_reduce", "rmsnorm", "add", "scale" })
            want.push_back(s);
    for (const char* s : { "rmsnorm", "logits_gemm", "gemm_nt" }) want.push_back(s);
    CHECK_EQ(join(got), join(want));
}

TEST(gemma4_head_refuses_a_history_pass_shaped_for_a_cache_of_its_own) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    RadBatch batch = head_batch(-1);
    batch.n_draft_out = 1;                 /* a history pass with non-drafting rows */
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4_bf16::step(&c, &batch);
    CHECK(!c.step_fail.empty());
    CHECK(c.issues.empty());
}

TEST(gemma4_probe_names_the_heads_operating_depth) {
    RadArchProbe p{};
    RadModelMeta with = gemma4_draft_meta();
    CHECK_OK(gemma4_bf16::probe(&with, &p));
    CHECK_EQ(p.draft_depth, gemma4_bf16::kHeadDepth);
    CHECK_EQ(p.draft_depth_fixed, 0);
    CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);   /* with the head and without one */
    RadArchProbe q{};
    RadModelMeta without = gemma4_meta();
    CHECK_OK(gemma4_bf16::probe(&without, &q));
    CHECK_EQ(q.draft_depth, 0);
}

/* The 12B's assistant: the same head under the name Gemma4UnifiedAssistantForCausalLM, width
 * 1024 over a 3840 trunk, 16 q heads over the trunk's 8 sliding and one global KV head. */
const char* k_gem12d_keys[] = {
    "model_type", "sliding_window", "global_head_dim", "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "attention_k_eq_v", "hidden_size_per_layer_input", "layer_types",
    "draft.architectures", "draft.backbone_hidden_size", "draft.use_ordered_embeddings",
    "draft.text_config.hidden_size", "draft.text_config.num_hidden_layers",
    "draft.text_config.layer_types", "draft.text_config.num_attention_heads",
    "draft.text_config.num_key_value_heads", "draft.text_config.num_global_key_value_heads",
    "draft.text_config.head_dim", "draft.text_config.global_head_dim",
    "draft.text_config.intermediate_size", "draft.text_config.sliding_window",
    "draft.text_config.vocab_size", "draft.text_config.rms_norm_eps",
    "draft.text_config.rope_parameters.sliding_attention.rope_theta",
    "draft.text_config.rope_parameters.full_attention.rope_theta",
    "draft.text_config.rope_parameters.full_attention.partial_rotary_factor",
};
std::vector<const char*> gem12d_vals() {
    std::vector<const char*> v(std::begin(k_gem12_vals), std::end(k_gem12_vals));
    for (const char* s : { "Gemma4UnifiedAssistantForCausalLM", "3840", "0", "1024", "4",
                           "sliding_attention sliding_attention sliding_attention full_attention",
                           "16", "8", "1", "256", "512", "8192", "1024", "262144", "1e-06", "10000",
                           "1000000", "0.25" })
        v.push_back(s);
    return v;
}

/* Declared at one, two and four ranks: the head's full layer is gqa 16, 8 and 4 over the trunk's
 * one (replicated) global KV head, its sliding layers gqa 2. A pass reads layers 10 and 11. A head
 * stating another global KV count than the trunk's is still refused. */
TEST(the_gemma4_12b_assistant_is_declared_at_one_two_and_four_ranks) {
    const std::vector<const char*> vals = gem12d_vals();
    REQUIRE_EQ(vals.size(), sizeof(k_gem12d_keys) / sizeof(k_gem12d_keys[0]));
    for (int world : { 1, 2, 4 }) {
        for (int rank : { 0, world - 1 }) {
            RadBuilder b;
            RadModelMeta meta = gemma4_12b_meta();
            meta.n_kv = (int)vals.size(); meta.kv_key = k_gem12d_keys; meta.kv_val = vals.data();
            RadBuildCtx ctx = build_ctx(rank, world);
            ctx.max_spec = 3;
            CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
            CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
            CHECK_EQ(b.drafter.depth, 3);
            CHECK_EQ(count_op(b, "attn_paged"), 16);
            const int64_t nq = 16 / world;
            for (int l = 0; l < 4; ++l) {
                const RecOp* at = find_op(b, "attn_paged", 12 + l);
                REQUIRE(at != nullptr);
                CHECK_EQ(pint(at, "n_head"), nq);
                CHECK_EQ(pint(at, "head_dim"), l == 3 ? 512 : 256);
                CHECK_EQ(pint(at, "gqa"), l == 3 ? nq : 2);
            }
            const RecWeight* wq3 = find_w(b, "mtp.blk.3.attn_q.weight");
            const RecWeight* pre = find_w(b, "mtp.pre_proj.weight");
            const RecWeight* post = find_w(b, "mtp.post_proj.weight");
            REQUIRE(wq3 && pre && post);
            CHECK_EQ(wq3->d.shape[0], nq * 512); CHECK_EQ(wq3->d.shape[1], 1024);
            CHECK_EQ(pre->d.shape[0], 1024);     CHECK_EQ(pre->d.shape[1], 2 * 3840);
            CHECK_EQ(post->d.shape[0], 3840);    CHECK_EQ(post->d.shape[1], 1024);
            const RecMap* mpre = find_map(b, "mtp.pre_proj.weight");
            REQUIRE(mpre != nullptr);
            CHECK_EQ(mpre->src[0], std::string("draft.pre_projection.weight"));

            RadBatch batch = head_batch(-1);
            RadCtx c; c.b = &b; c.rank = rank; c.world = world; c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
            CHECK(c.step_fail.empty());
            int seen = 0;
            for (const auto& i : c.issues) {
                const RecOp* o = b.op_of(i.op);
                if (!o || o->op != "attn_paged") continue;
                REQUIRE(i.opd.size() > 1);
                CHECK_EQ((int)i.opd[1].handle, seen < 3 ? 1 : 2);
                CHECK_EQ((long long)i.opd[1].offset, seen < 3 ? 10LL : 11LL);
                ++seen;
            }
            CHECK_EQ(seen, 4);
        }
    }
    std::vector<const char*> two = gem12d_vals();
    two[20] = "2";   /* draft.text_config.num_global_key_value_heads */
    for (int world : { 1, 2 }) {
        RadBuilder b;
        RadModelMeta meta = gemma4_12b_meta();
        meta.n_kv = (int)two.size(); meta.kv_key = k_gem12d_keys; meta.kv_val = two.data();
        RadBuildCtx ctx = build_ctx(0, world);
        ctx.max_spec = 3;
        CHECK_EQ(stderr_of_rc([&] { return gemma4_bf16::declare(&b, &meta, &ctx); }),
                 (int)RAD_E_INVAL);
    }
    RadArchProbe p{};
    RadModelMeta with = gemma4_12b_meta();
    with.n_kv = (int)vals.size(); with.kv_key = k_gem12d_keys; with.kv_val = vals.data();
    CHECK_OK(gemma4_bf16::probe(&with, &p));
    CHECK_EQ(p.draft_depth, gemma4_bf16::kHeadDepth);
}

/* ===================== gemma4: the 26B-A4B, a dense and a routed arm side by side ===================== */

namespace {

/* gemma-4-26B-A4B-it's text geometry cut to six layers (five sliding then one full, the model's
 * 5:1) with the routed arm on. Keys are the checkpoint's flattened text_config, as arch/gemma4
 * reads them; quant is empty, the container this plugin claims. */
const char* k_g4moe_keys[] = {
    "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "global_head_dim",
    "num_global_key_value_heads", "attention_k_eq_v", "vocab_size", "intermediate_size",
    "rms_norm_eps", "sliding_window", "final_logit_softcapping", "hidden_activation", "layer_types",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.sliding_attention.rope_type",
    "rope_parameters.full_attention.rope_theta", "rope_parameters.full_attention.rope_type",
    "rope_parameters.full_attention.partial_rotary_factor", "enable_moe_block", "num_experts",
    "top_k_experts", "moe_intermediate_size",
};
const char* k_g4moe_vals[] = {
    "2816", "16", "8", "256", "512",
    "2", "true", "262144", "2112",
    "1e-06", "1024", "30.0", "gelu_pytorch_tanh",
    "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
    "\"sliding_attention\", \"full_attention\"]",
    "10000.0", "default",
    "1000000.0", "proportional",
    "0.25", "true", "128",
    "8", "704",
};
static_assert(sizeof(k_g4moe_keys) == sizeof(k_g4moe_vals), "one value a key");

RadModelMeta gemma4_moe_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma4";
    m.name        = "test-gemma4-26b-a4b";
    m.quant       = "";
    m.n_layers    = 6;
    m.n_embd      = 2816;
    m.n_head      = 16;
    m.n_head_kv   = 8;
    m.head_dim    = 256;
    m.n_ff        = 2112;
    m.n_vocab     = 262144;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.n_kv        = (int)(sizeof(k_g4moe_keys) / sizeof(k_g4moe_keys[0]));
    m.kv_key      = k_g4moe_keys;
    m.kv_val      = k_g4moe_vals;
    return m;
}

int count_resolved(const RadBuilder& b, const std::string& name) {
    int n = 0;
    for (const auto& o : b.ops) if (o.op == name && o.resolved) ++n;
    return n;
}

}  /* namespace */

TEST(a_gemma4_moe_trunk_declares_both_arms_and_the_joins) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    /* five sliding layers read q from the fused row (attn_paged_qrow); the full one attn_paged */
    CHECK_EQ(count_op(b, "attn_paged_qrow"), 5);
    CHECK_EQ(count_op(b, "attn_paged"),      1);
    CHECK_EQ(count_op(b, "kv_store"),   6);
    /* Per layer, one join after attention and one after the two feed-forward arms. */
    CHECK_EQ(count_resolved(b, "norm_add_norm"),  6);
    CHECK_EQ(count_resolved(b, "join_add_scale"), 6);
    /* The routed arm: experts are bf16 where the container has no encoding for them. */
    CHECK(count_op(b, "moe_gemm") > 0);
    CHECK_EQ(count_op(b, "moe_gemm_q"), 0);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);
    CHECK_EQ(count_op(b, "all_reduce"),    0);   /* world 1 */

    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

/* The sliding group is a window group at the width the attention op reads, the full layers'
 * group is RAD_KV_FULL (arch/gemma4/gemma4.cpp, the KV groups). */
TEST(gemma4_moe_kv_groups_are_window_and_full_and_the_window_rides_the_op) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    /* The sliding group is a window group: the pool caps at what attn_paged_qrow can read. */
    REQUIRE((int)b.kv_groups.size() == 2);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_swa"));
    CHECK_EQ((int)b.kv_groups[0].d.kind, RAD_KV_WINDOW);
    CHECK_EQ(b.kv_groups[0].d.window, 1024);
    CHECK_EQ(b.kv_groups[1].name, std::string("kv_full"));
    CHECK_EQ((int)b.kv_groups[1].d.kind, RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[1].d.window, 0);
    const RecOp* sl = find_op(b, "attn_paged_qrow", 0);
    REQUIRE(sl != nullptr);
    CHECK_EQ(pint(sl, "head_dim"), 256);
    CHECK_EQ(pint(sl, "window"),   1024);
    const RecOp* fu = find_op(b, "attn_paged", 0);
    REQUIRE(fu != nullptr);
    CHECK_EQ(pint(fu, "head_dim"), 512);
    CHECK_EQ(pint(fu, "gqa"),      8);
    CHECK_EQ(pint(fu, "window"),   0);
}

/* Proportional RoPE is not a partial rotary: full layers rotate the whole 512-wide head as NeoX
 * pairs (j, j + 256), with frequencies past pair 63 removed by the YaRN ramp. rotary_dim 128 would
 * pair (j, j + 64), which is wrong. */
TEST(gemma4_full_rope_is_the_whole_head_with_the_frequencies_past_64_pairs_cut) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    int full = 0;
    for (const auto& o : b.ops) {
        if (o.op != "rope" || pint(&o, "head_dim") != 512) continue;
        ++full;
        CHECK_EQ(pint(&o, "rotary_dim"), 512);
        CHECK_EQ(pstr(&o, "mode"), std::string("neox"));
        CHECK_NEAR(pflt(&o, "theta"),     1e6, 1.0);
        CHECK_NEAR(pflt(&o, "yarn_low"),  63.5, 1e-9);
        CHECK_NEAR(pflt(&o, "yarn_high"), 64.0, 1e-9);
        CHECK(pflt(&o, "yarn_factor") >= 1e30);
    }
    CHECK(full > 0);
}

/* Where no kernel library serves a join, the plugin declares and issues the rmsnorm / add chain;
 * libref deliberately has no row for the joins so this path stays reachable (docs/OPS.md). */
TEST(gemma4_falls_back_to_the_chain_where_no_join_resolves) {
    std::vector<std::string> fused, chain;
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) { b.refuse.insert("norm_add_norm"); b.refuse.insert("join_add_scale"); }
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        (refuse ? chain : fused) = issued_ops(b, c);
    }
    CHECK_EQ(count_issued(fused, "norm_add_norm"),  6);
    CHECK_EQ(count_issued(fused, "join_add_scale"), 6);
    CHECK_EQ(count_issued(chain, "norm_add_norm"),  0);
    CHECK_EQ(count_issued(chain, "join_add_scale"), 0);
    for (const auto* v : { &fused, &chain }) {
        CHECK_EQ(count_issued(*v, "attn_paged_qrow"), 5);
        CHECK_EQ(count_issued(*v, "attn_paged"),      1);
    }
    /* The chain is longer by the norms and adds the joins fold in. */
    CHECK(count_issued(chain, "rmsnorm") > count_issued(fused, "rmsnorm"));
    CHECK(count_issued(chain, "add") > count_issued(fused, "add"));
}

/* Both Gemma 4 plugins export the same reply-format declaration. */
TEST(both_gemma4_plugins_export_one_reply_format) {
    RadModelMeta moe = gemma4_moe_meta(), dense = gemma4_meta();
    CHECK(gemma4::chat_format(&moe) == rad::arch::gemma4_chat_format());
    CHECK(gemma4_bf16::chat_format(&dense) == rad::arch::gemma4_chat_format());
}

/* ================== gemma4 + DFlash: the drafter's taps, live where they are written ================== */

namespace {

/* The MoE trunk above with z-lab's DFlash drafter merged: five layers (four sliding, one full)
 * reading the trunk at layers 1, 3 and 5. */
std::vector<const char*> g4df_keys() {
    std::vector<const char*> k(std::begin(k_g4moe_keys), std::end(k_g4moe_keys));
    for (const char* s : { "draft.architectures", "draft.num_hidden_layers", "draft.head_dim",
                           "draft.num_attention_heads", "draft.num_key_value_heads",
                           "draft.intermediate_size", "draft.rms_norm_eps", "draft.rope_theta",
                           "draft.hidden_size", "draft.vocab_size", "draft.block_size",
                           "draft.dflash_config.target_layer_ids",
                           "draft.dflash_config.mask_token_id", "draft.layer_types",
                           "draft.sliding_window" })
        k.push_back(s);
    return k;
}
std::vector<const char*> g4df_vals() {
    std::vector<const char*> v(std::begin(k_g4moe_vals), std::end(k_g4moe_vals));
    for (const char* s : { "[\"DFlashDraftModel\"]", "5", "128", "32", "8", "5632", "1e-06",
                           "1000000.0", "2816", "262144", "16", "[1, 3, 5]", "4",
                           "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                           "\"sliding_attention\", \"full_attention\"]",
                           "2048" })
        v.push_back(s);
    return v;
}

int buf_named(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.bufs.size(); ++i)
        if (b.bufs[i].name == name) return (int)i + 1;
    return 0;
}

struct G4MoeVision {
    std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();

    G4MoeVision() {
        const char* vk[] = { "vision_config.model_type", "vision_config.hidden_size",
                             "vision_config.num_attention_heads", "vision_config.head_dim",
                             "vision_config.intermediate_size", "vision_config.pooling_kernel_size",
                             "vision_config.position_embedding_size", "vision_config.num_hidden_layers",
                             "vision_config.patch_size", "vision_config.rope_parameters.rope_theta",
                             "vision_config.rope_parameters.rope_type", "vision_config.hidden_activation",
                             "vision_config.standardize", "text_config.use_bidirectional_attention" };
        const char* vv[] = { "gemma4_vision", "1152", "16", "72", "4304", "3", "10240", "27", "16",
                             "100", "default", "gelu_pytorch_tanh", "1", "vision" };
        keys.insert(keys.end(), std::begin(vk), std::end(vk));
        vals.insert(vals.end(), std::begin(vv), std::end(vv));
    }

    RadModelMeta meta() const {
        RadModelMeta m = gemma4_moe_meta();
        m.n_kv = (int)keys.size();
        m.kv_key = keys.data();
        m.kv_val = vals.data();
        return m;
    }
};

}  /* namespace */

TEST(gemma4_moe_declares_and_runs_the_26b_vision_tower_only_when_enabled) {
    const G4MoeVision f;
    const RadModelMeta meta = f.meta();
    RadArchProbe p{};
    CHECK_OK(gemma4::probe(&meta, &p));
    CHECK_EQ(p.enc_merge, 9);
    CHECK_EQ(p.draft_depth, 15);
    CHECK_EQ(p.chain_pays, RAD_CHAIN_SPEC); // retain current upstream policy pending device evidence

    std::vector<std::string> text_ops;
    for (int patches : { 0, 18 }) {
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        ctx.max_enc_patches = patches;
        REQUIRE_EQ(gemma4::declare(&b, &meta, &ctx), RAD_OK);
        CHECK_EQ(b.encoder.bidirectional, patches ? 1 : 0);
        CHECK_EQ(count_op(b, "scatter_rows"), patches ? 1 : 0);

        RadBatch text = prefill_batch();
        RadCtx tc{ .b = &b, .batch = &text };
        gemma4::step(&tc, &text);
        CHECK(tc.step_fail.empty());
        CHECK_LINT(b, &tc);
        if (!patches) text_ops = issued_ops(b, tc);
        else CHECK(issued_ops(b, tc) == text_ops);

        static uint16_t pixels[18 * 768];
        static int32_t coords[2 * 18];
        static const int32_t cu[] = { 0, 18 };
        RadBatch enc{};
        enc.enc = RAD_MM_IMAGE | RAD_MM_VIDEO; /* BatchBuilder uses the encoder mask. */
        enc.enc_pixels = pixels;
        enc.enc_coord = coords;
        enc.enc_cu = cu;
        enc.enc_n_patch = enc.enc_max_seg = 18;
        enc.enc_n_seg = 1;
        RadCtx ec{ .b = &b, .batch = &enc };
        gemma4::step(&ec, &enc);
        if (!patches) {
            CHECK(find_w(b, "v.embed.weight") == nullptr);
            CHECK(ec.step_fail.find("no vision tower was declared") != std::string::npos);
            CHECK(ec.issues.empty());
            continue;
        }

        CHECK_EQ(b.encoder.n_embd, 2816);
        CHECK_EQ(b.encoder.patch_dim, 768);
        CHECK_EQ(b.encoder.merge, 9);
        CHECK_EQ(b.encoder.modalities, (uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO));
        const RecWeight* first = find_w(b, "v.blk.0.qkv");
        REQUIRE(first != nullptr);
        CHECK_EQ(first->d.group.layer, meta.n_layers + 5);
        CHECK(find_w(b, "dflash.fc.weight") == nullptr);
        const RecWeight* proj = find_w(b, "v.embed.weight");
        REQUIRE(proj != nullptr);
        CHECK_EQ(proj->d.shape[0], 2816);
        CHECK_EQ(proj->d.shape[1], 1152);
        CHECK_EQ(proj->d.shard, RAD_SHARD_NONE);
        const RecMap* map = find_map(b, proj->name);
        REQUIRE(map && map->src.size() == 1);
        CHECK_EQ(map->src.front(), std::string("model.embed_vision.embedding_projection.weight"));
        for (const RecWeight& w : b.weights)
            if (w.name.starts_with("v.")) CHECK(find_map(b, w.name) != nullptr);

        CHECK(ec.step_fail.empty());
        CHECK_LINT(b, &ec);
        REQUIRE(!ec.issues.empty());
        const auto ops = issued_ops(b, ec);
        CHECK_EQ(count_issued(ops, "attn_dense"), 27);
        CHECK_EQ(count_issued(ops, "kv_store"), 0);
        CHECK_EQ(ec.issues.back().n, 2);
        CHECK_EQ(ec.issues.back().opd.back().handle, b.encoder.out);
        for (int modality : {int(RAD_MM_IMAGE), int(RAD_MM_VIDEO), int(RAD_MM_AUDIO),
                             int(RAD_MM_IMAGE | RAD_MM_AUDIO)}) {
            enc.enc = modality;
            RadCtx mc{ .b = &b, .batch = &enc };
            gemma4::step(&mc, &enc);
            if (modality & ~(RAD_MM_IMAGE | RAD_MM_VIDEO)) {
                CHECK(mc.step_fail.find("unsupported encoder modality") != std::string::npos);
                CHECK(mc.issues.empty());
            } else {
                CHECK(mc.step_fail.empty());
                CHECK_LINT(b, &mc);
                CHECK_EQ(count_issued(issued_ops(b, mc), "attn_dense"), 27);
            }
        }
    }
}

TEST(gemma4_moe_text_only_container_with_vision_metadata_needs_no_tower_weights) {
    const G4MoeVision f;
    const RadModelMeta meta = f.meta();
    const RadEncoding bf16 = rad_enc_plain(RAD_BF16);
    for (int patches : { 0, 18 }) {
        RadBuilder b;
        b.encs = { { "token_embd.weight", bf16 } };
        RadBuildCtx ctx = build_ctx();
        ctx.max_enc_patches = patches;
        ctx.max_spec = 15;
        CHECK_EQ(gemma4::declare(&b, &meta, &ctx), patches ? RAD_E_NOTFOUND : RAD_OK);
        if (!patches) {
            CHECK_EQ(b.encoder.modalities, 0u);
            CHECK(find_w(b, "v.patch.weight") == nullptr);
            CHECK(find_w(b, "dflash.fc.weight") != nullptr);
        }
    }
}

TEST(gemma4_moe_vision_without_bidirectional_metadata_uses_causal_attention) {
    G4MoeVision f;
    f.keys.pop_back();  /* text_config.use_bidirectional_attention */
    f.vals.pop_back();
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_enc_patches = 18;
    REQUIRE_EQ(gemma4::declare(&b, &meta, &ctx), RAD_OK);
    CHECK_EQ(b.encoder.bidirectional, 0);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);
    CHECK(find_w(b, "v.patch.weight") != nullptr);
    for (const RecOp& op : b.ops)
        if (op.op.starts_with("attn_paged")) CHECK_EQ(pint(&op, "causal"), 1);
}

TEST(gemma4_moe_defensively_refuses_an_overlong_span_without_ahead_attention) {
    G4MoeVision f;
    for (size_t i = 0; i < f.keys.size(); ++i)
        if (!std::strcmp(f.keys[i], "sliding_window")) f.vals[i] = "8";
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    b.undeclared = { "attn_paged_qrow", "attn_paged_ahead", "attn_paged_qrow_ahead" };
    RadBuildCtx ctx = build_ctx();
    ctx.max_enc_patches = 126;
    REQUIRE_EQ(gemma4::declare(&b, &meta, &ctx), RAD_OK);
    CHECK(b.asked_undeclared.empty());

    static const int32_t row = 4, seq = 0;
    for (int32_t n : { 8, 9 }) {
        const int32_t used = row + n;
        RadBatch batch = prefill_batch();
        batch.n_bidi = 1;
        batch.bidi_row = &row;
        batch.bidi_n = &n;
        batch.bidi_seq = &seq;
        batch.bidi_used = &used;
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        if (n == 8) {
            CHECK(c.step_fail.empty());
            CHECK_LINT(b, &c);
        } else {
            CHECK(c.step_fail.find("unrestricted lookahead") != std::string::npos);
        }
    }
}

TEST(gemma4_moe_tp2_image_spans_and_int8_dflash_share_the_trunk) {
    G4MoeVision f;
    for (size_t i = 0; i < f.keys.size(); ++i)
        if (!std::strcmp(f.keys[i], "sliding_window")) f.vals[i] = "8";
    const RadModelMeta meta = f.meta();
    static const int32_t rows[] = { 12, 40 }, lengths[] = { 3, 10 };
    static const int32_t seqs[] = { 0, 1 }, used[] = { 15, 18 };
    static const int32_t mm_rows[] = { 12, 13, 14, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49 };
    static uint16_t mm_embd[13 * 2816];
    static int32_t tables[2 * 4];

    /* Exercise both dense-Q and fused-row queries, and a span longer than the window. */
    for (bool i8 : { false, true })
    for (bool qrow : { false, true })
    for (int rank : { 0, 1 }) {
        RadBuilder b;
        if (!qrow) b.undeclared.insert("attn_paged_qrow");
        const RadEncoding draft_i8 = rad_enc_affine(RAD_I8, RAD_F16, 1, 128);
        RadEncoding expert = i8 ? rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) : rad_enc_plain(RAD_BF16);
        if (i8) std::strcpy(expert.transform, "fwht128");
        b.encs = { { "ffn_gate_up_exps.", expert }, { "dflash.fc.", draft_i8 }, { "dflash.blk.", draft_i8 },
                   { "dflash.draft_head.", draft_i8 } };
        RadBuildCtx ctx = build_ctx(rank, 2);
        ctx.max_enc_patches = 126;
        ctx.max_spec = 15;
        ctx.kv_dtype = RAD_I8;
        REQUIRE_EQ(gemma4::declare(&b, &meta, &ctx), RAD_OK);
        CHECK(b.asked_undeclared.empty());
        CHECK(find_w(b, "dflash.draft_head.weight") != nullptr);
        const RecWeight* vis = find_w(b, "v.blk.0.qkv");
        REQUIRE(vis != nullptr);
        CHECK_EQ(vis->d.group.layer, meta.n_layers + 5);

        RadKVGroupBatch sliding{};
        sliding.group = b.binds.front().second;
        sliding.block_table = tables;
        sliding.block_table_pitch = 4;
        RadBatch batch = prefill_batch();
        batch.kv = &sliding;
        batch.n_kv_groups = 1;
        batch.n_mm_rows = 13;
        batch.mm_rows = mm_rows;
        batch.mm_embd = mm_embd;
        batch.n_bidi = 2;
        batch.bidi_row = rows;
        batch.bidi_n = lengths;
        batch.bidi_seq = seqs;
        batch.bidi_used = used;
        RadCtx c{ .b = &b, .batch = &batch, .rank = rank, .world = 2 };
        gemma4::step(&c, &batch);
        CHECK(c.step_fail.empty());
        CHECK_LINT(b, &c);
        const auto ops = issued_ops(b, c);
        REQUIRE(ops.size() >= 3);
        CHECK_EQ(ops[0], std::string("embed_lookup"));
        CHECK_EQ(ops[1], std::string("scatter_rows"));
        CHECK_EQ(ops[2], std::string("all_reduce"));
        CHECK_EQ(c.issues[1].opd[0].raw, mm_embd);
        CHECK_EQ(count_issued(ops, "moe_gemm_q"), i8 ? 6 : 0);
        CHECK_EQ(count_issued(ops, "moe_gemm"), i8 ? 6 : 12);

        int causal = 0, corrections = 0;
        for (const RecIssue& issue : c.issues) {
            const RecOp* op = b.op_of(issue.op);
            if (!op || !op->op.starts_with("attn_paged")) continue;
            if (pint(op, "causal")) {
                ++causal;
                continue;
            }
            const int span = corrections++ % 2;
            CHECK_EQ(causal, (corrections + 1) / 2);
            CHECK_EQ(pint(op, "head_dim"), 256);
            CHECK_EQ(pint(op, "window"), 8);
            CHECK_EQ(pstr(op, "kv_dtype"), std::string("i8"));
            CHECK_EQ(op->op, std::string(qrow ? "attn_paged_qrow" : "attn_paged") +
                                  (span ? "_ahead" : ""));
            CHECK_EQ(issue.n, lengths[span]);
            REQUIRE(issue.opd.size() == 8);
            CHECK_EQ(issue.opd[0].offset, rows[span] * (qrow ? 4096 : 2048));
            CHECK_EQ(issue.opd[2].raw, tables + 4 * seqs[span]);
            CHECK_EQ(issue.opd[3].raw, used + span);
            CHECK_EQ(issue.opd[7].offset, rows[span] * 2048);
            CHECK_EQ(issue.opd[7].rows, lengths[span]);
        }
        CHECK_EQ(causal, 6);
        CHECK_EQ(corrections, 10);

        for (int pass : { -1, 1 }) {
            RadBatch draft = decode_batch();
            draft.draft_pass = pass;
            RadCtx dc{ .b = &b, .batch = &draft, .rank = rank, .world = 2 };
            gemma4::step(&dc, &draft);
            CHECK(dc.step_fail.empty());
            const auto draft_ops = issued_ops(b, dc);
            CHECK_EQ(count_issued(draft_ops, "kv_store"), 5);
            CHECK_EQ(count_issued(draft_ops, "scatter_rows"), 0);
            CHECK_EQ(count_issued(draft_ops, "attn_dense"), 0);
            CHECK_EQ(count_issued(draft_ops, "moe_gemm_q"), 0);
            CHECK(count_issued(draft_ops, "gemm_nt_q") > 0);
        }
    }
}

/* DFlash proposes a block in one pass, so the probe's depth is the trained block size minus the
 * anchor, and fixed. Without a drafter, or with DFlash2 (not served by this plugin), it is 0. */
TEST(gemma4_probe_names_the_dflash_blocks_depth) {
    std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    RadModelMeta meta = gemma4_moe_meta();
    meta.n_kv   = (int)keys.size();
    meta.kv_key = keys.data();
    meta.kv_val = vals.data();
    RadArchProbe p{};
    CHECK_OK(gemma4::probe(&meta, &p));
    CHECK_EQ(p.draft_depth, 15);
    CHECK_EQ(p.draft_depth_fixed, 1);
    CHECK_EQ(p.chain_pays, 1);   /* chained launches measured a gain with DFlash */

    for (size_t i = 0; i < keys.size(); ++i)
        if (!std::strcmp(keys[i], "draft.architectures")) vals[i] = "[\"DFlash2DraftModel\"]";
    meta.kv_val = vals.data();
    RadArchProbe q{};
    CHECK_OK(gemma4::probe(&meta, &q));
    CHECK_EQ(q.draft_depth, 0);

    RadModelMeta plain = gemma4_moe_meta();
    RadArchProbe r{};
    CHECK_OK(gemma4::probe(&plain, &r));
    CHECK_EQ(r.draft_depth, 0);
}

/* Each tap is declared where it is issued. The buffer planner takes liveness from declaration
 * order, so `dflash_src` is live from its first declared write; a tap declared after the trunk but
 * issued inside it would let the plan alias it with `moe_edn`, which every MoE layer writes,
 * silently corrupting what the drafter reads. The two live ranges must overlap. */
TEST(gemma4_dflash_taps_are_live_from_the_layer_that_writes_them) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    RadModelMeta meta = gemma4_moe_meta();
    meta.n_kv   = (int)keys.size();
    meta.kv_key = keys.data();
    meta.kv_val = vals.data();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    const int src = buf_named(b, "dflash_src"), edn = buf_named(b, "moe_edn");
    REQUIRE(src > 0 && edn > 0);
    int first_src = -1, last_edn = -1;
    for (size_t i = 0; i < b.ops.size(); ++i)
        for (rad_buf w : b.ops[i].writes) {
            if ((int)w == src && first_src < 0) first_src = (int)i;
            if ((int)w == edn) last_edn = (int)i;
        }
    REQUIRE(first_src >= 0 && last_edn >= 0);
    CHECK(first_src < last_edn);

    /* A trunk step writes each tap once: three copies into `dflash_src`. */
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    int taps = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && std::find(o->writes.begin(), o->writes.end(), (rad_buf)src) != o->writes.end())
            ++taps;
    }
    CHECK_EQ(taps, 3);
}

/* Sliding attention reads q in place in the fused projection row where attn_paged_qrow is served,
 * and lifts q into its own buffer for attn_paged where it is not. */
TEST(gemma4_sliding_attention_reads_q_in_the_fused_row) {
    /* Served: attn_paged_qrow reads the slice and nothing is lifted. Refused: q is lifted into q_s
     * for attn_paged. attn_paged never gets the slice, since its kernel may read q only dense. */
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) b.refuse.insert("attn_paged_qrow");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        const int qkv = buf_named(b, "qkv_s"), qs = buf_named(b, "q_s");
        REQUIRE(qkv > 0);
        CHECK_EQ(qs > 0, refuse == 1);
        auto reads = [](const RecOp& o, int buf) {
            return buf > 0 && std::find(o.reads.begin(), o.reads.end(), (rad_buf)buf) != o.reads.end();
        };
        int qrow = 0, paged_slice = 0, paged_lifted = 0;
        for (const auto& o : b.ops) {
            if (!o.resolved) continue;
            if (o.op == "attn_paged_qrow" && reads(o, qkv)) ++qrow;
            if (o.op == "attn_paged" && reads(o, qkv)) ++paged_slice;
            if (o.op == "attn_paged" && reads(o, qs)) ++paged_lifted;
        }
        const int sliding = refuse ? paged_lifted : qrow;
        CHECK(sliding > 0);
        CHECK_EQ(refuse ? qrow : paged_lifted, 0);
        CHECK_EQ(paged_slice, 0);
        CHECK_EQ(count_op(b, "cast"), refuse ? sliding : 0);   /* the lifts (no drafter taps) */

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "attn_paged_qrow"), refuse ? 0 : sliding);
        CHECK_EQ(count_issued(issued, "cast"), refuse ? sliding : 0);
        for (const auto& o : b.ops)
            if (o.resolved) CHECK(o.uses_declared);
    }
}

/* The feed-forward join writes the next layer's input norm. With joins served, the only rmsnorm
 * from residual `x` into `h` is layer 0's, the last join produces the final norm, and every join
 * writes `h`. Refused, the chain declares them all. */
TEST(gemma4_feed_forward_join_writes_the_next_input_norm) {
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) { b.refuse.insert("norm_add_norm"); b.refuse.insert("join_add_scale"); }
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        const int x = buf_named(b, "x"), h = buf_named(b, "h");
        REQUIRE(x > 0 && h > 0);
        int x_to_h = 0, joins_to_h = 0;
        for (const auto& o : b.ops) {
            if (!o.resolved) continue;
            const bool rx = std::find(o.reads.begin(), o.reads.end(), (rad_buf)x) != o.reads.end();
            const bool wh = std::find(o.writes.begin(), o.writes.end(), (rad_buf)h) != o.writes.end();
            if (o.op == "rmsnorm" && rx && wh) ++x_to_h;
            if (o.op == "join_add_scale" && wh) ++joins_to_h;
        }
        /* fused: layer 0's input norm. Chained: six input norms, six pre-feedforward norms and the
         * final norm. */
        CHECK_EQ(x_to_h, refuse ? 13 : 1);
        CHECK_EQ(joins_to_h, refuse ? 0 : 6);

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "join_add_scale"), refuse ? 0 : 6);
        CHECK_EQ(count_issued(issued, "logits_gemm"), 1);
    }
}

/* At decode heights the router GEMM, top-k and sort are one launch (libr4d's
 * router_gemm_topk_scatter); if refused, router_topk_scatter follows the router GEMM; if both are
 * refused, three separate ops. The routing buffers are the same in all cases. */
TEST(gemma4_routes_a_decode_step_in_one_launch) {
    for (int refuse = 0; refuse < 3; ++refuse) {
        RadBuilder b;
        if (refuse >= 1) b.refuse.insert("router_gemm_topk_scatter");
        if (refuse >= 2) b.refuse.insert("router_topk_scatter");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "router_gemm_topk_scatter"), refuse == 0 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "router_topk_scatter"), refuse == 1 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "router_topk"), refuse == 2 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "moe_scatter"), refuse == 2 ? 6 : 0);
        /* the per-expert scale and gather run whichever routing form was used */
        CHECK_EQ(count_issued(issued, "expert_scale"), 6);
        CHECK_EQ(count_issued(issued, "moe_gather"), 6);
    }
}

/* The per-expert scale is one launch where `expert_scale` is served, else a table lookup and a
 * multiply over the same buffers. */
TEST(gemma4_scales_the_routing_weights_in_one_launch) {
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) b.refuse.insert("expert_scale");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "expert_scale"), refuse ? 0 : 6);
        CHECK_EQ(count_issued(issued, "embed_lookup"), refuse ? 1 + 6 : 1);   /* 1: the token embedding */
        CHECK_EQ(count_issued(issued, "mul"), refuse ? 6 : 0);
        for (const auto& o : b.ops)
            if (o.resolved) CHECK(o.uses_declared);
    }
}

/* The drafter's head reads the container's block-fp8 copy of the target's table
 * (`dflash.draft_head.weight`) through quant_act_fp8 and gemm_nt_q where present, else the bf16
 * table through gemm_nt. The copy is only declared when present, so older containers are served
 * unchanged. */
TEST(gemma4_dflash_head_reads_the_fp8_copy_where_the_container_holds_one) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    for (int fp8 = 0; fp8 < 2; ++fp8) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuilder b;
        if (fp8)
            b.encs = { { "dflash.draft_head", rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128) } };
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        CHECK_EQ(find_w(b, "dflash.draft_head.weight") != nullptr, fp8 == 1);

        const int dlog = buf_named(b, "df_dlog"), selh = buf_named(b, "df_selh");
        REQUIRE(dlog > 0 && selh > 0);
        const RecOp* head = nullptr;
        for (const auto& o : b.ops)
            if (o.resolved &&
                std::find(o.writes.begin(), o.writes.end(), (rad_buf)dlog) != o.writes.end())
                head = &o;
        REQUIRE(head != nullptr);
        CHECK_EQ(head->op, std::string(fp8 ? "gemm_nt_q" : "gemm_nt"));
        const bool reads_selh =
            std::find(head->reads.begin(), head->reads.end(), (rad_buf)selh) != head->reads.end();
        CHECK_EQ(reads_selh, fp8 == 0);

        /* a draft pass issues it */
        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK(c.step_fail.empty());
        int head_issues = 0;
        for (const auto& i : c.issues)
            if (b.op_of(i.op) == head) ++head_issues;
        CHECK_EQ(head_issues, 1);
    }
}

/* The drafter fuses residual add, norm and quantiser into `rmsnorm_quant_fp8`: ten per draft pass
 * over five layers (each layer's two inputs; layer 0's attention input takes x as the embedding
 * left it), leaving one `add`, the last layer's, before the final norm. Refused: two adds and two
 * quantisers of x per layer. */
TEST(gemma4_dflash_folds_the_residual_add_norm_and_quantiser_into_one_launch) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuilder b;
        if (refuse) b.refuse.insert("rmsnorm_quant_fp8");
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));

        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        CHECK_EQ(count_issued(ops, "rmsnorm_quant_fp8"), refuse ? 0 : 10);
        CHECK_EQ(count_issued(ops, "add"), refuse ? 10 : 1);
        /* chain: x quantised twice per layer; attention output quantised once per layer always */
        CHECK_EQ(count_issued(ops, "quant_act_fp8"), refuse ? 15 : 5);
        if (refuse) continue;

        /* the delta then the residual x past layer 0; layer 0 reads x and adds nothing */
        const uint32_t x = (uint32_t)buf_named(b, "df_x"), h = (uint32_t)buf_named(b, "df_h");
        int k = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "rmsnorm_quant_fp8") continue;
            REQUIRE(i.opd.size() == 6);
            if (k++ == 0) {
                CHECK_EQ(i.opd[0].handle, x);
                CHECK_EQ((int)i.opd[1].kind, (int)RAD_OPK_NONE);
            } else {
                CHECK_EQ(i.opd[0].handle, h);
                CHECK_EQ(i.opd[1].handle, x);
            }
        }
    }
}

/* ============ gemma4: the prefill overlap -- two slices, the reductions on lane 1 ============ */

namespace {

/* The elements [lo, hi) of its buffer an operand can reach, in whole rows at the declared width.
 * False for a non-buffer operand. */
bool reach(const RadBuilder& b, const RadOperand& o, rad_buf* buf, int64_t* lo, int64_t* hi) {
    if (o.kind != RAD_OPK_BUF || o.handle == 0 || o.handle > b.bufs.size()) return false;
    const RadBufDecl& d = b.bufs[o.handle - 1].d;
    int64_t pitch = 1;
    for (uint32_t i = 1; i < d.rank; ++i) pitch *= d.shape[i];
    *buf = o.handle;
    *lo  = o.offset - o.offset % pitch;
    *hi  = *lo + (o.rows > 0 ? o.rows : d.shape[0]) * pitch;
    return true;
}

/* Whether issue a runs before issue b (a < b): on one lane by issue order; across lanes only via a
 * join from a's lane to b's issued between them. */
bool ordered(const RadCtx& c, size_t a, size_t b) {
    const int la = c.issues[a].lane, lb = c.issues[b].lane;
    if (la == lb) return true;
    for (const RecJoin& j : c.joins)
        if (j.from == la && j.to == lb && j.at > a && j.at <= b) return true;
    return false;
}

/* Pairs of unordered issues on different lanes that reach a common element. Every operand counts
 * as read and written, so this can only over-count. */
int races(const RadBuilder& b, const RadCtx& c) {
    int n = 0;
    for (size_t i = 0; i < c.issues.size(); ++i)
        for (size_t j = i + 1; j < c.issues.size(); ++j) {
            if (ordered(c, i, j)) continue;
            bool meet = false;
            for (const RadOperand& p : c.issues[i].opd)
                for (const RadOperand& q : c.issues[j].opd) {
                    rad_buf bp = 0, bq = 0;
                    int64_t pl = 0, ph = 0, ql = 0, qh = 0;
                    if (reach(b, p, &bp, &pl, &ph) && reach(b, q, &bq, &ql, &qh) && bp == bq &&
                        pl < qh && ql < ph)
                        meet = true;
                }
            n += meet ? 1 : 0;
        }
    return n;
}

/* One prompt of T tokens, sampling its last. */
RadBatch one_prompt(int64_t T, const int32_t* out) {
    RadBatch b{};
    b.phase       = RAD_PHASE_PREFILL;
    b.n_tok       = T;
    b.n_seq       = 1;
    b.max_q_len   = T;
    b.max_ctx_len = T;
    b.n_out       = 1;
    b.out_ids     = out;
    return b;
}

/* Rows issued, summed by op: a slice's n is its rows; RAD_N_BATCH counts as T. */
std::map<std::string, int64_t> rows_by_op(const RadBuilder& b, const RadCtx& c, int64_t T) {
    std::map<std::string, int64_t> w;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        w[o ? o->op : "<null>"] += i.n < 0 ? T : i.n;
    }
    return w;
}

/* The n of each issue of one op, in issue order. */
std::vector<int64_t> ns_of(const RadBuilder& b, const RadCtx& c, const char* op) {
    std::vector<int64_t> v;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == op) v.push_back(i.n);
    }
    return v;
}

bool noted(const RadBuilder& b, const char* text) {
    for (const auto& n : b.notes)
        if (n.find(text) != std::string::npos) return true;
    return false;
}

}  /* namespace */

TEST(gemma4_moe_overlap_attends_an_image_span_across_the_slice_boundary) {
    const G4MoeVision f;
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx(0, 2);
    ctx.max_tok = 4096;
    ctx.max_enc_patches = 36;
    REQUIRE_EQ(gemma4::declare(&b, &meta, &ctx), RAD_OK);
    REQUIRE(noted(b, "prefill overlap: a chunk of two routed passes"));

    static const int32_t rows[] = { 2046, 2047, 2048, 2049 };
    static const int32_t n = 4, seq = 0, used = 2050, out = 4095;
    static uint16_t image[4 * 2816];
    RadBatch batch = one_prompt(4096, &out);
    batch.n_mm_rows = 4;
    batch.mm_rows = rows;
    batch.mm_embd = image;
    batch.n_bidi = 1;
    batch.bidi_row = rows;
    batch.bidi_n = &n;
    batch.bidi_seq = &seq;
    batch.bidi_used = &used;
    RadCtx c{ .b = &b, .batch = &batch, .world = 2 };
    gemma4::step(&c, &batch);
    CHECK(c.step_fail.empty());
    CHECK_LINT(b, &c);
    CHECK_EQ(races(b, c), 0);
    int corrections = 0, overlap_reductions = 0;
    for (const RecIssue& issue : c.issues) {
        const RecOp* op = b.op_of(issue.op);
        REQUIRE(op != nullptr);
        if (issue.lane == 1 && op->op == "all_reduce") ++overlap_reductions;
        if (op->op != "attn_paged_qrow" || pint(op, "causal") != 0) continue;
        ++corrections;
        CHECK_EQ(issue.n, n);
        CHECK_EQ(issue.opd.back().offset, 2046 * 2048);
        CHECK_EQ(issue.opd.back().rows, n);
    }
    CHECK_EQ(corrections, 5);
    CHECK_EQ(overlap_reductions, 24);
}

/* A prefill chunk in two slices (arch/gemma4, trunk_overlap). Checked: every reduction runs on
 * lane 1 after its producer and before its consumer, and nothing else does; the cut is a whole
 * number of routed passes; and the slices issue exactly the whole step's rows (same routed
 * passes, same drafter taps), so the output matches. A chunk under two passes, and a decode step,
 * run whole: reductions on lane 0 and the dense arm on lane 1 beside the routed arm, all ordered. */
TEST(a_gemma4_prefill_chunk_runs_as_two_slices_with_its_reductions_on_the_second_lane) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    const int64_t E = 2816, L = 6;
    auto model = [&](RadBuilder& b, bool whole) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuildCtx ctx = build_ctx(0, 4);
        ctx.max_tok  = 8192;
        ctx.max_spec = 15;
        /* The lossy wire runs every chunk whole, since a slice is not shown to get the same bits
         * as the whole step. */
        if (whole) ctx.tp_wire_lossy = 1;
        return gemma4::declare(&b, &meta, &ctx);
    };
    auto run = [&](RadBuilder& b, RadCtx& c, RadBatch& batch) {
        c = RadCtx{ .b = &b, .batch = &batch, .world = 4 };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
    };

    struct Case { int64_t T, cut; };
    for (const Case k : { Case{8192, 4096}, Case{6144, 4096}, Case{6143, 2048}, Case{4096, 2048},
                          Case{4095, 0} }) {
        const int32_t out[1] = { (int32_t)(k.T - 1) };
        RadBuilder b;
        CHECK_OK(model(b, false));
        CHECK(noted(b, "prefill overlap: a chunk of two routed passes"));
        CHECK_EQ((int)b.concurrent.size(), 5);
        for (const char* name : { "o", "ff2", "h", "gate_up", "ffn" })
            CHECK(std::count(b.concurrent.begin(), b.concurrent.end(),
                             (rad_buf)buf_named(b, name)) == 1);
        RadBatch batch = one_prompt(k.T, out);
        RadCtx c;
        run(b, c, batch);
        CHECK(c.step_fail.empty());
        CHECK_EQ(c.lane, 0);
        CHECK_EQ(races(b, c), 0);

        std::vector<int64_t> lane1;   /* reductions on lane 1 */
        int dense1 = 0;               /* dense-arm launches on lane 1 */
        for (const auto& i : c.issues) {
            if (i.lane != 1) continue;
            const RecOp* o = b.op_of(i.op);
            REQUIRE(o != nullptr);
            if (o->op == "all_reduce") lane1.push_back(i.n);
            else ++dense1;
        }
        if (k.cut == 0) {
            CHECK(lane1.empty());
            CHECK_EQ(dense1, 3 * (int)L);    /* gate_up, act_mul, down beside the routed arm */
        } else {
            CHECK_EQ(dense1, 0);
            REQUIRE(lane1.size() == (size_t)(4 * L));
            for (int64_t l = 0; l < L; ++l) {
                CHECK_EQ(lane1[4 * l + 0], k.cut * E);               /* attention, slice A */
                CHECK_EQ(lane1[4 * l + 1], (k.T - k.cut) * E);       /* attention, slice B */
                CHECK_EQ(lane1[4 * l + 2], 2 * k.cut * E);           /* both arms, slice A */
                CHECK_EQ(lane1[4 * l + 3], 2 * (k.T - k.cut) * E);   /* both arms, slice B */
            }
        }

        /* The whole step on the lossy wire: the same rows by op and routed passes, reductions on
         * lane 0, only the dense arm on lane 1. */
        RadBuilder w;
        CHECK_OK(model(w, true));
        CHECK(noted(w, "prefill overlap: off (the lossy all-reduce wire)"));
        CHECK(std::count(w.concurrent.begin(), w.concurrent.end(), (rad_buf)buf_named(w, "o")) == 0);
        RadBatch wbatch = one_prompt(k.T, out);
        RadCtx wc;
        run(w, wc, wbatch);
        CHECK(wc.step_fail.empty());
        CHECK_EQ(races(w, wc), 0);
        for (const auto& i : wc.issues) {
            const RecOp* o = w.op_of(i.op);
            if (i.lane == 1) CHECK(o == nullptr || o->op != "all_reduce");
        }
        CHECK(rows_by_op(b, c, k.T) == rows_by_op(w, wc, k.T));
        CHECK(ns_of(b, c, "moe_gather") == ns_of(w, wc, "moe_gather"));
        CHECK_EQ((int)ns_of(b, c, "cast").size(), k.cut ? 6 : 3);   /* three taps, per slice */
    }

    /* A decode step: reductions on lane 0, the dense arm on lane 1 joined before the feed-forward
     * reduction, all ordered. */
    RadBuilder b;
    CHECK_OK(model(b, false));
    CHECK(noted(b, "the two arms: in a step that runs whole"));
    RadBatch d = decode_batch();
    RadCtx c;
    run(b, c, d);
    CHECK(c.step_fail.empty());
    CHECK_EQ(c.lane, 0);
    CHECK_EQ(races(b, c), 0);
    int dense1 = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        REQUIRE(o != nullptr);
        if (o->op == "all_reduce") CHECK_EQ(i.lane, 0);
        if (i.lane == 1) {
            ++dense1;
            CHECK(o->op == "gemm_nt" || o->op == "act_mul");
        }
    }
    CHECK_EQ(dense1, 3 * (int)L);
    CHECK_EQ((int)c.joins.size(), 2 * (int)L);
}

/* --kv-cache-dtype i8 on the 26B-A4B: both attention kinds' caches are int8 groups at their own
 * head_dim (256 sliding, 512 full; the core adds a row's four scale bytes), and every op that
 * names a cache dtype bands on i8. A decode step lints clean. */
TEST(gemma4_i8_cache_bands_every_cache_op) {
    RadModelMeta meta = gemma4_moe_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.kv_dtype = RAD_I8;
    REQUIRE_EQ(gemma4::declare(&b, &meta, &ctx), RAD_OK);
    REQUIRE(b.kv_groups.size() >= 2);
    CHECK_EQ(b.kv_groups[0].d.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(b.kv_groups[0].d.head_dim, 256);
    CHECK_EQ(b.kv_groups[1].d.head_dim, 512);
    int i8 = 0, other = 0;
    for (const auto& o : b.ops) {
        if (!param(&o, "kv_dtype")) continue;
        const std::string kd = pstr(&o, "kv_dtype");
        i8 += kd == "i8";
        other += kd != "i8";
    }
    CHECK(i8 > 0);
    CHECK_EQ(other, 0);
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
}

/* Where no loaded library declares the fused ops (CPU-only, or no libr4d), asking for one would
 * stop the engine, so the plugin checks rad_op_declared first and declares the chains: rmsnorm /
 * add joins, embed_lookup and mul for the expert scale, and q lifted for attn_paged. */
TEST(gemma4_without_the_fused_ops_declares_the_chains_and_asks_for_no_undeclared_op) {
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", fp8 } };
    for (const char* op : { "norm_add_norm", "norm_add_norm_route", "join_add_scale",
                            "expert_scale", "attn_paged_qrow" })
        b.undeclared.insert(op);
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));
    CHECK(b.asked_undeclared.empty());
    CHECK(noted(b, "sandwich joins: 0 of 12 fused"));
    CHECK(noted(b, "routed inputs: 0 of 6"));
    CHECK(noted(b, "sliding attention: q lifted"));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    const std::vector<std::string> ops = issued_ops(b, c);
    CHECK_EQ(count_issued(ops, "rmsnorm_quant_fp8"), 6);
    CHECK_EQ(count_issued(ops, "cast"), 5);
    CHECK(count_issued(ops, "add") > 0);
}

/* Rotated int8 gate_up experts are the int8 table GEMM (i8a8h, a scale a 128 of K on both sides)
 * over codes in an int8 moe_hq, which the attention join's fused route writes
 * (norm_add_norm_route_i8g) and, where no library declares that, the rotating norm. No E4M3
 * quantiser is issued either way. A model whose quantised experts are block fp8 on one layer and
 * int8 on the others is refused. */
TEST(gemma4_serves_rotated_int8_experts_from_the_join_or_the_rotating_norm) {
    RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(i8.transform, "fwht128");
    for (const bool route : { true, false }) {
        RadBuilder b;
        b.encs = { { "ffn_gate_up_exps", i8 } };
        if (!route) b.undeclared.insert("norm_add_norm_route_i8g");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        CHECK(b.asked_undeclared.empty());
        CHECK_EQ(count_resolved(b, "norm_add_norm_route"), 0);
        CHECK_EQ(count_resolved(b, "norm_add_norm_route_i8g"), route ? 6 : 0);
        CHECK_EQ(count_resolved(b, "norm_add_norm"), route ? 0 : 6);
        CHECK_EQ(count_op(b, "rmsnorm_had_quant_i8g"), 6);
        CHECK(noted(b, "gate_up rotated int8"));
        int gu = 0;
        for (const auto& o : b.ops) {
            if (o.op != "moe_gemm_q") continue;
            CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
            CHECK_EQ(pint(&o, "a_group"), 128);
            ++gu;
        }
        CHECK_EQ(gu, 6);
        const RecBuf* hq = find_b(b, "moe_hq");
        REQUIRE(hq != nullptr);
        CHECK_EQ(hq->d.dtype, (uint32_t)RAD_I8);
        if (route) {
            const RecOp* r = find_op(b, "norm_add_norm_route_i8g");
            REQUIRE(r != nullptr);
            CHECK(in(r->writes, (rad_buf)buf_named(b, "moe_hq")));
        }

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        std::map<std::string, int> issued;
        for (const auto& n : issued_ops(b, c)) ++issued[n];
        CHECK_EQ(issued["norm_add_norm_route_i8g"], route ? 6 : 0);
        CHECK_EQ(issued["rmsnorm_had_quant_i8g"], route ? 0 : 6);
        CHECK_EQ(issued["rmsnorm_quant_fp8"], 0);
        CHECK_EQ(issued["quant_act_fp8"], 0);
        CHECK_EQ(issued["moe_gemm_q"], 6);
    }
    {
        RadBuilder b;
        b.encs = { { "blk.0.ffn_gate_up_exps", rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128) },
                   { "ffn_gate_up_exps", i8 } };
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return gemma4::declare(&b, &meta, &ctx); }), RAD_E_DTYPE);
    }
}

/* On a layer whose gate_up is block fp8 the attention join (norm_add_norm_route) also writes the
 * routed arm's inputs: the router's norm into moe_rn and the experts' codes into moe_hq / moe_hqs,
 * so neither the router's rmsnorm nor rmsnorm_quant_fp8 is issued. Refused: norm_add_norm plus
 * those two launches. A tp-4 prefill through the overlap keeps every reduction ordered too. */
TEST(gemma4_attention_join_writes_the_routed_inputs_on_fp8_experts) {
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    std::map<std::string, int> issued[2];
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        b.encs = { { "ffn_gate_up_exps", fp8 } };
        if (refuse) b.refuse.insert("norm_add_norm_route");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        CHECK_EQ(count_resolved(b, "norm_add_norm_route"), refuse ? 0 : 6);
        CHECK_EQ(count_resolved(b, "norm_add_norm"), refuse ? 6 : 0);
        CHECK(noted(b, refuse ? "routed inputs: 0 of 6" : "routed inputs: 6 of 6"));
        if (!refuse) {
            const RecOp* r = find_op(b, "norm_add_norm_route");
            REQUIRE(r != nullptr);
            for (const char* name : { "moe_rn", "moe_hq", "moe_hqs", "x", "h" })
                CHECK(std::count(r->writes.begin(), r->writes.end(),
                                 (rad_buf)buf_named(b, name)) == 1);
        }

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        for (const auto& n : issued_ops(b, c)) ++issued[refuse][n];
    }
    CHECK_EQ(issued[0]["norm_add_norm_route"], 6);
    CHECK_EQ(issued[0]["norm_add_norm"], 0);
    CHECK_EQ(issued[0]["rmsnorm_quant_fp8"], 0);
    CHECK_EQ(issued[1]["norm_add_norm_route"], 0);
    CHECK_EQ(issued[1]["norm_add_norm"], 6);
    CHECK_EQ(issued[1]["rmsnorm_quant_fp8"], 6);
    CHECK_EQ(issued[1]["rmsnorm"] - issued[0]["rmsnorm"], 6);   /* the router's input norm */
    CHECK_EQ(issued[0]["moe_gemm_q"], 6);

    /* tp 4, an 8192-token chunk in two slices */
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", fp8 } };
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx(0, 4);
    ctx.max_tok = 8192;
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));
    CHECK(noted(b, "prefill overlap: a chunk of two routed passes"));
    const int32_t out[1] = { 8191 };
    RadBatch batch = one_prompt(8192, out);
    RadCtx c{ .b = &b, .batch = &batch, .world = 4 };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    CHECK_EQ(races(b, c), 0);
    CHECK_EQ(count_issued(issued_ops(b, c), "norm_add_norm_route"), 12);
    CHECK_EQ(count_issued(issued_ops(b, c), "rmsnorm_quant_fp8"), 0);
}

/* At four ranks the junctions carry their reductions at decode heights, in the same issue order as
 * at two ranks (only the embedding's and the last layer's feed-forward reduction stand alone),
 * through ar_rmsnorm_quant_i8 declared at world_size 4 up to 16 rows (kAr4FoldRows: a DFlash
 * verify's rows), and beside each the same op with `sp` (the all-reduce split around the
 * junction) over the same 16 rows, which a step from 4 rows takes. A taller step issues each
 * reduction and its junction apart, and where no library declares the fold every step issues the
 * pair. */
TEST(a_gemma4_w4a8_step_at_four_ranks_folds_the_all_reduce_into_the_junction_at_decode_heights) {
    const int L = 12;
    std::vector<std::string> folded = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    std::vector<std::string> apart  = folded;
    /* beside each reduction, up to a step of 16 rows: a prefetch (prefetch_beside) */
    auto pf = [](std::vector<std::string>& v, std::vector<std::string> s, bool next) {
        std::vector<std::string> out = { "prefetch" };   /* o_proj's, beside the attention */
        for (const auto& x : s) {
            if ((x == "ar_rmsnorm_quant_i8" || x == "all_reduce") && (next || &x != &s.back()))
                out.push_back("prefetch");
            out.push_back(x);
        }
        v.insert(v.end(), out.begin(), out.end());
    };
    std::vector<std::string> apart_tall = folded, folded_bare = folded;
    for (int l = 0; l < L; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i)
            for (auto* v : { &folded, &apart, &apart_tall, &folded_bare }) v->push_back("gemm_nt_q");
        const std::vector<std::string> fs = { "qkv_norm_rope_kv_store", "attn_paged", "quant_act_i8",
                                              "gemm_nt_q", "ar_rmsnorm_quant_i8", "gemm_nt_q",
                                              "gelu_mul_quant_i8", "gemm_nt_q" };
        const std::vector<std::string> as = { "qkv_norm_rope_kv_store", "attn_paged", "quant_act_i8",
                                              "gemm_nt_q", "all_reduce", "rmsnorm_quant_i8",
                                              "gemm_nt_q", "gelu_mul_quant_i8", "gemm_nt_q",
                                              "all_reduce" };
        pf(folded, fs, true);
        folded_bare.insert(folded_bare.end(), fs.begin(), fs.end());
        pf(apart, as, true);
        apart_tall.insert(apart_tall.end(), as.begin(), as.end());
        if (l < L - 1) {
            folded.push_back("prefetch");
            apart_tall.push_back("rmsnorm_quant_i8");
            folded_bare.push_back("ar_rmsnorm_quant_i8");
            folded.push_back("ar_rmsnorm_quant_i8");
            apart.push_back("rmsnorm_quant_i8");
        } else {
            folded.push_back("prefetch");   /* the lm_head's */
            for (const char* s : { "all_reduce", "rmsnorm", "add", "scale" }) {
                folded.push_back(s);
                folded_bare.push_back(s);
            }
            for (const char* s : { "rmsnorm", "add", "scale" }) {
                apart.push_back(s);
                apart_tall.push_back(s);
            }
        }
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        for (auto* v : { &folded, &apart, &apart_tall, &folded_bare }) v->push_back(s);
    /* a decode step folds */
    CHECK_EQ(join(gemma4_w4_step({}, 4)), join(folded));
    /* where the library refuses `prefetch`, no reads */
    CHECK_EQ(join(gemma4_w4_step({ "prefetch" }, 4)), join(folded_bare));

    auto declare4 = [](RadBuilder& b) {
        const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
        b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
                   { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
        RadModelMeta meta = gemma4_meta();
        RadBuildCtx  ctx  = build_ctx(0, 4);
        return gemma4_bf16::declare(&b, &meta, &ctx);
    };
    RadBuilder b;
    CHECK_OK(declare4(b));
    CHECK(noted(b, "folded in at steps of 16 rows or fewer"));
    auto fold_hi = [](const RadBuilder& bb, int* n) {
        int64_t hi = -1;
        *n = 0;
        for (const auto& o : bb.ops) {
            if (o.op != "ar_rmsnorm_quant_i8" || pint(&o, "sp") != -12345) continue;
            ++*n;
            CHECK_EQ(pint(&o, "world_size"), 4);
            for (const auto& p : o.p)
                if (p.key == "M") hi = p.ihi;
        }
        return hi;
    };
    int n_ar = 0;
    CHECK_EQ(fold_hi(b, &n_ar), 16);
    CHECK_EQ(n_ar, 2 * L - 1);
    /* the split form beside each fold, over its 16 rows; the residual on every row (sp 2) only
     * before the last layer's feed-forward, with no drafter tapping the stream here */
    auto sp_hi = [](const RadBuilder& bb, int* n, int* n2) {
        int64_t hi = -1;
        *n = 0; *n2 = 0;
        for (const auto& o : bb.ops) {
            if (o.op != "ar_rmsnorm_quant_i8" || pint(&o, "sp") == -12345) continue;
            ++*n;
            if (pint(&o, "sp") == 2) ++*n2;
            else CHECK_EQ(pint(&o, "sp"), 1);
            CHECK_EQ(pint(&o, "world_size"), 4);
            for (const auto& p : o.p)
                if (p.key == "M") hi = p.ihi;
        }
        return hi;
    };
    int n_sp = 0, n_sp2 = 0;
    CHECK_EQ(sp_hi(b, &n_sp, &n_sp2), 16);
    CHECK_EQ(n_sp, 2 * L - 1);
    CHECK_EQ(n_sp2, 1);
    /* which form a step's junctions take: the split one from 4 rows, the fold below */
    auto sp_issued = [&b](const RadCtx& c) {
        int n = 0, folds = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "ar_rmsnorm_quant_i8") continue;
            ++folds;
            if (pint(o, "sp") != -12345) ++n;
        }
        return std::make_pair(n, folds);
    };

    /* 20 MB of gate_up beside the attention's reduction and of the next layer's q beside the
     * feed-forward's, and 9 MB of down_proj beside the attention */
    int n_pf = 0, n_pf9 = 0;
    for (const auto& o : b.ops)
        if (o.op == "prefetch") {
            ++n_pf;
            if (pint(&o, "bytes") == (int64_t)9 << 20) ++n_pf9;
            else CHECK_EQ(pint(&o, "bytes"), (int64_t)20 << 20);
        }
    CHECK_EQ(n_pf, 3 * L);
    CHECK_EQ(n_pf9, (int)L);

    /* a verify's height folds (libr4d's tail form) and a step past 16 rows issues the pair; the
     * prefetches ride a decode step's (a verify's) 16 rows, never a prompt's rows and not a step
     * past 16 */
    const int32_t out[1] = { 127 };
    for (int rows : { 16, 128 }) {
        RadBatch tall = one_prompt(rows, out);
        RadCtx c; c.b = &b; c.rank = 0; c.world = 4; c.batch = &tall;
        gemma4_bf16::step(&c, &tall);
        CHECK(c.step_fail.empty());
        /* a prompt's rows: no reads */
        CHECK_EQ(join(issued_ops(b, c)), join(rows <= 16 ? folded_bare : apart_tall));
        /* the same rows as a decode step (a verify's) take them, up to 16 */
        tall.phase = RAD_PHASE_DECODE;
        RadCtx d; d.b = &b; d.rank = 0; d.world = 4; d.batch = &tall;
        gemma4_bf16::step(&d, &tall);
        CHECK(d.step_fail.empty());
        CHECK_EQ(join(issued_ops(b, d)), join(rows <= 16 ? folded : apart_tall));
        if (rows <= 16) CHECK(sp_issued(d) == std::make_pair(2 * L - 1, 2 * L - 1));
    }
    /* three rows: the fold; eight users' verify (64 rows): the pair */
    for (int rows : { 3, 64 }) {
        RadBatch v = one_prompt(rows, out);
        v.phase = RAD_PHASE_DECODE;
        RadCtx d; d.b = &b; d.rank = 0; d.world = 4; d.batch = &v;
        gemma4_bf16::step(&d, &v);
        CHECK(d.step_fail.empty());
        if (rows == 64) CHECK_EQ(join(issued_ops(b, d)), join(apart_tall));
        CHECK(sp_issued(d) == (rows == 3 ? std::make_pair(0, 2 * L - 1) : std::make_pair(0, 0)));
    }


    /* no library declares the fold: none declared, every step issues the pair */
    RadBuilder off;
    off.undeclared.insert("ar_rmsnorm_quant_i8");
    CHECK_OK(declare4(off));
    CHECK(noted(off, "issued apart"));
    for (const auto& o : off.ops) CHECK(o.op != "ar_rmsnorm_quant_i8");
}

/* ============ gemma4 31B + DFlash: z-lab's block drafter on the w4a8 / w8a8 trunk ============ */

namespace {

/* The twelve-layer 31B above with z-lab/gemma-4-31B-it-DFlash merged under `draft.` as
 * `rad-convert --draft-model` writes it: its own geometry (5376 wide, 64 heads / 8 kv x 128, ffn
 * 10752, five layers -- four sliding over 2048, one full -- at block 16), reading the trunk at three
 * of the fixture's twelve layers where the real one reads six of sixty. */
std::vector<const char*> g431df_keys() {
    std::vector<const char*> k(std::begin(k_gem4_keys), std::end(k_gem4_keys));
    for (const char* s : { "draft.architectures", "draft.num_hidden_layers", "draft.head_dim",
                           "draft.num_attention_heads", "draft.num_key_value_heads",
                           "draft.intermediate_size", "draft.rms_norm_eps", "draft.rope_theta",
                           "draft.hidden_size", "draft.vocab_size", "draft.block_size",
                           "draft.dflash_config.target_layer_ids",
                           "draft.dflash_config.mask_token_id", "draft.layer_types",
                           "draft.sliding_window" })
        k.push_back(s);
    return k;
}
std::vector<const char*> g431df_vals() {
    std::vector<const char*> v(std::begin(k_gem4_vals), std::end(k_gem4_vals));
    for (const char* s : { "[\"DFlashDraftModel\"]", "5", "128", "64", "8", "10752", "1e-06",
                           "1000000", "5376", "262144", "16", "[1, 5, 9]", "4",
                           "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                           "\"sliding_attention\", \"full_attention\"]",
                           "2048" })
        v.push_back(s);
    return v;
}

struct G431Df {
    std::vector<const char*> keys = g431df_keys(), vals = g431df_vals();
    RadModelMeta meta() const {
        RadModelMeta m = gemma4_meta();
        m.n_kv   = (int)keys.size();
        m.kv_key = keys.data();
        m.kv_val = vals.data();
        return m;
    }
};

/* The trunk at w4a8 as the -df recipe makes it, the drafter's linears on the int8 grid (w8a8;
 * `i8` false gives DFlash2's rotated int8 instead) and, with `head`, the table's 8-bit copy at
 * w8a8. The drafter's keys come FIRST: the fake answers the first key a name contains, and
 * `attn_q.` is in `dflash.blk.0.attn_q.weight` too. With `fused`, the trunk's q, k and v are one
 * linear, as a recipe naming `blk.*.attn_qkv.weight` converts them. */
void g431df_encs(RadBuilder& b, bool i8, bool head, bool fused = false) {
    const RadEncoding w4  = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    const RadEncoding q8  = rad_enc_affine(RAD_I8, RAD_F16, 1, 128);
    RadEncoding r8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(r8.transform, "fwht128");
    const RadEncoding& d  = i8 ? q8 : r8;
    b.encs = { { "dflash.fc.", d }, { "dflash.blk.", d } };
    if (head) b.encs.push_back({ "dflash.draft_head.", q8 });
    if (fused) b.encs.push_back({ "attn_qkv.", w4 });
    else
        for (const char* s : { "attn_q.", "attn_k.", "attn_v." }) b.encs.push_back({ s, w4 });
    for (const char* s : { "attn_o.", "ffn_gate_up.", "ffn_down." })
        b.encs.push_back({ s, w4 });
}

int count_writes_to(const RadBuilder& b, const RadCtx& c, int buf) {
    int n = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && std::find(o->writes.begin(), o->writes.end(), (rad_buf)buf) != o->writes.end())
            ++n;
    }
    return n;
}

rad_kvgroup kv_named(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.kv_groups.size(); ++i)
        if (b.kv_groups[i].name == name) return (rad_kvgroup)(i + 1);
    return 0;
}

}  /* namespace */

/* The 31B declares z-lab's drafter as a block drafter past its trunk: five layers at 12..16 on a KV
 * group of their own, the trunk's taps in one plane, its embedding read at the reference's scale,
 * and the depth the block was trained at, the only one it serves, which probe names for rad-convert
 * and `--num-speculative-tokens auto`. Without speculation none of it is declared. */
TEST(gemma4_31b_declares_dflash_as_a_block_drafter_past_its_trunk) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (const bool served : { true, false }) {
    RadBuilder b;
    g431df_encs(b, true, false);
    if (!served) b.refuse.insert("attn_paged_qrow");
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.drafter_name, std::string("dflash"));
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
    CHECK_EQ(b.drafter.depth, 15);
    CHECK_EQ((int)b.drafter.block, 16);
    CHECK_EQ(b.drafter.mask_token, 4);
    CHECK(!b.drafter_shared);
    CHECK_EQ(b.drafter.proposal, (rad_buf)buf_named(b, "dflash_tokens"));

    /* its own cache, the five layers past the trunk's twelve bound to it */
    const rad_kvgroup kvd = kv_named(b, "kv_draft");
    REQUIRE(kvd != 0);
    CHECK_EQ((int)b.kv_groups[kvd - 1].d.kind, (int)RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[kvd - 1].d.head_dim, 128);
    /* 64 heads over 8 kv heads is gqa 8. Where a card serves it under attn_paged_qrow (libr4d's
     * head-128 kernel with two query groups a kv head), the cache holds each kv head once and
     * attention runs at gqa 8. Where not, libr4d's head-128 attention takes 4 queries per kv head
     * at a block of 16: the cache holds each kv head twice, the projections stay 8 heads wide, and
     * every attention and cache write of the drafter sees 16 kv heads at gqa 4. */
    const int64_t heads = served ? 8 : 16;
    CHECK_EQ(b.kv_groups[kvd - 1].d.n_head_kv, heads);
    CHECK_EQ(b.bufs[buf_named(b, "df_k") - 1].d.shape[1], 8 * 128);
    if (served) CHECK_EQ(buf_named(b, "df_k4"), 0);
    else        CHECK_EQ(b.bufs[buf_named(b, "df_k4") - 1].d.shape[1], 16 * 128);
    {
        int n_attn = 0, n_store = 0;
        for (const auto& o : b.ops) {
            /* the asking ops (qrow_serves) name no buffer */
            if (o.op == (served ? "attn_paged_qrow" : "attn_paged") && pint(&o, "head_dim") == 128 &&
                !o.reads.empty()) {
                ++n_attn;
                CHECK_EQ(pint(&o, "gqa"), served ? 8 : 4);
                CHECK_EQ(pint(&o, "n_head"), 64);
            }
            if (o.op == "kv_store" && pint(&o, "head_dim") == 128) {
                ++n_store;
                CHECK_EQ(pint(&o, "n_head_kv"), heads);
            }
        }
        CHECK_EQ(n_attn, 2);    /* the sliding layers' and the full layer's */
        CHECK_EQ(n_store, 2);   /* the context pass's and the query pass's */
    }
    std::vector<int> bound;
    for (const auto& [layer, g] : b.binds) if (g == kvd) bound.push_back(layer);
    CHECK(bound == (std::vector<int>{ 12, 13, 14, 15, 16 }));

    /* three taps of 5376 a row */
    const int src = buf_named(b, "dflash_src");
    REQUIRE(src > 0);
    CHECK_EQ(b.bufs[src - 1].d.shape[1], 3 * 5376);

    /* the drafter embeds the anchor and the masks through the TRUNK's table at 73.5 */
    const RecOp* te = find_op(b, "embed_lookup", 0);
    const RecOp* de = find_op(b, "embed_lookup", 1);
    REQUIRE(te != nullptr && de != nullptr);
    CHECK(te->w == de->w);
    CHECK_NEAR(pflt(de, "mult"), 73.5, 1e-9);

    /* one name map a declared weight: a COPY map repeated is refused by the real builder */
    std::map<std::string, int> maps;
    for (const auto& mp : b.maps) if (mp.mode == RAD_MAP_COPY) ++maps[mp.declared];
    for (const auto& [name, n] : maps) CHECK_EQ(name + " x" + std::to_string(n), name + " x1");

    RadArchProbe p{};
    CHECK_OK(gemma4_bf16::probe(&meta, &p));
    CHECK_EQ(p.draft_depth, 15);
    CHECK_EQ(p.draft_depth_fixed, 1);
    CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);

    /* no speculation: no drafter, no taps */
    RadBuilder b0;
    g431df_encs(b0, true, false);
    RadBuildCtx c0 = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b0, &meta, &c0));
    CHECK_EQ(buf_named(b0, "dflash_src"), 0);
    CHECK(b0.drafter_name.empty());

    /* and a depth the block was not trained at is refused by name */
    RadBuilder b4;
    g431df_encs(b4, true, false);
    RadBuildCtx c4 = build_ctx();
    c4.max_spec = 4;
    CHECK_EQ(gemma4_bf16::declare(&b4, &meta, &c4), RAD_E_UNSUPPORTED);
    }
}

/* The taps on the w4a8 trunk are declared in the layer they copy, so `dflash_src` is live from its
 * first write on (the arch/gemma4 trap that cost 98% of the drafts). Each is issued once its
 * layer's output is whole, after the junction that adds the feed-forward, applies the layer scalar
 * and quantises the next layer's input, so the copy is HF's hidden_states[l + 1]. */
TEST(gemma4_31b_dflash_taps_the_stream_the_junction_leaves) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    g431df_encs(b, true, false);
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const int src = buf_named(b, "dflash_src"), gu = buf_named(b, "gate_up");
    REQUIRE(src > 0 && gu > 0);
    int first_src = -1, last_gu = -1;
    for (size_t i = 0; i < b.ops.size(); ++i)
        for (rad_buf w : b.ops[i].writes) {
            if ((int)w == src && first_src < 0) first_src = (int)i;
            if ((int)w == gu) last_gu = (int)i;
        }
    REQUIRE(first_src >= 0 && last_gu >= 0);
    CHECK(first_src < last_gu);

    RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
    RadBatch batch = decode_batch();
    c.batch = &batch;
    gemma4_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());
    CHECK_EQ(count_writes_to(b, c, src), 3);
    int t = 0;
    for (size_t i = 0; i < c.issues.size(); ++i) {
        const RecOp* o = b.op_of(c.issues[i].op);
        if (!o || std::find(o->writes.begin(), o->writes.end(), (rad_buf)src) == o->writes.end())
            continue;
        REQUIRE(i > 0);
        CHECK_EQ(b.op_of(c.issues[i - 1].op)->op, std::string("rmsnorm_quant_i8"));
        REQUIRE(c.issues[i].opd.size() == 2);
        CHECK_EQ(c.issues[i].opd[1].handle, (uint32_t)src);
        CHECK_EQ(c.issues[i].opd[1].offset, (int64_t)t * 5376);   /* the tap's own column slice */
        ++t;
    }
}

/* A draft pass runs only the drafter: the context pass from the taps into the drafter's cache,
 * the query pass over the block to the proposal. */
TEST(gemma4_31b_dflash_passes_run_the_drafter_and_nothing_else) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    g431df_encs(b, true, true);
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    const int src = buf_named(b, "dflash_src"), tok = buf_named(b, "dflash_tokens");

    for (int pass : { -1, 1 }) {
        RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
        RadBatch batch = decode_batch();
        batch.draft_pass = pass;
        c.batch = &batch;
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        REQUIRE(!ops.empty());
        CHECK_EQ(count_issued(ops, "gelu_mul_quant_i8"), 0);   /* no trunk layer */
        CHECK_EQ(count_issued(ops, "kv_store"), 5);
        if (pass < 0) {
            CHECK_EQ(ops.front(), std::string("quant_act_i8"));
            const RecOp* o = b.op_of(c.issues.front().op);
            CHECK(std::find(o->reads.begin(), o->reads.end(), (rad_buf)src) != o->reads.end());
            CHECK_EQ(count_issued(ops, "attn_paged") + count_issued(ops, "attn_paged_qrow"), 0);
        } else {
            CHECK_EQ(ops.front(), std::string("embed_lookup"));
            CHECK_EQ(count_issued(ops, "attn_paged") + count_issued(ops, "attn_paged_qrow"), 5);
            CHECK_EQ(count_writes_to(b, c, tok), 1);
        }
    }
}

/* Q8, the drafter on the int8 grid. Every linear is gemm_nt_q at w8a8 over quant_act_i8's per-row
 * activation, each residual add, norm and quantiser is one rmsnorm_quant_i8 (ten per draft pass, as
 * rmsnorm_quant_fp8 is on the E4M3 grid), and nothing on the E4M3 grid is declared. The
 * gate_up|silu|quant fold is fp8a8 only, so the int8 grid keeps silu_mul and the quantiser. Where
 * rmsnorm_quant_i8 is refused, the chain runs. */
TEST(gemma4_31b_dflash_on_the_int8_grid_reads_w8a8_and_folds_its_norms) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        g431df_encs(b, true, true);
        if (refuse) b.refuse.insert("rmsnorm_quant_i8");
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        CHECK(b.bufs[buf_named(b, "df_hq") - 1].d.dtype == RAD_I8);
        CHECK_EQ(b.bufs[buf_named(b, "df_hs") - 1].d.shape[1], 1);   /* a scale a row */
        const RecWeight* wq = find_w(b, "dflash.blk.0.attn_q.weight");
        const RecWeight* sq = find_w(b, "dflash.blk.0.attn_q.scale");
        REQUIRE(wq != nullptr && sq != nullptr);
        CHECK(wq->d.dtype == RAD_I8 && sq->d.dtype == RAD_F16);
        CHECK_EQ(sq->d.shape[1], 5376 / 128);

        RadCtx c; c.b = &b; c.rank = 0; c.world = 1;
        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        c.batch = &batch;
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        CHECK_EQ(count_issued(ops, "rmsnorm_quant_i8"), refuse ? 0 : 10);
        CHECK_EQ(count_issued(ops, "add"), refuse ? 10 : 1);
        /* the attention output's and the gated product's a layer, x's twice a layer in the chain,
         * and the head's copy of the gathered rows */
        CHECK_EQ(count_issued(ops, "quant_act_i8"), (refuse ? 20 : 10) + 1);
        CHECK_EQ(count_issued(ops, "silu_mul"), 5);
        for (const char* fp : { "quant_act_fp8", "rmsnorm_quant_fp8", "gemm_nt_q_gated" })
            CHECK_EQ(count_issued(ops, fp), 0);
        int q = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "gemm_nt_q") continue;
            CHECK_EQ(pstr(o, "dtype"), std::string("w8a8"));
            ++q;
        }
        CHECK_EQ(q, 5 * 6 + 1);   /* six linears a layer, and the head */
        if (refuse) continue;

        /* the fused form's operands: rmsnorm_quant_i8's eight, the delta and the residual past
         * layer 0, and no junction gains */
        const uint32_t x = (uint32_t)buf_named(b, "df_x"), h = (uint32_t)buf_named(b, "df_h");
        int k = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "rmsnorm_quant_i8") continue;
            REQUIRE(i.opd.size() == 8);
            CHECK_EQ(i.opd[0].handle, k++ == 0 ? x : h);
            CHECK_EQ((int)i.opd[6].kind, (int)RAD_OPK_NONE);
            CHECK_EQ((int)i.opd[7].kind, (int)RAD_OPK_NONE);
        }

        /* and the context pass: two quantisers, eleven w8a8 GEMMs (fc, then k and v a layer) */
        RadCtx cc; cc.b = &b; cc.rank = 0; cc.world = 1;
        RadBatch cb = decode_batch();
        cb.draft_pass = -1;
        cc.batch = &cb;
        gemma4_bf16::step(&cc, &cb);
        CHECK(cc.step_fail.empty());
        const std::vector<std::string> cops = issued_ops(b, cc);
        CHECK_EQ(count_issued(cops, "quant_act_i8"), 2);
        CHECK_EQ(count_issued(cops, "gemm_nt_q"), 11);
    }
}

/* The drafter's linears take one grid: one quantised activation feeds several of them, so an fc on
 * the E4M3 grid and layers on the int8 one would read pairs nothing writes. Refused by name. */
TEST(gemma4_31b_dflash_refuses_linears_on_two_grids) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    RadBuilder b;
    g431df_encs(b, true, false);
    b.encs[0].second = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);   /* fc rotated, the rest not */
    rad_enc_copy_str(b.encs[0].second.transform, "fwht128");
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_E_DTYPE);
}

/* The head's int8 copy: where the container holds one, the gathered rows are quantised a row at a
 * time and read by gemm_nt_q at w8a8 over it; otherwise the head reads the tied bf16 table. */
TEST(gemma4_31b_dflash_head_reads_the_int8_copy_where_the_container_holds_one) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int copy = 0; copy < 2; ++copy) {
        RadBuilder b;
        g431df_encs(b, true, copy == 1);
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        const int dlog = buf_named(b, "df_dlog");
        const RecOp* head = nullptr;
        for (const auto& o : b.ops)
            if (o.resolved &&
                std::find(o.writes.begin(), o.writes.end(), (rad_buf)dlog) != o.writes.end())
                head = &o;
        REQUIRE(head != nullptr);
        CHECK_EQ(head->op, std::string(copy ? "gemm_nt_q" : "gemm_nt"));
        if (!copy) continue;
        CHECK_EQ(pstr(head, "dtype"), std::string("w8a8"));
        const int selq = buf_named(b, "df_selq"), sels = buf_named(b, "df_sels");
        REQUIRE(selq > 0 && sels > 0);
        CHECK(b.bufs[selq - 1].d.dtype == RAD_I8);
        CHECK_EQ(b.bufs[sels - 1].d.shape[1], 1);
        const RecWeight* hs = find_w(b, "dflash.draft_head.scale");
        REQUIRE(hs != nullptr);
        CHECK(hs->d.dtype == RAD_F16);
        CHECK_EQ(hs->d.shape[0], 262144);
        CHECK_EQ(hs->d.shape[1], 5376 / 128);
    }
}

/* At four ranks, as the R9700 box serves it: the drafter's heads split 16 / 2 kv per rank, its head
 * is vocab-sharded with the trunk's and merged by pair_argmax, and both 8-bit grids declare (the
 * int8 one's prefill GEMM takes N in 256s, which a 256-row k and v still are). */
TEST(gemma4_31b_dflash_at_four_ranks) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int i8 = 1; i8 < 2; ++i8)
    for (const bool served : { true, false }) {
        RadBuilder b;
        g431df_encs(b, i8 == 1, true);
        if (!served) b.refuse.insert("attn_paged_qrow");
        RadBuildCtx ctx = build_ctx(1, 4);
        ctx.max_spec = 15;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        const rad_kvgroup kvd = kv_named(b, "kv_draft");
        REQUIRE(kvd != 0);
        /* two kv heads a rank: once at gqa 8 where attn_paged_qrow serves it, else each held twice
         * at gqa 4 (the declares test says why) */
        CHECK_EQ(b.kv_groups[kvd - 1].d.n_head_kv, served ? 2 : 4);
        CHECK_EQ(b.bufs[buf_named(b, "df_q") - 1].d.shape[1], 16 * 128);
        CHECK_EQ(b.bufs[buf_named(b, "df_k") - 1].d.shape[1], 2 * 128);
        CHECK(buf_named(b, "df_dgath") > 0);
        for (const auto& o : b.ops)
            if ((o.op == "attn_paged" || o.op == "attn_paged_qrow") && pint(&o, "head_dim") == 128 &&
                !o.reads.empty())
                CHECK_EQ(pint(&o, "gqa"), served ? 8 : 4);

        RadCtx c; c.b = &b; c.rank = 1; c.world = 4;
        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        c.batch = &batch;
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        CHECK_EQ(count_issued(ops, "pair_argmax"), 1);
        CHECK_EQ(count_issued(ops, "all_gather"), 1);
        /* the embedding's reduction and two a layer */
        CHECK_EQ(count_issued(ops, "all_reduce"), 1 + 2 * 5);
        /* k and v into the cache's layout: two heads, two copies each, a layer -- or none */
        CHECK_EQ(count_issued(ops, "cast"), served ? 0 : 5 * 2 * (2 * 2));
    }
}

/* The 31B's prefill in two slices (gemma4_bf16, trunk_overlap). The order and the work: every
 * trunk reduction rides lane 1 behind its producer and ahead of its consumer and nothing else does;
 * no two issues on different lanes reach the same rows unordered (`h` carries both reductions); the
 * cut leaves both slices at least 512 rows; and the slices issue exactly the rows the whole step
 * issues, the drafter's taps a slice each. Only `h` is taken out of the arena's sharing. A step
 * under two slices, and a decode step, run whole with their reductions on lane 0. At two ranks the
 * slices take the plain junctions, the whole step the ones with the all-reduce folded in. */
TEST(a_gemma4_31b_prefill_step_runs_as_two_slices_with_its_reductions_on_the_second_lane) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    const int64_t E = 5376, L = 12;
    for (int world : { 4, 2 }) {
        auto model = [&](RadBuilder& b, bool lossy) {
            g431df_encs(b, true, true);
            RadBuildCtx ctx = build_ctx(0, world);
            ctx.max_tok  = 4096;
            ctx.max_spec = 15;
            if (lossy) ctx.tp_wire_lossy = 1;   /* nothing shows a slice its bits on that wire */
            return gemma4_bf16::declare(&b, &meta, &ctx);
        };
        auto run = [&](RadBuilder& b, RadCtx& c, RadBatch& batch) {
            c.b = &b; c.rank = 0; c.world = world; c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
        };

        struct Case { int64_t T, cut; };
        for (const Case k : { Case{4096, 2048}, Case{1100, 512}, Case{1024, 512}, Case{1023, 0} }) {
            const int32_t out[1] = { (int32_t)(k.T - 1) };
            RadBuilder b;
            CHECK_OK(model(b, false));
            CHECK(noted(b, "prefill overlap: a step of 1024 rows or more runs as two slices"));
            /* `h`, and slice A's context end a KV group the attention reads (per-slice
             * attention: the sliding and the global class) */
            REQUIRE(b.concurrent.size() == 3);
            CHECK_EQ(b.concurrent[0], (rad_buf)buf_named(b, "h"));
            CHECK_EQ(b.concurrent[1], (rad_buf)buf_named(b, "slice_used_0"));
            CHECK_EQ(b.concurrent[2], (rad_buf)buf_named(b, "slice_used_1"));
            RadBatch batch = one_prompt(k.T, out);
            RadCtx c;
            run(b, c, batch);
            CHECK(c.step_fail.empty());
            CHECK_EQ(c.lane, 0);
            CHECK_EQ(races(b, c), 0);

            std::vector<int64_t> lane1;
            int other1 = 0;
            for (const auto& i : c.issues) {
                if (i.lane != 1) continue;
                const RecOp* o = b.op_of(i.op);
                REQUIRE(o != nullptr);
                if (o->op == "all_reduce") {
                    lane1.push_back(i.n);
                    /* the narrow launch beside compute, not the wide one op_ar resolves to */
                    CHECK_EQ(pint(o, "overlap"), 1);
                } else {
                    ++other1;
                }
            }
            CHECK_EQ(other1, 0);
            if (k.cut == 0) {
                CHECK(lane1.empty());
            } else {
                REQUIRE(lane1.size() == (size_t)(4 * L));
                for (int64_t l = 0; l < L; ++l) {
                    CHECK_EQ(lane1[4 * l + 0], k.cut * E);              /* attention, slice A */
                    CHECK_EQ(lane1[4 * l + 1], (k.T - k.cut) * E);      /* attention, slice B */
                    CHECK_EQ(lane1[4 * l + 2], k.cut * E);              /* feed-forward, slice A */
                    CHECK_EQ(lane1[4 * l + 3], (k.T - k.cut) * E);      /* feed-forward, slice B */
                }
                CHECK_EQ(count_issued(issued_ops(b, c), "ar_rmsnorm_quant_i8"), 0);
            }

            /* The whole step, on the lossy wire: the same rows by op -- every junction, GEMM and
             * quantiser over the same rows, the reductions over the same elements. */
            RadBuilder w;
            CHECK_OK(model(w, true));
            CHECK(noted(w, "prefill overlap: off (the lossy all-reduce wire)"));
            CHECK(w.concurrent.empty());
            RadBatch wbatch = one_prompt(k.T, out);
            RadCtx wc;
            run(w, wc, wbatch);
            CHECK(wc.step_fail.empty());
            for (const auto& i : wc.issues) CHECK_EQ(i.lane, 0);
            std::map<std::string, int64_t> sliced = rows_by_op(b, c, k.T), whole = rows_by_op(w, wc, k.T);
            if (k.cut) {
                /* attention once a slice, slice A's over its context end: one kv_len_trim a KV
                 * group, which the whole step has no use for */
                CHECK_EQ((int)ns_of(b, c, "kv_len_trim").size(), 2);
                CHECK_EQ((int)ns_of(b, c, "attn_paged").size() + (int)ns_of(b, c, "attn_paged_qrow").size(),
                         2 * (int)L);
                sliced.erase("kv_len_trim");
            }
            if (world == 2 && k.cut == 0) {
                /* At two ranks on the exact wire a step that runs whole folds all but two of its
                 * reductions -- the embedding's and the last layer's feed-forward -- into its
                 * 2L - 1 junctions; the lossy wire folds none. */
                CHECK_EQ(sliced["ar_rmsnorm_quant_i8"], (2 * L - 1) * k.T);
                sliced.erase("ar_rmsnorm_quant_i8");
                sliced["all_reduce"] += (2 * L - 1) * k.T * E;
                sliced["rmsnorm_quant_i8"] += (2 * L - 1) * k.T;
            }
            CHECK(sliced == whole);
            CHECK_EQ((int)ns_of(b, c, "cast").size(), k.cut ? 6 : 3);   /* three taps, a slice each */
        }

        /* a library without kv_len_trim: the slices keep each layer's attention whole between them,
         * with only `h` out of the arena's sharing and the reductions still on lane 1 */
        {
            const int32_t out[1] = { 4095 };
            RadBuilder b;
            b.undeclared.insert("kv_len_trim");
            CHECK_OK(model(b, false));
            REQUIRE(b.concurrent.size() == 1);
            CHECK_EQ(b.concurrent[0], (rad_buf)buf_named(b, "h"));
            RadBatch batch = one_prompt(4096, out);
            RadCtx c;
            run(b, c, batch);
            CHECK(c.step_fail.empty());
            CHECK_EQ(races(b, c), 0);
            CHECK(ns_of(b, c, "kv_len_trim").empty());
            CHECK_EQ((int)ns_of(b, c, "attn_paged").size() + (int)ns_of(b, c, "attn_paged_qrow").size(),
                     (int)L);
            int lane1 = 0;
            for (const auto& i : c.issues) lane1 += i.lane == 1;
            CHECK_EQ(lane1, 4 * (int)L);
        }

        /* a decode step runs whole: every reduction on lane 0; the prefetches beside them
         * (prefetch_beside) follow lane 0's GEMM before each -- on lane 0 itself where launches
         * chain (prefetch_lane0), else on lane 1 behind a join from that GEMM */
        for (const bool pf_lane0 : { false, true }) {
            const ChainOn chain(pf_lane0);
            RadBuilder b;
            CHECK_OK(model(b, false));
            RadBatch d = decode_batch();
            RadCtx c;
            run(b, c, d);
            CHECK(c.step_fail.empty());
            const std::vector<std::string> ops = issued_ops(b, c);
            int n_pf = 0;
            for (size_t i = 0; i < c.issues.size(); ++i) {
                const bool pf = ops[i] == "prefetch";
                CHECK_EQ(c.issues[i].lane, pf && !pf_lane0 ? 1 : 0);
                if (!pf) continue;
                ++n_pf;
                CHECK(i > 0 && ops[i - 1] == "gemm_nt_q" && ordered(c, i - 1, i));
            }
            CHECK_EQ(n_pf, 3 * (int)L);
            CHECK_EQ((int)c.joins.size(), pf_lane0 ? 0 : n_pf);
            CHECK_EQ(races(b, c), 0);
            CHECK_EQ(count_issued(ops, "ar_rmsnorm_quant_i8"), 2 * (int)L - 1);
        }
    }

    /* the overlap needs the fused junctions: a bf16 trunk keeps its chain and runs whole */
    RadBuilder b;
    RadModelMeta plain = gemma4_meta();
    RadBuildCtx ctx = build_ctx(0, 4);
    ctx.max_tok = 4096;
    CHECK_OK(gemma4_bf16::declare(&b, &plain, &ctx));
    CHECK(noted(b, "prefill overlap: off (the residual junctions are not fused)"));
    CHECK(b.concurrent.empty());

    /* and the lossy wire, which gives no slice the whole step's bits: every step whole, nothing on
     * the second lane */
    RadBuilder sw;
    g431df_encs(sw, true, true);
    RadBuildCtx sctx = build_ctx(0, 4);
    sctx.max_tok       = 4096;
    sctx.max_spec      = 15;
    sctx.tp_wire_lossy = 1;
    CHECK_OK(gemma4_bf16::declare(&sw, &meta, &sctx));
    CHECK(noted(sw, "prefill overlap: off (the lossy all-reduce wire)"));
    CHECK(sw.concurrent.empty());
    const int32_t out[1] = { 4095 };
    RadBatch big = one_prompt(4096, out);
    RadCtx sc; sc.b = &sw; sc.rank = 0; sc.world = 4; sc.batch = &big;
    gemma4_bf16::step(&sc, &big);
    CHECK(sc.step_fail.empty());
    for (const auto& i : sc.issues) CHECK_EQ(i.lane, 0);
}

/* The 31B on the grids that read quant_act_i8g's pair -- int8 a 128 (i8a8), a4 and its five-bit
 * form -- fuses its residual junctions as w4a8 does, through the `_i8g` twins: one
 * rmsnorm_quant_i8g a junction (2L), one gelu_mul_quant_i8g a feed-forward, at two ranks the
 * all-reduce folded in (ar_rmsnorm_quant_i8g on all but the embedding's and the last
 * feed-forward's) and a prefill of 1024 rows or more in two slices. An a4 or w5 linear reads bf16
 * up to 64 rows: in such a step every junction also writes the norm output into `h`, the row each
 * linear's cast reads, and the feed-forward keeps gelu_mul and down's own cast; above that no
 * junction writes `h` and no linear casts. Either way the fused step runs the unfused step's GEMMs
 * and casts over the same rows. Without the twins (a CPU-only engine) the chains are declared and no
 * undeclared op is asked for. */
TEST(the_gemma4_31b_fuses_its_junctions_on_the_grouped_int8_grids) {
    RadEncoding u4 = rad_enc_affine(RAD_U4, RAD_F16, 1, 128);
    rad_enc_add_plane(&u4, "zero", RAD_U8, 1, 128);
    RadEncoding u5 = rad_enc_affine(RAD_U5, RAD_F16, 1, 128);
    rad_enc_add_plane(&u5, "zero", RAD_U8, 1, 128);
    struct Fmt { const char* name; RadEncoding e; bool band; };
    const Fmt fmts[] = { { "i8g", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128), false },
                         { "a4", u4, true }, { "w5", u5, true } };
    const int64_t L = 12;
    const char* const twins[] = { "rmsnorm_quant_i8g", "gelu_mul_quant_i8g", "ar_rmsnorm_quant_i8g" };
    for (const Fmt& f : fmts)
        for (int world : { 1, 2 }) {
            auto model = [&](RadBuilder& b, bool fused, bool lossy) {
                b.encs = { { "attn_qkv.", f.e }, { "attn_o.", f.e }, { "ffn_gate_up.", f.e },
                           { "ffn_down.", f.e } };
                if (!fused) for (const char* op : twins) b.undeclared.insert(op);
                RadModelMeta meta = gemma4_meta();
                RadBuildCtx ctx = build_ctx(0, world);
                ctx.max_tok = 4096;
                if (lossy) ctx.tp_wire_lossy = 1;
                return gemma4_bf16::declare(&b, &meta, &ctx);
            };
            {
                RadBuilder d;
                CHECK_OK(model(d, true, false));
                const gemma4_bf16::Model& m = gemma4_bf16::g_model[0];
                CHECK(m.fuse);
                CHECK_EQ(m.fuse_ar, world == 2);
                CHECK_EQ(m.overlap, world == 2);
                if (world == 2)
                    CHECK(noted(d, "prefill overlap: a step of 1024 rows or more runs as two slices"));
                CHECK_EQ(count_op(d, "rmsnorm_quant_i8g"), 2 * (int)L);
                CHECK_EQ(count_op(d, "gelu_mul_quant_i8g"), (int)L);
                CHECK_EQ(count_op(d, "ar_rmsnorm_quant_i8g"), world == 2 ? 2 * (int)L - 1 : 0);
                for (const char* op : { "rmsnorm_quant_i8", "gelu_mul_quant_i8", "ar_rmsnorm_quant_i8" })
                    CHECK_EQ(count_op(d, op), 0);
                /* a junction declares `h` written only where a linear after it has a decode band;
                 * the all-reduce form writes `h` (its `x`, the reduced sum) either way, and the bf16
                 * row besides in the band */
                const rad_buf h = (rad_buf)buf_named(d, "h");
                for (const RecOp& o : d.ops) {
                    if (o.op == "rmsnorm_quant_i8g")
                        CHECK_EQ(std::find(o.writes.begin(), o.writes.end(), h) != o.writes.end(), f.band);
                    if (o.op == "ar_rmsnorm_quant_i8g") {
                        CHECK_EQ(o.writes.size(), (size_t)(f.band ? 5 : 4));
                        CHECK(param(&o, "sp") == nullptr);
                    }
                }
            }
            {
                RadBuilder u;
                CHECK_OK(model(u, false, false));
                CHECK(!gemma4_bf16::g_model[0].fuse);
                CHECK(u.asked_undeclared.empty());
            }

            /* step() runs the model declare last made, so each step below follows its own */
            auto linears = [](const RadBuilder& bb, const RadCtx& cc, int64_t TT) {
                std::map<std::string, int64_t> w;
                for (const auto& i : cc.issues) {
                    const RecOp* o = bb.op_of(i.op);
                    if (!o) continue;
                    if (o->op == "gemm_nt_q") w["gemm_nt_q " + pstr(o, "dtype")] += i.n < 0 ? TT : i.n;
                    if (o->op == "cast" && pstr(o, "to") == "f16") w["cast f16"] += i.n < 0 ? TT : i.n;
                }
                return w;
            };
            for (const int64_t T : { (int64_t)1, (int64_t)16, (int64_t)64, (int64_t)65, (int64_t)200,
                                     (int64_t)4096 }) {
                const int32_t out[1] = { (int32_t)(T - 1) };
                const bool band = f.band && T <= 64;
                const bool sliced = world == 2 && T >= 1024;
                RadBuilder b;
                CHECK_OK(model(b, true, false));
                RadBatch batch = one_prompt(T, out);
                RadCtx c; c.b = &b; c.rank = 0; c.world = world; c.batch = &batch;
                gemma4_bf16::step(&c, &batch);
                CHECK(c.step_fail.empty());
                CHECK_LINT(b, &c);
                CHECK_EQ(races(b, c), 0);
                const std::vector<std::string> ops = issued_ops(b, c);
                const rad_buf h = (rad_buf)buf_named(b, "h");
                const int k = sliced ? 2 : 1;   /* issues a row op takes: one a slice */
                const int n_ar = count_issued(ops, "ar_rmsnorm_quant_i8g");
                CHECK_EQ(count_issued(ops, "rmsnorm_quant_i8g") + n_ar, 1 + (k * (2 * (int)L - 1)));
                CHECK_EQ(n_ar, world == 2 && !sliced ? 2 * (int)L - 1 : 0);
                CHECK_EQ(count_issued(ops, "gelu_mul_quant_i8g"), band ? 0 : k * (int)L);
                CHECK_EQ(count_issued(ops, "gelu_mul"), band ? (int)L : 0);
                /* o_proj's own quantiser only: every other linear reads a junction's pair */
                CHECK_EQ(count_issued(ops, "quant_act_i8g"), band ? 0 : k * (int)L);
                int y_written = 0;
                for (size_t i = 0; i < c.issues.size(); ++i) {
                    if (ops[i] != "rmsnorm_quant_i8g" && ops[i] != "ar_rmsnorm_quant_i8g") continue;
                    REQUIRE(c.issues[i].opd.size() > 5);
                    const RadOperand& y = c.issues[i].opd[5];
                    if (y.kind != RAD_OPK_BUF) continue;
                    ++y_written;
                    CHECK_EQ((rad_buf)y.handle, h);
                }
                CHECK_EQ(y_written, band ? 2 * (int)L : 0);
                /* Every int8 pair a step writes or reads, a slice's included, is the same rows of codes
                 * and of scales: a scale row is a codes row's 128th. */
                int pairs = 0;
                for (size_t i = 0; i < c.issues.size(); ++i) {
                    const RecOp* o = b.op_of(c.issues[i].op);
                    if (!o) continue;
                    int iq = -1;
                    if (o->op == "rmsnorm_quant_i8g" || o->op == "ar_rmsnorm_quant_i8g") iq = 3;
                    else if (o->op == "gelu_mul_quant_i8g" || o->op == "quant_act_i8g") iq = 1;
                    else if (o->op == "gemm_nt_q" && pstr(o, "dtype") != "w4a16_asym" &&
                             pstr(o, "dtype") != "w5a16") iq = 0;
                    if (iq < 0) continue;
                    REQUIRE(c.issues[i].opd.size() > (size_t)iq + 1);
                    const RadOperand& q = c.issues[i].opd[(size_t)iq];
                    const RadOperand& s = c.issues[i].opd[(size_t)iq + 1];
                    CHECK_EQ(s.offset * 128, q.offset);
                    CHECK_EQ(s.rows, q.rows);
                    ++pairs;
                }
                CHECK(pairs > 0);
                int lane1 = 0;
                for (const auto& i : c.issues) lane1 += i.lane == 1;
                CHECK_EQ(lane1, sliced ? 4 * (int)L : 0);
                const std::map<std::string, int64_t> fused_lin = linears(b, c, T);
                const int64_t casts = fused_lin.count("cast f16") ? fused_lin.at("cast f16") : 0;
                CHECK_EQ(casts, band ? 4 * L * T : 0);

                /* the same GEMMs and casts over the same rows as the unfused step */
                RadBuilder u;
                CHECK_OK(model(u, false, false));
                RadBatch ub = one_prompt(T, out);
                RadCtx uc; uc.b = &u; uc.rank = 0; uc.world = world; uc.batch = &ub;
                gemma4_bf16::step(&uc, &ub);
                CHECK(uc.step_fail.empty());
                CHECK_LINT(u, &uc);
                CHECK(fused_lin == linears(u, uc, T));

                /* the slices issue the rows the whole step issues (the lossy wire runs it whole) */
                if (sliced) {
                    std::map<std::string, int64_t> sl = rows_by_op(b, c, T);
                    RadBuilder w;
                    CHECK_OK(model(w, true, true));
                    CHECK(!gemma4_bf16::g_model[0].overlap);
                    RadBatch wb = one_prompt(T, out);
                    RadCtx wc; wc.b = &w; wc.rank = 0; wc.world = world; wc.batch = &wb;
                    gemma4_bf16::step(&wc, &wb);
                    CHECK(wc.step_fail.empty());
                    sl.erase("kv_len_trim");
                    CHECK(sl == rows_by_op(w, wc, T));
                }
            }
        }

    /* At four ranks libr4d has no ar_rmsnorm_quant_i8g row: the junctions fuse, the all-reduce in
     * front of each is issued apart, and the prefill still runs as two slices. */
    for (const Fmt& f : fmts)
        for (const int64_t T : { (int64_t)1, (int64_t)4096 }) {
            const int32_t out[1] = { (int32_t)(T - 1) };
            RadBuilder b;
            b.refuse.insert("ar_rmsnorm_quant_i8g");
            b.encs = { { "attn_qkv.", f.e }, { "attn_o.", f.e }, { "ffn_gate_up.", f.e }, { "ffn_down.", f.e } };
            RadModelMeta meta = gemma4_meta();
            RadBuildCtx ctx = build_ctx(0, 4);
            ctx.max_tok = 4096;
            CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
            const gemma4_bf16::Model& m = gemma4_bf16::g_model[0];
            CHECK(m.fuse);
            CHECK(!m.fuse_ar);
            CHECK(m.overlap);
            CHECK(noted(b, "residual junctions: the all-reduce in front of each issued apart"));
            RadBatch batch = one_prompt(T, out);
            RadCtx c; c.b = &b; c.rank = 0; c.world = 4; c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
            CHECK(c.step_fail.empty());
            CHECK_LINT(b, &c);
            CHECK_EQ(races(b, c), 0);
            const std::vector<std::string> ops = issued_ops(b, c);
            CHECK_EQ(count_issued(ops, "ar_rmsnorm_quant_i8g"), 0);
            CHECK_EQ(count_issued(ops, "rmsnorm_quant_i8g"), T > 1 ? 1 + 2 * (2 * 12 - 1) : 2 * 12);
            CHECK_EQ(count_issued(ops, "all_reduce"), T > 1 ? 1 + 4 * 12 : 1 + 2 * 12);
        }

    /* The E2B/E4B form (KV-shared layers, the per-layer branch) on the same grids. With q, k and v
     * apart, i8g's k and v read q's pair and the junctions fuse; an a4 linear shares no pair, so an
     * a4 trunk with q, k and v apart keeps its chain. With q|k|v one linear (and the KV-shared
     * layers' q on the same grid) it fuses, and its decode band reads the junctions' bf16 output. */
    struct E4 { const Fmt* f; bool qkv; bool fuse; };
    for (const E4 k : { E4{ &fmts[0], false, true }, E4{ &fmts[1], false, false },
                        E4{ &fmts[2], true, true } })
        for (int world : { 1, 2 })
            for (const int64_t T : { (int64_t)4, (int64_t)200 }) {
                RadBuilder b;
                const RadEncoding& e = k.f->e;
                b.encs = { { "attn_q.", e }, { "attn_o.", e }, { "ffn_gate_up.", e }, { "ffn_down.", e } };
                if (k.qkv) b.encs.push_back({ "attn_qkv.", e });
                else       for (const char* x : { "attn_k.", "attn_v." }) b.encs.push_back({ x, e });
                RadModelMeta meta = gemma4_e4b_meta();
                RadBuildCtx ctx = build_ctx(0, world);
                CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
                CHECK_EQ(gemma4_bf16::g_model[0].fuse, k.fuse);
                CHECK(!gemma4_bf16::g_model[0].overlap);
                static const int32_t outs[] = { 99, 199 };
                RadBatch batch = T == 4 ? decode_batch() : prefill_batch();
                if (T != 4) {
                    batch.n_tok = 200; batch.max_q_len = 100; batch.max_ctx_len = 100;
                    batch.out_ids = outs;
                }
                RadCtx c; c.b = &b; c.rank = 0; c.world = world; c.batch = &batch;
                gemma4_bf16::step(&c, &batch);
                CHECK(c.step_fail.empty());
                CHECK_LINT(b, &c);
                const std::vector<std::string> ops = issued_ops(b, c);
                CHECK_EQ(count_issued(ops, "rmsnorm_quant_i8g") + count_issued(ops, "ar_rmsnorm_quant_i8g"),
                         k.fuse ? 2 * 12 : 0);
            }
}

/* Where no loaded library declares libr4d's fused ops (a CPU-only build, where rad-convert and the
 * host backend run), asking for one is an error, so the 31B with its DFlash drafter asks first and
 * declares the chains: the junctions, the attention prologue, the gated quantiser, the folded
 * all-reduce, and the drafter's fused norm on the int8 grid. It declares on both grids, at one rank
 * and two, and never asks for an undeclared op. */
TEST(gemma4_31b_without_libr4d_declares_the_chains_and_asks_for_no_undeclared_op) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int i8 = 1; i8 < 2; ++i8)
        for (int world : { 1, 2 }) {
            RadBuilder b;
            g431df_encs(b, i8 == 1, true);
            for (const char* op : { "rmsnorm_quant_i8", "gelu_mul_quant_i8", "qkv_norm_rope",
                                    "qkv_norm_rope_kv_store", "ar_rmsnorm_quant_i8" })
                b.undeclared.insert(op);
            RadBuildCtx ctx = build_ctx(0, world);
            ctx.max_spec = 15;
            CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
            CHECK(b.asked_undeclared.empty());
            RadCtx c; c.b = &b; c.rank = 0; c.world = world;
            RadBatch batch = decode_batch();
            c.batch = &batch;
            gemma4_bf16::step(&c, &batch);
            CHECK(c.step_fail.empty());
            const std::vector<std::string> ops = issued_ops(b, c);
            CHECK_EQ(count_issued(ops, "rmsnorm_p"), 12);    /* the prologue's value norm, unfused */
            CHECK_EQ(count_issued(ops, "gelu_mul"), 12);
            batch.draft_pass = 1;
            RadCtx d; d.b = &b; d.rank = 0; d.world = world; d.batch = &batch;
            gemma4_bf16::step(&d, &batch);
            CHECK(d.step_fail.empty());
            CHECK_EQ(count_issued(issued_ops(b, d), "rmsnorm_quant_i8"), 0);
        }
}

/* The 31B's engine steps lint clean with z-lab's drafter, on both of the drafter's grids, at one,
 * two and four ranks (the first and last rank of each): a decode step, a mixed step, a prefill
 * chunk tall enough to run as two slices, and a verify step with the drafter's context and query
 * passes behind it. The lint holds the buffers each pass reads to what an earlier op of the step
 * wrote and the planner kept alive, across the passes, so a tap whose plane the planner let another
 * buffer share, or a drafter buffer read after its arena slot was reused, fails here. Eight
 * sequences at a block of 16 is 128 query rows, past every fused band the drafter has. Both trunk
 * layouts: q, k and v apart, and as one linear, where a one-row decode reads q in the row and a
 * taller step copies it out. */
TEST(gemma4_31b_steps_lint_clean_with_a_dflash_drafter) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    for (int fused = 0; fused < 2; ++fused)
    for (int i8 = 1; i8 < 2; ++i8)
        for (int world : { 1, 2, 4 })
            for (int rank : { 0, world - 1 }) {
                if (world == 1 && rank > 0) continue;
                RadBuilder b;
                g431df_encs(b, i8 == 1, true, fused == 1);
                RadBuildCtx ctx = build_ctx(rank, world);
                ctx.max_tok = 2048;
                ctx.max_spec = 15;
                REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
                REQUIRE_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
                for (const auto& k : { kit(b, { 1 }, {}), kit(b, { 1, 1, 1 }, {}),
                                       kit(b, { 8, 1 }, { 5 }), kit(b, {}, { 1500 }) }) {
                    const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, rank, world);
                    CHECK_PASSES(b, v);
                }
                for (int64_t n_seq : { 2, 8 }) {
                    const auto ver = kit(b, std::vector<int>((size_t)n_seq, 16), {});
                    const auto ctxp = context_pass(b, *ver);
                    const auto qry = query_pass(b, n_seq, 16, 15);
                    const auto v = run_passes(b, gemma4_bf16::step,
                                              { ver.get(), ctxp.get(), qry.get() }, rank, world);
                    CHECK_PASSES(b, v);
                }
            }
}

/* ============ gemma4 31B + DFlash2: RedHatAI's speculators-format drafter ============ */
namespace {
/* The 31B fixture with a DFlash2 drafter in vLLM's speculators layout, as rad-convert merges
 * RedHatAI/gemma-4-31B-it-speculator.dflash2's config.json: the speculator's own keys at the top of
 * `draft.`, the decoder layers' config under `draft.transformer_layer_config.`, booleans as 1/0.
 * Taps at aux ids 1 5 9 (the residual entering those layers) for the fixture's twelve layers. */
struct G431Df2 {
    std::vector<const char*> keys, vals;
    G431Df2() {
        keys.assign(std::begin(k_gem4_keys), std::end(k_gem4_keys));
        vals.assign(std::begin(k_gem4_vals), std::end(k_gem4_vals));
        const char* kv[][2] = {
            { "draft.architectures", "[\"DFlash2DraftModel\"]" },
            { "draft.speculators_model_type", "dflash2" },
            { "draft.block_size", "8" },
            { "draft.mask_token_id", "4" },
            { "draft.aux_hidden_state_layer_ids", "[1, 5, 9]" },
            { "draft.selector_rank", "256" },
            { "draft.selector_top_k", "16" },
            { "draft.conv_kernel_size", "2" },
            { "draft.conv_group_size", "16" },
            { "draft.sample_from_anchor", "0" },
            { "draft.sliding_window_non_causal", "1" },
            { "draft.draft_vocab_size", "262144" },
            { "draft.transformer_layer_config.num_hidden_layers", "5" },
            { "draft.transformer_layer_config.head_dim", "256" },
            { "draft.transformer_layer_config.num_attention_heads", "32" },
            { "draft.transformer_layer_config.num_key_value_heads", "16" },
            { "draft.transformer_layer_config.intermediate_size", "21504" },
            { "draft.transformer_layer_config.rms_norm_eps", "1e-06" },
            { "draft.transformer_layer_config.rope_parameters.rope_theta", "10000.0" },
            { "draft.transformer_layer_config.hidden_size", "5376" },
            { "draft.transformer_layer_config.vocab_size", "262144" },
            { "draft.transformer_layer_config.hidden_act", "silu" },
            { "draft.transformer_layer_config.sliding_window", "2048" },
            { "draft.transformer_layer_config.use_sliding_window", "1" },
            { "draft.transformer_layer_config.layer_types",
              "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
              "\"sliding_attention\", \"sliding_attention\"]" },
        };
        for (const auto& p : kv) { keys.push_back(p[0]); vals.push_back(p[1]); }
    }
    RadModelMeta meta() const {
        RadModelMeta m = gemma4_meta();
        m.n_kv   = (int)keys.size();
        m.kv_key = keys.data();
        m.kv_val = vals.data();
        return m;
    }
};

/* The -df2 recipe's grids: the drafter's linears rotated int8, its head copy int8 (or, without
 * `head8`, no 8-bit encoding, which leaves the 2-bit grid), and the trunk at w4a8 with q|k|v fused. */
void g431df2_encs(RadBuilder& b, bool head8) {
    g431df_encs(b, false, false, true);
    if (head8) b.encs.insert(b.encs.begin(), { "dflash.draft_head.", rad_enc_affine(RAD_I8, RAD_F16, 1, 128) });
}
}  /* namespace */

/* The 31B declares RedHatAI's DFlash2 drafter as a block drafter past its trunk, read from the
 * speculators layout: depth 7 (block 8, the anchor predicting nothing), mask 4, every layer on a
 * 2048-token window, non-causal, its own windowed KV group of 16 kv heads x 256, the taps on the
 * layers BEFORE the aux ids (1 5 9 -> 0 4 8), the trunk's table read unscaled, and its head on the
 * 8-bit copy, the candidates merged across ranks where the head is vocab-sharded. */
TEST(gemma4_31b_declares_redhats_dflash2_from_the_speculators_layout) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    RadBuildCtx pc = build_ctx();
    pc.max_spec = 7;
    int taps[8] = {};
    CHECK_EQ(gemma4::dflash2_taps(&meta, &pc, 12, taps, 8), 3);
    CHECK_EQ(taps[0], 0);
    CHECK_EQ(taps[1], 4);
    CHECK_EQ(taps[2], 8);
    RadArchProbe p{};
    CHECK_OK(gemma4_bf16::probe(&meta, &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
    CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);   /* and with DFlash2 */

    for (int world : { 1, 4 })
    for (int head8 = 0; head8 < 2; ++head8) {
        RadBuilder b;
        g431df2_encs(b, head8 == 1);
        RadBuildCtx ctx = build_ctx(0, world);
        ctx.max_spec = 7;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);

        CHECK_EQ(b.drafter_name, std::string("dflash2"));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
        CHECK_EQ(b.drafter.depth, 7);
        CHECK_EQ((int)b.drafter.block, 8);
        CHECK_EQ(b.drafter.mask_token, 4);
        CHECK_EQ(b.drafter.window, 2048);
        CHECK_EQ(b.drafter.proposal, (rad_buf)buf_named(b, "dflash_tokens"));

        const rad_kvgroup kvd = kv_named(b, "kv_draft");
        REQUIRE(kvd != 0);
        CHECK_EQ((int)b.kv_groups[kvd - 1].d.kind, (int)RAD_KV_WINDOW);
        CHECK_EQ(b.kv_groups[kvd - 1].d.head_dim, 256);
        CHECK_EQ(b.kv_groups[kvd - 1].d.n_head_kv, 16 / world);

        const int src = buf_named(b, "dflash_src");
        REQUIRE(src > 0);
        CHECK_EQ(b.bufs[src - 1].d.shape[1], 3 * 5376);

        /* attention: head 256, gqa 2, non-causal, the window */
        int n_attn = 0;
        for (const auto& o : b.ops)
            if (o.op == "attn_paged" && pint(&o, "head_dim") == 256 && pint(&o, "window") == 2048 &&
                pint(&o, "causal") == 0) {
                ++n_attn;
                CHECK_EQ(pint(&o, "gqa"), 2);
            }
        CHECK_EQ(n_attn, 1);

        /* the drafter's lookup reads the trunk's table without Gemma's scale */
        const RecOp* te = find_op(b, "embed_lookup", 0);
        const RecOp* de = find_op(b, "embed_lookup", 1);
        REQUIRE(te != nullptr && de != nullptr);
        CHECK(te->w == de->w);
        CHECK_NEAR(pflt(te, "mult"), 73.5, 1e-9);
        CHECK(param(de, "mult") == nullptr);

        /* the head: 8-bit where the copy is int8, else the 2-bit grid; merged across ranks */
        bool head_seen = false;
        for (const auto& o : b.ops)
            if (o.op == "gemm_nt_q" && pint(&o, "K") == 5376 && pint(&o, "N") == 262144 / world) {
                head_seen = true;
                CHECK_EQ(pstr(&o, "dtype"), std::string(head8 ? "w8a8" : "w2a8"));
            }
        CHECK(head_seen);
        CHECK_EQ(count_op(b, "row_topk_merge"), world > 1 ? 1 : 0);
        CHECK_EQ(count_op(b, "dflash_select"), 1);
    }
}

/* DFlash2's two passes each run their attention prologue as one launch a layer where
 * qkv_norm_rope_kv_store resolves: the query pass with its q heads (8 a rank at four ranks; v_norm
 * 0), the context pass with none (n_head 0: the keys alone, q passed as k's buffer, which it does
 * not read). With the fold refused, both chains return: the norms, the rotaries and kv_store. */
TEST(gemma4_31b_dflash2_folds_both_passes_prologues) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    const int L = 5;
    for (int off = 0; off < 2; ++off) {
        RadBuilder b;
        g431df2_encs(b, true);
        if (off) b.refuse.insert("qkv_norm_rope_kv_store");
        RadBuildCtx ctx = build_ctx(0, 4);
        ctx.max_tok = 2048;
        ctx.max_spec = 7;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
        const auto ver = kit(b, { 8 }, {});
        const auto ctxp = context_pass(b, *ver);
        const auto qry = query_pass(b, 1, 8, 7);
        const auto v = run_passes(b, gemma4_bf16::step, { ctxp.get(), qry.get() }, 0, 4);
        REQUIRE(steps_ok(v));
        for (int p = 0; p < 2; ++p) {
            const RadCtx& c = *v[(size_t)p];
            int fused = 0;
            for (const RecIssue& is : c.issues) {
                const RecOp* o = b.op_of(is.op);
                if (!o || o->op != "qkv_norm_rope_kv_store") continue;
                ++fused;
                CHECK_EQ(pint(o, "n_head"), p == 0 ? 0 : 8);
                CHECK_EQ(pint(o, "v_norm"), 0);
                if (p == 0) {
                    REQUIRE(is.opd.size() >= 2);
                    CHECK_EQ(is.opd[0].handle, is.opd[1].handle);
                }
            }
            const std::vector<std::string> ops = issued_ops(b, c);
            CHECK_EQ(fused, off ? 0 : L);
            CHECK_EQ(count_issued(ops, "kv_store"), off ? L : 0);
            CHECK_EQ(count_issued(ops, "rope"), off ? (p == 0 ? L : 2 * L) : 0);
        }
    }
}

/* The block-16 DFlash2 drafter (the DBloom fine-tune) as a tree: its query pass also runs at its
 * short block of 8 rows (rad_drafter_short_block), whose selector grows a 7-node tree at an anchor
 * stride of 8, beside its own block's 15 at 16; a block it has no ops for fails the step. A
 * 7-node tree over the block-16 drafter (--num-speculative-tokens 7) declares the same short
 * block. At block 8 the drafter declares none. */
TEST(gemma4_31b_dflash2_block16_tree_runs_its_short_block) {
    for (int b16 = 0; b16 < 3; ++b16)
        for (int world : { 1, 4 }) {
            G431Df2 f;
            for (size_t i = 0; i < f.keys.size(); ++i)
                if (!std::strcmp(f.keys[i], "draft.block_size")) f.vals[i] = b16 ? "16" : "8";
            const int64_t own = b16 == 2 ? 7 : b16 ? 15 : 7;   /* the tree's nodes */
            const RadModelMeta meta = f.meta();
            RadBuilder b;
            g431df2_encs(b, true);
            RadBuildCtx ctx = build_ctx(0, world);
            ctx.max_tok = 2048;
            ctx.max_spec = (int)own;
            ctx.draft_tree = 1;
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            REQUIRE(b.drafter_parents != 0);
            CHECK_EQ(b.drafter_short, b16 ? 8 : 0);
            const int64_t full = b16 ? 16 : 8;
            std::vector<int64_t> blocks = { full };
            if (b16) blocks.push_back(8);
            for (int64_t rows : blocks) {
                const auto qry = query_pass(b, 2, rows, rows - 1);
                const auto v = run_passes(b, gemma4_bf16::step, { qry.get() }, 0, world);
                REQUIRE(steps_ok(v));
                int64_t nodes = -1, stride = -1, n_tree = 0;
                for (const RecIssue& is : v[0]->issues)
                    if (const RecOp* o = b.op_of(is.op); o && o->op == "dflash_tree") {
                        nodes = pint(o, "nodes");
                        stride = pint(o, "anchor_stride");
                        ++n_tree;
                    }
                CHECK_EQ(n_tree, 1);
                CHECK_EQ(nodes, std::min<int64_t>(own, rows - 1));
                CHECK_EQ(stride, rows);
            }
            const auto qbad = query_pass(b, 2, 4, 3);
            const auto vb = run_passes(b, gemma4_bf16::step, { qbad.get() }, 0, world);
            CHECK(!vb[0]->step_fail.empty());
        }
}

/* DFlash2's query pass issues each residual add with the norm after it, as one add_rmsnorm where it
 * resolves: after every attention (the feed-forward norm) and every feed-forward (the next layer's
 * attention norm, after the last the final norm), ten on the 31B's five layers, no add left and
 * one plain norm (layer 0's, after the embedding). With add_rmsnorm refused, the pairs. */
TEST(gemma4_31b_dflash2_folds_each_residual_add_into_the_next_norm) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    const int L = 5;
    for (int off = 0; off < 2; ++off) {
        RadBuilder b;
        g431df2_encs(b, true);
        if (off) b.refuse.insert("add_rmsnorm");
        RadBuildCtx ctx = build_ctx(0, 4);
        ctx.max_tok = 2048;
        ctx.max_spec = 7;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
        const auto qry = query_pass(b, 1, 8, 7);
        const auto v = run_passes(b, gemma4_bf16::step, { qry.get() }, 0, 4);
        REQUIRE(steps_ok(v));
        const std::vector<std::string> ops = issued_ops(b, *v[0]);
        CHECK_EQ(count_issued(ops, "add_rmsnorm"), off ? 0 : 2 * L);
        CHECK_EQ(count_issued(ops, "add"), off ? 2 * L : 0);
        CHECK_EQ(count_issued(ops, "rmsnorm"), off ? 2 * L + 1 : 1);
    }
}

/* DFlash2's query pass reads the next GEMM's weight into the cache beside each of its all-reduces
 * (prefetch_beside, rad_arch.h): the layer's gate_up beside the attention's, the next layer's q --
 * after the last layer, the draft head -- beside the feed-forward's, and o's beside the attention,
 * 20 MB each, after the GEMM before it: on lane 0 where launches chain (prefetch_lane0), else on
 * lane 1 behind a join. One rank has no reduction to sit beside and declares none, and neither
 * does a library without `prefetch`. */
TEST(gemma4_31b_dflash2_query_pass_prefetches_beside_its_all_reduces) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    const int L = 5;
    for (int world : { 1, 4 })
        for (int k = 0; k < 4; ++k) {
            const int off = k & 1, pf_lane = k < 2 ? 1 : 0;   /* lane 0 where launches chain */
            const ChainOn chain(pf_lane == 0);
            RadBuilder b;
            g431df2_encs(b, true);
            if (off) b.undeclared.insert("prefetch");
            RadBuildCtx ctx = build_ctx(0, world);
            ctx.max_tok = 2048;
            ctx.max_spec = 7;
            const int rc = gemma4_bf16::declare(&b, &meta, &ctx);
            REQUIRE_EQ(rc, RAD_OK);
            const auto qry = query_pass(b, 1, 8, 7);
            const auto v = run_passes(b, gemma4_bf16::step, { qry.get() }, 0, world);
            REQUIRE(steps_ok(v));
            const RadCtx& c = *v[0];
            const std::vector<std::string> ops = issued_ops(b, c);
            std::vector<std::string> read;
            for (size_t i = 0; i < c.issues.size(); ++i) {
                if (ops[i] != "prefetch") {
                    CHECK_EQ(c.issues[i].lane, 0);
                    continue;
                }
                CHECK_EQ(c.issues[i].lane, pf_lane);
                size_t g = i;   /* the GEMM before this read (and any read issued with it) */
                while (g > 0 && ops[g - 1] == "prefetch") --g;
                CHECK(g > 0 && ops[g - 1] == "gemm_nt_q" &&
                      ordered(c, g - 1, i));
                CHECK(i + 1 < ops.size() &&
                      (ops[i + 1] == "all_reduce" || ops[i + 1] == "qkv_norm_rope_kv_store" ||
                       ops[i + 1] == "prefetch"));
                const RecOp* o = b.op_of(c.issues[i].op);
                REQUIRE(o != nullptr && o->w.size() == 1);
                CHECK_EQ(pint(o, "bytes"), (int64_t)20 << 20);
                read.push_back(b.weights[o->w[0] - 1].name);
            }
            CHECK_EQ(races(b, c), 0);
            if (world == 1 || off) {
                CHECK(read.empty());
                continue;
            }
            /* a layer: o; the feed-forward's coefficient projection and gate_up; the next
             * layer's attention coefficient projection and q -- after the last, the draft head */
            REQUIRE_EQ((int)read.size(), 5 * L - 1);
            std::vector<std::string> want;
            for (int l = 0; l < L; ++l) {
                const std::string bl = "dflash.blk." + std::to_string(l) + ".";
                const std::string nx = "dflash.blk." + std::to_string(l + 1) + ".";
                for (const std::string& s : { bl + "attn_output", bl + "ffn_conv_proj",
                                              bl + "ffn_gate_up" })
                    want.push_back(s);
                if (l + 1 < L) { want.push_back(nx + "attn_conv_proj"); want.push_back(nx + "attn_q"); }
                else           want.push_back("dflash.draft_head");
            }
            for (size_t k = 0; k < want.size(); ++k) CHECK(read[k].find(want[k]) == 0);
        }
}

/* Every engine step with RedHatAI's DFlash2 behind the trunk lints clean at one, two and four ranks:
 * decode, a mixed step, a prefill chunk, and a verify step with the drafter's context and query
 * passes behind it (block 8). */
TEST(gemma4_31b_steps_lint_clean_with_a_dflash2_drafter) {
    G431Df2 f;
    const RadModelMeta meta = f.meta();
    for (int world : { 1, 2, 4 })
        for (int rank : { 0, world - 1 }) {
            if (world == 1 && rank > 0) continue;
            RadBuilder b;
            g431df2_encs(b, true);
            RadBuildCtx ctx = build_ctx(rank, world);
            ctx.max_tok = 2048;
            ctx.max_spec = 7;
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            for (const auto& k : { kit(b, { 1 }, {}), kit(b, { 1, 1, 1 }, {}),
                                   kit(b, { 8, 1 }, { 5 }), kit(b, {}, { 1500 }) }) {
                const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            for (int64_t n_seq : { 2, 8 }) {
                const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
                const auto ctxp = context_pass(b, *ver);
                const auto qry = query_pass(b, n_seq, 8, 7);
                const auto v = run_passes(b, gemma4_bf16::step,
                                          { ver.get(), ctxp.get(), qry.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
        }
}

/* Where the container holds q, k and v as one linear (a recipe naming blk.*.attn_qkv.weight), the
 * 31B declares one projection a layer -- q | k | v on a sliding layer, q | k on a global one, each
 * rank its slice of every part -- and none of the three apart, and the junction writes that
 * linear's int8 pair. The step: four GEMMs a layer instead of six (five on a global layer). A
 * global layer's attention reads q in the row (attn_paged_qrow); a sliding layer's reads it there
 * at one row and from attn_qs, copied out, at sixteen. */
TEST(gemma4_31b_takes_q_k_and_v_as_one_linear_where_the_container_holds_them_so) {
    G431Df f;
    const RadModelMeta meta = f.meta();
    const int64_t E = 5376, L = 12;
    for (int world : { 1, 4 }) {
        RadBuilder b;
        g431df_encs(b, true, true, true);
        RadBuildCtx ctx = build_ctx(0, world);
        ctx.max_tok = 2048;
        ctx.max_spec = 15;
        REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);

        for (int64_t l = 0; l < L; ++l) {
            const bool global = (l + 1) % 6 == 0;
            const std::string pre = "blk." + std::to_string(l) + ".";
            const RecWeight* qkv = nullptr;
            for (const RecWeight& w : b.weights) {
                if (w.name == pre + "attn_qkv.weight") qkv = &w;
                for (const char* sep : { "attn_q.weight", "attn_k.weight", "attn_v.weight" })
                    CHECK(w.name != pre + sep);
            }
            REQUIRE(qkv);
            CHECK_EQ(qkv->d.shape[0], (global ? 32 * 512 + 4 * 512 : 32 * 256 + 2 * 16 * 256) / world);
            CHECK_EQ(qkv->d.shape[1], E);
            CHECK_EQ((int)qkv->d.n_row_parts, global ? 2 : 3);
        }

        for (const int q : { 1, 16, 40 }) {
            const auto k = kit(b, { q }, {});
            const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, 0, world);
            CHECK_PASSES(b, v);
            const std::vector<std::string> ops = issued_ops(b, *v[0]);
            CHECK_EQ(count_issued(ops, "gemm_nt_q"), (int)(4 * L));
            /* the sliding layers read the row up to 32 rows a sequence (gqa 2), copy q out above */
            const bool dec = q <= 32;
            CHECK_EQ(count_issued(ops, "attn_paged_qrow"), dec ? 12 : 2);
            CHECK_EQ(count_issued(ops, "attn_paged"), dec ? 0 : 10);
            CHECK_EQ(count_issued(ops, "quant_act_i8"), (int)L);   /* o_proj's input only */
            CHECK_EQ(count_writes_to(b, *v[0], buf_named(b, "attn_qs")), dec ? 0 : 10);
            /* the GEMM, then qkv_norm_rope in place */
            CHECK_EQ(count_writes_to(b, *v[0], buf_named(b, "attn_qkvs")), 2 * 10);
            CHECK_EQ(count_writes_to(b, *v[0], buf_named(b, "attn_qkvg")), 2 * 2);
        }

        /* Where attn_paged_qrow is not served: attn_paged everywhere, q in the row at one row and
         * copied out above it. */
        RadBuilder r;
        g431df_encs(r, true, true, true);
        r.refuse.insert("attn_paged_qrow");
        REQUIRE_EQ(gemma4_bf16::declare(&r, &meta, &ctx), RAD_OK);
        for (const int q : { 1, 16 }) {
            const auto k = kit(r, { q }, {});
            const auto v = run_passes(r, gemma4_bf16::step, { k.get() }, 0, world);
            CHECK_PASSES(r, v);
            const std::vector<std::string> ops = issued_ops(r, *v[0]);
            CHECK_EQ(count_issued(ops, "attn_paged_qrow"), 0);
            CHECK_EQ(count_issued(ops, "attn_paged"), 12);
            CHECK_EQ(count_writes_to(r, *v[0], buf_named(r, "attn_qs")), q == 1 ? 0 : 10);
            CHECK_EQ(count_writes_to(r, *v[0], buf_named(r, "attn_qg")), q == 1 ? 0 : 2);
        }
    }
}

/* ==================================================================== the step lint, checked */

namespace {

/* A hand-built program for the lint's own checks: two transients of one arena, A and X, and four
 * ops declared in the order write A, read A, write X, read X, so A lives over ops 1..2 and X over
 * 3..4 and the plan may give them the same bytes. A KV group binds layer 0 only. */
struct LintProgram {
    RadBuilder b;
    rad_buf    A = 0, X = 0;
    rad_op     wA = 0, rA = 0, wX = 0, rX = 0;
    rad_kvgroup kv = 0;
    LintProgram() {
        RadBufDecl d{};
        d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 8; d.shape[1] = 16;
        A = rad_decl_buffer(&b, "A", &d);
        X = rad_decl_buffer(&b, "X", &d);
        wA = rw(b, "fill", {}, { A });
        rA = rw(b, "copy", { A }, {});
        wX = rw(b, "fill", {}, { X });
        rX = rw(b, "copy", { X }, {});
        RadKVGroupDecl g{};
        g.kind = RAD_KV_FULL; g.dtype = RAD_BF16; g.n_head_kv = 1; g.head_dim = 16;
        kv = rad_decl_kv_group(&b, "kv", &g);
        rad_bind_layer_kv(&b, 0, kv);
    }
    static rad_op rw(RadBuilder& b, const char* name, std::initializer_list<rad_buf> r,
                     std::initializer_list<rad_buf> w) {
        const rad_op h = rad_decl_op(&b, name, nullptr, 0, nullptr, 0);
        rad_op_reads(&b, h, r.begin(), (int)r.size());
        rad_op_writes(&b, h, w.begin(), (int)w.size());
        return h;
    }
    std::vector<std::string> lint(std::initializer_list<std::pair<rad_op, RadOperand>> issues) {
        RadCtx c{ .b = &b };
        for (const auto& [op, o] : issues) rad_issue(&c, op, &o, 1, RAD_N_BATCH);
        return lint_step(b, { &c });
    }
};

bool any_has(const std::vector<std::string>& v, const char* s) {
    for (const auto& x : v) if (has(x, s)) return true;
    return false;
}

}  /* namespace */

/* The lint's rules, each on a program built to break exactly one of them. A lint that cannot
 * fire would let every plugin test below pass vacuously. */
TEST(the_step_lint_reports_each_rule_it_checks) {
    {   /* Issued in declaration order: nothing to report. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.rA, RAD_B(p.A) },
                                { p.wX, RAD_B(p.X) }, { p.rX, RAD_B(p.X) } });
        CHECK_EQ(report_lint(f), (size_t)0);
    }
    {   /* X written between A's write and its read, with disjoint planned ranges. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.X) },
                                { p.rA, RAD_B(p.A) }, { p.rX, RAD_B(p.X) } });
        REQUIRE_EQ(f.size(), (size_t)1);
        CHECK(has(f[0], "reads 'A' (ops 1..2) after a write of 'X' (ops 3..4)"));
    }
    {   /* The same order is safe when A stays live through X's ops: the plan keeps them apart. */
        LintProgram p;
        rad_op_reads(&p.b, p.rX, &p.A, 1);
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.X) },
                                { p.rA, RAD_B(p.A) }, { p.rX, RAD_B(p.X) } });
        CHECK_EQ(report_lint(f), (size_t)0);
    }
    {   /* An op issued with a buffer outside the buffer's planned ops. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.A) } });
        CHECK(any_has(f, "undeclared use of 'A' (planned ops 1..2)"));
    }
    {   /* Rows past the declared extent, and an offset that runs a slice off the end. */
        LintProgram p;
        CHECK(any_has(p.lint({ { p.wA, RAD_B_N(p.A, 0, 9) } }), "span past 'A'"));
        CHECK(any_has(p.lint({ { p.wA, RAD_B_N(p.A, 16 * 7 + 1, 1) } }), "span past 'A'"));
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_B_N(p.A, 16 * 7, 1) } })), (size_t)0);
        /* A column slice keeps the declared pitch: 8 rows of the last 4 columns fit. */
        RadOperand col = RAD_B_COL(p.A, 12, 4);
        col.rows = 8;
        CHECK_EQ(report_lint(p.lint({ { p.wA, col } })), (size_t)0);
        col.offset = 13;
        CHECK(any_has(p.lint({ { p.wA, col } }), "span past 'A'"));
    }
    {   /* A fused op that did not resolve, issued anyway. */
        LintProgram p;
        CHECK(any_has(p.lint({ { RAD_NULL_HANDLE, RAD_B(p.A) } }), "null op issued"));
    }
    {   /* A transient read before anything in the step wrote it. */
        LintProgram p;
        CHECK(any_has(p.lint({ { p.rA, RAD_B(p.A) } }),
                      "reads 'A' before any issue of this step wrote it"));
    }
    {   /* A cache operand naming a layer no binding gave the group. */
        LintProgram p;
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_KV(p.kv, 0) } })), (size_t)0);
        CHECK(any_has(p.lint({ { p.wA, RAD_KV(p.kv, 1) } }), "layer 1 is not bound"));
    }
    {   /* A GEMM fed a column slice narrower than its buffer; the whole width is fine. */
        LintProgram p;
        const rad_op mm = LintProgram::rw(p.b, "gemm_nt", { p.A }, { p.X });
        const RadOperand half = RAD_B_COL(p.A, 0, 8), whole = RAD_B_COL(p.A, 0, 16);
        CHECK(any_has(p.lint({ { p.wA, RAD_B(p.A) }, { mm, half } }), "not contiguous"));
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_B(p.A) }, { mm, whole } })), (size_t)0);
    }
    {   /* A raw pointer into a buffer counts as that buffer. */
        LintProgram p;
        RadCtx c{ .b = &p.b };
        const RadOperand o = RAD_P_T((char*)rad_buf_ptr(&c, p.A) + 32, RAD_BF16);
        CHECK(any_has(p.lint({ { p.wX, o } }), "undeclared use of 'A' through a raw pointer"));
    }
}

/* ====================================================== qwen35_fp8 and qwen35moe_fp8 */

namespace {

/* Qwen3.8-27B-FP8's shape cut to the eight-layer fixture, its MTP head declared. */
MetaKV q35fp8_meta() {
    MetaKV k(qwen35_meta());
    k.m.quant = "fp8_e4m3";
    k.m.name  = "test-qwen35-27b-fp8";
    k.set("qwen35.nextn_predict_layers", "1");
    return k;
}

/* The 35B-A3B's routed feed-forward on the same trunk: 16 experts of 512, 4 a token, and a shared
 * expert of 512 (cut from 256 / 8 at 512). `n_ff` is the expert width, as rad-convert writes it. */
MetaKV q35moe_meta() {
    MetaKV k = q35fp8_meta();
    k.m.name = "test-qwen35-35b-a3b";
    k.m.n_ff = 512;
    k.m.n_expert = 16;
    k.m.n_expert_used = 4;
    k.set("shared_expert_intermediate_size", "512");
    return k;
}

/* The 35B-A3B-VL: the routed trunk with the Qwen3.5 vision tower beside it, as Sage-A1.5 and
 * Ornith-1.5 carry it. The fixture's trunk is q35fp8_meta's (5120 wide), so the tower emits
 * that width; the shipped checkpoints' tower is 2048, the real trunk's width. */
MetaKV q35moe_vl_meta() {
    MetaKV k = q35moe_meta();
    k.set("vision_config.depth", "2")
     .set("vision_config.hidden_size", "1152")
     .set("vision_config.num_heads", "16")
     .set("vision_config.intermediate_size", "4304")
     .set("vision_config.patch_size", "16")
     .set("vision_config.temporal_patch_size", "2")
     .set("vision_config.in_channels", "3")
     .set("vision_config.spatial_merge_size", "2")
     .set("vision_config.out_hidden_size", "5120")
     .set("vision_config.num_position_embeddings", "2304")
     .set("vision_config.deepstack_visual_indexes", "")
     .set("vision_config.hidden_act", "gelu_pytorch_tanh");
    return k;
}

/* A DFlash2 drafter merged under `draft.`: two layers reading trunk layers 1, 3 and 5, a block
 * of 8 (seven drafts), the selector of rank 64 over 8 candidates. */
MetaKV with_dflash2(MetaKV k) {
    k.set("draft.architectures", "DFlash2DraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "32")
     .set("draft.num_key_value_heads", "8")
     .set("draft.intermediate_size", "8192")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "10000000")
     .set("draft.vocab_size", "248320")
     .set("draft.dflash_config.target_layer_ids", "1 3 5")
     .set("draft.dflash_config.block_size", "8")
     .set("draft.dflash_config.mask_token_id", "248070")
     .set("draft.dflash_config.selector_rank", "64")
     .set("draft.dflash_config.selector_top_k", "8")
     .set("draft.dflash_config.conv_kernel_size", "2")
     .set("draft.dflash_config.conv_group_size", "16");
    return k;
}

/* Every gemm_nt handle a pass issues is issued once: the bf16 GEMMs take a whole pass in one
 * launch (libr4d's bf16 row serves M to 65536), so a slice loop would be a stale band. */
bool each_gemm_nt_once(const RadBuilder& b, const RadCtx& c) {
    std::map<rad_op, int> n;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gemm_nt") ++n[i.op];
    }
    for (const auto& [op, k] : n)
        if (k != 1) {
            fprintf(stderr, "    gemm_nt op %u issued %d times in one pass\n", (unsigned)op, k);
            return false;
        }
    return true;
}

/* The engine steps a serial-drafter or block-drafter model runs, each linted whole: decode,
 * prefill, a mixed step, and a verify followed by the drafter's passes. `short_block` false for a
 * block drafter whose architecture declares no short block (rad_drafter_short_block is its own
 * choice). */
template <class Fn>
void lint_qwen35_fp8_family(Fn declare, StepFn step, const RadModelMeta* meta, int max_spec,
                            bool block_drafter, int64_t block, bool short_block = true) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = spec_ctx(rank, world, max_spec);
            REQUIRE_EQ(declare(&b, meta, &ctx), RAD_OK);
            const auto dec = kit(b, { 1, 1, 1, 1 }, {});
            const auto pre = kit(b, {}, { 32, 32 });
            const auto mix = kit(b, { 1, 1 + max_spec }, { 6 });
            const auto ver = kit(b, std::vector<int>(4, 1 + max_spec), {});
            for (const Kit* k : { dec.get(), pre.get(), mix.get() }) {
                const auto v = run_passes(b, step, { k }, rank, world);
                CHECK_PASSES(b, v);
            }
            std::vector<std::unique_ptr<Kit>> passes;
            if (block_drafter) {
                passes.push_back(context_pass(b, *ver));
                passes.push_back(query_pass(b, 4, block, block - 1));
            } else {
                passes.push_back(history_pass(b, *ver));
                for (int r = 2; r <= max_spec; ++r) passes.push_back(round_pass(b, 4, r));
            }
            std::vector<const Kit*> all = { ver.get() };
            for (const auto& p : passes) all.push_back(p.get());
            const auto v = run_passes(b, step, all, rank, world);
            CHECK_PASSES(b, v);
            for (size_t i = 1; i < v.size(); ++i) CHECK(each_gemm_nt_once(b, *v[i]));
            if (!block_drafter || block <= 4 || !short_block) continue;
            /* The short block (rad_drafter_short_block): a query pass staged at 4 rows a sequence
             * runs the short ops, its selector proposing 3; a block with no ops fails the step. */
            CHECK_EQ(b.drafter_short, 4);
            const auto ctx2 = context_pass(b, *ver);
            const auto qs = query_pass(b, 4, b.drafter_short, b.drafter_short - 1);
            const auto vs = run_passes(b, step, { ver.get(), ctx2.get(), qs.get() }, rank, world);
            CHECK_PASSES(b, vs);
            CHECK(vs[2]->step_fail.empty());
            int64_t sel_steps = -1, sel_stride = -1;
            for (const RecIssue& i : vs[2]->issues)
                if (const RecOp* o = b.op_of(i.op); o && o->op == "dflash_select") {
                    sel_steps = pint(o, "steps");
                    sel_stride = pint(o, "anchor_stride");
                }
            CHECK_EQ(sel_steps, 3);
            CHECK_EQ(sel_stride, 4);
            const auto qbad = query_pass(b, 4, 6, 5);
            const auto vb = run_passes(b, step, { qbad.get() }, rank, world);
            CHECK(!vb[0]->step_fail.empty());
        }
}

}  /* namespace */

/* The GDN state cache's width is a deployment's choice: f32 by default, f16 under
 * RadBuildCtx::state_dtype (--state-cache-dtype f16), which the probe says the plugin serves. */
TEST(qwen35_fp8_gdn_state_is_f32_unless_the_deployment_asks_f16) {
    MetaKV mk = q35fp8_meta();
    RadArchProbe pb{};
    CHECK_OK(qwen35_fp8::probe(mk.meta(), &pb));
    CHECK_EQ(pb.state_f16_ok, 1);
    for (uint32_t want : { (uint32_t)0, (uint32_t)RAD_F16 }) {
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        ctx.state_dtype = want;
        const int rc = stderr_of_rc([&] { return qwen35_fp8::declare(&b, mk.meta(), &ctx); });
        REQUIRE_EQ(rc, RAD_OK);
        int n = 0;
        for (const RecKV& g : b.kv_groups)
            if (g.name == "kv_gdn_state") {
                ++n;
                CHECK_EQ(g.d.dtype, want == RAD_F16 ? RAD_F16 : RAD_F32);
            }
        CHECK_EQ(n, 1);
        CHECK_EQ(noted(b, "GDN state cache f16"), want == RAD_F16);
    }
}

/* The fp8 sibling of qwen35_bf16: the same layers with every large linear a block-fp8 gemm_nt_q
 * read at fp8a8, one quantiser per activation, and the attention's output gate, multiply and
 * quantiser fused into gate_quant_fp8 (the sigmoid/mul pair is the bf16-linear form only). The
 * delta net's a|b projection stays bf16: its 48 rows are not a whole 128-row scale block. */
TEST(qwen35_fp8_declares_fp8_linears_and_the_fused_output_gate) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    int q = 0;
    for (const auto& o : b.ops) {
        if (o.op != "gemm_nt_q") continue;
        ++q;
        CHECK_EQ(pstr(&o, "dtype"), std::string("fp8a8"));
        CHECK_EQ(pint(&o, "group"), 128);
    }
    /* q|gate, k, v, o for 2 attention layers; qkv|z and out for 6 delta-net layers; gate_up and
     * down for 8 feed-forwards. */
    CHECK_EQ(q, 2 * 4 + 6 * 2 + 8 * 2);
    CHECK_EQ(count_op(b, "gate_quant_fp8"), 2);
    CHECK_EQ(count_op(b, "sigmoid"), 0);
    CHECK_EQ(count_op(b, "mul"), 0);
    CHECK_EQ(count_op(b, "gemm_nt"), 6);
    for (const auto& o : b.ops)
        if (o.op == "gemm_nt") CHECK(in(o.writes, (rad_buf)buf_named(b, "gdn_ab")));
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
    /* The lfm2-only rows are that model's own declarations: nothing here names one, so every op
     * resolves to the row it always did (gemm_nt keeps its band, rmsnorm and rope their launches). */
    for (const auto& o : b.ops) {
        CHECK(o.op != "gemm_nt_q_short");
        CHECK(o.op != "gemm_nt_q_gated_wide");
        CHECK(o.op != "rmsnorm_rows");
        CHECK(o.op != "rope_rows");
    }
    /* No speculation: no head, no drafter. */
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
    CHECK(find_b(b, "mtp_x") == nullptr);
}

/* Every engine step the dense fp8 model runs lints clean at one and two ranks, the MTP head's
 * history pass and rounds included. */
TEST(qwen35_fp8_steps_lint_clean_with_its_mtp_head) {
    MetaKV mk = q35fp8_meta();
    lint_qwen35_fp8_family(qwen35_fp8::declare, qwen35_fp8::step, mk.meta(), 3, false, 0);
}

/* A mixed step's dense attention is issued once per band (AttnGatedFP8::step): the decode
 * sequences as a prefix at their own band, then the prefill chunks at the op's top band with
 * their per-sequence arrays past the decode sequences and q, out and cu at absolute rows. A pure
 * step issues one attention a layer. */
TEST(qwen35_fp8_issues_a_mixed_steps_attention_once_per_band) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    /* A verify of 4 and a plain decode beside a 6-row chunk over 20 tokens of history. */
    const auto mixed = kit(b, { 4, 1 }, { 6 }, 20);
    const auto dec   = kit(b, { 4, 1 }, {});
    const auto pre   = kit(b, {}, { 6 }, 20);
    auto attn = [&](const Kit& k) {
        const auto v = run_passes(b, qwen35_fp8::step, { &k });
        CHECK(steps_ok(v));
        CHECK_LINT(b, v[0].get());
        std::vector<RecIssue> out;
        for (const auto& i : v[0]->issues) {
            const RecOp* o = b.op_of(i.op);
            if (o && o->op == "attn_paged") out.push_back(i);
        }
        return out;
    };
    /* two attention layers in the fixture */
    CHECK_EQ(attn(*dec).size(), (size_t)2);
    CHECK_EQ(attn(*pre).size(), (size_t)2);
    const auto m = attn(*mixed);
    REQUIRE_EQ(m.size(), (size_t)4);
    const int32_t* table = mixed->table.data();
    const int32_t* used  = mixed->used.data();
    const int32_t* cu    = mixed->cu.data();
    for (size_t l = 0; l < 2; ++l) {
        const RecIssue& d = m[2 * l];
        const RecIssue& p = m[2 * l + 1];
        REQUIRE(d.opd.size() == 8 && p.opd.size() == 8);
        CHECK_EQ(d.n, (int64_t)4);                       /* the decode band: the verify's 4 */
        CHECK_EQ(d.opd[0].rows, (int64_t)5);             /* q: the 5 decode rows */
        CHECK(d.opd[2].raw == (void*)table);  CHECK_EQ(d.opd[2].rows, (int64_t)2);
        CHECK(d.opd[3].raw == (void*)used);   CHECK_EQ(d.opd[3].rows, (int64_t)2);
        CHECK(d.opd[6].raw == (void*)cu);     CHECK_EQ(d.opd[6].rows, (int64_t)3);
        CHECK_EQ(d.opd[7].rows, (int64_t)5);
        CHECK_EQ(p.n, (int64_t)256);                     /* the top band, whatever the chunk */
        CHECK_EQ(p.opd[0].rows, (int64_t)11);            /* q and out whole: cu is absolute */
        CHECK(p.opd[2].raw == (void*)(table + 2 * 16));  CHECK_EQ(p.opd[2].rows, (int64_t)1);
        CHECK(p.opd[3].raw == (void*)(used + 2));        CHECK_EQ(p.opd[3].rows, (int64_t)1);
        CHECK(p.opd[6].raw == (void*)(cu + 2));          CHECK_EQ(p.opd[6].rows, (int64_t)2);
        CHECK_EQ(p.opd[7].rows, (int64_t)11);
        CHECK_EQ(d.opd[1].handle, p.opd[1].handle);      /* the same layer's cache */
        CHECK_EQ(d.opd[1].offset, p.opd[1].offset);
    }
    /* A chunk within the decode row's band still goes to the top band (the decode row's scratch
     * is sized for its own band); a chunk no longer than the verify blocks beside it keeps one
     * launch at their band (r4d_selftest attn_mixed_step checks both halves' scratch). */
    const auto shortc = kit(b, { 3, 2 }, { 4 }, 20);
    const auto m2 = attn(*shortc);
    REQUIRE_EQ(m2.size(), (size_t)4);
    for (size_t l = 0; l < 2; ++l) {
        CHECK_EQ(m2[2 * l].n, (int64_t)3);
        CHECK_EQ(m2[2 * l + 1].n, (int64_t)256);
        CHECK_EQ(m2[2 * l + 1].opd[0].rows, (int64_t)9);
    }
    const auto tiny = kit(b, { 4, 3 }, { 2 }, 20);
    const auto m3 = attn(*tiny);
    REQUIRE_EQ(m3.size(), (size_t)2);
    for (const RecIssue& i : m3) {
        CHECK_EQ(i.n, (int64_t)4);
        CHECK_EQ(i.opd[0].rows, (int64_t)9);
        CHECK_EQ(i.opd[6].rows, (int64_t)4);
    }
}

namespace {

/* Every trunk and MTP linear of a qwen35 container in the 4-bit form an int4 W4A16 release
 * imports to: u4 codes, an f16 scale and a u4 zero point per 128 of K. */
std::vector<std::pair<std::string, RadEncoding>> int4_linears() {
    RadEncoding w4 = rad_enc_affine(RAD_U4, RAD_F16, 1, 128);
    rad_enc_add_plane(&w4, "zero", RAD_U4, 1, 128);
    std::vector<std::pair<std::string, RadEncoding>> e;
    for (const char* k : { "attn_qg.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up.weight", "ffn_down.weight" })
        e.emplace_back(k, w4);
    return e;
}

int qwen35_fp8_int4_declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* c) {
    b->encs = int4_linears();
    return qwen35_fp8::declare(b, m, c);
}

}  /* namespace */

/* An int4 container's linears are read by the 4-bit GEMMs over their input's bf16 plane: at or
 * below 64 rows w4a16_asym narrows that plane on load, above that quant_act_i8g's int8 codes and
 * w4a8i8_asym, both over the codes and the joined (scale, zero) view. No linear keeps an fp8a8
 * GEMM, and the gated fold (an fp8a8 tile) is not declared over a 4-bit gate_up. */
TEST(qwen35_fp8_reads_an_int4_container_with_the_4bit_gemms) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35_fp8_int4_declare(&b, mk.meta(), &ctx), RAD_OK);
    int f16 = 0, i8 = 0, other = 0;
    for (const auto& o : b.ops) {
        if (o.op != "gemm_nt_q") continue;
        const std::string dt = pstr(&o, "dtype");
        const RecParam* m = param(&o, "M");
        REQUIRE(m != nullptr);
        if (dt == "w4a16_asym") {
            ++f16;
            CHECK_EQ(m->ival, 1LL);
            CHECK_EQ(m->ihi, 64LL);
        } else if (dt == "w4a8i8_asym") {
            ++i8;
            CHECK_EQ(m->ival, 65LL);
            CHECK_EQ(m->ihi, 256LL);
        } else {
            ++other;
        }
        CHECK_EQ(pint(&o, "group"), 128);
    }
    /* q|gate, k, v, o for 2 attention layers; qkv|z and out for 6 delta-net layers; gate_up and
     * down for 8 feed-forwards: 36 linears, each with both GEMMs and a quantiser. The decode GEMM
     * consumes bf16 directly. */
    CHECK_EQ(f16, 36);
    CHECK_EQ(i8, 36);
    CHECK_EQ(other, 0);
    for (const auto& o : b.ops)
        CHECK(o.op != "cast" || pstr(&o, "to") != "f16");
    CHECK_EQ(count_op(b, "quant_act_i8g"), 36);
    CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 0);
    int joined = 0;
    for (const RecWeight& w : b.weights)
        if (w.d.planes && std::string(w.d.planes) == "scale,zero") {
            ++joined;
            CHECK_EQ(w.d.dtype, (uint32_t)RAD_F16);
        }
    CHECK_EQ(joined, 36);
}

/* Every engine step an int4 container runs lints clean at one and two ranks, the MTP head's
 * history pass and rounds included: the 4-bit linears' row parts and shards are cut like the
 * block-fp8 ones. */
TEST(qwen35_fp8_int4_steps_lint_clean_with_its_mtp_head) {
    MetaKV mk = q35fp8_meta();
    lint_qwen35_fp8_family(qwen35_fp8_int4_declare, qwen35_fp8::step, mk.meta(), 3, false, 0);
}

/* The head is a serial drafter at the deployment's depth, declared after the trunk's lm_head:
 * the planner reads declaration order as liveness, and the head's passes run after the trunk. */
TEST(qwen35_fp8_declares_its_mtp_head_after_the_trunk) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 3);
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
    CHECK_EQ(b.drafter.depth, 3);
    CHECK_EQ(b.drafter_name, std::string("mtp"));
    size_t logits = 0, first_head = 0;
    for (size_t i = 0; i < b.ops.size() && !logits; ++i)
        if (b.ops[i].op == "logits_gemm") logits = i + 1;
    const rad_buf hx = (rad_buf)buf_named(b, "mtp_x");
    for (size_t i = 0; i < b.ops.size() && !first_head; ++i)
        if (in(b.ops[i].reads, hx) || in(b.ops[i].writes, hx)) first_head = i + 1;
    REQUIRE(logits > 0 && first_head > 0);
    CHECK(first_head > logits);
    /* The head binds one more layer of the trunk's attention group, at index n_layers. */
    bool bound = false;
    for (const auto& [layer, g] : b.binds) bound |= layer == 8 && g == (rad_kvgroup)1;
    CHECK(bound);
    /* A draft round-1 pass is refused: the history pass carries it. */
    const auto one = round_pass(b, 4, 1);
    const auto v = run_passes(b, qwen35_fp8::step, { one.get() });
    CHECK(!v[0]->step_fail.empty());
}

/* DFlash2 replaces the MTP head when the container carries one: a block drafter whose block is
 * one longer than the depth, a power of two (its convolution masks positions with block - 1),
 * reading trunk taps copied out after layers 1, 3 and 5. */
TEST(qwen35_fp8_dflash2_drafter_takes_a_power_of_two_block) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 7);
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
        CHECK_EQ(b.drafter_name, std::string("dflash2"));
        CHECK_EQ(b.drafter.depth, 7);
        CHECK_EQ(b.drafter.block, 8);
        CHECK_EQ(b.drafter.mask_token, 248070);
        CHECK(find_b(b, "mtp_x") == nullptr);           /* no MTP head beside it */
        CHECK(noted(b, "the DFlash2 drafter replaces the 1-layer MTP head"));
        const RecBuf* src = find_b(b, "dflash_src");
        REQUIRE(src != nullptr);
        CHECK_EQ(src->d.shape[1], 3 * 5120);
        /* The drafter's layers follow the trunk's in its own group. */
        std::set<int> drafted;
        for (const auto& [layer, g] : b.binds)
            if (g == (rad_kvgroup)b.kv_groups.size()) drafted.insert(layer);
        CHECK(drafted == (std::set<int>{ 8, 9 }));
    }
    {   /* Six drafts make a block of seven, which the convolution cannot mask. */
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 6);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {   /* Another power of two runs, with a note that it was trained at 8. */
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 15);
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(b.drafter.block, 16);
        CHECK(noted(b, "DFlash2 was trained at block_size 8"));
    }
    {   /* The probe fixes the depth at the trained block, so the engine asks for 7. */
        RadArchProbe p{};
        CHECK_OK(qwen35_fp8::probe(mk.meta(), &p));
        CHECK_EQ(p.draft_depth, 7);
        CHECK_EQ(p.draft_depth_fixed, 1);
        CHECK_EQ(p.chain_pays, 0);   /* chained launches measured even here */
    }
}

/* The block drafter's engine step (verify, context pass, query pass) lints clean at one and two
 * ranks, and its bf16 GEMMs run once a pass. */
TEST(qwen35_fp8_steps_lint_clean_with_a_dflash2_drafter) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    lint_qwen35_fp8_family(qwen35_fp8::declare, qwen35_fp8::step, mk.meta(), 7, true, 8);
}

/* Qwen3.8's DFlash2 drafter has head_dim 128, outside every qkv_norm_rope_kv_store a library serves
 * (libr4d's: 256 to 512), and an op declared for a shape that resolves to no device kernel refuses
 * startup whether or not it is issued. So the block declares no prologue fold for it, and each pass
 * keeps its norm, rope and kv_store chain; the residual adds, which add_rmsnorm serves at n_embd,
 * still fold, two a layer. */
TEST(qwen35_fp8_dflash2_head_128_drafter_declares_no_prologue_fold) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    int folds_128 = 0;
    for (const auto& o : b.ops)
        if (o.op == "qkv_norm_rope_kv_store" && pint(&o, "head_dim") == 128) ++folds_128;
    CHECK_EQ(folds_128, 0);
    CHECK_EQ(count_op(b, "add_rmsnorm"), 2 * 2);
}

/* qwen35_bf16 declares the same drafter for a bf16 container converted with --draft-model, beside
 * a trunk whose ops are the same with the drafter as without it: the tap copies are all it adds
 * there. The trunk is compared by op and parameters; buffer handles move with the taps' buffer. */
TEST(qwen35_bf16_declares_a_dflash2_drafter_beside_an_unchanged_trunk) {
    MetaKV plain(qwen35_meta());
    MetaKV df = with_dflash2(MetaKV(qwen35_meta()));
    RadBuilder a, b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    REQUIRE_EQ(qwen35_bf16::declare(&a, plain.meta(), &ctx), RAD_OK);
    REQUIRE_EQ(qwen35_bf16::declare(&b, df.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)a.drafter.kind, (int)RAD_DRAFT_NONE);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
    CHECK_EQ(b.drafter_name, std::string("dflash2"));
    CHECK_EQ(b.drafter.depth, 7);
    CHECK_EQ(b.drafter.block, 8);
    const RecBuf* src = find_b(b, "dflash_src");
    REQUIRE(src != nullptr);
    CHECK_EQ(src->d.shape[1], 3 * 5120);
    CHECK_EQ(count_op(b, "cast"), 1);
    /* The bf16 head rescores the drafter's candidates (logit_rerank_bf16 reads it). */
    CHECK_EQ(count_op(b, "logit_rerank"), 1);
    std::set<int> drafted;
    for (const auto& [layer, g] : b.binds)
        if (g == (rad_kvgroup)b.kv_groups.size()) drafted.insert(layer);
    CHECK(drafted == (std::set<int>{ 8, 9 }));

    std::vector<const RecOp*> ta, tb;
    for (const auto& o : a.ops) ta.push_back(&o);
    for (const auto& o : b.ops) {
        if (o.op == "cast") continue;
        tb.push_back(&o);
        if (o.op == "logits_gemm") break;
    }
    REQUIRE_EQ(ta.size(), tb.size());
    for (size_t i = 0; i < ta.size(); ++i) {
        CHECK_EQ(ta[i]->op, tb[i]->op);
        CHECK_EQ(ta[i]->w.size(), tb[i]->w.size());
        REQUIRE_EQ(ta[i]->p.size(), tb[i]->p.size());
        for (size_t k = 0; k < ta[i]->p.size(); ++k) {
            const RecParam &x = ta[i]->p[k], &y = tb[i]->p[k];
            CHECK(x.key == y.key && x.kind == y.kind && x.ival == y.ival && x.ihi == y.ihi &&
                  x.dval == y.dval && x.sval == y.sval);
        }
    }
}

/* Its engine steps (verify, context pass, query pass) lint clean at one and two ranks, and the
 * probe fixes the depth at the trained block, or answers 0 for a container without a drafter. */
TEST(qwen35_bf16_steps_lint_clean_with_a_dflash2_drafter) {
    MetaKV df = with_dflash2(MetaKV(qwen35_meta()));
    /* No short block: qwen35_bf16 does not call rad_drafter_short_block, so its graph is the
     * drafter's own block alone. */
    lint_qwen35_fp8_family(qwen35_bf16::declare, qwen35_bf16::step, df.meta(), 7, true, 8, false);
    RadArchProbe p{};
    CHECK_OK(qwen35_bf16::probe(df.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
    MetaKV plain(qwen35_meta());
    RadArchProbe q{};
    CHECK_OK(qwen35_bf16::probe(plain.meta(), &q));
    CHECK_EQ(q.draft_depth, 0);
}

/* data/recipes/q38-27b-bf16x12-df2.recipe's encodings: the nine linears of the bf16 blocks packed
 * as bf16x12, everything else plain. Each packed linear is declared over every plane at its codes'
 * dtype (no view, the shape, shard and row parts as before) and its gemm_nt reads bf16x12
 * (decl_linear_bf16). Every other weight, op and parameter is the plain container's, the drafter's
 * included, at one rank and two. A linear the recipe leaves bf16 (down here) keeps the bf16 GEMM,
 * and a model holding every weight plain declares what one with no encodings does. The drafter's
 * keys come first: its names contain the trunk's. Its head is u2 in both, as in every recipe
 * (codes=u2 zero=u8 group=128 scale=f16): a head the container holds plain is a bf16 head of its
 * own, declared without the scale view (rad_block_dflash2.h). */
TEST(qwen35_bf16_reads_a_bf16x12_trunk_at_bf16x12_and_changes_nothing_else) {
    RadEncoding x12;
    rad_enc_clear(&x12);
    rad_enc_copy_str(x12.scheme, "bf16x12");
    rad_enc_add_plane(&x12, "codes", RAD_U8, 1, 1);
    rad_enc_add_plane(&x12, "d", RAD_U4, 1, 1);
    rad_enc_add_plane(&x12, "e", RAD_U8, 1, 32);
    rad_enc_add_table(&x12, "esc", RAD_BF16, 64, 32);
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    REQUIRE(rad_enc_valid(&u2));
    static const char* const kLinears[] = { "attn_qg.weight", "attn_k.weight", "attn_v.weight",
                                            "attn_output.weight", "ssm_in.weight", "ssm_z.weight",
                                            "ssm_out.weight", "ffn_gate_up.weight",
                                            "ffn_down.weight" };
    auto trunk_linear = [](const std::string& n, const char* t) {
        const size_t k = std::strlen(t);
        return n.rfind("blk.", 0) == 0 && n.size() > k && n.compare(n.size() - k, k, t) == 0;
    };
    MetaKV df = with_dflash2(MetaKV(qwen35_meta()));

    /* `b` against `a`: the weights packed in `b` are those `packed` names; `*read_x12` counts
     * b's gemm_nt at bf16x12. */
    auto same_but_packed = [&](const RadBuilder& a, const RadBuilder& b,
                               const std::function<bool(const std::string&)>& packed,
                               int* read_x12) {
        *read_x12 = -1;
        REQUIRE_EQ(a.weights.size(), b.weights.size());
        for (size_t i = 0; i < a.weights.size(); ++i) {
            const RecWeight &x = a.weights[i], &y = b.weights[i];
            REQUIRE_EQ(x.name, y.name);
            const bool p = packed(y.name);
            CHECK_EQ(y.d.dtype, p ? (uint32_t)RAD_U8 : x.d.dtype);
            if (p) CHECK(x.d.dtype == RAD_BF16 && !y.d.source && !y.d.planes);
            CHECK((x.d.source == nullptr) == (y.d.source == nullptr));
            CHECK((x.d.planes == nullptr) == (y.d.planes == nullptr));
            REQUIRE_EQ(x.d.rank, y.d.rank);
            for (uint32_t r = 0; r < x.d.rank; ++r) CHECK_EQ(x.d.shape[r], y.d.shape[r]);
            CHECK(x.d.access == y.d.access && x.d.shard == y.d.shard &&
                  x.d.optional == y.d.optional && x.d.group.layer == y.d.group.layer &&
                  x.d.group.expert == y.d.group.expert && x.d.group.slot == y.d.group.slot);
            REQUIRE_EQ(x.d.n_row_parts, y.d.n_row_parts);
            for (int r = 0; r < x.d.n_row_parts; ++r) CHECK_EQ(x.d.row_parts[r], y.d.row_parts[r]);
        }
        *read_x12 = 0;
        REQUIRE_EQ(a.ops.size(), b.ops.size());
        for (size_t i = 0; i < a.ops.size(); ++i) {
            const RecOp &x = a.ops[i], &y = b.ops[i];
            REQUIRE_EQ(x.op, y.op);
            CHECK(x.w == y.w && x.reads == y.reads && x.writes == y.writes);
            const bool p = y.op == "gemm_nt" && y.w.size() == 1 &&
                           packed(b.weights[y.w[0] - 1].name);
            *read_x12 += p;
            REQUIRE_EQ(x.p.size(), y.p.size());
            for (size_t k = 0; k < x.p.size(); ++k) {
                const RecParam &u = x.p[k], &v = y.p[k];
                CHECK(u.key == v.key && u.kind == v.kind && u.ival == v.ival && u.ihi == v.ihi &&
                      u.dval == v.dval);
                if (p && u.key == "dtype")
                    CHECK(u.sval == "bf16" && v.sval == "bf16x12");
                else
                    CHECK_EQ(u.sval, v.sval);
            }
        }
    };

    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank)
            for (bool keep_down : { false, true }) {
                RadBuildCtx ctx = spec_ctx(rank, world, 7);
                RadBuilder a, b, c;
                b.encs = { { "dflash.draft_head", u2 }, { "dflash.", plain } };
                for (const char* t : kLinears)
                    b.encs.push_back({ t, keep_down && !std::strcmp(t, "ffn_down.weight") ? plain
                                                                                       : x12 });
                c.encs = { { "dflash.draft_head", u2 }, { "", plain } };
                REQUIRE_EQ(qwen35_bf16::declare(&a, df.meta(), &ctx), RAD_OK);
                REQUIRE_EQ(qwen35_bf16::declare(&b, df.meta(), &ctx), RAD_OK);
                REQUIRE_EQ(qwen35_bf16::declare(&c, df.meta(), &ctx), RAD_OK);
                auto packed = [&](const std::string& n) {
                    for (const char* t : kLinears)
                        if (trunk_linear(n, t))
                            return !(keep_down && !std::strcmp(t, "ffn_down.weight"));
                    return false;
                };
                /* The 8-layer fixture: q|gate, k, v and o in 2 attention layers, in, z and out in
                 * 6 delta-net layers, gate|up and down in 8 feed-forwards. */
                int n = 0;
                same_but_packed(a, b, packed, &n);
                CHECK_EQ(n, 2 * 4 + 6 * 3 + 8 * (keep_down ? 1 : 2));
                same_but_packed(a, c, [](const std::string&) { return false; }, &n);
                CHECK_EQ(n, 0);
            }
}

/* What a step issues at two ranks: one ar_rmsnorm per folded boundary (rad_norm_bf16.h), and the
 * standalone all-reduce and add only where `x` is read between blocks: after a drafter tap layer
 * and after the last layer (the final norm), besides the embedding's all-reduce. Exact wire here,
 * the fixture's default, so decode and prefill both fold. */
TEST(qwen35_bf16_folds_each_boundary_into_one_ar_rmsnorm_at_two_ranks) {
    auto issued = [](const RadBuilder& b, const RadCtx& c, const char* op) {
        int n = 0;
        for (const std::string& o : issued_ops(b, c)) n += o == op;
        return n;
    };
    MetaKV plain(qwen35_meta());
    MetaKV df = with_dflash2(MetaKV(qwen35_meta()));
    struct Case { MetaKV* m; int max_spec, arn, ar, add; };
    /* 8 layers. Without a drafter: 8 feed-forwards and 7 mixers fold; the last layer keeps its
     * add and all-reduce. With taps after layers 1, 3 and 5: those feed-forwards keep theirs too,
     * and the mixers of layers 2, 4 and 6 do not fold. */
    for (const Case& k : { Case{ &plain, 0, 15, 1 + 1, 1 }, Case{ &df, 7, 12, 1 + 4, 4 } }) {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 2, k.max_spec);
        REQUIRE_EQ(qwen35_bf16::declare(&b, k.m->meta(), &ctx), RAD_OK);
        const auto dec = kit(b, { 1, 1, 1, 1 }, {});
        const auto pre = kit(b, {}, { 32, 32 });
        for (const Kit* kt : { dec.get(), pre.get() }) {
            const auto v = run_passes(b, qwen35_bf16::step, { kt }, 0, 2);
            REQUIRE_EQ(v.size(), (size_t)1);
            CHECK_EQ(issued(b, *v[0], "ar_rmsnorm"), k.arn);
            CHECK_EQ(issued(b, *v[0], "all_reduce"), k.ar);
            CHECK_EQ(issued(b, *v[0], "add"), k.add);
        }
    }
}

/* On a lossy wire a step whose message goes out compressed cannot take the exact-only
 * ar_rmsnorm: there the producer issues its own all-reduce and the next norm `add` then `rmsnorm`,
 * the three ops as before. A decode step below the floor still folds. */
TEST(qwen35_bf16_falls_back_to_the_three_ops_where_the_wire_is_lossy) {
    auto issued = [](const RadBuilder& b, const RadCtx& c, const char* op) {
        int n = 0;
        for (const std::string& o : issued_ops(b, c)) n += o == op;
        return n;
    };
    MetaKV plain(qwen35_meta());
    RadBuilder b;
    RadBuildCtx ctx = build_ctx(0, 2);
    ctx.tp_wire_lossy = 1;
    ctx.tp_wire_min_bytes = 64 << 10;   /* decode's 4 x 5120 bf16 is under, prefill's 64 rows over */
    REQUIRE_EQ(qwen35_bf16::declare(&b, plain.meta(), &ctx), RAD_OK);
    const auto dec = kit(b, { 1, 1, 1, 1 }, {});
    const auto pre = kit(b, {}, { 32, 32 });
    const auto vd = run_passes(b, qwen35_bf16::step, { dec.get() }, 0, 2);
    CHECK_EQ(issued(b, *vd[0], "ar_rmsnorm"), 15);
    const auto vp = run_passes(b, qwen35_bf16::step, { pre.get() }, 0, 2);
    CHECK_EQ(issued(b, *vp[0], "ar_rmsnorm"), 0);
    CHECK_EQ(issued(b, *vp[0], "all_reduce"), 1 + 16);
    CHECK_EQ(issued(b, *vp[0], "add"), 16);
}

/* Under calibration (--calib-dir) a step issues each block's input norm ahead of the block for the
 * tap, so the blocks keep their own all-reduce and add: no ar_rmsnorm is declared at two ranks.
 * Declare only; the tap's step needs a device. */
TEST(qwen35_bf16_declares_no_ar_rmsnorm_under_calibration) {
    MetaKV plain(qwen35_meta());
    for (bool calib : { false, true }) {
        RadBuilder b;
        RadBuildCtx ctx = build_ctx(0, 2);
        if (calib) {
            ctx.calib_dir    = "calib-test";
            ctx.calib_tokens = 1 << 20;
        }
        REQUIRE_EQ(qwen35_bf16::declare(&b, plain.meta(), &ctx), RAD_OK);
        if (calib) CHECK_EQ(count_op(b, "ar_rmsnorm"), 0);
        else       CHECK(count_op(b, "ar_rmsnorm") > 0);
    }
}

/* At two ranks a decode step reads the next GEMM's leading weight bytes beside each reduction
 * (kPfGateUpBytes and the rest): gate_up's after the mixer, the next layer's first GEMM after the
 * feed-forward (after the last layer the lm_head's), and the delta net's out before its recurrent
 * update. Each read rides lane 1 behind a join from lane 0, right after the GEMM before it, and
 * lane 0 goes on to the reduction or the update. A verify of one or two sequences reads; one of
 * three (24 rows), a prompt and a mixed step do not. The drafter's query pass reads its own, its
 * context pass none. */
TEST(qwen35_bf16_reads_the_next_weights_beside_its_decode_reductions) {
    MetaKV df = with_dflash2(MetaKV(qwen35_meta()));
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 2, 7);
    REQUIRE_EQ(qwen35_bf16::declare(&b, df.meta(), &ctx), RAD_OK);
    auto wname = [&](const RecOp* o) {
        return o && o->w.size() == 1 ? b.weights[o->w[0] - 1].name : std::string("-");
    };
    /* a read: the GEMM before it > the weight read : MiB > the lane-0 op after it */
    auto site = [](const std::string& gemm, const std::string& read, int64_t mib,
                   const std::string& next) {
        return gemm + ">" + read + ":" + std::to_string(mib) + ">" + next;
    };

    /* The 8-layer fixture: full attention at 3 and 7, the drafter's taps after 1, 3 and 5, so the
     * feed-forwards of 1, 3, 5 and 7 issue their own all-reduce and the others leave it to the
     * next norm's ar_rmsnorm. */
    std::vector<std::string> want;
    for (int l = 0; l < 8; ++l) {
        const std::string bl = "blk." + std::to_string(l) + ".";
        if ((l + 1) % 4) {
            want.push_back(site(bl + "ssm_z.weight", bl + "ssm_out.weight", 12,
                                "gdn_recurrent_update"));
            want.push_back(site(bl + "ssm_out.weight", bl + "ffn_gate_up.weight", 12, "ar_rmsnorm"));
        } else {
            want.push_back(site(bl + "attn_output.weight", bl + "ffn_gate_up.weight", 12,
                                "ar_rmsnorm"));
        }
        const std::string nx = "blk." + std::to_string(l + 1) + ".";
        const std::string next = l == 7            ? std::string("output.weight")
                               : (l + 2) % 4 == 0  ? nx + "attn_qg.weight"
                                                   : nx + "ssm_in.weight";
        const bool own_ar = l == 1 || l == 3 || l == 5 || l == 7;
        want.push_back(site(bl + "ffn_down.weight", next, 8, own_ar ? "all_reduce" : "ar_rmsnorm"));
    }

    /* A pass's reads as sites; every other issue is on lane 0. */
    auto reads = [&](const RadCtx& c) {
        std::vector<std::string> got;
        for (size_t i = 0; i < c.issues.size(); ++i) {
            const RecOp* o = b.op_of(c.issues[i].op);
            if (!o || o->op != "prefetch") {
                CHECK_EQ(c.issues[i].lane, 0);
                continue;
            }
            CHECK_EQ(c.issues[i].lane, 1);
            if (i == 0 || i + 1 == c.issues.size()) {
                got.push_back("a read at the pass's end");
                continue;
            }
            CHECK(ordered(c, i - 1, i));
            const int64_t bytes = pint(o, "bytes");
            CHECK_EQ(bytes % (1 << 20), (int64_t)0);
            const RecOp* g = b.op_of(c.issues[i - 1].op);
            const RecOp* n = b.op_of(c.issues[i + 1].op);
            CHECK(g && g->op == "gemm_nt");
            got.push_back(site(wname(g), wname(o), bytes >> 20, n ? n->op : std::string("-")));
        }
        return got;
    };
    for (int seqs : { 1, 2 }) {
        const auto k = kit(b, std::vector<int>((size_t)seqs, 8), {});
        const auto v = run_passes(b, qwen35_bf16::step, { k.get() }, 0, 2);
        CHECK_PASSES(b, v);
        CHECK_EQ(join(reads(*v[0])), join(want));
        CHECK_EQ(v[0]->joins.size(), want.size());
        CHECK_EQ(races(b, *v[0]), 0);
    }

    /* three sequences (24 rows), a prompt and a mixed step: no read and no join */
    const auto k3 = kit(b, { 8, 8, 8 }, {}), kp = kit(b, {}, { 32 }), km = kit(b, { 8 }, { 6 });
    for (const Kit* k : { k3.get(), kp.get(), km.get() }) {
        const auto v = run_passes(b, qwen35_bf16::step, { k }, 0, 2);
        CHECK_PASSES(b, v);
        CHECK_EQ(count_issued(issued_ops(b, *v[0]), "prefetch"), 0);
        CHECK(v[0]->joins.empty());
    }

    /* The drafter's query pass reads beside its all-reduces at 12 MiB (rad_block_dflash2.h): o,
     * the feed-forward's coefficient projection and gate_up, then the next layer's attention
     * coefficient projection and q, after the last layer the draft head. Its context pass reads
     * none. */
    const auto ver = kit(b, { 8 }, {});
    const auto cp = context_pass(b, *ver);
    const auto qp = query_pass(b, 1, 8, 7);
    const auto vd = run_passes(b, qwen35_bf16::step, { ver.get(), cp.get(), qp.get() }, 0, 2);
    CHECK_PASSES(b, vd);
    REQUIRE_EQ(vd.size(), (size_t)3);
    CHECK_EQ(count_issued(issued_ops(b, *vd[1]), "prefetch"), 0);
    std::vector<std::string> dread;
    for (const RecIssue& r : vd[2]->issues) {
        const RecOp* o = b.op_of(r.op);
        if (!o || o->op != "prefetch") continue;
        CHECK_EQ(r.lane, 1);
        CHECK_EQ(pint(o, "bytes"), (int64_t)12 << 20);
        dread.push_back(wname(o));
    }
    CHECK_EQ(join(dread),
             join({ "dflash.blk.0.attn_output.weight", "dflash.blk.0.ffn_conv_proj.weight",
                    "dflash.blk.0.ffn_gate_up.weight", "dflash.blk.1.attn_conv_proj.weight",
                    "dflash.blk.1.attn_q.weight", "dflash.blk.1.attn_output.weight",
                    "dflash.blk.1.ffn_conv_proj.weight", "dflash.blk.1.ffn_gate_up.weight",
                    "dflash.draft_head.weight" }));
    /* and nothing else is declared: 22 the trunk's, 9 the drafter's */
    CHECK_EQ(count_op(b, "prefetch"), 22 + 9);
}

/* Where launches chain (rad_chain_on) every read is the chained form on lane 0, right behind the
 * GEMM before it, with no join, and the delta net's out read is not declared: in front of the
 * plain recurrent update it would run in series with it. The other reads are those above. */
TEST(qwen35_bf16_reads_on_lane_0_and_skips_the_out_read_where_launches_chain) {
    MetaKV df = with_dflash2(MetaKV(qwen35_meta()));
    const ChainOn chain(1);
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 2, 7);
    REQUIRE_EQ(qwen35_bf16::declare(&b, df.meta(), &ctx), RAD_OK);
    auto wname = [&](const RecOp* o) {
        return o && o->w.size() == 1 ? b.weights[o->w[0] - 1].name : std::string("-");
    };
    auto site = [](const std::string& gemm, const std::string& read, int64_t mib,
                   const std::string& next) {
        return gemm + ">" + read + ":" + std::to_string(mib) + ">" + next;
    };
    std::vector<std::string> want;
    for (int l = 0; l < 8; ++l) {
        const std::string bl = "blk." + std::to_string(l) + ".";
        want.push_back(site(bl + ((l + 1) % 4 ? "ssm_out.weight" : "attn_output.weight"),
                            bl + "ffn_gate_up.weight", 12, "ar_rmsnorm"));
        const std::string nx = "blk." + std::to_string(l + 1) + ".";
        const std::string next = l == 7            ? std::string("output.weight")
                               : (l + 2) % 4 == 0  ? nx + "attn_qg.weight"
                                                   : nx + "ssm_in.weight";
        const bool own_ar = l == 1 || l == 3 || l == 5 || l == 7;
        want.push_back(site(bl + "ffn_down.weight", next, 8, own_ar ? "all_reduce" : "ar_rmsnorm"));
    }
    for (int seqs : { 1, 2 }) {
        const auto k = kit(b, std::vector<int>((size_t)seqs, 8), {});
        const auto v = run_passes(b, qwen35_bf16::step, { k.get() }, 0, 2);
        CHECK_PASSES(b, v);
        const RadCtx& c = *v[0];
        std::vector<std::string> got;
        for (size_t i = 0; i < c.issues.size(); ++i) {
            CHECK_EQ(c.issues[i].lane, 0);
            const RecOp* o = b.op_of(c.issues[i].op);
            if (!o || o->op != "prefetch") continue;
            REQUIRE(i > 0 && i + 1 < c.issues.size());
            CHECK(ordered(c, i - 1, i));
            const RecOp* g = b.op_of(c.issues[i - 1].op);
            const RecOp* n = b.op_of(c.issues[i + 1].op);
            CHECK(g && g->op == "gemm_nt");
            got.push_back(site(wname(g), wname(o), pint(o, "bytes") >> 20,
                               n ? n->op : std::string("-")));
        }
        CHECK_EQ(join(got), join(want));
        CHECK(c.joins.empty());
        CHECK_EQ(races(b, c), 0);
    }
    /* 16 the trunk's (22 less the six delta-net layers' out reads), 9 the drafter's */
    CHECK_EQ(count_op(b, "prefetch"), 16 + 9);
}

/* None of those reads at one rank, under --weight-prefetch off (the drafter's included), or where
 * no library serves `prefetch`. With them off, a verify and the drafter's two passes issue what
 * they issue with them, less the reads, and no join. */
TEST(qwen35_bf16_declares_no_weight_reads_at_one_rank_or_under_weight_prefetch_off) {
    MetaKV df = with_dflash2(MetaKV(qwen35_meta()));
    struct Case { int world; bool off, no_op; int reads; };
    for (const Case& k : { Case{ 2, false, false, 22 + 9 }, Case{ 1, false, false, 0 },
                           Case{ 2, true, false, 0 }, Case{ 2, false, true, 0 } }) {
        RadBuilder b;
        if (k.no_op) b.undeclared.insert("prefetch");
        RadBuildCtx ctx = spec_ctx(0, k.world, 7);
        ctx.weight_prefetch_off = k.off ? 1 : 0;
        REQUIRE_EQ(stderr_of_rc([&] { return qwen35_bf16::declare(&b, df.meta(), &ctx); }), RAD_OK);
        CHECK_EQ(count_op(b, "prefetch"), k.reads);
    }

    std::vector<std::string> ops[2][3];   /* [off][verify, context pass, query pass] */
    size_t joins[2][3] = {};
    for (int off : { 0, 1 }) {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 2, 7);
        ctx.weight_prefetch_off = off;
        REQUIRE_EQ(qwen35_bf16::declare(&b, df.meta(), &ctx), RAD_OK);
        const auto ver = kit(b, { 8 }, {});
        const auto cp = context_pass(b, *ver);
        const auto qp = query_pass(b, 1, 8, 7);
        const auto v = run_passes(b, qwen35_bf16::step, { ver.get(), cp.get(), qp.get() }, 0, 2);
        CHECK_PASSES(b, v);
        REQUIRE_EQ(v.size(), (size_t)3);
        for (size_t p = 0; p < 3; ++p) {
            ops[off][p]   = issued_ops(b, *v[p]);
            joins[off][p] = v[p]->joins.size();
        }
    }
    for (size_t p = 0; p < 3; ++p) {
        std::vector<std::string> rest;
        for (const std::string& s : ops[0][p])
            if (s != "prefetch") rest.push_back(s);
        CHECK_EQ(joins[0][p], (size_t)count_issued(ops[0][p], "prefetch"));   /* one a read */
        CHECK_EQ(joins[1][p], (size_t)0);
        CHECK_EQ(join(ops[1][p]), join(rest));
    }
    CHECK_EQ(count_issued(ops[0][0], "prefetch"), 22);
    CHECK_EQ(count_issued(ops[0][2], "prefetch"), 9);
}

/* =============================== gemma3: TranslateGemma 27B's trunk =========================== */

namespace {

/* TranslateGemma 27B's text geometry cut to twelve layers, two global (5 and 11) over ten sliding
 * ones by the legacy (l + 1) % 6 rule. One head geometry for both classes (head_dim 128, 32 / 16),
 * scores scaled by query_pre_attn_scalar^-0.5, linear rotary scaling on the global class only. */
MetaKV gemma3_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma3";
    m.name        = "test-translategemma-27b";
    m.quant       = "";
    m.n_layers    = 12;
    m.n_embd      = 5376;
    m.n_head      = 32;
    m.n_head_kv   = 16;
    m.head_dim    = 128;
    m.n_ff        = 21504;
    m.n_vocab     = 262208;
    m.n_ctx_train = 131072;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    MetaKV k(m);
    k.set("sliding_window", "1024")
     .set("_sliding_window_pattern", "6")
     .set("rope_local_base_freq", "10000")
     .set("rope_theta", "1000000")
     .set("query_pre_attn_scalar", "168")
     .set("rope_scaling.rope_type", "linear")
     .set("rope_scaling.factor", "8");
    return k;
}

MetaKV gemma3_dflash2_meta() {
    MetaKV k = gemma3_meta();
    k.set("draft.architectures", "DFlash2DraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "32")
     .set("draft.num_key_value_heads", "8")
     .set("draft.hidden_size", "5376")
     .set("draft.intermediate_size", "8192")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "1000000")
     .set("draft.vocab_size", "262208")
     .set("draft.use_sliding_window", "1")
     .set("draft.sliding_window", "2048")
     .set("draft.dflash_config.input_embedding_scale", "73.5")
     .set("draft.dflash_config.target_layer_ids", "1 5 10")
     .set("draft.dflash_config.block_size", "8")
     .set("draft.dflash_config.mask_token_id", "6")
     .set("draft.dflash_config.selector_rank", "64")
     .set("draft.dflash_config.selector_top_k", "8")
     .set("draft.dflash_config.conv_kernel_size", "2")
     .set("draft.dflash_config.conv_group_size", "16");
    return k;
}

}  /* namespace */

/* Sliding layers in a window group declared first, the global ones in a full group, both at the
 * one head geometry; layers bound by the (l + 1) % 6 rule. */
TEST(gemma3_declares_two_kv_groups_and_binds_the_schedule) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    REQUIRE_EQ((int)b.kv_groups.size(), 2);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_sliding"));
    CHECK_EQ((int)b.kv_groups[0].d.kind, RAD_KV_WINDOW);
    CHECK_EQ(b.kv_groups[0].d.window, 1024);
    CHECK_EQ(b.kv_groups[0].d.n_head_kv, 16);
    CHECK_EQ(b.kv_groups[0].d.head_dim, 128);
    CHECK_EQ(b.kv_groups[1].name, std::string("kv_global"));
    CHECK_EQ((int)b.kv_groups[1].d.kind, RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[1].d.window, 0);
    CHECK_EQ(b.kv_groups[1].d.head_dim, 128);
    CHECK_EQ((int)b.binds.size(), 12);
    for (int l = 0; l < 12; ++l) {
        CHECK_EQ((int)b.binds[(size_t)l].first, l);
        CHECK_EQ((int)b.binds[(size_t)l].second, (l + 1) % 6 == 0 ? 2 : 1);
    }
}

/* layer_types, where the config has it, names the schedule instead of the legacy pattern, and a
 * count that is not the model's is refused. */
TEST(gemma3_layer_types_name_the_global_layers) {
    MetaKV mk = gemma3_meta();
    mk.set("layer_types", "[\"full_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"full_attention\"]");
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    for (int l = 0; l < 12; ++l)
        CHECK_EQ((int)b.binds[(size_t)l].second, (l == 0 || l == 11) ? 2 : 1);

    mk.set("layer_types", "[\"full_attention\", \"sliding_attention\"]");
    RadBuilder c;
    CHECK(stderr_of_rc([&] { return gemma3::declare(&c, mk.meta(), &ctx); }) != RAD_OK);
}

/* Both classes attend at 1/sqrt(query_pre_attn_scalar), not 1/sqrt(head_dim); the sliding class
 * is windowed and rotates at rope_local_base_freq unscaled, the global class at rope_theta with
 * the linear factor; every norm is GemmaRMSNorm (wadd 1). */
TEST(gemma3_attention_scale_rope_and_norms_per_class) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    std::vector<const RecOp*> at;
    for (const auto& o : b.ops) if (o.op == "attn_paged") at.push_back(&o);
    REQUIRE(!at.empty());
    bool windowed = false, full = false;
    for (const RecOp* o : at) {
        CHECK_EQ(pint(o, "head_dim"), 128);
        CHECK_EQ(pint(o, "gqa"), 2);
        CHECK(std::fabs(pflt(o, "scale") - 1.0 / std::sqrt(168.0)) < 1e-9);
        if (pint(o, "window") == 1024) windowed = true;
        if (pint(o, "window") == 0) full = true;
    }
    CHECK(windowed && full);
    std::set<std::pair<double, double>> rot;   /* (theta, scale) of every rotary op */
    for (const auto& o : b.ops)
        if (param(&o, "theta")) rot.insert({ pflt(&o, "theta"), pflt(&o, "scale") });
    CHECK(rot.count({ 10000.0, 1.0 }) == 1);
    CHECK(rot.count({ 1000000.0, 0.125 }) == 1 || rot.count({ 1000000.0, 8.0 }) == 1);
    for (const auto& o : b.ops)
        if (param(&o, "wadd")) CHECK_EQ(pflt(&o, "wadd"), 1.0);
}

/* Gemma 2's soft-capping and a YaRN scaling are refused rather than served without them. */
TEST(gemma3_refuses_softcapping_and_yarn) {
    RadBuildCtx ctx = build_ctx();
    {
        MetaKV mk = gemma3_meta();
        mk.set("final_logit_softcapping", "30");
        RadBuilder b;
        CHECK_EQ(stderr_of_rc([&] { return gemma3::declare(&b, mk.meta(), &ctx); }), RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = gemma3_meta();
        mk.set("rope_scaling.rope_type", "yarn").set("rope_scaling.original_max_position_embeddings", "8192");
        RadBuilder b;
        CHECK(stderr_of_rc([&] { return gemma3::declare(&b, mk.meta(), &ctx); }) != RAD_OK);
    }
}

TEST(gemma3_every_op_declares_its_buffers_and_every_weight_has_a_name_map) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
}

/* A decode step and a prefill step lint clean at one and two ranks. */
TEST(gemma3_steps_lint_clean_at_one_and_two_ranks) {
    MetaKV mk = gemma3_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = build_ctx(rank, world);
            REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
            RadBatch batch = decode_batch();
            RadCtx c{ .b = &b, .batch = &batch };
            gemma3::step(&c, &batch);
            CHECK_LINT(b, &c);
            CHECK(c.step_fail.empty());
            CHECK_EQ(count_op(b, "all_reduce") + count_op(b, "ar_rmsnorm_quant_fp8") +
                     count_op(b, "ar_rmsnorm_post_quant_fp8") > 0, world == 2);
        }
}

/* On quant_act_i8g's grid (int8 codes with a scale per 128 columns: TranslateGemma's i8a8 recipe)
 * the trunk takes the `_i8g` twins of the fused junction and of the GELU quantiser, with the
 * all-reduce folded in at two ranks: one junction in front of the first block, two a layer but the
 * last layer's second. Where no library declares the twins every linear quantises its own input;
 * with them only o_proj's does, which no junction writes. */
TEST(gemma3_fuses_its_junctions_on_the_grouped_int8_grid) {
    const RadEncoding i8g = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    const char* const twins[] = { "rmsnorm_quant_i8g", "gelu_mul_quant_i8g", "ar_rmsnorm_quant_i8g" };
    for (int world : { 1, 2 })
        for (bool fused : { true, false }) {
            MetaKV mk = gemma3_meta();
            RadBuilder b;
            b.encs = { { "blk.", i8g } };
            if (!fused) for (const char* op : twins) b.undeclared.insert(op);
            RadBuildCtx ctx = build_ctx(0, world);
            ctx.max_tok = 2048;
            REQUIRE_EQ(stderr_of_rc([&] { return gemma3::declare(&b, mk.meta(), &ctx); }), RAD_OK);
            const gemma3::Model& m = gemma3::g_model[0];
            const int L = (int)m.layers.size();
            CHECK_EQ(m.fuse, fused);
            CHECK_EQ(m.fuse_ar, fused && world == 2);
            CHECK_EQ(count_op(b, "rmsnorm_quant_i8g"), fused ? 2 * L : 0);
            CHECK_EQ(count_op(b, "gelu_mul_quant_i8g"), fused ? L : 0);
            CHECK_EQ(count_op(b, "ar_rmsnorm_quant_i8g"), fused && world == 2 ? 2 * L - 1 : 0);
            for (const char* op : { "rmsnorm_quant_i8", "gelu_mul_quant_i8", "ar_rmsnorm_quant_i8" })
                CHECK_EQ(count_op(b, op), 0);
            const auto dec = kit(b, {}, { 4 });
            const auto v = run_passes(b, gemma3::step, { dec.get() }, 0, world);
            CHECK_PASSES(b, v);
            int own = 0;
            for (const RecIssue& i : v[0]->issues)
                if (const RecOp* o = b.op_of(i.op); o && o->op == "quant_act_i8g") ++own;
            /* Fused, only o_proj's input is quantised by an op of its own. */
            if (fused) CHECK_EQ(own, L);
            else       CHECK(own >= 3 * L);
        }
}

/* Under a lossy --tp-wire a decode step's few rows, whose message goes out exact, fold every
 * junction's all-reduce in; a taller step, like a prefill chunk, keeps the lossy all-reduce,
 * unfused: the int8 junction sends the exact wire only. */
TEST(gemma3_under_a_lossy_wire_small_steps_keep_the_fused_junction) {
    MetaKV mk = gemma3_meta();
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    for (int rank = 0; rank < 2; ++rank) {
        RadBuilder b;
        b.encs = { { "attn_", w4 }, { "ffn_", w4 } };
        RadBuildCtx ctx = build_ctx(rank, 2);
        ctx.tp_wire_lossy = 1;
        ctx.tp_wire_min_bytes = 128 * 1024;
        REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
        const gemma3::Model& m = gemma3::g_model[rank];
        CHECK(!m.fuse_mx && m.fuse && m.fuse_ar);
        auto issued = [&](const RadCtx& c, std::initializer_list<const char*> names) {
            int n = 0;
            for (const RecIssue& i : c.issues)
                if (const RecOp* o = b.op_of(i.op))
                    for (const char* nm : names) n += o->op == nm;
            return n;
        };
        const auto dec = kit(b, { 1 }, {});
        const auto pre = kit(b, {}, { 64 });           /* 64 x 5376 x 2 bytes, above the floor */
        const auto v = run_passes(b, gemma3::step, { dec.get(), pre.get() }, rank, 2);
        CHECK_PASSES(b, v);
        const int fused_dec = issued(*v[0], { "ar_rmsnorm_quant_i8" });
        const int fused_pre = issued(*v[1], { "ar_rmsnorm_quant_i8" });
        CHECK(fused_dec > 0);
        CHECK_EQ(fused_pre, 0);
        /* Every junction the decode step fused is an all-reduce the prefill step issues alone. */
        CHECK_EQ(issued(*v[1], { "all_reduce" }), issued(*v[0], { "all_reduce" }) + fused_dec);
        /* 24 rows (24 x 5376 x 2 bytes, above the floor) are unfused as well. */
        const auto mid = kit(b, {}, { 24 });
        const auto vm = run_passes(b, gemma3::step, { mid.get() }, rank, 2);
        CHECK_PASSES(b, vm);
        CHECK_EQ(issued(*vm[0], { "ar_rmsnorm_quant_i8" }), 0);
    }
}

/* The DFlash2 drafter: a block of 8 (depth 7, fixed), taps copied out after the named layers into
 * one column-sliced buffer, the mask token from the config, and the drafter's own KV group after
 * the trunk's two. Its input_embedding_scale must be the trunk's sqrt(hidden) held in bf16
 * (73.5 on the 27B), which the embedding lookup applies. */
TEST(gemma3_dflash2_drafter_taps_and_block) {
    MetaKV mk = gemma3_dflash2_meta();
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
    CHECK_EQ(b.drafter.depth, 7);
    CHECK_EQ(b.drafter.block, 8);
    CHECK_EQ(b.drafter.mask_token, 6);
    const RecBuf* src = find_b(b, "dflash_src");
    REQUIRE(src != nullptr);
    CHECK_EQ(src->d.shape[1], 3 * 5376);
    CHECK_EQ((int)b.kv_groups.size(), 3);
    CHECK(!noted(b, "input_embedding_scale"));
    {   /* A drafter trained at another embedding scale still serves, with a note. */
        MetaKV other = gemma3_dflash2_meta();
        other.set("draft.dflash_config.input_embedding_scale", "64");
        RadBuilder o;
        REQUIRE_EQ(gemma3::declare(&o, other.meta(), &ctx), RAD_OK);
        CHECK(noted(o, "input_embedding_scale"));
    }

    RadArchProbe p{};
    CHECK_OK(gemma3::probe(mk.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
    CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);   /* with a drafter and without one */
    MetaKV plain = gemma3_meta();
    RadArchProbe q{};
    CHECK_OK(gemma3::probe(plain.meta(), &q));
    CHECK_EQ(q.draft_depth, 0);
}

/* --kv-cache-dtype i8: int8 K and V with one f16 scale each per token and head. The trunk's
 * group is int8 at head_dim 256 and the DFlash2 drafter's at its 128 (the core adds a row's four
 * scale bytes), and every op that names a cache dtype bands on i8. */
TEST(qwen35_i8_cache_is_an_int8_group_at_the_models_head_dim) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    ctx.kv_dtype = RAD_I8;
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(f.head_dim, 256);
    CHECK_EQ(b.kv_groups.back().d.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(b.kv_groups.back().d.head_dim, 128);
    int i8 = 0, other = 0;
    for (const auto& o : b.ops) {
        if (!param(&o, "kv_dtype")) continue;
        const std::string kd = pstr(&o, "kv_dtype");
        i8 += kd == "i8";
        other += kd != "i8";
    }
    CHECK(i8 > 0);
    CHECK_EQ(other, 0);
}

/* The int8 cache streams: the gated attention declares the streamed store and the fold for it as
 * it does for a bf16 cache. */
TEST(an_int8_kv_cache_declares_the_streamed_store_and_the_fold) {
    RadModelMeta q4 = qwen4exp_meta();
    RadBuildCtx ctx = build_ctx();
    ctx.kv_dtype = RAD_I8;
    ctx.kv_resident_blocks = 64;
    ctx.kv_stream_ok = 1;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &q4, &ctx), RAD_OK);
    int store = 0, fold = 0;
    for (const auto& o : b.ops) {
        const std::string op = o.op;
        if (op == "kv_store_streamed" || op == "qk_norm_rope_gate_kv_store_streamed") {
            CHECK_EQ(std::string(pstr(&o, "kv_dtype")), std::string("i8"));
            (op == "kv_store_streamed" ? store : fold)++;
        }
    }
    CHECK(store > 0);
    CHECK(fold > 0);
}

/* Flash-Next takes chained launches with and without its draft head: its pass is dependent
 * launches nearly throughout, and the kernels of its decode heights have chained forms. */
TEST(qwen4exp_probe_asks_for_chained_launches_always) {
    RadModelMeta q4 = qwen4exp_meta();
    RadArchProbe p{};
    CHECK_OK(qwen4exp_fp8::probe(&q4, &p));
    CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);
}

/* The int8 cache on the routed build at Qwen3.6-35B-A3B's 8 query heads a kv head: an int8 group
 * at head_dim 256, every cache op banded on i8. */
TEST(the_int8_cache_declares_on_the_routed_build_at_gqa8) {
    MetaKV m35 = q35moe_meta();
    m35.m.n_head = 8 * m35.m.n_head_kv;
    RadBuildCtx ctx = build_ctx();
    ctx.kv_dtype = RAD_I8;
    RadBuilder b;
    REQUIRE_EQ(qwen35moe_fp8::declare(&b, m35.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.kv_groups[0].d.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(b.kv_groups[0].d.head_dim, 256);
    int i8 = 0, other = 0;
    for (const auto& o : b.ops) {
        if (!param(&o, "kv_dtype")) continue;
        i8 += pstr(&o, "kv_dtype") == "i8";
        other += pstr(&o, "kv_dtype") != "i8";
    }
    CHECK(i8 > 0);
    CHECK_EQ(other, 0);
}

/* The 35B-A3B: a replicated router over every expert, a normalised top-k (norm_topk_prob
 * defaults on in this family; without it the weights sum to about 0.4), fp8a8 grouped GEMMs over
 * the expert planes, and the shared expert folded into the gather behind a sigmoid gate. */
TEST(qwen35moe_fp8_routes_every_expert_and_folds_the_shared_arm) {
    MetaKV mk = q35moe_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35moe_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(count_op(b, "router_topk"), 8);
        for (const auto& o : b.ops) {
            if (o.op == "router_topk") {
                CHECK_EQ(pint(&o, "n_expert"), 16);
                CHECK_EQ(pint(&o, "top_k"), 4);
                CHECK_EQ(pint(&o, "norm"), 1);
            }
            if (o.op == "moe_gemm_q") {
                CHECK_EQ(pstr(&o, "dtype"), std::string("fp8a8"));
                CHECK_EQ(pint(&o, "n_expert"), 16);
                const long long N = pint(&o, "N"), K = pint(&o, "K");
                CHECK((N == 2 * 512 / world && K == 5120) || (N == 5120 && K == 512 / world));
            }
            if (o.op == "moe_gather") CHECK_EQ(pstr(&o, "act"), std::string("sigmoid"));
        }
        CHECK_EQ(count_op(b, "moe_gather"), 8);
        CHECK(count_op(b, "moe_gemm_q") >= 16);
        /* The shared arm's buffers sit on the concurrent list, out of the linear packing. */
        CHECK(std::count(b.concurrent.begin(), b.concurrent.end(),
                         (rad_buf)buf_named(b, "shexp_gate_up")) > 0);
    }
    /* Without experts this is the dense plugin's checkpoint, and the reverse. */
    {
        MetaKV dense = q35fp8_meta();
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(qwen35moe_fp8::declare(&b, dense.meta(), &ctx), RAD_E_UNSUPPORTED);
        RadBuilder b2;
        CHECK_EQ(qwen35_fp8::declare(&b2, mk.meta(), &ctx), RAD_E_UNSUPPORTED);
    }
}

/* The experts are served in the format the container stores, read from layer 0's first expert:
 * rotated four-bit experts take the w4a8h grouped GEMM, whose gate_up reads a rotated copy of the
 * block input (`moe_hr`), at one rank and two. Plain bf16 experts are refused, since this plugin
 * has no bf16 expert path. */
TEST(qwen35moe_fp8_serves_the_experts_in_their_stored_format) {
    MetaKV mk = q35moe_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    REQUIRE(rad_enc_valid(&w4));
    for (int world : { 1, 2 }) {
        RadBuilder b;
        b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35moe_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        const rad_buf hrq = (rad_buf)buf_named(b, "moe_hr.q");
        REQUIRE(hrq != 0);
        int rotated = 0, fp8 = 0, reads_hr = 0;
        for (const auto& o : b.ops) {
            if (o.op.rfind("moe_gemm", 0) != 0) continue;
            rotated += pstr(&o, "dtype") == "w4a8h";
            fp8 += pstr(&o, "dtype") == "fp8a8";
            reads_hr += in(o.reads, hrq);
        }
        CHECK_EQ(rotated, 2 * 8);
        CHECK_EQ(fp8, 0);
        CHECK_EQ(reads_hr, 8);
    }
    {
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        RadBuilder b;
        b.encs = { { "ffn_gate_up_exps", plain }, { "ffn_down_exps", plain } };
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return qwen35moe_fp8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* The routed model's engine steps, its routed MTP head included, lint clean at one and two
 * ranks. */
TEST(qwen35moe_fp8_steps_lint_clean_with_its_routed_mtp_head) {
    MetaKV mk = q35moe_meta();
    lint_qwen35_fp8_family(qwen35moe_fp8::declare, qwen35moe_fp8::step, mk.meta(), 3, false, 0);
}

/* The 35B-A3B-VL (Sage-A1.5's shape): the routed trunk and the vision tower declare together,
 * the tower once and not a copy a rank, and the routed MTP head beside them. */
TEST(qwen35moe_fp8_declares_the_vision_tower_beside_the_routed_trunk) {
    MetaKV mk = q35moe_vl_meta();
    /* With no encoder budget the tower is not declared, whatever the container carries. */
    {
        RadBuilder b;
        RadBuildCtx ctx = vl_ctx(0);
        REQUIRE_EQ(qwen35moe_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK(b.encoder_name.empty());
        CHECK(find_w(b, "v.patch.weight") == nullptr);
        CHECK_EQ(count_op(b, "router_topk"), 8);
    }
    RadBuilder b;
    RadBuildCtx ctx = vl_ctx(1024);
    ctx.max_spec = 3;
    REQUIRE_EQ(qwen35moe_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);

    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO));
    CHECK_EQ(b.encoder.max_patches, 1024);
    CHECK_EQ(b.encoder.n_embd, 5120);
    CHECK_EQ(b.encoder.merge, 4);

    /* The trunk is still the routed one (eight blocks, plus the MTP layer's own router under
     * --max-spec 3), and the shared expert is still folded in. */
    CHECK_EQ(count_op(b, "router_topk"), 9);
    CHECK(count_op(b, "moe_gemm_q") >= 16);

    /* 3 stem + 12 a block + 6 merger, each mapped from the checkpoint's model.visual.*. */
    int nv = 0;
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            ++nv;
            CHECK(find_map(b, w.name) != nullptr);
        }
    CHECK_EQ(nv, 3 + 12 * 2 + 6);
}

/* A DFlash2 drafter merged into the routed VL trunk (data/recipes/sage-a1.5-fp8-df2.recipe)
 * replaces the routed MTP head: the block drafter is declared at the trained block, the MTP
 * layer and its router are not, and the vision tower still is, at one rank and two. The probe
 * fixes the depth at the block. */
TEST(qwen35moe_fp8_dflash2_drafter_replaces_the_routed_mtp_head) {
    MetaKV mk = with_dflash2(q35moe_vl_meta());
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = spec_ctx(rank, world, 7);
            ctx.max_enc_patches = 1024;
            REQUIRE_EQ(qwen35moe_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
            CHECK_EQ(b.drafter_name, std::string("dflash2"));
            CHECK_EQ(b.drafter.depth, 7);
            CHECK_EQ(b.drafter.block, 8);
            CHECK(find_b(b, "mtp_x") == nullptr);
            CHECK(noted(b, "the DFlash2 drafter replaces the 1-layer MTP head"));
            /* Eight routed blocks and no router for the MTP layer. */
            CHECK_EQ(count_op(b, "router_topk"), 8);
            for (const auto& w : b.weights) CHECK(w.name.rfind("blk.8.", 0) != 0);
            const RecBuf* src = find_b(b, "dflash_src");
            REQUIRE(src != nullptr);
            CHECK_EQ(src->d.shape[1], 3 * 5120);
            CHECK_EQ(b.encoder_name, std::string("vision"));
        }
    RadArchProbe p{};
    CHECK_OK(qwen35moe_fp8::probe(mk.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
    /* The same container under --num-speculative-tokens 0 declares neither drafter. */
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35moe_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
    CHECK(find_b(b, "dflash_src") == nullptr);
    CHECK(find_b(b, "mtp_x") == nullptr);
}

/* The routed trunk's engine steps with the DFlash2 drafter (verify, context pass, query pass)
 * lint clean at one and two ranks. */
TEST(qwen35moe_fp8_steps_lint_clean_with_a_dflash2_drafter) {
    MetaKV mk = with_dflash2(q35moe_meta());
    lint_qwen35_fp8_family(qwen35moe_fp8::declare, qwen35moe_fp8::step, mk.meta(), 7, true, 8);
}

/* `layer_types` is checked against the interval rule, not ignored: a model built with the wrong
 * layers linear still generates fluent text. A disagreeing, short or unknown schedule is refused
 * by every plugin of the family; an agreeing one declares. */
TEST(the_qwen35_family_refuses_a_layer_schedule_that_contradicts_its_interval) {
    const char* agree = "linear_attention linear_attention linear_attention full_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    const char* wrong = "linear_attention linear_attention full_attention linear_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    const char* shrt  = "linear_attention linear_attention linear_attention full_attention";
    const char* odd   = "linear_attention linear_attention linear_attention sliding_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    for (const char* s : { agree, wrong, shrt, odd }) {
        const int want = s == agree ? RAD_OK : RAD_E_UNSUPPORTED;
        MetaKV b16(qwen35_meta());
        b16.set("layer_types", s);
        MetaKV f8 = q35fp8_meta();
        f8.set("layer_types", s);
        RadBuildCtx ctx = build_ctx();
        RadBuilder x, y;
        CHECK_EQ(stderr_of_rc([&] { return qwen35_bf16::declare(&x, b16.meta(), &ctx); }), want);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&y, f8.meta(), &ctx); }), want);
    }
    /* Flash-Next reads the same key under its own spelling. */
    MetaKV q4(qwen4exp_meta());
    q4.set("layer_types", wrong);
    RadBuildCtx c4 = build_ctx(0, 1);
    c4.max_tok = 2048; c4.max_seqs = 8; c4.max_spec = 3; c4.max_ctx = 200000;
    RadBuilder z;
    CHECK_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&z, q4.meta(), &c4); }),
             RAD_E_UNSUPPORTED);
}

/* ==================================================================== qwen35_w8a8 */

namespace {

/* The dense fixture as data/recipes/q38-27b-w8a8-df2.recipe states it: w8a8 / fwht128. */
MetaKV q35w8a8_meta() {
    MetaKV k = q35fp8_meta();
    k.m.quant = "w8a8_fwht128";
    k.m.name  = "test-qwen35-27b-w8a8";
    k.set("quantization_config.quant_method", "w8a8").set("quantization_config.fmt", "fwht128");
    return k;
}

/* The recipe's encodings by declared name: the eight trunk linears int8 (`/fwht128` unless
 * `rotated` is false, which the plugin refuses), an int8 lm_head, and the DFlash2 drafter's MXFP4
 * linears with its 2-bit head. The drafter's keys come first because its names contain the
 * trunk's (dflash.blk.0.attn_k.weight), and the head's last because `output.weight` is in
 * `attn_output.weight`. */
void w8a8_encs(RadBuilder* b, bool rotated = true) {
    RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    if (rotated) rad_enc_copy_str(i8.transform, "fwht128");
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    b->encs = { { "dflash.draft_head", u2 },
                { "dflash.", rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32) } };
    for (const char* t : { "attn_qg.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up.weight", "ffn_down.weight" })
        b->encs.push_back({ t, i8 });
    b->encs.push_back({ "output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
}

/* qwen35_w8a8::declare with the recipe's encodings, in lint_qwen35_fp8_family's shape. */
int w8a8_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    w8a8_encs(b);
    return qwen35_w8a8::declare(b, meta, ctx);
}

}  /* namespace */

/* The `_had_quant_i8g` producers and the ops that write those codes in their epilogue (the
 * attention gate's, the delta net's decode conv fold's), which libr4d serves and libref does not. */
const char* const kW8a8Fused[] = { "rmsnorm_had_quant_i8g", "ar_rmsnorm_had_quant_i8g",
                                   "gated_had_quant_i8g", "gate_had_quant_i8g",
                                   "gdn_conv_recurrent_update_had" };

/* Where the `_had_quant_i8g` producers are not served, every trunk linear of a rotated int8
 * container is an i8a8h GEMM right behind its own had_quant_act_i8g, at one rank and two: 2
 * attention layers of four linears, 6 delta-net layers of two and 8 feed-forwards of two. None is
 * left at fp8, which is what a linear that missed the encoding falls back to, and the MLP declares
 * no gated fold (an fp8a8 tile over E4M3 codes). The four trunk activations carry an int8 pair;
 * the delta net's `gdn_in` feeds no int8 linear. */
TEST(qwen35_w8a8_serves_every_trunk_linear_rotated_int8) {
    MetaKV mk = q35w8a8_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        for (const char* op : kW8a8Fused) b.undeclared.insert(op);
        w8a8_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        int rot = 0, paired = 0;
        for (size_t i = 0; i < b.ops.size(); ++i) {
            const RecOp& o = b.ops[i];
            if (o.op != "gemm_nt_q") continue;
            CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
            if (pstr(&o, "dtype") != "i8a8h") continue;
            ++rot;
            paired += i > 0 && b.ops[i - 1].op == "had_quant_act_i8g";
        }
        CHECK_EQ(rot, 2 * 4 + 6 * 2 + 8 * 2);
        CHECK_EQ(paired, rot);
        CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
        CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 0);
        for (const char* a : { "h.q8", "attn_out.q8", "gdn_o.q8", "ffn.q8" })
            CHECK(find_b(b, a) != nullptr);
        CHECK(find_b(b, "gdn_in.q8") == nullptr);
    }
}

/* Where they are served, the norms, the gated product, the attention gate and the delta net's
 * decode conv fold write had_quant_act_i8g's codes into the int8 pair, and the projections reading
 * them declare no quantiser: had_quant_act_i8g is left only as the delta net's quantiser of the
 * rows its decode fold does not serve (6), every norm is the rotated twin (at two ranks with its
 * fused all-reduce form beside it), every feed-forward's product is gated_had_quant_i8g and every
 * attention gate gate_had_quant_i8g. No E4M3 norm or gated quantiser is left, and the 36 GEMMs are
 * unchanged. */
TEST(qwen35_w8a8_feeds_its_rotated_codes_from_the_producers_where_libr4d_serves_them) {
    MetaKV mk = q35w8a8_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        w8a8_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        int rot = 0;
        for (const RecOp& o : b.ops) rot += o.op == "gemm_nt_q" && pstr(&o, "dtype") == "i8a8h";
        CHECK_EQ(rot, 2 * 4 + 6 * 2 + 8 * 2);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), 6);
        CHECK_EQ(count_op(b, "rmsnorm_had_quant_i8g"), 8 + 8);
        CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 8);
        CHECK_EQ(count_op(b, "gate_had_quant_i8g"), 2);
        CHECK_EQ(count_op(b, "gdn_conv_recurrent_update_had"), 6);
        for (const char* op : { "rmsnorm_quant_fp8", "ar_rmsnorm_quant_fp8", "gated_quant_fp8",
                                "quant_act_i8g" })
            CHECK_EQ(count_op(b, op), 0);
        CHECK_EQ(count_op(b, "ar_rmsnorm_had_quant_i8g") > 0, world == 2);
    }
}

/* What the int8 build refuses: a trunk that is not int8 (no stated encoding, which a block-fp8
 * plugin reads as fp8), a plain int8 trunk (the descriptor names the rotation, and a plain weight
 * served behind the rotated quantiser would be wrong), and a container stated for another format
 * or without the rotation's `fmt`. */
TEST(qwen35_w8a8_refuses_what_is_not_a_rotated_int8_trunk) {
    RadBuildCtx ctx = build_ctx();
    {
        MetaKV mk = q35w8a8_meta();
        RadBuilder b;
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_DTYPE);
    }
    {
        MetaKV mk = q35w8a8_meta();
        RadBuilder b;
        w8a8_encs(&b, false);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_DTYPE);
    }
    {
        MetaKV mk = q35w8a8_meta();
        mk.set("quantization_config.quant_method", "fp8").set("quantization_config.fmt", "e4m3");
        RadBuilder b;
        w8a8_encs(&b);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = q35w8a8_meta();
        mk.drop("quantization_config.fmt");
        RadBuilder b;
        w8a8_encs(&b);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* Every engine step lints clean at one and two ranks, with the MTP head's passes and with the
 * DFlash2 drafter's, whose depth the probe fixes at its trained block as qwen35_fp8's does. */
TEST(qwen35_w8a8_steps_lint_clean_with_its_mtp_head_and_a_dflash2_drafter) {
    MetaKV mtp = q35w8a8_meta();
    lint_qwen35_fp8_family(w8a8_declare, qwen35_w8a8::step, mtp.meta(), 3, false, 0);
    MetaKV df = with_dflash2(q35w8a8_meta());
    lint_qwen35_fp8_family(w8a8_declare, qwen35_w8a8::step, df.meta(), 7, true, 8);
    RadArchProbe p{};
    CHECK_OK(qwen35_w8a8::probe(df.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
}

namespace {

/* Split target as prepared by vhead.py; each class covers complete tiles on both TP ranks. */
void q35_split_head(RadBuilder& b, bool draft, int draft_group = 128) {
    w8a8_encs(&b);
    const int64_t rows[] = {130560, 31744, 41472, 44544};
    const uint32_t dt[] = {RAD_U3, RAD_U4, RAD_U5, RAD_U6};
    for (int i = 0; i < 4; ++i) {
        RadEncoding e{};
        rad_enc_copy_str(e.scheme, "trellis");
        rad_enc_copy_str(e.transform, "fwht128");
        rad_enc_add_plane(&e, "codes", dt[i], 1, 1);
        rad_enc_add_plane(&e, "scale", RAD_F32, 1, 0);
        const int t = rad_enc_add_table(&e, "trellis", RAD_I8, 1 << 16, 1);
        e.plane[t].block[0] = 256;
        rad_enc_add_table(&e, "trellis.mad", RAD_U32, 2, 1);
        const std::string name = "output.k" + std::to_string(i + 3) + ".weight";
        b.encs.emplace_back(name, e);
        b.shapes[name] = {rows[i], 5120};
    }
    if (draft) {
        RadEncoding e = rad_enc_affine(RAD_U2, RAD_F16, 1, draft_group);
        rad_enc_add_plane(&e, "zero", RAD_U8, 1, draft_group);
        b.encs.emplace_back("mtp.draft_head.weight", e);
    }
}

}  /* namespace */

TEST(qwen35_split_target_verifies_independent_u2_mtp_proposals) {
    MetaKV mk = q35w8a8_meta();
    for (int world : {1, 2}) for (int rank = 0; rank < world; ++rank) {
        RadBuilder b;
        q35_split_head(b, true);
        RadBuildCtx ctx = spec_ctx(rank, world, 3);
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(b.drafter_name, std::string("mtp"));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
        CHECK_EQ(b.drafter.proposal, (rad_buf)buf_named(b, "mtp_didx"));
        CHECK_EQ(b.drafter.proposal_pitch, 1);
        CHECK_EQ(count_op(b, "mtp_couple_propose"), 0);
        CHECK(find_b(b, "mtp_couple_state") == nullptr);
        const RecBuf* dlog = find_b(b, "mtp_dlog");
        REQUIRE(dlog != nullptr);
        CHECK_EQ(dlog->d.shape[0], ctx.max_seqs);
        CHECK_EQ(dlog->d.shape[1], mk.m.n_vocab);
        CHECK(find_w(b, "output.weight") == nullptr);
        REQUIRE(find_w(b, "mtp.draft_head.weight") != nullptr);
        CHECK_EQ(count_op(b, "logits_gemm"), 0);  /* no phantom single-weight head */
        const auto ver = kit(b, {4, 4}, {});
        const auto hist = history_pass(b, *ver);
        const auto next = round_pass(b, 2, 2);
        const auto v = run_passes(b, qwen35_w8a8::step, {ver.get(), hist.get(), next.get()}, rank, world);
        CHECK_PASSES(b, v);
        CHECK(issue_of(b, *v[0], "logits_gemm_vocab") != nullptr);
        for (int p : {1, 2}) {
            CHECK(issue_of(b, *v[p], "logits_gemm_vocab") == nullptr);
            CHECK(issue_of(b, *v[p], "row_topk") != nullptr);
        }
    }
}

TEST(qwen35_mtp_draft_head_calibration_taps_full_history) {
    MetaKV mk = q35w8a8_meta();
    const char* out_dir = "/tmp/rad-qwen35-mtp-calib-test";
    for (int rank = 0; rank < 2; ++rank) {
        RadBuilder b;
        q35_split_head(b, true);
        RadBuildCtx ctx = spec_ctx(rank, 2, 3);
        ctx.max_spec = 0;  /* calibration needs the MTP context pass, not speculative proposals */
        ctx.calib_dir = out_dir;
        ctx.calib_layer_lo = 8;  /* n_main: the virtual MTP layer */
        ctx.calib_layer_hi = 9;
        ctx.calib_tokens = 100;  /* keep the fake stream below the write threshold */
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(b.drafter.kind, RAD_DRAFT_HISTORY);
        CHECK_EQ(b.drafter.depth, 0);
        CHECK_EQ(b.drafter.proposal, RAD_NULL_HANDLE);
        CHECK_EQ(b.drafter.proposal_pitch, 0);
        CHECK(find_w(b, "mtp.draft_head.weight") == nullptr);
        CHECK(find_b(b, "mtp_dlog") == nullptr);
        auto& model = qwen35_w8a8::g_model[rank];
        CHECK(model.mtp.draft_head_calib_on);
        CHECK_EQ(model.mtp.op_dsel, RAD_NULL_HANDLE);
        CHECK_EQ(model.mtp.op_dquant, RAD_NULL_HANDLE);  /* all ranks run the same replicated-head path */
        if (rank == 0) {
            CHECK(model.mtp.draft_head_calib.op != 0);
            CHECK(model.mtp.draft_head_calib.h != 0);
        } else {
            CHECK_EQ(model.mtp.draft_head_calib.op, 0u);
        }

        auto k = kit(b, {}, { 8 }, 10);
        k->b.phase = RAD_PHASE_DECODE;
        k->out = { 0, 1, 2, 3, 4, 5, 6, 7 };
        k->b.out_ids = k->out.data();
        k->b.n_out = 8;
        k->b.n_draft_out = 0;  /* the trunk step has no proposal rows */
        k->b.draft_pass = 0;
        RadCtx trunk{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
        qwen35_w8a8::step(&trunk, &k->b);
        RadBatch history_batch = k->b;
        history_batch.draft_pass = -1;  /* every row is history; no proposal row */
        RadCtx history{ .b = &b, .batch = &history_batch, .rank = rank, .world = 2 };
        qwen35_w8a8::step(&history, &history_batch);
        CHECK(trunk.step_fail.empty());
        CHECK(history.step_fail.empty());
        CHECK(!history.issues.empty());
        CHECK(issue_of(b, history, "row_topk") == nullptr);

        RadBatch invalid = history_batch;
        invalid.n_draft_out = 1;
        RadCtx bad{ .b = &b, .batch = &invalid, .rank = rank, .world = 2 };
        model.mtp.step(&bad, &invalid);
        CHECK(!bad.step_fail.empty());
        CHECK(bad.issues.empty());
        if (rank == 0) CHECK(issue_of(b, history, "gram_accum") != nullptr);
        else           CHECK(issue_of(b, history, "gram_accum") == nullptr);
        const auto lint = lint_step(b, { &trunk, &history });
        for (const std::string& problem : lint)
            std::fprintf(stderr, "MTP draft-head tap lint: %s\n", problem.c_str());
        CHECK(lint.empty());
    }
}

TEST(qwen35_refuses_a_calibration_range_that_mixes_trunk_and_mtp) {
    MetaKV mk = q35w8a8_meta();
    RadBuilder b;
    q35_split_head(b, true);
    RadBuildCtx ctx = spec_ctx(0, 1, 0);
    ctx.calib_dir = "/tmp/rad-qwen35-mtp-calib-test";
    ctx.calib_layer_lo = 7;
    ctx.calib_layer_hi = 9;  /* includes the last trunk layer and the virtual MTP layer */
    ctx.calib_tokens = 100;
    CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }), RAD_E_INVAL);
}

TEST(qwen35_u2_mtp_couples_each_round_after_the_global_shortlist) {
    MetaKV mk = q35w8a8_meta();
    for (int world : {1, 2}) for (int rank = 0; rank < world; ++rank) {
        RadBuilder b;
        q35_split_head(b, true);
        RadBuildCtx ctx = spec_ctx(rank, world, 3);
        ctx.draft_coupling = 1;
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(b.drafter.proposal_pitch, 8);
        const RecBuf* st = find_b(b, "mtp_couple_state");
        REQUIRE(st != nullptr);
        CHECK_EQ(st->d.kind, (int)RAD_BUF_PERSIST);
        CHECK_EQ(st->d.shape[0], 3);
        CHECK_EQ(st->d.shape[1], ctx.max_seqs);
        CHECK_EQ(st->d.shape[2], (int64_t)RAD_MTP_COUPLE_STATE_WORDS);
        const auto ver = kit(b, {4, 4}, {});
        const auto hist = history_pass(b, *ver);
        const auto r2 = round_pass(b, 2, 2), r3 = round_pass(b, 2, 3);
        const auto v = run_passes(b, qwen35_w8a8::step,
                                  {ver.get(), hist.get(), r2.get(), r3.get()}, rank, world);
        CHECK_PASSES(b, v);
        CHECK(issue_of(b, *v[0], "mtp_couple_propose") == nullptr);
        for (int rd = 1; rd <= 3; ++rd) {
            const RecIssue* top = issue_of(b, *v[rd], "row_topk");
            const RecIssue* prop = issue_of(b, *v[rd], "mtp_couple_propose");
            REQUIRE(top && prop);
            CHECK(top < prop);
            CHECK_EQ(pint(b.op_of(top->op), "R"), 8);
            CHECK_EQ(pint(b.op_of(prop->op), "round"), rd);
            CHECK_EQ(pint(b.op_of(prop->op), "n_vocab"), mk.m.n_vocab);
        }
        hist->b.n_draft_out = 0;  /* a KV-only history pass samples nothing */
        const auto empty = run_passes(b, qwen35_w8a8::step, {hist.get()}, rank, world);
        CHECK(issue_of(b, *empty[0], "mtp_couple_propose") == nullptr);
        const auto past = round_pass(b, 2, 4);
        const auto invalid = run_passes(b, qwen35_w8a8::step, {past.get()}, rank, world);
        CHECK(!invalid[0]->step_fail.empty());
    }
}

TEST(qwen35_u2_mtp_rejects_unservable_coupling) {
    MetaKV mk = q35w8a8_meta();
    auto check = [&](bool head, bool kernels, int depth) {
        RadBuilder b;
        q35_split_head(b, head);
        if (!kernels) b.undeclared.insert("mtp_couple_propose");
        RadBuildCtx ctx = spec_ctx(0, 1, depth);
        ctx.draft_coupling = 1;
        return stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); });
    };
    CHECK_EQ(check(false, true, 3), RAD_E_UNSUPPORTED);
    CHECK_EQ(check(true, false, 3), RAD_E_UNSUPPORTED);
    CHECK_EQ(check(true, true, RAD_MTP_COUPLE_MAX_ROUNDS + 1), RAD_E_UNSUPPORTED);
}

TEST(qwen35_split_target_loads_only_a_usable_requested_drafter) {
    MetaKV mk = q35w8a8_meta();
    for (int spec : {0, 3}) {
        RadBuilder b;
        q35_split_head(b, spec == 0);  /* real head with spec off, old container with spec on */
        RadBuildCtx ctx = spec_ctx(0, 1, spec);
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK(b.drafter_name.empty());
        CHECK(find_w(b, "mtp.fc.weight") == nullptr);
        CHECK(find_b(b, "mtp_dlog") == nullptr);
        CHECK_EQ(count_op(b, "logits_gemm_vocab"), 1);
    }
    RadBuilder bad;
    q35_split_head(bad, true, 64);
    RadBuildCtx ctx = spec_ctx(0, 1, 3);
    CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&bad, mk.meta(), &ctx); }), RAD_E_DTYPE);

    MetaKV df = with_dflash2(q35w8a8_meta());
    RadBuilder b;
    q35_split_head(b, true);
    ctx.max_spec = 7;
    REQUIRE_EQ(qwen35_w8a8::declare(&b, df.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.drafter_name, std::string("dflash2"));
    CHECK(find_w(b, "mtp.fc.weight") == nullptr);
}

/* ==================================================================== llama_fp8 and DSpark */

namespace {

/* MiniCPM5-2B's graph cut to four layers at a quarter of its width, with the DSpark drafter
 * MiniCPM5-2B-DSpark ships merged under `draft.`: two layers reading trunk layers 0, 2 and 3, a
 * block of seven, the rank-64 bigram head. */
MetaKV llama_meta(const char* quant = "fp8_e4m3") {
    RadModelMeta m{};
    m.arch_id = "llama"; m.name = "test-minicpm5"; m.quant = quant;
    m.n_layers = 4; m.n_embd = 1024; m.n_head = 8; m.n_head_kv = 2; m.head_dim = 128;
    m.n_ff = 2048; m.n_vocab = 32000; m.n_ctx_train = 32768;
    m.rms_eps = 1e-5f; m.rope_theta = 5e6f; m.rope_scale = 1.0f;
    return MetaKV(m);
}

MetaKV with_dspark(MetaKV k) {
    k.set("draft.architectures", "DSparkDraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "8")
     .set("draft.num_key_value_heads", "2")
     .set("draft.intermediate_size", "2048")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "1000000")
     .set("draft.vocab_size", "32000")
     .set("draft.target_layer_ids", "0 2 3")
     .set("draft.block_size", "7")
     .set("draft.mask_token_id", "31999")
     .set("draft.markov_rank", "64")
     .set("draft.markov_head_type", "vanilla");
    return k;
}

RadBuildCtx dspark_ctx(int rank, int world) {
    RadBuildCtx c = spec_ctx(rank, world, 7);
    c.max_seqs = 16;     /* room for a query pass past 64 rows */
    return c;
}

/* Moves the declarations of `moved`, in their order, to just before `before`, as a plugin that
 * declared them there would have, and renumbers the recorded issues to match. */
void redeclare_before(RadBuilder& b, const std::vector<std::unique_ptr<RadCtx>>& ctxs,
                      const std::vector<rad_op>& moved, rad_op before) {
    std::set<rad_op> mv(moved.begin(), moved.end());
    std::vector<rad_op> order;
    for (rad_op h = 1; h <= (rad_op)b.ops.size(); ++h) {
        if (mv.count(h)) continue;
        if (h == before) order.insert(order.end(), moved.begin(), moved.end());
        order.push_back(h);
    }
    std::vector<rad_op> to(b.ops.size() + 1, 0);
    std::vector<RecOp> ops;
    for (size_t i = 0; i < order.size(); ++i) {
        to[order[i]] = (rad_op)(i + 1);
        ops.push_back(b.ops[order[i] - 1]);
    }
    b.ops = ops;
    for (const auto& c : ctxs)
        for (RecIssue& is : c->issues) is.op = to[is.op];
}

}  /* namespace */

/* The plain LLaMA graph's engine steps lint clean at one and two ranks, with the DSpark drafter's
 * context and query passes behind a verify. Ten sequences at a block of seven is 70 query rows,
 * past the 64 the fused gate_up GEMM serves, so the unfused feed-forward path runs too. */
TEST(llama_fp8_steps_lint_clean_with_a_dspark_drafter) {
    MetaKV mk = with_dspark(llama_meta());
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = dspark_ctx(rank, world);
            REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
            CHECK_EQ(b.drafter.depth, 7);
            CHECK_EQ(b.drafter.block, 7);       /* every row predicts, the anchor included */
            CHECK_EQ(b.drafter.mask_token, 31999);
            const int64_t gated = llama_fp8::g_model[rank].ds.gated_rows;
            for (const auto& k : { kit(b, { 1, 1, 1 }, {}), kit(b, {}, { 40 }),
                                   kit(b, { 8, 1 }, { 5 }) }) {
                const auto v = run_passes(b, llama_fp8::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            for (int64_t n_seq : { 4, 10 }) {
                const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
                const auto ctxp = context_pass(b, *ver);
                const auto qry = query_pass(b, n_seq, 7, 7);
                const auto v = run_passes(b, llama_fp8::step, { ver.get(), ctxp.get(), qry.get() },
                                          rank, world);
                CHECK_PASSES(b, v);
                /* The feed-forward of the query pass takes the fold up to its band, the pair past
                 * it. */
                const bool fold = gated > 0 && n_seq * 7 <= gated;
                const auto ops = issued_ops(b, *v[2]);
                CHECK_EQ(count_issued(ops, "silu_mul") > 0, !fold);
            }
        }
}

/* Declaration order is liveness, and DSpark's norms are declared where its step issues them. Moved
 * back to where they once were, all ahead of the layers, the residual stream's planned life ends
 * before the first layer's temporaries begin; past the fused GEMM's 64 rows the unfused
 * feed-forward writes those temporaries, and the lint reports the stream being read after them. */
TEST(the_step_lint_catches_dspark_norms_declared_ahead_of_their_layers) {
    MetaKV mk = with_dspark(llama_meta());
    for (int64_t n_seq : { 4, 10 }) {
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        const rad::arch::Dspark& ds = llama_fp8::g_model[0].ds;
        const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
        const auto ctxp = context_pass(b, *ver);
        const auto qry = query_pass(b, n_seq, 7, 7);
        const auto v = run_passes(b, llama_fp8::step, { ver.get(), ctxp.get(), qry.get() });
        CHECK_PASSES(b, v);

        std::vector<rad_op> norms;
        for (int64_t l = 0; l < ds.g.n_layer; ++l) {
            norms.push_back(ds.op_anorm[l]);
            norms.push_back(ds.op_fnorm[l]);
        }
        norms.push_back(ds.op_outnorm);
        redeclare_before(b, v, norms, ds.op_qat);
        const auto f = lint_step(b, view(v));
        bool stream = false;
        for (const auto& s : f)
            stream |= has(s, "reads 'ds_x'") && (has(s, "'ds_gate_up'") || has(s, "'ds_ffn'"));
        if (n_seq * 7 > ds.gated_rows) CHECK(stream);
        else                           CHECK(!stream);
    }
}

/* A drafter's taps name trunk layers. Out-of-range ids are dropped, not clamped (the drafter
 * belongs to another target), parsing stops at the first non-number, and a drafter naming no
 * layer of this model is refused. So are a trained block other than the serve's depth, a Markov
 * rank the walk cannot apply, and a head type other than the bigram. */
TEST(llama_reads_its_drafters_taps_and_refuses_what_dspark_does_not_implement) {
    {
        MetaKV mk = with_dspark(llama_meta());
        mk.set("draft.target_layer_ids", "[1, 99, 3, -2]");
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(llama_fp8::g_model[0].n_taps, 2);
        CHECK_EQ(llama_fp8::g_model[0].tap_layer[0], 1);
        CHECK_EQ(llama_fp8::g_model[0].tap_layer[1], 3);
        CHECK_EQ(find_b(b, "draft_src")->d.shape[1], 2 * 1024);
    }
    auto refused = [](MetaKV mk, int max_spec = 7) {
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        ctx.max_spec = max_spec;
        return stderr_of_rc([&] { return llama_fp8::declare(&b, mk.meta(), &ctx); });
    };
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.target_layer_ids", "7 9")),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()), 6), RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.markov_rank", "48")), RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.markov_head_type", "mlp")),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).drop("draft.block_size")), RAD_E_FORMAT);
    CHECK_EQ(refused(with_dspark(llama_meta()).drop("draft.mask_token_id")), RAD_E_FORMAT);
    /* Without speculation the drafter is ignored and the trunk declares alone. */
    RadBuilder b;
    MetaKV mk = with_dspark(llama_meta());
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
    CHECK(find_b(b, "draft_src") == nullptr);
    /* The probe reports the trained block, fixed. */
    RadArchProbe p{};
    CHECK_OK(llama_fp8::probe(mk.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
    CHECK_EQ(p.chain_pays, 0);
}

/* One graph, two descriptors: llama_dense_fp8 serves the dense checkpoint a recipe quantises
 * (descriptor ""), llama_fp8 the fp8 one, each refusing the other's, and both declare the same
 * ops and weights. */
TEST(llama_dense_fp8_declares_the_fp8_graph_for_the_dense_descriptor) {
    MetaKV fp8 = with_dspark(llama_meta());
    MetaKV dense = with_dspark(llama_meta(""));
    RadBuildCtx ctx = dspark_ctx(0, 1);
    RadBuilder a, d;
    REQUIRE_EQ(llama_fp8::declare(&a, fp8.meta(), &ctx), RAD_OK);
    REQUIRE_EQ(llama_dense_fp8::declare(&d, dense.meta(), &ctx), RAD_OK);
    CHECK_EQ(join(op_names(a)), join(op_names(d)));
    REQUIRE_EQ(a.weights.size(), d.weights.size());
    for (size_t i = 0; i < a.weights.size(); ++i) {
        CHECK_EQ(a.weights[i].name, d.weights[i].name);
        CHECK_EQ(a.weights[i].d.dtype, d.weights[i].d.dtype);
    }
    RadBuilder x, y;
    CHECK_EQ(stderr_of_rc([&] { return llama_fp8::declare(&x, dense.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(stderr_of_rc([&] { return llama_dense_fp8::declare(&y, fp8.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    /* Its steps are llama_fp8's. A plugin keeps one model per rank, so declare again after the
     * refusal above replaced it. */
    d = RadBuilder{};
    REQUIRE_EQ(llama_dense_fp8::declare(&d, dense.meta(), &ctx), RAD_OK);
    const auto ver = kit(d, std::vector<int>(4, 8), {});
    const auto ctxp = context_pass(d, *ver);
    const auto qry = query_pass(d, 4, 7, 7);
    const auto v = run_passes(d, llama_dense_fp8::step, { ver.get(), ctxp.get(), qry.get() });
    CHECK_PASSES(d, v);
}

namespace {

/* IQ8R as libquant states it: int8 codes rotated by fwht128, a u8 scale per [1 x 256] of K under
 * a bf16 scale a row. */
RadEncoding iq8r_enc() {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_U8, 1, 256);
    rad_enc_add_plane(&e, "scale.1", RAD_BF16, 1, 0);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

/* data/recipes/minicpm5-2b-iq8r.recipe's encodings by declared name: every trunk and drafter
 * linear IQ8R, an unrotated int8 head and the drafter's 2-bit copy of it. The head comes last
 * because `output.weight` is in `attn_output.weight`. */
void llama_iq8r_encs(RadBuilder* b) {
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    b->encs = { { "dspark.draft_head", u2 } };
    for (const char* t : { "dspark.fc.weight", "attn_qkv.weight", "attn_q.weight",
                           "attn_k.weight", "attn_v.weight", "attn_output.weight",
                           "ffn_gate_up.weight", "ffn_down.weight" })
        b->encs.push_back({ t, iq8r_enc() });
    b->encs.push_back({ "output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
}

}  /* namespace */

namespace {

/* data/recipes/q36-35b-a3b-iq8r.recipe's encodings by declared name: the trunk's linears and the
 * shared expert's gate/up IQ8R, the routed experts and the shared expert's down projection
 * rotated int8 at a bf16 scale a (row, 128 of K), an unrotated int8 head and the MTP head's 2-bit
 * copy of it. The head comes last because `output.weight` is in
 * `attn_output.weight`. */
void moe_iq8r_encs(RadBuilder* b) {
    RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(i8.transform, "fwht128");
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    b->encs = { { "mtp.draft_head", u2 } };
    for (const char* t : { "ffn_gate_up_exps", "ffn_down_exps", "ffn_down_shexp" })
        b->encs.push_back({ t, i8 });
    for (const char* t : { "attn_qg.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up_shexp" })
        b->encs.push_back({ t, iq8r_enc() });
    b->encs.push_back({ "output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
}

/* q35moe_meta as a container converted with that recipe states its format. */
MetaKV q35moe_w8a8_meta() {
    MetaKV k = q35moe_meta();
    k.set("quantization_config.quant_method", "w8a8").set("quantization_config.fmt", "fwht128");
    return k;
}

int moe_w8a8_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    moe_iq8r_encs(b);
    return qwen35moe_w8a8::declare(b, meta, ctx);
}

}  /* namespace */

/* The routed build with an int8 trunk, where libr4d serves the rotating quantisers: every grouped
 * GEMM is the int8 row (i8a8h), gate_up reading the block input's own rotated codes at IQ8R's 256
 * (no second quantiser and no rotated copy of the input) and down the gated product's at a 128;
 * the gated product is gated_had_quant_i8g for the routed arm and the shared one; and no E4M3
 * quantiser, and no E4M3 plane of the routed or the shared product, is left. One rank and two. */
TEST(qwen35moe_w8a8_reads_the_trunks_int8_codes_in_its_routed_experts) {
    MetaKV mk = q35moe_w8a8_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        moe_iq8r_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35moe_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        const rad_buf hq = (rad_buf)buf_named(b, "h.q8");
        const rad_buf fq = (rad_buf)buf_named(b, "moe_ffn.q8");
        REQUIRE(hq != 0);
        REQUIRE(fq != 0);
        int gu = 0, dn = 0;
        for (const auto& o : b.ops) {
            if (o.op != "moe_gemm_q") continue;
            CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
            const bool token = pstr(&o, "a_order") == "token";
            CHECK_EQ(pint(&o, "a_group"), token ? 256 : 128);
            CHECK(in(o.reads, token ? hq : fq));
            (token ? gu : dn) += 1;
        }
        CHECK_EQ(gu, 8);
        CHECK_EQ(dn, 8);
        /* The routed arm's and the shared arm's products, a layer each. */
        CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 8 + 8);
        for (const char* op : { "gated_quant_fp8", "gated_had_quant_fp8", "had_quant_act_fp8",
                                "quant_act_fp8", "rmsnorm_quant_fp8" })
            CHECK_EQ(count_op(b, op), 0);
        for (const char* a : { "moe_ffn.q", "moe_hr.q", "moe_hr.q8", "shexp_ffn.q" })
            CHECK(find_b(b, a) == nullptr);
    }
}

/* Where the rotating norm is not served (the CPU kernels), the routed experts read a rotated int8
 * copy of the block input that had_quant_act_i8g writes a layer, at a scale a 128. */
TEST(qwen35moe_w8a8_quantises_the_expert_input_itself_without_the_rotating_norm) {
    MetaKV mk = q35moe_w8a8_meta();
    RadBuilder b;
    for (const char* op : { "rmsnorm_had_quant_i8g", "ar_rmsnorm_had_quant_i8g" })
        b.undeclared.insert(op);
    moe_iq8r_encs(&b);
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35moe_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
    const rad_buf hr = (rad_buf)buf_named(b, "moe_hr.q8");
    REQUIRE(hr != 0);
    int reads_hr = 0, writes_hr = 0;
    for (const auto& o : b.ops) {
        if (o.op == "moe_gemm_q" && pstr(&o, "a_order") == "token") {
            CHECK_EQ(pint(&o, "a_group"), 128);
            reads_hr += in(o.reads, hr);
        }
        writes_hr += o.op == "had_quant_act_i8g" && in(o.writes, hr);
    }
    CHECK_EQ(reads_hr, 8);
    CHECK_EQ(writes_hr, 8);
}

/* What the build refuses: a container that does not state the int8 trunk, block-fp8 experts under
 * one (they read E4M3 activations), and the dense checkpoint. */
TEST(qwen35moe_w8a8_refuses_another_format_by_name) {
    {
        MetaKV mk = q35moe_meta();
        RadBuilder b;
        moe_iq8r_encs(&b);
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return qwen35moe_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = q35moe_w8a8_meta();
        RadBuilder b;
        moe_iq8r_encs(&b);
        const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
        for (auto& e : b.encs)
            if (e.first == "ffn_gate_up_exps" || e.first == "ffn_down_exps") e.second = fp8;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return qwen35moe_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* The routed int8 model's engine steps, its routed MTP head included, lint clean at one and two
 * ranks. */
TEST(qwen35moe_w8a8_steps_lint_clean_with_its_routed_mtp_head) {
    MetaKV mk = q35moe_w8a8_meta();
    lint_qwen35_fp8_family(moe_w8a8_declare, qwen35moe_w8a8::step, mk.meta(), 3, false, 0);
}

/* An IQ8R container through the dense descriptor: every linear of the trunk and of the DSpark
 * drafter is an i8a8h GEMM at one scale a 256 of K (the drafter's 2-bit head is the one other
 * gemm_nt_q), its int8 codes written by the rotating norms, the gated product and
 * had_quant_act_i8g at that group. No op that writes or reads E4M3 codes is declared and no
 * activation plane is E4M3. The engine's steps lint clean at one and two ranks, the drafter's
 * query pass past its 64 rows included. */
TEST(llama_dense_fp8_serves_an_iq8r_trunk_and_drafter_on_int8_activations) {
    MetaKV mk = with_dspark(llama_meta(""));
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            llama_iq8r_encs(&b);
            RadBuildCtx ctx = dspark_ctx(rank, world);
            REQUIRE_EQ(llama_dense_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
            int rot = 0, other = 0;
            for (const RecOp& o : b.ops) {
                if (o.op != "gemm_nt_q") continue;
                const bool h = pstr(&o, "dtype") == "i8a8h";
                rot += h;
                other += !h;
                if (h) CHECK_EQ(pint(&o, "group"), 256);
            }
            /* Four trunk layers of q|k|v, o, gate|up and down; the drafter's fc, and its two
             * layers' k and v twice (the context pass and the query pass), q, o, gate|up, down. */
            CHECK_EQ(rot, 4 * 4 + 1 + 2 * 8);
            CHECK_EQ(other, 1);
            for (const char* op : { "quant_act_fp8", "rmsnorm_quant_fp8", "ar_rmsnorm_quant_fp8",
                                    "gated_quant_fp8", "gemm_nt_q_gated", "quant_act_i8g" })
                CHECK_EQ(count_op(b, op), 0);
            CHECK(count_op(b, "rmsnorm_had_quant_i8g") > 0);
            CHECK(count_op(b, "gated_had_quant_i8g") > 0);
            for (const auto& buf : b.bufs) CHECK(buf.d.dtype != (uint32_t)RAD_F8E4M3);
            for (const auto& k : { kit(b, { 1, 1, 1 }, {}), kit(b, {}, { 40 }),
                                   kit(b, { 8, 1 }, { 5 }) }) {
                const auto v = run_passes(b, llama_dense_fp8::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            for (int64_t n_seq : { 1, 2, 4, 10 }) {
                const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
                const auto ctxp = context_pass(b, *ver);
                const auto qry = query_pass(b, n_seq, 7, 7);
                const auto v = run_passes(b, llama_dense_fp8::step,
                                          { ver.get(), ctxp.get(), qry.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
        }
}

/* --kv-cache-dtype i8 on the LLaMA graph at MiniCPM5-2B's ratio, 8 query heads a kv head at
 * head_dim 128: the trunk's cache and the DSpark drafter's are int8 groups at head_dim 128, and
 * every op that names a cache dtype bands on i8. */
TEST(llama_i8_cache_is_an_int8_group_at_the_models_head_dim) {
    MetaKV mk = with_dspark(llama_meta(""));
    mk.m.n_head = 16;
    mk.set("draft.num_attention_heads", "16");

    RadBuilder b;
    llama_iq8r_encs(&b);
    RadBuildCtx ctx = dspark_ctx(0, 1);
    ctx.kv_dtype = RAD_I8;
    REQUIRE_EQ(llama_dense_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.kv_groups[0].d.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(b.kv_groups[0].d.head_dim, 128);
    CHECK_EQ(b.kv_groups.back().d.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(b.kv_groups.back().d.head_dim, 128);
    int i8 = 0, other = 0;
    for (const auto& o : b.ops) {
        if (!param(&o, "kv_dtype")) continue;
        const std::string kd = pstr(&o, "kv_dtype");
        i8 += kd == "i8";
        other += kd != "i8";
    }
    CHECK(i8 > 0);
    CHECK_EQ(other, 0);
    for (int64_t n_seq : { 4, 10 }) {
        const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
        const auto ctxp = context_pass(b, *ver);
        const auto qry = query_pass(b, n_seq, 7, 7);
        const auto v = run_passes(b, llama_dense_fp8::step, { ver.get(), ctxp.get(), qry.get() });
        CHECK_PASSES(b, v);
    }
}

/* ==================================================================== qwen4exp_fp8 stepped */

namespace {

RadBuildCtx q4_ctx(int rank, int world, int64_t max_seqs = 8) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_tok = 2048; c.max_seqs = max_seqs; c.max_spec = 3; c.max_ctx = 200000;
    return c;
}

/* The 2-bit draft head's encoding (u2 codes, an f16 scale and a u8 zero per 128), as the recipe
 * writes `mtp.draft_head.weight`. */
RadEncoding u2_head() {
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    return u2;
}

/* The Flash-Next fixture with 32 experts instead of 512: the step and its lint do not depend on
 * the expert count, and declaring 512 experts on every rank is most of a test's time. */
MetaKV q4_small() {
    MetaKV k(qwen4exp_meta());
    k.set("num_experts", "32");
    return k;
}

/* The issues of op `name` in a pass. */
std::vector<const RecIssue*> issues_of(const RadBuilder& b, const RadCtx& c, const char* name) {
    std::vector<const RecIssue*> v;
    for (const auto& i : c.issues)
        if (const RecOp* o = b.op_of(i.op); o && o->op == name) v.push_back(&i);
    return v;
}

}  /* namespace */

/* Flash-Next's engine steps lint clean at one, two and four ranks: decode, a prefill chunk after
 * a long history (the sparse attention), a mixed step, and a verify followed by the MTP head's
 * history pass and rounds, the 2-bit draft head and its vocab-parallel gather included. */
TEST(flashnext_steps_lint_clean_at_one_two_and_four_ranks) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.encs = { { "draft_head", u2_head() } };
            RadBuildCtx ctx = q4_ctx(rank, world);
            REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
            REQUIRE_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
            for (const auto& k : { kit(b, { 1, 1, 1, 1 }, {}), kit(b, {}, { 64, 64 }, 9000),
                                   kit(b, { 1, 4 }, { 6 }, 300) }) {
                const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            const auto his = history_pass(b, *ver);
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, qwen4exp_fp8::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, rank, world);
            CHECK_PASSES(b, v);
        }
}

/* --draft-suffix: the verify window is widened to 7 drafts (`max_spec` 7: the delta-net fold serves 8
 * query rows) while the head still drafts 3 rounds. A suffix step is an eight-row verify, then the
 * head's history pass over all eight rows (round 1 on the last) and rounds 2 and 3; the head's own
 * step in the same deployment is four rows and the same passes. Both lint clean, and every pass issues
 * the same ops by name, the same number of times, as the head's step at its own window does: only row
 * counts differ, and round 3 of a deployment whose head declared more rounds than it runs carries one
 * `cast` more (its hidden state on to a round 4 the core does not issue; a few microseconds). */
TEST(flashnext_a_suffix_step_issues_the_heads_ops_over_eight_rows_and_lints_clean) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    std::vector<std::map<std::string, size_t>> pattern[3];
    int idx = 0;
    for (const auto& cfg : { std::pair<int, int>{ 3, 4 }, std::pair<int, int>{ 7, 8 }, std::pair<int, int>{ 7, 4 } }) {
        RadBuilder b;
        b.encs = { { "draft_head", u2_head() } };
        RadBuildCtx ctx = q4_ctx(0, 1);
        ctx.max_spec = cfg.first;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
        REQUIRE_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
        const auto ver = kit(b, std::vector<int>(1, cfg.second), {});
        CHECK_EQ((int)ver->b.n_spec, cfg.second - 1);
        const auto his = history_pass(b, *ver);
        CHECK_EQ((int)his->b.n_tok, cfg.second);              /* every verified position */
        CHECK_EQ(his->dout[0], cfg.second - 1);               /* round 1 on the last row */
        const auto r2 = round_pass(b, 1, 2), r3 = round_pass(b, 1, 3);
        const auto v = run_passes(b, qwen4exp_fp8::step, { ver.get(), his.get(), r2.get(), r3.get() });
        CHECK_PASSES(b, v);
        std::set<std::string> names;
        for (const RecOp& o : b.ops) names.insert(o.op);
        for (size_t pass = 0; pass < v.size(); ++pass) {
            std::map<std::string, size_t> m;
            for (const std::string& nm : names) m[nm] = issues_of(b, *v[pass], nm.c_str()).size();
            pattern[idx].push_back(m);
        }
        ++idx;
    }
    for (size_t pass = 0; pass < 4; ++pass)
        for (int o = 1; o < 3; ++o) {
            std::map<std::string, std::pair<size_t, size_t>> d;
            for (const auto& [nm, n] : pattern[0][pass])
                if (pattern[o][pass].at(nm) != n) d[nm] = { n, pattern[o][pass].at(nm) };
            if (pass == 3) {
                CHECK_EQ(d.size(), (size_t)1);
                CHECK_EQ(d.at("cast").first, (size_t)0);
                CHECK_EQ(d.at("cast").second, (size_t)1);
            } else {
                CHECK_EQ(d.size(), (size_t)0);
            }
        }
}

/* Sparse attention issues one row per query, chunked by the split-KV scratch it declared, never
 * by `max_seqs`: with four sequences allowed, a 1024-token chunk after a long history is one
 * launch of the gated form, or four of 256 rows where only the plain form resolves. Rows of four
 * would be 256 launches a layer. */
TEST(flashnext_sparse_attention_chunks_by_its_scratch_not_by_max_seqs) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int gated = 0; gated < 2; ++gated) {
        RadBuilder b;
        if (!gated) b.refuse.insert("attn_paged_gate_quant");
        RadBuildCtx ctx = q4_ctx(0, 1, 4);
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
        const auto k = kit(b, {}, { 1024 }, 9000);
        const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
        CHECK_PASSES(b, v);
        const auto at = issues_of(b, *v[0], gated ? "attn_paged_gate_quant" : "attn_paged");
        /* Two full-attention layers. */
        REQUIRE_EQ(at.size(), (size_t)(gated ? 2 : 2 * 4));
        for (const RecIssue* i : at) {
            CHECK_EQ(i->n, 1);                               /* q_len 1: a row is a query */
            CHECK_EQ(i->opd[0].rows, gated ? 1024 : 256);    /* queries in this launch */
            CHECK_EQ(i->opd[2].rows, i->opd[0].rows);        /* one selection row a query */
        }
    }
}

/* The indexer's unfinished block is state, not scratch: each QSA layer stores it in the linear
 * `kv_qsa_tail` group at its own layer, so two layers never share one tail and a slot survives
 * across steps. */
TEST(flashnext_qsa_tail_is_per_layer_state) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    rad_kvgroup tail = 0;
    for (size_t g = 0; g < b.kv_groups.size(); ++g)
        if (b.kv_groups[g].name == "kv_qsa_tail") tail = (rad_kvgroup)(g + 1);
    REQUIRE(tail != 0);
    CHECK_EQ((int)b.kv_groups[tail - 1].d.kind, (int)RAD_KV_LINEAR);
    const auto k = kit(b, { 1, 1 }, {});
    const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
    CHECK_PASSES(b, v);
    std::set<int64_t> layers;
    for (const RecIssue* i : issues_of(b, *v[0], "qsa_tail_store"))
        for (const RadOperand& o : i->opd)
            if (o.kind == RAD_OPK_KV) {
                CHECK_EQ(o.handle, tail);
                layers.insert(o.offset);
            }
    CHECK(layers == (std::set<int64_t>{ 3, 7 }));
}

/* The rope table has a row for every position a step can name: a verify and the draft rounds run
 * up to max_spec past max_ctx, so a table ending at max_ctx would rotate them by unwritten rows.
 * Every draft pass fills its own rows before anything reads them, since a round's positions are
 * ones the trunk step never wrote. */
TEST(flashnext_rope_table_reaches_past_the_context_and_draft_passes_fill_it_first) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    const RecBuf* cs = find_b(b, "rope_cs");
    REQUIRE(cs != nullptr);
    CHECK(cs->d.shape[0] >= ctx.max_ctx + ctx.max_spec);
    const rad_buf h = (rad_buf)buf_named(b, "rope_cs");
    const auto ver = kit(b, std::vector<int>(4, 4), {});
    const auto his = history_pass(b, *ver);
    const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
    const auto v = run_passes(b, qwen4exp_fp8::step, { ver.get(), his.get(), r2.get(), r3.get() });
    CHECK_PASSES(b, v);
    for (size_t p = 1; p < v.size(); ++p) {
        int first_write = -1, first_read = -1;
        for (size_t i = 0; i < v[p]->issues.size(); ++i) {
            const RecIssue& is = v[p]->issues[i];
            const RecOp* o = b.op_of(is.op);
            bool names = false;
            for (const RadOperand& od : is.opd) names |= od.kind == RAD_OPK_BUF && od.handle == h;
            if (!names) continue;
            if (in(o->writes, h) && first_write < 0) first_write = (int)i;
            if (in(o->reads, h) && first_read < 0) first_read = (int)i;
        }
        CHECK(first_write >= 0);
        CHECK(first_read < 0 || first_write < first_read);
        CHECK_EQ(b.op_of(v[p]->issues[0].op)->op, std::string("rope_table"));
    }
}

/* A draft pass refills the rope table before the head's first op, while the carry `mtp_hin` holds
 * the rows the previous pass left, so the table's planned life must reach the carry's or the
 * planner may give the two one set of bytes. Without qk_norm_rope_gate (the fused attention
 * prologue and the indexer's) no head op reads the table, so only the head's own declaration of
 * the refill carries its life there. At one rank and two, with the fused prologue and without, the
 * table is live at the carry's first op and a verify, the history pass and two rounds lint clean. */
TEST(flashnext_draft_passes_refill_the_rope_table_where_the_carry_is_live) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int fused = 0; fused < 2; ++fused)
        for (int world : { 1, 2 }) {
            RadBuilder b;
            if (!fused) b.undeclared.insert("qk_norm_rope_gate");
            RadBuildCtx ctx = q4_ctx(0, world);
            REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
            CHECK_EQ(count_op(b, "qk_norm_rope_gate") > 0, fused == 1);
            const std::vector<Live> L = liveness(b);
            const rad_buf cs = (rad_buf)buf_named(b, "rope_cs");
            const rad_buf hin = (rad_buf)buf_named(b, "mtp_hin");
            REQUIRE(cs && hin);
            CHECK(L[cs].planned && L[hin].planned);
            CHECK(L[cs].lo <= L[hin].lo && L[hin].lo <= L[cs].hi);
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            const auto his = history_pass(b, *ver);
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, qwen4exp_fp8::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, 0, world);
            CHECK_PASSES(b, v);
        }
}

/* The vocab-parallel draft head gathers each rank's top-R as (id, value) pairs: a row is 2R
 * floats, whole 16-byte words, and the gather moves every draft row's. A row of one pair (8 bytes)
 * is what the two-rank gather cannot take. */
/* --draft-depth adaptive (core/sched/draft_depth.h): a deployment declared for the deepest chain D
 * verifies, step by step, the 1 .. D drafts the chain of that step really made. The plugin learns the
 * depth from the batch's rows alone; a verify of d drafts under the deep declaration issues exactly
 * the ops, in the order, of a chain deployment declared at depth d, and so does the history pass
 * behind it (whose rows mirror the verify's). Sequences of one step may differ in depth. */
TEST(flashnext_a_shorter_verify_under_a_deeper_coupled_chain_issues_that_chains_ops) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    const int D = 5;
    auto seq = [](const RadBuilder& b, const RadCtx& c) {
        std::vector<std::string> v;
        for (const auto& i : c.issues) if (const RecOp* o = b.op_of(i.op)) v.push_back(o->op);
        return v;
    };
    /* One plugin instance keeps its handles per slot: each deployment is declared right before its
     * steps run. */
    auto run = [&](int declared, const std::vector<int>& rows, std::vector<std::string>* ver,
                   std::vector<std::string>* his) {
        RadBuilder b;
        b.encs = { { "draft_head", u2_head() } };
        RadBuildCtx c = q4_ctx(0, 1, 4);
        c.max_spec = declared;
        c.draft_coupling = 1;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
        CHECK_EQ(b.drafter.depth, declared);
        const auto k = kit(b, rows, {});
        const auto h = history_pass(b, *k);
        const auto v = run_passes(b, qwen4exp_fp8::step, { k.get(), h.get() });
        CHECK_PASSES(b, v);
        *ver = seq(b, *v[0]);
        *his = seq(b, *v[1]);
    };
    for (int d = 1; d <= D; ++d) {
        const std::vector<int> rows((size_t)4, d + 1);
        std::vector<std::string> fv, fh, av, ah;
        run(d, rows, &fv, &fh);
        /* A head of one round ends in its history pass; with more rounds that pass also hands its
         * output to round 2 (one cast), whatever the verify behind it carried. The history pass of a
         * one-draft verify is therefore compared under a two-round declaration. */
        if (d == 1) {
            std::vector<std::string> v2;
            run(2, rows, &v2, &fh);
            CHECK(v2 == fv);
        }
        run(D, rows, &av, &ah);
        CHECK(!fv.empty() && !fh.empty());
        CHECK(av == fv);
        CHECK(ah == fh);
        if (av != fv || ah != fh) std::fprintf(stderr, "    depth %d under %d: issue pattern differs\n", d, D);
    }
    /* depths that differ between the sequences of one step: the ops of the full-depth step. The
     * rows add up past kPrefetchRows, as the full-depth step's do: below it the model issues a
     * step by its row count (the decode-height issue order), and that is not this property. */
    std::vector<std::string> mv, mh, dv, dh;
    run(D, { 2, D + 1, D + 1, D + 1 }, &mv, &mh);
    run(D, { D + 1, D + 1, D + 1, D + 1 }, &dv, &dh);
    CHECK(mv == dv);
    CHECK(mh == dh);
}

TEST(flashnext_draft_head_gathers_whole_candidate_rows_at_two_ranks) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    b.encs = { { "draft_head", u2_head() } };
    RadBuildCtx ctx = q4_ctx(0, 2);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    const RecOp* topk = nullptr;
    for (const auto& o : b.ops)
        if (o.op == "row_topk" && param(&o, "vocab_off")) topk = &o;
    const RecOp* ag = find_op(b, "all_gather");
    REQUIRE(topk != nullptr && ag != nullptr);
    const long long R = pint(topk, "R");
    CHECK(R >= 2);
    CHECK_EQ(pint(ag, "row"), 2 * R);
    CHECK_EQ(pint(ag, "row") * 4 % 16, 0);
    CHECK_EQ(pint(ag, "world_size"), 2);
    const long long rows = ctx.max_seqs * (1 + ctx.max_spec);
    CHECK_EQ(pint(ag, "numel"), rows * 2 * R);
    const RecBuf* pair = find_b(b, "mtp_dpair");
    const RecBuf* gath = find_b(b, "mtp_dgath");
    REQUIRE(pair != nullptr && gath != nullptr);
    CHECK_EQ(pair->d.shape[1] * pair->d.shape[2], 2 * R);
    CHECK_EQ(gath->d.shape[1] * gath->d.shape[2], 2 * 2 * R);
}

/* ==================================================================== embgemma2 */

namespace {

/* EmbeddingGemma 2 cut to six layers (five sliding, one whole-input, its 5:1), with its vision
 * tower and audio encoder cut to two layers each. */
MetaKV eg2_meta() {
    RadModelMeta m{};
    m.arch_id = "embeddinggemma2"; m.name = "test-embgemma2"; m.quant = "";
    m.n_layers = 6; m.n_embd = 512; m.n_head = 4; m.n_head_kv = 2; m.head_dim = 256;
    m.n_ff = 2048; m.n_vocab = 262144; m.n_ctx_train = 2048;
    m.rms_eps = 1e-6f; m.rope_theta = 1e4f; m.rope_scale = 1.0f;
    MetaKV k(m);
    k.set("layer_types", "sliding_attention sliding_attention sliding_attention sliding_attention "
                         "sliding_attention full_attention")
     .set("rope_parameters.sliding_attention.rope_theta", "10000")
     .set("rope_parameters.full_attention.rope_theta", "1000000")
     .set("sliding_window", "512")
     .set("per_layer_config.05.head_dim", "512")
     .set("per_layer_config.05.num_key_value_heads", "1")
     .set("hidden_size_per_layer_input", "512")
     .set("embedding_dim", "768")
     .set("vision_config.model_type", "gemma4_vision")
     .set("vision_config.hidden_size", "768")
     .set("vision_config.num_attention_heads", "12")
     .set("vision_config.head_dim", "64")
     .set("vision_config.intermediate_size", "3072")
     .set("vision_config.pooling_kernel_size", "3")
     .set("vision_config.position_embedding_size", "10240")
     .set("vision_config.num_hidden_layers", "2")
     .set("vision_config.patch_size", "16")
     .set("vision_config.rope_parameters.rope_theta", "100")
     .set("vision_config.rope_parameters.rope_type", "default")
     .set("vision_config.hidden_activation", "gelu_pytorch_tanh")
     .set("audio_config.model_type", "gemma4_audio")
     .set("audio_config.hidden_size", "1024")
     .set("audio_config.num_attention_heads", "8")
     .set("audio_config.output_proj_dims", "1536")
     .set("audio_config.attention_chunk_size", "12")
     .set("audio_config.attention_context_left", "13")
     .set("audio_config.attention_context_right", "0")
     .set("audio_config.attention_logit_cap", "50")
     .set("audio_config.num_hidden_layers", "2")
     .set("audio_config.conv_kernel_size", "5")
     .set("audio_config.residual_weight", "0.5")
     .set("audio_config.hidden_act", "silu")
     .set("audio_config.subsampling_conv_channels", "128 32")
     .set("preprocessor.feature_size", "128");
    return k;
}

RadBuildCtx eg2_ctx() {
    RadBuildCtx c = build_ctx();
    c.max_enc_patches = 9 * 64;
    c.max_audio_frames = 400;
    return c;
}

/* An encoder pass of `n` patches in one segment, every coordinate plane filled. */
struct EncKit {
    std::vector<uint16_t> pix;
    std::vector<int32_t>  coord, cu;
    RadBatch b{};
};
std::unique_ptr<EncKit> enc_pass(int32_t mod, int64_t n, int64_t patch_dim) {
    auto k = std::make_unique<EncKit>();
    k->pix.assign((size_t)(n * patch_dim), 0);
    k->coord.assign((size_t)(4 * n), 1);
    k->cu = { 0, (int32_t)n };
    k->b.enc = mod;
    k->b.enc_n_patch = n;
    k->b.enc_pixels = k->pix.data();
    k->b.enc_coord = k->coord.data();
    k->b.enc_n_seg = 1;
    k->b.enc_cu = k->cu.data();
    k->b.enc_max_seg = (int32_t)n;
    return k;
}

}  /* namespace */

/* An embedding model: a pooled f32 output and no logits, attention bidirectional on every layer
 * (a window on the sliding ones, one more than the reference's radius since attn_dense's bound is
 * strict), each layer's residual gated by its slice of the per-layer inputs and scaled by its
 * learned scalar, and both media encoders declared. */
TEST(embgemma2_declares_a_pooled_encoder_with_both_media_towers) {
    MetaKV mk = eg2_meta();
    RadBuilder b;
    RadBuildCtx ctx = eg2_ctx();
    REQUIRE_EQ(embgemma2::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    REQUIRE(b.pooled != 0);
    const RadBufDecl& p = b.bufs[b.pooled - 1].d;
    CHECK_EQ(p.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(p.shape[0], ctx.max_seqs);
    CHECK_EQ(p.shape[1], 768);
    CHECK_EQ(count_op(b, "logits_gemm"), 0);
    std::vector<long long> win;
    for (const auto& o : b.ops)
        if (o.op == "attn_dense" && pint(&o, "causal") == 0 && pint(&o, "head_dim") >= 256) {
            win.push_back(pint(&o, "window"));
            CHECK_NEAR(pflt(&o, "scale"), 1.0, 0.0);
        }
    CHECK(win == (std::vector<long long>{ 513, 513, 513, 513, 513, 0 }));
    CHECK_EQ(count_op(b, "mul_weight"), 6);       /* the layer scalars */
    CHECK_EQ(count_op(b, "seg_mean"), 1);
    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.audio_encoder_name, std::string("audio"));
    CHECK_EQ(b.audio_encoder.max_patches, 400);
    CHECK_EQ(b.audio_encoder.merge, 4);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights) CHECK(find_map(b, w.name) != nullptr);
}

/* What it refuses: a second rank, a container without layer_types or with a kind it does not
 * know or a count that is not the model's, and an audio tower that is not the conformer it
 * implements. */
TEST(embgemma2_refuses_what_it_does_not_serve) {
    RadBuildCtx ctx = eg2_ctx();
    auto rc = [&](MetaKV mk, const RadBuildCtx& c) {
        RadBuilder b;
        return stderr_of_rc([&] { return embgemma2::declare(&b, mk.meta(), &c); });
    };
    CHECK_EQ(rc(eg2_meta(), build_ctx(0, 2)), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().drop("layer_types"), ctx), RAD_E_INVAL);
    CHECK_EQ(rc(eg2_meta().set("layer_types", "sliding_attention sliding_attention "
                                              "sliding_attention sliding_attention "
                                              "sliding_attention chunked_attention"), ctx),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().set("layer_types", "sliding_attention full_attention"), ctx),
             RAD_E_INVAL);
    CHECK_EQ(rc(eg2_meta().set("audio_config.residual_weight", "1.0"), ctx), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().drop("embedding_dim"), ctx), RAD_E_INVAL);
}

/* The text pass over three whole inputs (one with an image's rows scattered in), the vision pass
 * and the audio pass each lint clean, and every GEMM reads whole rows: the audio encoder's clamp
 * copy feeding a GEMM is a buffer of its own width, not a slice of a wider one. */
TEST(embgemma2_text_and_media_passes_lint_clean_and_feed_gemms_whole_rows) {
    MetaKV mk = eg2_meta();
    RadBuilder b;
    RadBuildCtx ctx = eg2_ctx();
    REQUIRE_EQ(embgemma2::declare(&b, mk.meta(), &ctx), RAD_OK);
    const auto text = kit(b, {}, { 12, 7, 30 });
    static const int32_t rows[] = { 2, 3, 4 };
    static uint16_t embd[3 * 512];
    text->b.n_mm_rows = 3;
    text->b.mm_rows = rows;
    text->b.mm_embd = embd;
    const auto vis = enc_pass(RAD_MM_IMAGE, 9 * 16, 3 * 16 * 16);
    const auto aud = enc_pass(RAD_MM_AUDIO, 400, 128);
    std::vector<const RadBatch*> passes = { &text->b, &vis->b, &aud->b };
    for (const RadBatch* pb : passes) {
        RadCtx c{ .b = &b, .batch = pb };
        embgemma2::step(&c, pb);
        CHECK(c.step_fail.empty());
        CHECK(!c.issues.empty());
        CHECK_LINT(b, &c);
        if (pb == &text->b) {
            /* the pooled rows, one per input */
            const RecIssue& last = c.issues.back();
            CHECK_EQ(last.n, 3);
            CHECK_EQ(last.opd.back().handle, b.pooled);
        }
    }
    /* A pass past the declared audio frames, or not whole output rows, is refused. */
    for (int64_t n : { 404, 398 }) {
        const auto bad = enc_pass(RAD_MM_AUDIO, n, 128);
        RadCtx c{ .b = &b, .batch = &bad->b };
        embgemma2::step(&c, &bad->b);
        CHECK(!c.step_fail.empty());
        CHECK(c.issues.empty());
    }
}

/* Gemma 4 E4B with its audio encoder: the conformer EmbeddingGemma 2 declares
 * (rad_block_gemma4_audio.h), read under the conditional-generation checkpoint's `model.` prefix,
 * its mel bins from processor_config.json's feature_extractor. Its rows go through the same media
 * scatter as an image's, the pass lints clean, and the probe states 30 s of frames. */
TEST(gemma4_e4b_declares_its_audio_encoder) {
    std::vector<const char*> keys(std::begin(k_gem4e_keys), std::end(k_gem4e_keys));
    std::vector<const char*> vals(std::begin(k_gem4e_vals), std::end(k_gem4e_vals));
    const char* ak[] = { "audio_config.model_type", "audio_config.hidden_size",
                         "audio_config.num_attention_heads", "audio_config.output_proj_dims",
                         "audio_config.attention_chunk_size", "audio_config.attention_context_left",
                         "audio_config.attention_context_right", "audio_config.attention_logit_cap",
                         "audio_config.num_hidden_layers", "audio_config.conv_kernel_size",
                         "audio_config.residual_weight", "audio_config.hidden_act",
                         "audio_config.subsampling_conv_channels",
                         "processor.feature_extractor.feature_size" };
    const char* av[] = { "gemma4_audio", "1024", "8", "1536", "12", "13", "0", "50", "2", "5", "0.5",
                         "silu", "128 32", "128" };
    keys.insert(keys.end(), std::begin(ak), std::end(ak));
    vals.insert(vals.end(), std::begin(av), std::end(av));
    RadModelMeta meta = gemma4_e4b_meta();
    meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = 400;
    REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
    CHECK_EQ(b.audio_encoder_name, std::string("audio"));
    CHECK_EQ(b.audio_encoder.max_patches, 400);
    CHECK_EQ(b.audio_encoder.merge, 4);
    CHECK_EQ(b.audio_encoder.n_embd, meta.n_embd);
    const RecMap* emb = find_map(b, "a.embed");
    REQUIRE(emb != nullptr);
    CHECK(emb->src == std::vector<std::string>{ "model.embed_audio.embedding_projection.weight" });
    const RecMap* c0 = find_map(b, "a.sscp.c0");
    REQUIRE(c0 != nullptr);
    CHECK(c0->src == std::vector<std::string>{
                         "model.audio_tower.subsample_conv_projection.layer0.conv.weight" });
    /* the media scatter, and the per-layer inputs' pad rows over the same positions */
    CHECK_EQ(count_op(b, "scatter_rows"), 2);

    const auto aud = enc_pass(RAD_MM_AUDIO, 400, 128);
    RadCtx c{ .b = &b, .batch = &aud->b };
    gemma4_bf16::step(&c, &aud->b);
    CHECK(c.step_fail.empty());
    CHECK(!c.issues.empty());
    CHECK_LINT(b, &c);

    RadArchProbe pr{};
    REQUIRE_EQ(gemma4_bf16::probe(&meta, &pr), RAD_OK);
    CHECK_EQ(pr.audio_frames, 3000);
}

/* ==================================================================== nomicbert */

namespace {

/* Nomic BERT cut to two layers: one dense and one routed at moe_every_n_layers 2, two heads of
 * 32, ffn 128, four experts top 2, vocab 128. Its config spells the dimensions n_* and reads
 * post-norm. */
MetaKV nb_meta() {
    RadModelMeta m{};
    m.arch_id = "nomicbert"; m.name = "test-nomicbert"; m.quant = "";
    m.n_layers = 2; m.n_embd = 64; m.n_head = 2; m.n_head_kv = 2; m.head_dim = 32;
    m.n_ff = 128; m.n_vocab = 128; m.n_ctx_train = 2048;
    m.rms_eps = 1e-5f; m.rope_theta = 1e4f; m.rope_scale = 1.0f;
    m.n_expert = 4; m.n_expert_used = 2;
    MetaKV k(m);
    k.set("model_type", "nomic_bert")
     .set("moe_every_n_layers", "2")
     .set("moe_normalize_expert_weights", "0")
     .set("num_shared_experts", "0")
     .set("prenorm", "0")
     .set("causal", "0")
     .set("activation_function", "gelu")
     .set("rotary_emb_interleaved", "0")
     .set("rotary_emb_fraction", "1.0")
     .set("type_vocab_size", "1");
    return k;
}

}  /* namespace */

/* A post-norm encoder: word and token-type embeddings, emb_ln, then two layers, one dense and one
 * with a top-2 router over four experts, each layer's attention bidirectional at 1/sqrt(64) over
 * NeoX rotary, and a mean pool cast into the f32 pooled buffer. No logits and no lm_head. */
TEST(nomicbert_declares_a_pooled_post_norm_encoder) {
    MetaKV mk = nb_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(nomicbert::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    REQUIRE(b.pooled != 0);
    const RadBufDecl& p = b.bufs[b.pooled - 1].d;
    CHECK_EQ(p.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(p.shape[0], ctx.max_seqs);
    CHECK_EQ(p.shape[1], (int64_t)64);

    CHECK_EQ(count_op(b, "logits_gemm"), 0);
    CHECK_EQ(count_op(b, "embed_lookup"), 1);
    CHECK_EQ(count_op(b, "table_add"), 1);          /* the token-type table only */
    CHECK_EQ(count_op(b, "layernorm"), 5);          /* emb_ln and two norms on each of two layers */
    CHECK_EQ(count_op(b, "gemm_nt_bias"), 6);       /* qkv and out twice, fc1 and fc2 once */
    CHECK_EQ(count_op(b, "gemm_nt"), 1);            /* the router */
    CHECK_EQ(count_op(b, "rope"), 2);
    CHECK_EQ(count_op(b, "attn_dense"), 2);
    CHECK_EQ(count_op(b, "router_topk"), 1);
    CHECK_EQ(count_op(b, "moe_scatter"), 1);
    CHECK_EQ(count_op(b, "moe_gemm"), 2);
    CHECK_EQ(count_op(b, "gelu"), 1);
    CHECK_EQ(count_op(b, "moe_gather"), 1);
    CHECK_EQ(count_op(b, "add"), 1);                /* the routed residual */
    CHECK_EQ(count_op(b, "seg_mean"), 1);
    CHECK_EQ(count_op(b, "cast"), 1);

    std::vector<std::string> orders;
    for (const auto& o : b.ops) {
        if (o.op == "attn_dense") {
            CHECK_EQ(pint(&o, "head_dim"), 32);
            CHECK_EQ(pint(&o, "gqa"), 1);
            CHECK_EQ(pint(&o, "causal"), 0);
            CHECK_EQ(pint(&o, "max_seqlen"), (long long)ctx.max_tok);
            CHECK_NEAR(pflt(&o, "scale"), 0.125, 0.0);
        }
        if (o.op == "router_topk") {
            CHECK_EQ(pint(&o, "norm"), 0);
            CHECK_EQ(pint(&o, "n_expert"), 4);
            CHECK_EQ(pint(&o, "top_k"), 2);
        }
        if (o.op == "moe_gemm") {
            CHECK_EQ(pint(&o, "n_expert"), 4);
            CHECK_EQ(pint(&o, "top_k"), 2);
            orders.push_back(pstr(&o, "a_order"));
        }
        if (o.op == "table_add") {
            CHECK_EQ(pint(&o, "n"), 64);
            CHECK_EQ(pint(&o, "L"), 1);
            CHECK_EQ(pstr(&o, "dtype"), std::string("bf16"));
            REQUIRE_EQ(o.w.size(), (size_t)1);
            CHECK_EQ(b.weights[o.w[0] - 1].name, std::string("token_type.weight"));
            CHECK_EQ(b.weights[o.w[0] - 1].d.dtype, (uint32_t)RAD_BF16);
        }
    }
    CHECK(orders == (std::vector<std::string>{ "token", "sorted" }));

    /* Every declared weight has a name map, and each projection's expert handles are one
     * consecutive run with a slot per projection. */
    for (const auto& w : b.weights) CHECK(find_map(b, w.name) != nullptr);
    /* The dtypes the device rows require. table_add_bf16 reads a bf16 table, so the token-type
     * table is bf16; every linear and the token table are bf16. layernorm and gemm_nt_bias take
     * an f32 gain or bias, so norms and biases stay f32. */
    const RecWeight* tt = find_w(b, "token_type.weight");
    REQUIRE(tt != nullptr);
    CHECK_EQ(tt->d.dtype, (uint32_t)RAD_BF16);
    const RecWeight* tw = find_w(b, "token_embd.weight");
    REQUIRE(tw != nullptr);
    CHECK_EQ(tw->d.dtype, (uint32_t)RAD_BF16);
    for (const char* n : { "emb_ln.weight", "emb_ln.bias", "blk.0.attn_norm.weight",
                           "blk.0.attn_norm.bias", "blk.0.attn_qkv.bias",
                           "blk.0.attn_out.bias", "blk.0.ffn_norm.weight",
                           "blk.0.ffn_norm.bias", "blk.0.ffn_fc1.bias",
                           "blk.0.ffn_fc2.bias", "blk.1.ffn_norm.weight",
                           "blk.1.ffn_norm.bias" }) {
        const RecWeight* w = find_w(b, n);
        REQUIRE(w != nullptr);
        CHECK_EQ(w->d.dtype, (uint32_t)RAD_F32);
    }
    for (const char* n : { "blk.0.attn_qkv.weight", "blk.0.attn_out.weight",
                           "blk.0.ffn_fc1.weight", "blk.0.ffn_fc2.weight",
                           "blk.1.attn_qkv.weight", "blk.1.attn_out.weight",
                           "blk.1.ffn_router.weight", "blk.1.ffn_exp_w1.0.weight",
                           "blk.1.ffn_exp_w2.0.weight" }) {
        const RecWeight* w = find_w(b, n);
        REQUIRE(w != nullptr);
        CHECK_EQ(w->d.dtype, (uint32_t)RAD_BF16);
    }
    auto index_of = [&](const std::string& name) {
        for (size_t i = 0; i < b.weights.size(); ++i) if (b.weights[i].name == name) return (int)i;
        return -1;
    };
    const int w10 = index_of("blk.1.ffn_exp_w1.0.weight");
    REQUIRE(w10 >= 0);
    for (int e = 0; e < 4; ++e) {
        const std::string n1 = "blk.1.ffn_exp_w1." + std::to_string(e) + ".weight";
        const std::string n2 = "blk.1.ffn_exp_w2." + std::to_string(e) + ".weight";
        const RecWeight* a = find_w(b, n1);
        const RecWeight* c = find_w(b, n2);
        REQUIRE(a != nullptr && c != nullptr);
        CHECK_EQ(index_of(n1), w10 + e);
        CHECK_EQ(index_of(n2), w10 + 4 + e);
        CHECK_EQ(a->d.group.expert, e);
        CHECK_EQ(a->d.group.slot, 0);
        CHECK_EQ(c->d.group.expert, e);
        CHECK_EQ(c->d.group.slot, 1);
        CHECK_EQ(a->d.shape[0], (int64_t)128);
        CHECK_EQ(a->d.shape[1], (int64_t)64);
        CHECK_EQ(c->d.shape[0], (int64_t)64);
        CHECK_EQ(c->d.shape[1], (int64_t)128);
    }
}

/* The text pass over three whole inputs lints clean and its last issue writes the pooled rows. */
TEST(nomicbert_text_pass_lints_clean_and_ends_at_the_pooled_buffer) {
    MetaKV mk = nb_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(nomicbert::declare(&b, mk.meta(), &ctx), RAD_OK);
    const auto text = kit(b, {}, { 12, 7, 30 });
    RadCtx c{ .b = &b, .batch = &text->b };
    nomicbert::step(&c, &text->b);
    CHECK(c.step_fail.empty());
    CHECK(!c.issues.empty());
    CHECK_LINT(b, &c);
    const RecIssue& last = c.issues.back();
    CHECK_EQ(last.n, (int64_t)3);
    CHECK_EQ(last.opd.back().handle, b.pooled);
    for (const auto& op : issued_ops(b, c)) CHECK(op != std::string("logits_gemm"));
}

/* The probe accepts the config's model_type spelling and refuses another architecture. */
TEST(nomicbert_probe_reads_the_model_type) {
    MetaKV mk = nb_meta();
    RadArchProbe p{};
    CHECK_OK(nomicbert::probe(mk.meta(), &p));
    MetaKV other = nb_meta().set("model_type", "bert");
    CHECK_EQ(stderr_of_rc([&] { return nomicbert::probe(other.meta(), &p); }),
             RAD_E_UNSUPPORTED);
}

/* Gemma 4 12B's media embedders (rad_block_gemma4_unified.h) in place of the tower and the
 * conformer: a picture patch is a 48 x 48 window and an audio frame 640 samples, each one row. The
 * position table [npos, 2, width] is read as its taps 0 (column) and 1 (row) of dimension 1; a
 * vision pass is LayerNorm, the dense with its bias, LayerNorm, the two position rows (the column
 * from coordinate plane 1, the row from plane 0) added in, LayerNorm (eps 1e-5, no config states it),
 * the weightless RMS norm and the projection; an audio pass the norm and its projection. Every rank
 * runs both whole and both lint clean; the probe says one patch a row and 750 frames. */
TEST(the_gemma4_12b_embeds_images_and_audio_without_an_encoder) {
    std::vector<const char*> keys(std::begin(k_gem12_keys), std::end(k_gem12_keys));
    std::vector<const char*> vals(std::begin(k_gem12_vals), std::end(k_gem12_vals));
    const char* mk[] = { "vision_config.model_type", "vision_config.mm_embed_dim",
                         "vision_config.mm_posemb_size", "vision_config.model_patch_size",
                         "vision_config.output_proj_dims", "vision_config.rms_norm_eps",
                         "vision_config.patch_size", "vision_config.pooling_kernel_size",
                         "audio_config.model_type", "audio_config.audio_embed_dim",
                         "audio_config.rms_norm_eps", "processor.feature_extractor.sampling_rate",
                         "processor.feature_extractor.audio_samples_per_token",
                         "processor.audio_seq_length" };
    const char* mv[] = { "gemma4_unified_vision", "3840", "1120", "48", "3840", "1e-06", "16", "3",
                         "gemma4_unified_audio", "640", "1e-06", "16000", "640", "750" };
    keys.insert(keys.end(), std::begin(mk), std::end(mk));
    vals.insert(vals.end(), std::begin(mv), std::end(mv));
    RadModelMeta meta = gemma4_12b_meta();
    meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();

    RadArchProbe pr{};
    REQUIRE_EQ(gemma4_bf16::probe(&meta, &pr), RAD_OK);
    CHECK_EQ(pr.enc_merge, 1);
    CHECK_EQ(pr.audio_frames, 750);

    for (int world : { 1, 2, 4 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = build_ctx(rank, world);
            ctx.max_enc_patches = 1120;
            ctx.max_audio_frames = 750;
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            CHECK_EQ(b.encoder_name, std::string("vision"));
            CHECK_EQ(b.encoder.patch_dim, 6912);
            CHECK_EQ(b.encoder.merge, 1);
            CHECK_EQ(b.encoder.max_patches, 1120);
            CHECK_EQ(b.encoder.n_embd, 3840);
            CHECK_EQ(b.audio_encoder_name, std::string("audio"));
            CHECK_EQ(b.audio_encoder.patch_dim, 640);
            CHECK_EQ(b.audio_encoder.merge, 1);
            CHECK_EQ(b.audio_encoder.max_patches, 750);
            CHECK_EQ(count_op(b, "scatter_rows"), 1);
            for (int t = 0; t < 2; ++t) {
                const RecMap* pos = find_map(b, t ? "v.pos_y.weight" : "v.pos_x.weight");
                REQUIRE(pos != nullptr);
                CHECK_EQ(pos->mode, (int)RAD_MAP_TAP);
                CHECK_EQ(pos->concat_dim, 1);
                CHECK(pos->src == std::vector<std::string>{ "model.vision_embedder.pos_embedding" });
                REQUIRE_EQ(pos->src_index.size(), (size_t)1);
                CHECK_EQ(pos->src_index[0], t);
                const RecWeight* w = find_w(b, t ? "v.pos_y.weight" : "v.pos_x.weight");
                REQUIRE(w != nullptr);
                CHECK_EQ(w->d.shape[0], 1120);
                CHECK_EQ(w->d.shape[1], 3840);
            }
            const struct { const char* d; const char* s; } src[] = {
                { "v.ln1.weight", "model.vision_embedder.patch_ln1.weight" },
                { "v.ln1.bias", "model.vision_embedder.patch_ln1.bias" },
                { "v.in.weight", "model.vision_embedder.patch_dense.weight" },
                { "v.in.bias", "model.vision_embedder.patch_dense.bias" },
                { "v.ln2.weight", "model.vision_embedder.patch_ln2.weight" },
                { "v.ln2.bias", "model.vision_embedder.patch_ln2.bias" },
                { "v.pn.weight", "model.vision_embedder.pos_norm.weight" },
                { "v.pn.bias", "model.vision_embedder.pos_norm.bias" },
                { "v.embed.weight", "model.embed_vision.embedding_projection.weight" },
                { "a.embed.weight", "model.embed_audio.embedding_projection.weight" } };
            for (const auto& e : src) {
                const RecMap* mp = find_map(b, e.d);
                REQUIRE(mp != nullptr);
                CHECK(mp->src == std::vector<std::string>{ e.s });
                CHECK(find_w(b, e.d) != nullptr);
            }
            const RecWeight* dense = find_w(b, "v.in.weight");
            REQUIRE(dense != nullptr);
            CHECK_EQ(dense->d.shape[0], 3840);
            CHECK_EQ(dense->d.shape[1], 6912);
            CHECK_EQ(dense->d.shard, (int)RAD_SHARD_NONE);
            CHECK(find_w(b, "v.blk.0.qkv") == nullptr);           /* no tower */
            for (const auto& o : b.ops)
                if (o.op == "layernorm") CHECK_NEAR(pflt(&o, "eps"), 1e-5, 0.0);

            const auto vis = enc_pass(RAD_MM_IMAGE, 280, 6912);
            RadCtx cv{ .b = &b, .batch = &vis->b };
            cv.rank = rank; cv.world = world;
            gemma4_bf16::step(&cv, &vis->b);
            CHECK(cv.step_fail.empty());
            CHECK_LINT(b, &cv);
            const std::vector<std::string> want = { "layernorm", "gemm_nt_bias", "layernorm",
                                                    "embed_lookup", "embed_lookup", "add", "add",
                                                    "layernorm", "rmsnorm", "gemm_nt" };
            CHECK_EQ(join(issued_ops(b, cv)), join(want));
            REQUIRE_EQ(cv.issues.size(), want.size());
            CHECK(cv.issues[0].opd[0].raw == vis->pix.data());
            CHECK(cv.issues[3].opd[0].raw == vis->coord.data() + 280);   /* the column */
            CHECK(cv.issues[4].opd[0].raw == vis->coord.data());         /* the row */
            CHECK_EQ(cv.issues[3].opd[1].handle, (uint32_t)gemma4_bf16::g_model[rank].uvis.w_px);
            CHECK_EQ(cv.issues[4].opd[1].handle, (uint32_t)gemma4_bf16::g_model[rank].uvis.w_py);
            CHECK_EQ(cv.issues.back().n, 280);
            CHECK_EQ(cv.issues.back().opd.back().handle, b.encoder.out);

            const auto aud = enc_pass(RAD_MM_AUDIO, 125, 640);
            RadCtx ca{ .b = &b, .batch = &aud->b };
            ca.rank = rank; ca.world = world;
            gemma4_bf16::step(&ca, &aud->b);
            CHECK(ca.step_fail.empty());
            CHECK_LINT(b, &ca);
            CHECK_EQ(join(issued_ops(b, ca)), join({ "rmsnorm", "gemm_nt" }));
            CHECK_EQ(ca.issues.back().n, 125);
            CHECK_EQ(ca.issues.back().opd.back().handle, b.audio_encoder.out);

            /* More patches than declared is refused, issuing nothing. */
            const auto big = enc_pass(RAD_MM_IMAGE, 1121, 6912);
            RadCtx cb{ .b = &b, .batch = &big->b };
            gemma4_bf16::step(&cb, &big->b);
            CHECK(!cb.step_fail.empty());
            CHECK(cb.issues.empty());
        }
    }
}

/* The 12B (and the 31B) read a picture's placeholders both ways (use_bidirectional_attention
 * "vision") on the sliding layers; the global layers stay causal, as transformers' Gemma 4 masks
 * and KerasHub's have it. Every sliding layer declares a second attention without the causal
 * bound (its window two-sided), the encoder says so to the core (RadEncoderDecl::bidirectional),
 * and on a pass with a run (RadBatch::n_bidi) each sliding layer attends the run's rows again over
 * their sequence's cache up to the run's end, written over the causal result in the same rows.
 * Without the config key, nothing of it. */
TEST(the_gemma4_12b_reads_a_pictures_rows_both_ways) {
    std::vector<const char*> keys(std::begin(k_gem12_keys), std::end(k_gem12_keys));
    std::vector<const char*> vals(std::begin(k_gem12_vals), std::end(k_gem12_vals));
    const char* mk[] = { "vision_config.model_type", "vision_config.mm_embed_dim",
                         "vision_config.mm_posemb_size", "vision_config.model_patch_size",
                         "use_bidirectional_attention" };
    const char* mv[] = { "gemma4_unified_vision", "3840", "1120", "48", "vision" };
    keys.insert(keys.end(), std::begin(mk), std::end(mk));
    vals.insert(vals.end(), std::begin(mv), std::end(mv));
    static const int32_t row[] = { 40 }, seq[] = { 1 }, used[] = { 30 };
    const int64_t L_sliding = 10;                    /* layers 5 and 11 are global */
    /* A run of 10 rows within the window, and one longer than a window of 8, whose rows read the
     * keys behind them only through the window (attn_paged_ahead). */
    const int32_t run = 10;
    static const int32_t rows[] = { run };
    for (const int win : { 1024, 8 })
    for (int mode = 0; mode < 2; ++mode) {          /* 0 on, 1 no key */
        std::vector<const char*> wvals = vals;
        for (size_t k = 0; k < keys.size(); ++k)
            if (!std::strcmp(keys[k], "sliding_window")) wvals[k] = win == 8 ? "8" : "1024";
        for (bool fused : { false, true }) {
            for (int world : { 1, 2 }) {
                for (int rank = 0; rank < world; ++rank) {
                    RadBuilder b;
                    if (fused) {
                        const RadEncoding e = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
                        b.encs = { { "attn_qkv.", e }, { "attn_o.", e }, { "ffn_gate_up.", e },
                                   { "ffn_down.", e } };
                    }
                    RadModelMeta meta = gemma4_12b_meta();
                    meta.n_kv = (int)keys.size() - (mode == 1 ? 1 : 0);
                    meta.kv_key = keys.data(); meta.kv_val = wvals.data();
                    RadBuildCtx ctx = build_ctx(rank, world);
                    ctx.max_enc_patches = 1120;
                    const int rc = gemma4_bf16::declare(&b, &meta, &ctx);
                    REQUIRE_EQ(rc, RAD_OK);
                    const bool on = mode == 0;
                    CHECK_EQ(b.encoder.bidirectional, on ? 1 : 0);
                    int nc = 0;
                    for (const auto& o : b.ops)
                        if ((o.op == "attn_paged" || o.op == "attn_paged_qrow") && pint(&o, "causal") == 0) {
                            ++nc;
                            CHECK_EQ(pint(&o, "window"), win);
                            CHECK_EQ(pint(&o, "head_dim"), 256);
                        }
                    CHECK_EQ(nc, on ? (int)L_sliding : 0);
                    int na = 0;
                    for (const auto& o : b.ops)
                        if (o.op == "attn_paged_ahead" || o.op == "attn_paged_qrow_ahead") {
                            ++na;
                            CHECK_EQ(pint(&o, "causal"), 0);
                            CHECK_EQ(pint(&o, "window"), win);
                        }
                    CHECK_EQ(na, on ? (int)L_sliding : 0);

                    RadBatch batch = prefill_batch();
                    batch.n_bidi = 1;
                    batch.bidi_row = row; batch.bidi_n = rows; batch.bidi_seq = seq;
                    batch.bidi_used = used;
                    RadCtx c; c.b = &b; c.rank = rank; c.world = world; c.batch = &batch;
                    gemma4_bf16::step(&c, &batch);
                    CHECK_LINT(b, &c);
                    CHECK(c.step_fail.empty());
                    int again = 0;
                    rad_op last_attn = 0;
                    for (const auto& i : c.issues) {
                        const RecOp* o = b.op_of(i.op);
                        if (!o || (o->op != "attn_paged" && o->op != "attn_paged_qrow" &&
                                   o->op != "attn_paged_ahead" && o->op != "attn_paged_qrow_ahead"))
                            continue;
                        if (pint(o, "causal") == 1) { last_attn = i.op; continue; }
                        ++again;
                        CHECK(last_attn != 0);                       /* behind the causal pass */
                        const bool ahead = o->op.size() > 6 && o->op.compare(o->op.size() - 6, 6, "_ahead") == 0;
                        CHECK_EQ(ahead, run > win);
                        CHECK_EQ(i.n, run);
                        REQUIRE(i.opd.size() == 8);
                        CHECK_EQ((int)i.opd[1].kind, (int)RAD_OPK_KV);
                        CHECK(i.opd[3].raw == used);
                        CHECK_EQ(i.opd[3].rows, 1);
                        CHECK_EQ(i.opd[2].rows, 1);                  /* the run's sequence's table */
                        CHECK(i.opd[6].kind == RAD_OPK_NONE);
                        const int64_t qd = (16 / world) * (pint(o, "head_dim"));
                        CHECK_EQ(i.opd[7].offset, 40 * qd);          /* its rows of the output */
                        CHECK_EQ(i.opd[7].rows, run);
                        last_attn = 0;
                    }
                    CHECK_EQ(again, on ? (int)L_sliding : 0);
                }
            }
        }
    }
}

/* ==================================================================== qwenimage21 */

namespace {

/* Qwen-Image 2.1 with every network cut: a two-layer conditioning language model of 512, a
 * two-block DiT of four 128-wide heads (rotary axes 16, 56, 56), and the residual VAE's own
 * shape. */
MetaKV qimg_meta() {
    RadModelMeta m{};
    m.arch_id = "qwenimage21"; m.name = "test-qwen-image"; m.quant = "";
    m.n_layers = 2; m.n_embd = 512; m.n_head = 4; m.n_head_kv = 2; m.head_dim = 128;
    m.n_ff = 1024; m.n_vocab = 151936; m.n_ctx_train = 32768;
    m.rms_eps = 1e-6f; m.rope_theta = 5e6f; m.rope_scale = 1.0f;
    MetaKV k(m);
    std::string mean, sd;
    for (int i = 0; i < 64; ++i) {
        mean += (i ? " " : "") + std::to_string(0.01 * i);
        sd += (i ? " " : "") + std::to_string(1.0 + 0.01 * i);
    }
    k.set("pipeline._class_name", "QwenImage21Pipeline")
     .set("transformer.patch_size", "1")
     .set("transformer.causal_condition", "1")
     .set("transformer.num_layers", "2")
     .set("transformer.attention_head_dim", "128")
     .set("transformer.num_attention_heads", "4")
     .set("transformer.mlp_ratio", "4")
     .set("transformer.in_channels", "64")
     .set("transformer.context_in_dim", "512")
     .set("transformer.axes_dims_rope", "16 56 56")
     .set("vae.is_residual", "1")
     .set("vae.z_dim", "64")
     .set("vae.latents_mean", mean.c_str())
     .set("vae.latents_std", sd.c_str())
     .set("vae.dim_mult", "1 2 4 4 4")
     .set("vae.temperal_downsample", "0 1 1 0");
    return k;
}

/* A generator pass: COND over a text prompt of `n` tokens, DENOISE over `entries` latents of `n`
 * rows each, or DECODE of a 64 x 64 picture. */
struct GenKit {
    std::vector<int32_t> tok, rope, cu, src, seg, seg_h, seg_bidi, seg_cu, seg_start, slot;
    std::vector<int32_t> pair, lane, flags, table, used;
    std::vector<uint32_t> seed;
    std::vector<float> sigma, sigma_next, cfg;
    RadKVGroupBatch kv{};
    RadGenBatch g{};
    RadBatch b{};
};

std::unique_ptr<GenKit> gen_pass(const RadBuilder& bld, int pass, int64_t n, int64_t entries = 1) {
    auto k = std::make_unique<GenKit>();
    const int64_t S = pass == RAD_GEN_PASS_DENOISE ? entries : 1, N = n * S;
    for (int64_t i = 0; i < N; ++i) {
        k->tok.push_back((int32_t)(3 + i % 50));
        k->src.push_back((int32_t)i);
        k->slot.push_back((int32_t)i);
    }
    k->rope.assign((size_t)(3 * N), 0);
    for (int64_t s = 0; s <= S; ++s) k->cu.push_back((int32_t)(s * n));
    k->seg = { 0, (int32_t)n };
    k->seg_h = k->seg;
    k->seg_bidi = { 0 };
    k->seg_cu = { 0, (int32_t)n };
    k->seg_start = { 0 };
    k->sigma.assign((size_t)S, pass == RAD_GEN_PASS_COND ? 0.0f : 0.5f);
    k->sigma_next.assign((size_t)S, 0.25f);
    k->cfg.assign((size_t)S, 1.0f);
    k->pair.assign((size_t)S, -1);
    for (int64_t s = 0; s < S; ++s) k->lane.push_back((int32_t)s);
    k->flags.assign((size_t)S, RAD_GEN_F_FIRST);
    k->seed.assign((size_t)(2 * S), 7u);
    k->table.assign((size_t)(S * 16), 0);
    k->used.assign((size_t)S, (int32_t)n);
    k->kv.group = 1;
    k->kv.slot_mapping = k->slot.data();
    k->kv.block_table = k->table.data();
    k->kv.block_table_pitch = 16;
    k->kv.seqused = k->used.data();
    k->kv.block_size = bld.kv_block;
    k->kv.max_blocks = 16;
    RadGenBatch& g = k->g;
    g.rows = n;
    g.src = k->src.data();
    g.rope = k->rope.data();
    g.n_seg = 1;
    g.seg = k->seg.data();
    g.seg_h = k->seg_h.data();
    g.seg_bidi_h = k->seg_bidi.data();
    g.seg_cu = k->seg_cu.data();
    g.seg_start = k->seg_start.data();
    g.slot_mapping = k->slot.data();
    g.sigma = k->sigma.data();
    g.sigma_next = k->sigma_next.data();
    g.cfg = k->cfg.data();
    g.pair = k->pair.data();
    g.lane = k->lane.data();
    g.lane_dh = k->lane.data();
    g.n_lane = S;
    g.flags = k->flags.data();
    g.seed = k->seed.data();
    g.dec_lane = 0;
    g.dec_height = g.dec_width = 64;
    RadBatch& b = k->b;
    b.gen = pass;
    b.gen_b = &k->g;
    b.n_tok = pass == RAD_GEN_PASS_DECODE ? 16 : N;
    b.n_seq = S;
    b.token_ids = k->tok.data();
    b.rope_pos = k->rope.data();
    b.cu_seqlens = k->cu.data();
    b.n_kv_groups = 1;
    b.kv = &k->kv;
    return k;
}

RadBuildCtx qimg_ctx(int rank, int world, int lossy = 0) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_tok = 512;
    c.max_seqs = 2;
    c.tp_wire_lossy = lossy;
    c.tp_wire_min_bytes = 64 * 1024;
    return c;
}

}  /* namespace */

/* An image generator, not a language model: no logits, the generator declared with the model's
 * prompt and sizes. One modulation projection is shared by every block; the DiT rotates three
 * sections of adjacent pairs from axes 16, 56, 56; the prefix is cached in a full KV group, read
 * causally by text and both ways by a picture. A Turbo container's fixed schedule names it. */
TEST(qwenimage21_limits_preserve_defaults_and_bound_single_rank_declarations) {
    MetaKV mk = qimg_meta();
    for (const int64_t pixels : {int64_t(0), int64_t(256 * 256), int64_t(1) << 30}) {
        RadBuilder b;
        RadBuildCtx ctx = qimg_ctx(0, 1);
        ctx.gen_max_pixels = pixels;
        ctx.gen_max_refs = pixels == 0 ? 0 : 2;
        REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
        const int64_t expected = pixels == 256 * 256 ? 256 : qwenimage21::kMaxImageTokens;
        CHECK_EQ(b.generator.max_tokens, expected);
        CHECK_EQ(b.generator.max_refs, pixels == 0 ? qwenimage21::kMaxRefs : 2);
        CHECK_EQ(b.bufs[b.generator.image_out - 1].d.shape[0], expected * 256 * 4);
    }
}

TEST(qwenimage21_sizes_its_passes_for_a_bounded_deployment) {
    MetaKV mk = qimg_meta();
    RadArchProbe probe{};
    probe.gen_max_pixels = 1024 * 1024;
    probe.gen_max_refs = 1;
    REQUIRE_EQ(qwenimage21::probe(mk.meta(), &probe), RAD_OK);
    CHECK_EQ(probe.gen_rows, 8192);
    CHECK_EQ(probe.gen_patches, 4508);

    for (int rank = 0; rank < 2; ++rank) {
        RadBuilder b;
        RadBuildCtx ctx = qimg_ctx(rank, 2);
        ctx.gen_max_pixels = probe.gen_max_pixels;
        ctx.gen_max_refs = probe.gen_max_refs;
        ctx.max_tok = probe.gen_rows;
        REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(b.generator.max_tokens, 4096);
        CHECK_EQ(b.generator.max_refs, 1);
        CHECK_EQ(b.generator.default_width, 1024);
        CHECK_EQ(b.generator.default_height, 1024);
        REQUIRE(b.generator.image_out > 0);
        CHECK_EQ(b.bufs[b.generator.image_out - 1].d.shape[0], 1024 * 1024 * 4);
        for (const auto& k : { gen_pass(b, RAD_GEN_PASS_DENOISE, 16, 2),
                               gen_pass(b, RAD_GEN_PASS_DECODE, 0) }) {
            k->g.dec_lane = 1;
            RadCtx c{ .b = &b, .batch = &k->b, .rank = rank, .world = 2 };
            qwenimage21::step(&c, &k->b);
            CHECK(c.step_fail.empty());
            CHECK_LINT(b, &c);
        }
    }

    RadBuilder b;
    RadBuildCtx ctx = qimg_ctx(0, 1);
    ctx.gen_max_pixels = 1023;
    CHECK_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_E_INVAL);
}


TEST(qwenimage21_declares_a_generator_with_one_shared_modulation) {
    MetaKV mk = qimg_meta();
    RadBuilder b;
    RadBuildCtx ctx = qimg_ctx(0, 1);
    REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK_EQ(b.generator_name, std::string("qwen-image-2.1"));
    CHECK_EQ(b.generator.layout, (int32_t)RAD_GEN_LAYOUT_JOINT_GRID);
    CHECK_EQ(b.generator.latent_channels, 64);
    CHECK_EQ(b.generator.n_sigmas, 0);
    CHECK(b.generator.image_out != 0);
    int mods = 0;
    for (const auto& w : b.weights) mods += has(w.name, "dit.mod.weight");
    CHECK_EQ(mods, 1);
    const rad_buf mod = (rad_buf)buf_named(b, "dit.mod");
    int readers = 0;
    for (const auto& o : b.ops)
        if ((o.op == "ln_mod" || o.op == "gate_add" || o.op == "ln_mod_quant_fp8") &&
            in(o.reads, mod))
            ++readers;
    CHECK(readers >= 2 * 2);          /* both blocks' norms read the one modulation */
    bool dit_rope = false;
    for (const auto& o : b.ops)
        if (pstr(&o, "sections") == "8 28 28") dit_rope = true;
    CHECK(dit_rope);
    /* Per block: the causal and the bidirectional form, and (fp8 linears) the denoising pass's
     * bidirectional form with int8 scores. */
    int causal = 0, bidi = 0, bidi8 = 0;
    for (const auto& o : b.ops)
        if (o.op == "attn_prefix_self")
            (pint(&o, "causal") ? causal : pstr(&o, "qk") == "i8" ? bidi8 : bidi)++;
    CHECK_EQ(causal, 2);
    CHECK_EQ(bidi, 2);
    CHECK_EQ(bidi8, 2);
    REQUIRE_EQ(b.kv_groups.size(), (size_t)1);
    CHECK_EQ((int)b.kv_groups[0].d.kind, (int)RAD_KV_FULL);

    MetaKV turbo = qimg_meta();
    turbo.set("pipeline.sample_sigmas", "1.0 0.9 0.75 0.6 0.45 0.3 0.15 0.05");
    RadBuilder bt;
    REQUIRE_EQ(qwenimage21::declare(&bt, turbo.meta(), &ctx), RAD_OK);
    CHECK_EQ(bt.generator_name, std::string("qwen-image-2.1-turbo"));
    CHECK_EQ(bt.generator.n_sigmas, 8);
    CHECK_EQ(bt.generator.default_steps, 8);
}

/* Every collective follows --tp-wire: exact unless the lossy wire was accepted, and the floor
 * under which a message stays exact is the deployment's. */
/* The DiT's adapter names: one root before any GEMM, and every alias lands on a checkpoint tensor
 * the plugin maps, with the gate half of ComfyUI's `gate_up` on `gate_layer`. */
TEST(qwenimage21_names_its_adapter_keys_onto_mapped_tensors) {
    MetaKV mk = qimg_meta();
    RadBuilder b;
    const RadBuildCtx ctx = qimg_ctx(0, 1);
    REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
    REQUIRE_EQ(b.lora_roots.size(), (size_t)1);
    CHECK_EQ(b.lora_roots[0].first, std::string("transformer."));
    CHECK(b.lora_roots[0].second.find("lora_unet_") != std::string::npos);
    std::set<std::string> srcs;
    for (const RecMap& m : b.maps) for (const std::string& x : m.src) srcs.insert(x);
    REQUIRE_EQ(b.lora_aliases.size(), (size_t)2);
    for (const auto& a : b.lora_aliases) {
        std::string to = a.native + a.to;
        const size_t h = to.find('#');
        if (h != std::string::npos) to.replace(h, 1, "0");
        CHECK(srcs.count(to + ".weight"));
    }
    CHECK(b.lora_aliases[0].to.find("gate_layer") != std::string::npos);
    CHECK_EQ(b.lora_aliases[0].from_lo, (int64_t)0);
    CHECK_EQ(b.lora_aliases[1].from_lo, b.lora_aliases[0].from_hi);
}

TEST(qwenimage21_collectives_follow_the_tp_wire) {
    MetaKV mk = qimg_meta();
    for (int lossy : { 0, 1 }) {
        RadBuilder b;
        RadBuildCtx ctx = qimg_ctx(0, 2, lossy);
        REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
        int n = 0;
        for (const auto& o : b.ops) {
            if (o.op != "all_reduce") continue;
            ++n;
            CHECK_EQ(pint(&o, "exact"), lossy ? 0 : 1);
            CHECK_EQ(pint(&o, "min_bytes"), 64 * 1024);
        }
        CHECK(n >= 2 * 2 + 2 * 2);      /* two a layer in each network */
    }
}

/* The three passes the core drives (a text prompt's conditioning, denoising one image or two at
 * once, decoding one) lint clean at one and two ranks, the split all-reduces included. */
TEST(qwenimage21_generator_passes_lint_clean) {
    MetaKV mk = qimg_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = qimg_ctx(rank, world);
            REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
            for (const auto& k : { gen_pass(b, RAD_GEN_PASS_COND, 12),
                                   gen_pass(b, RAD_GEN_PASS_DENOISE, 16),
                                   gen_pass(b, RAD_GEN_PASS_DENOISE, 16, 2),
                                   gen_pass(b, RAD_GEN_PASS_DECODE, 0) }) {
                RadCtx c{ .b = &b, .batch = &k->b, .rank = rank, .world = world };
                qwenimage21::step(&c, &k->b);
                CHECK(c.step_fail.empty());
                CHECK(!c.issues.empty());
                CHECK_EQ(c.lane, 0);
                CHECK_LINT(b, &c);
            }
            /* A token step is not a generator pass. */
            const auto tok = kit(b, { 1 }, {});
            RadCtx c{ .b = &b, .batch = &tok->b, .rank = rank, .world = world };
            qwenimage21::step(&c, &tok->b);
            CHECK(!c.step_fail.empty());
        }
}

/* An IQ8R container: every linear of the text encoder and of the DiT is an i8a8h GEMM at one scale
 * a 256 of K, over int8 codes the rotating quantisers and the gated products write. The DiT's
 * norms are the fused op's rotating twin at that group where it is served, and the separate
 * gate_add, ln_mod and had_quant_act_i8g where it is not. No op that writes E4M3 is declared and no
 * activation plane is E4M3; the generator's passes lint clean at one and two ranks either way. */
TEST(qwenimage21_serves_iq8r_linears_on_int8_activations) {
    MetaKV mk = qimg_meta();
    for (int world : { 1, 2 })
      for (bool fused : { true, false })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            if (!fused) b.undeclared.insert("ln_mod_had_quant_i8g");
            for (const char* t : { "attn_qkv.weight", "attn_output.weight", "ffn_gate_up.weight",
                                   "ffn_down.weight" })
                b.encs.push_back({ t, iq8r_enc() });
            RadBuildCtx ctx = qimg_ctx(rank, world);
            REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
            int rot = 0;
            for (const RecOp& o : b.ops) {
                if (o.op != "gemm_nt_q") continue;
                CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
                CHECK_EQ(pint(&o, "group"), 256);
                ++rot;
            }
            CHECK(rot >= (2 + 2) * 4);
            for (const char* op : { "quant_act_fp8", "ln_mod_quant_fp8", "gated_quant_fp8" })
                CHECK_EQ(count_op(b, op), 0);
            CHECK(count_op(b, "had_quant_act_i8g") > 0);
            CHECK(count_op(b, "gated_had_quant_i8g") > 0);
            /* Two blocks of two norms; unfused, only norm_out's ln_mod is left beside them. */
            CHECK_EQ(count_op(b, "ln_mod_had_quant_i8g"), fused ? 4 : 0);
            CHECK_EQ(count_op(b, "ln_mod"), fused ? 1 : 5);
            for (const RecOp& o : b.ops)
                if (o.op == "ln_mod_had_quant_i8g") CHECK_EQ(pint(&o, "group"), 256);
            for (const auto& buf : b.bufs) CHECK(buf.d.dtype != (uint32_t)RAD_F8E4M3);
            for (const auto& k : { gen_pass(b, RAD_GEN_PASS_COND, 12),
                                   gen_pass(b, RAD_GEN_PASS_DENOISE, 16),
                                   gen_pass(b, RAD_GEN_PASS_DENOISE, 16, 2),
                                   gen_pass(b, RAD_GEN_PASS_DECODE, 0) }) {
                RadCtx c{ .b = &b, .batch = &k->b, .rank = rank, .world = world };
                qwenimage21::step(&c, &k->b);
                CHECK(c.step_fail.empty());
                CHECK(!c.issues.empty());
                CHECK_LINT(b, &c);
            }
        }
}

/* ==================================================================== anima */

namespace {

/* Anima's text encoder (Qwen3-0.6B-Base) as the container's config states it; the DiT, adapter
 * and VAE are the plugin's own constants. */
MetaKV anima_meta(const char* variant) {
    RadModelMeta m{};
    m.arch_id = "anima"; m.name = "test-anima"; m.quant = "";
    m.n_layers = 28; m.n_embd = 1024; m.n_head = 16; m.n_head_kv = 8; m.head_dim = 128;
    m.n_ff = 3072; m.n_vocab = 151936; m.n_ctx_train = 32768;
    m.rms_eps = 1e-6f; m.rope_theta = 1e6f; m.rope_scale = 1.0f;
    MetaKV k(m);
    k.set("num_hidden_layers", "28").set("hidden_size", "1024").set("anima.variant", variant);
    return k;
}

RadBuildCtx anima_ctx() {
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2 * (4096 + 512);
    c.max_seqs = 4;
    return c;
}

/* A JOINT_STEP pass as the driver lays one out for Anima: COND over a prompt of `nq` Qwen3 tokens
 * (padded to 512) and `nt` T5 ids, DENOISE over `entries` entries of 512 context rows and `ni`
 * image rows each, or DECODE of a 64 x 64 picture. */
struct AnimaKit {
    std::vector<int32_t> tok, pos, cu, src, lane, flags, pair, aux, aux_cu, valid_cu;
    std::vector<int32_t> img_cu, img_rows, mode_cu, mode_rows, mode_src, mode_ccu;
    std::vector<int64_t> n_mode;
    std::vector<uint32_t> seed;
    std::vector<float> sigma, sigma_next, cfg, rope_f, mode_rope_f;
    RadGenBatch g{};
    RadBatch b{};
};

std::unique_ptr<AnimaKit> anima_pass(int pass, int64_t nq = 12, int64_t nt = 9, int64_t ni = 16,
                                     int64_t entries = 1) {
    auto k = std::make_unique<AnimaKit>();
    RadGenBatch& g = k->g;
    RadBatch& b = k->b;
    b.gen = pass;
    b.gen_b = &k->g;
    if (pass == RAD_GEN_PASS_COND) {
        for (int64_t i = 0; i < 512; ++i) {
            k->tok.push_back(i < nq ? (int32_t)(100 + i) : 151643);
            k->pos.push_back(i < nq ? (int32_t)i : 1);
        }
        for (int64_t i = 0; i < nt; ++i) k->aux.push_back((int32_t)(3 + i));
        k->aux_cu = { 0, (int32_t)nt };
        k->valid_cu = { 0, (int32_t)nq };
        k->cu = { 0, 512 };
        k->lane = { 0, 1 };
        k->sigma = { 0.0f };
        b.n_tok = 512;
        b.n_seq = 1;
        b.token_ids = k->tok.data();
        b.positions = k->pos.data();
        b.cu_seqlens = k->cu.data();
        g.rows = 512;
        g.n_pad = (int32_t)(512 - nq);
        g.valid_cu = k->valid_cu.data();
        g.n_lane = 2;
        g.lane = k->lane.data();
        g.lane_h = k->lane.data();
        g.branch = 1;
        g.sigma = k->sigma.data();
        g.n_aux = nt;
        g.aux_ids = k->aux.data();
        g.aux_cu = k->aux_cu.data();
        return k;
    }
    if (pass == RAD_GEN_PASS_DECODE) {
        k->cu = { 0, 16 };
        b.n_tok = 16;
        b.n_seq = 1;
        b.cu_seqlens = k->cu.data();
        g.dec_lane = 1;
        g.dec_height = g.dec_width = 64;
        return k;
    }
    const int64_t S = entries, N = S * (512 + ni);
    k->cu.push_back(0);
    k->img_cu.push_back(0);
    for (int64_t s = 0; s < S; ++s) {
        k->mode_cu.push_back((int32_t)(s * (512 + ni)));
        k->mode_cu.push_back((int32_t)(s * (512 + ni) + 512));
        k->mode_cu.push_back((int32_t)(s * (512 + ni) + 512));
        for (int64_t r = 0; r < 512; ++r) k->src.push_back((int32_t)(s * 512 + r));
        for (int64_t r = 0; r < ni; ++r) {
            k->src.push_back((int32_t)(-1 - (s * ni + r)));
            k->img_rows.push_back((int32_t)(s * (512 + ni) + 512 + r));
        }
        k->cu.push_back((int32_t)((s + 1) * (512 + ni)));
        k->img_cu.push_back((int32_t)((s + 1) * ni));
        k->lane.push_back((int32_t)(s / 2));
        k->pair.push_back(S > 1 ? (s % 2 ? -2 : (int32_t)(s + 1)) : -1);
    }
    k->mode_cu.push_back((int32_t)N);
    for (int64_t s = 0; s < S; ++s)
        for (int64_t r = 0; r < 512; ++r) {
            k->mode_rows.push_back((int32_t)(s * (512 + ni) + r));
            k->mode_src.push_back((int32_t)(s * 512 + r));
        }
    for (int64_t s = 0; s < S; ++s)
        for (int64_t r = 0; r < ni; ++r) {
            k->mode_rows.push_back((int32_t)(s * (512 + ni) + 512 + r));
            k->mode_src.push_back((int32_t)(-1 - (s * ni + r)));
        }
    k->n_mode = { 512 * S, 0, ni * S };
    for (int m = 0; m < 3; ++m)
        for (int64_t s = 0; s <= S; ++s)
            k->mode_ccu.push_back((int32_t)(s * (m == 0 ? 512 : m == 1 ? 0 : ni)));
    k->rope_f.assign((size_t)(4 * N), 0.0f);
    k->mode_rope_f.assign((size_t)(4 * N), 0.0f);
    k->tok.assign((size_t)N, 0);
    k->pos.assign((size_t)N, 0);
    k->sigma.assign((size_t)S, 0.75f);
    k->sigma_next.assign((size_t)S, 0.5f);
    k->cfg.assign((size_t)S, 4.0f);
    k->flags.assign((size_t)S, RAD_GEN_F_FIRST);
    k->seed.assign((size_t)(2 * S), 7u);
    b.n_tok = N;
    b.n_seq = S;
    b.token_ids = k->tok.data();
    b.positions = k->pos.data();
    b.cu_seqlens = k->cu.data();
    g.src = k->src.data();
    g.n_img = ni * S;
    g.img_cu = k->img_cu.data();
    g.img_rows = k->img_rows.data();
    g.mode_cu = k->mode_cu.data();
    g.rope_f = k->rope_f.data();
    g.mode_rows = k->mode_rows.data();
    g.mode_src = k->mode_src.data();
    g.mode_ccu = k->mode_ccu.data();
    g.n_mode_h = k->n_mode.data();
    g.mode_rope_f = k->mode_rope_f.data();
    g.sigma = k->sigma.data();
    g.sigma_next = k->sigma_next.data();
    g.cfg = k->cfg.data();
    g.pair = k->pair.data();
    g.lane = k->lane.data();
    g.lane_dh = k->lane.data();
    g.flags = k->flags.data();
    g.seed = k->seed.data();
    return k;
}

/* The linears a LoRA for Anima names: every DiT block's attention, MLP and modulation linears
 * and the text encoder's. */
bool anima_lora_target(const std::string& w) {
    for (const char* t : { "attn_qkv", "attn_output", "cross_q", "cross_k", "cross_v",
                           "cross_output", "mlp_layer1", "mlp_layer2", ".mod_", "ffn_gate_up",
                           "ffn_down" })
        if (has(w, t) && !has(w, ".scale")) return true;
    return false;
}

}  /* namespace */

/* A JOINT_STEP generator that reads a second tokenizer, pads an empty prompt to one token and
 * samples on ComfyUI's simple schedule; the variant sets its defaults and the DiT keys' prefix is
 * the checkpoint's, `model.diffusion_model.` here and `net.` for Base. */
TEST(anima_declares_a_two_tokenizer_generator_under_the_checkpoints_prefix) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("aesthetic");
    RadBuilder b;
    b.encs = { { "dit.t_norm.mdm", plain }, { "dit.blk.0.attn_qkv", plain } };
    REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK_EQ(b.generator_name, std::string("anima-aesthetic"));
    const RadGeneratorDecl& d = b.generator;
    CHECK_EQ(d.layout, (int32_t)RAD_GEN_LAYOUT_JOINT_STEP);
    CHECK_EQ(d.aux_tokens, 1);
    CHECK_EQ(d.aux_max_tokens, (int64_t)512);
    CHECK_EQ(d.min_tokens, 1);
    CHECK_EQ(d.pad_multiple, 512);
    CHECK_EQ(d.cond_rows, (int64_t)512);
    CHECK_EQ(d.sigma_grid, 1000);
    CHECK_EQ(d.shift, 3.0);
    CHECK_EQ(d.dyn_shift, 0);
    CHECK_EQ(d.cfg_original, 0);
    CHECK_EQ(d.default_steps, (int64_t)30);
    CHECK_EQ(d.default_cfg, 4.0);
    CHECK_EQ(d.latent_channels, (int64_t)64);
    CHECK(b.kv_groups.empty());
    const RecMap* q = find_map(b, "dit.blk.27.attn_qkv.weight");
    REQUIRE(q != nullptr);
    REQUIRE_EQ(q->src.size(), (size_t)3);
    CHECK_EQ(q->src[0], std::string("model.diffusion_model.blocks.27.self_attn.q_proj.weight"));
    const RecMap* te = find_map(b, "te.blk.3.ffn_gate_up.weight");
    REQUIRE(te != nullptr);
    CHECK_EQ(te->src[1], std::string("model.layers.3.mlp.up_proj.weight"));
    /* The patch embedding over its first 64 columns, the final linear's rows in eight pieces. */
    const RecMap* xe = find_map(b, "dit.x_embed.weight");
    REQUIRE(xe != nullptr);
    CHECK_EQ(xe->mode, (int)RAD_MAP_COLS);
    int final_maps = 0;
    for (const auto& m : b.maps)
        if (m.declared == "dit.final.weight") { ++final_maps; CHECK_EQ(m.mode, (int)RAD_MAP_ROWS); }
    CHECK_EQ(final_maps, 8);
    /* The DiT's rotary: 22, 21 and 21 pairs, NeoX, over (frame, row, column). */
    int axial = 0;
    for (const auto& o : b.ops)
        if (o.op == "qk_norm_rope" && pstr(&o, "mode") == "axial") {
            CHECK_EQ(pstr(&o, "sections"), std::string("22 21 21"));
            ++axial;
        }
    CHECK_EQ(axial, 28);

    MetaKV turbo = anima_meta("turbo");
    RadBuilder bt;
    bt.encs = { { "dit.t_norm.mdm", plain } };
    REQUIRE_EQ(anima::declare(&bt, turbo.meta(), &ctx), RAD_OK);
    CHECK_EQ(bt.generator_name, std::string("anima-turbo"));
    CHECK_EQ(bt.generator.default_steps, (int64_t)10);
    CHECK_EQ(bt.generator.default_cfg, 1.0);

    MetaKV base = anima_meta("base");
    RadBuilder bb;
    bb.encs = { { "dit.t_norm.net", plain } };
    REQUIRE_EQ(anima::declare(&bb, base.meta(), &ctx), RAD_OK);
    const RecMap* nq = find_map(bb, "dit.blk.0.mlp_layer1.weight");
    REQUIRE(nq != nullptr);
    CHECK_EQ(nq->src[0], std::string("net.blocks.0.mlp.layer1.weight"));

    /* Neither spelling, or an unknown variant, is refused. */
    RadBuilder none;
    CHECK(anima::declare(&none, mk.meta(), &ctx) < 0);
    MetaKV odd = anima_meta("deluxe");
    RadBuilder bo;
    bo.encs = { { "dit.t_norm.mdm", plain } };
    CHECK(anima::declare(&bo, odd.meta(), &ctx) < 0);
}

/* Every linear a LoRA for Anima names is read by a plain GEMM a runtime LoRA can follow: gemm_nt
 * in a bf16 container, gemm_nt_q without rotated parts in a block-fp8 one, its activation a
 * separate op. */
TEST(anima_lora_targets_are_plain_gemms) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("base");
    {
        RadBuilder b;
        b.encs = { { "dit.t_norm.net", plain }, { "dit.blk.0.attn_qkv", plain } };
        REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
        int targets = 0;
        for (const auto& o : b.ops)
            for (rad_weight w : o.w) {
                const std::string& n = b.weights[w - 1].name;
                if (!anima_lora_target(n)) continue;
                ++targets;
                const bool ok = o.op == "gemm_nt";
                if (!ok) fprintf(stderr, "    '%s' read by %s\n", n.c_str(), o.op.c_str());
                CHECK(ok);
                CHECK(o.op != "gemm_nt_bias");
            }
        /* 28 blocks of 8 linears and 6 modulation linears; 28 text encoder layers of 4. */
        CHECK(targets >= 28 * (8 + 6) + 28 * 4);
    }
}

/* The three passes the core drives (a conditioning with its T5 ids, denoising one image or a
 * guided pair, decoding one) lint clean in a bf16 container. */
TEST(anima_generator_passes_lint_clean) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("aesthetic");
    {
        RadBuilder b;
        b.encs = { { "dit.t_norm.mdm", plain }, { "dit.blk.0.attn_qkv", plain } };
        REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
        for (const auto& k : { anima_pass(RAD_GEN_PASS_COND, 12, 9),
                               anima_pass(RAD_GEN_PASS_COND, 3, 40),
                               anima_pass(RAD_GEN_PASS_DENOISE, 12, 9, 16, 1),
                               anima_pass(RAD_GEN_PASS_DENOISE, 12, 9, 16, 2),
                               anima_pass(RAD_GEN_PASS_DECODE) }) {
            RadCtx c{ .b = &b, .batch = &k->b, .rank = 0, .world = 1 };
            anima::step(&c, &k->b);
            CHECK(c.step_fail.empty());
            if (!c.step_fail.empty()) fprintf(stderr, "    %s\n", c.step_fail.c_str());
            CHECK(!c.issues.empty());
            CHECK_LINT(b, &c);
        }
        const auto tok = kit(b, { 1 }, {});
        RadCtx c{ .b = &b, .batch = &tok->b, .rank = 0, .world = 1 };
        anima::step(&c, &tok->b);
        CHECK(!c.step_fail.empty());
    }
}

/* ======================================================== the conventional vocabulary */

namespace {

/* libref's vocabulary: every op its schemas name. A CPU-only engine, or a new hardware library
 * that implements only what libref does, declares exactly these (libavx's are a subset). Empty
 * when libref.so is not under $RADIANCE_HOME/kernels (ctest sets it to the build's plugin home). */
const std::set<std::string>& libref_vocab() {
    static std::set<std::string> v;
    static bool tried = false;
    if (tried) return v;
    tried = true;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home) return v;
    const std::string path = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "    cannot load %s: %s\n", path.c_str(), dlerror());
        return v;
    }
    auto count = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
    auto at = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
    if (count && at)
        for (int i = 0; i < count(); ++i)
            if (const RadOpSchema* s = at(i); s && s->op) v.insert(s->op);
    return v;
}

/* Nomic BERT declares only ops libref serves, so a CPU-only engine can start it. */
TEST(nomicbert_declares_on_librefs_vocabulary) {
    const std::set<std::string>& vocab = libref_vocab();
    if (vocab.empty()) {
        fprintf(stderr, "    SKIP: libref.so not found under $RADIANCE_HOME/kernels\n");
        return;
    }
    MetaKV mk = nb_meta();
    RadBuilder b;
    b.vocab = &vocab;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(stderr_of_rc([&] { return nomicbert::declare(&b, mk.meta(), &ctx); }), RAD_OK);
    for (const auto& op : b.asked_undeclared)
        fprintf(stderr, "    nomicbert asks for '%s'\n", op.c_str());
    CHECK(b.asked_undeclared.empty());
}

/* One architecture and configuration. `pinned` lists the device-only ops it may still ask for on
 * libref's vocabulary: an upper bound that may only shrink. An op libref gains drops out of the
 * expectation by itself; one the plugin stops asking for must be removed here. */
struct ArchCase {
    std::string name;
    std::function<int(RadBuilder*)> declare;
    StepFn step = nullptr;
    int max_spec = 0;
    std::vector<std::string> pinned;
    /* The declared graph's fingerprint with every op declared (graph_hash). */
    uint64_t graph = 0;
};

/* Every architecture in the tree, at the fixtures above. */
struct ArchCases {
    MetaKV q35 = q35fp8_meta(), q35df = with_dflash2(q35fp8_meta()), q35moe = q35moe_meta();
    MetaKV q35w8df = with_dflash2(q35w8a8_meta());
    MetaKV lla = with_dspark(llama_meta()), eg2 = eg2_meta(), qimg = qimg_meta();
    MetaKV q4s = q4_small();
    RadModelMeta q35b = qwen35_meta(), q4 = *q4s.meta(), lfm = lfm2_meta();
    RadModelMeta g4 = gemma4_meta(), g4m = gemma4_moe_meta(), g4df = gemma4_moe_meta();
    std::vector<const char*> g4dk = g4df_keys(), g4dv = g4df_vals();
    RadEncoding w4{}, w4g{}, fp8{};
    std::vector<ArchCase> list;

    ArchCases() {
        g4df.n_kv = (int)g4dk.size(); g4df.kv_key = g4dk.data(); g4df.kv_val = g4dv.data();
        w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
        rad_enc_copy_str(w4.transform, "fwht128");
        w4g = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
        fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
        auto sc = [](int world, int max_spec) { return spec_ctx(0, world, max_spec); };
        auto g4w4 = [this](RadBuilder* b) {
            b->encs = { { "attn_q.", w4g }, { "attn_k.", w4g }, { "attn_v.", w4g },
                        { "attn_o.", w4g }, { "ffn_gate_up.", w4g }, { "ffn_down.", w4g } };
        };
        /* The same with q, k and v as one linear, as a recipe naming attn_qkv converts it. */
        auto g4qkv = [this](RadBuilder* b) {
            b->encs = { { "attn_qkv.", w4g }, { "attn_o.", w4g }, { "ffn_gate_up.", w4g },
                        { "ffn_down.", w4g } };
        };
        /* The ops libref lacks that each still asks for, both two-rank fusions with no chain:
         * qwen35_fp8's all-reduce norm (layer_fold assumes it at two ranks) and Flash-Next's
         * all-reduce into the wide stream. */
        auto add = [&](const char* name, std::function<int(RadBuilder*)> d, StepFn step,
                       int max_spec, std::vector<std::string> pinned, uint64_t graph) {
            list.push_back({ name, std::move(d), step, max_spec, std::move(pinned), graph });
        };
        add("qwen35_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return qwen35_bf16::declare(b, &q35b, &c); },
            qwen35_bf16::step, 0, {}, 0xf82117051023e608ull);
        add("qwen35_fp8 + MTP", [this, sc](RadBuilder* b) {
                auto c = sc(1, 3); return qwen35_fp8::declare(b, q35.meta(), &c); },
            qwen35_fp8::step, 3, {}, 0x958b83102809266bull);
        add("qwen35_fp8 + MTP, two ranks", [this, sc](RadBuilder* b) {
                auto c = sc(2, 3); return qwen35_fp8::declare(b, q35.meta(), &c); },
            nullptr, 3, { "ar_rmsnorm_quant_fp8" }, 0x91408682a38a1754ull);
        add("qwen35_fp8 + DFlash2", [this, sc](RadBuilder* b) {
                auto c = sc(1, 7); return qwen35_fp8::declare(b, q35df.meta(), &c); },
            qwen35_fp8::step, 7, {}, 0x60d33ded99aa451eull);
        add("qwen35_w8a8 + DFlash2", [this, sc](RadBuilder* b) {
                auto c = sc(1, 7); return w8a8_declare(b, q35w8df.meta(), &c); },
            qwen35_w8a8::step, 7, {}, 0x570d3bc8fc15f701ull);
        add("qwen35_w8a8 + DFlash2, two ranks", [this, sc](RadBuilder* b) {
                auto c = sc(2, 7); return w8a8_declare(b, q35w8df.meta(), &c); },
            nullptr, 7, { "ar_rmsnorm_quant_fp8" }, 0x6c5267846652829full);
        add("qwen35moe_fp8 + MTP", [this, sc](RadBuilder* b) {
                auto c = sc(1, 3); return qwen35moe_fp8::declare(b, q35moe.meta(), &c); },
            qwen35moe_fp8::step, 3, {}, 0x325d0299495a0dd8ull);
        add("qwen4exp_fp8", [this](RadBuilder* b) {
                auto c = q4_ctx(0, 1); return qwen4exp_fp8::declare(b, &q4, &c); },
            qwen4exp_fp8::step, 3, {}, 0x0f402d93ca192adbull);
        add("qwen4exp_fp8, two ranks", [this](RadBuilder* b) {
                auto c = q4_ctx(0, 2); return qwen4exp_fp8::declare(b, &q4, &c); },
            nullptr, 3, { "ar_hc_write" }, 0x81339abe032a433bull);
        /* The cache dtype stated rather than left to the plugin's default. The streamed store and
         * the prologue that folds it are declared only where the vocabulary has them, and both are
         * written at the KV group's dtype -- bf16 being the copy-only one, and it is what
         * every case above already gets from the default, the "fp8" in this architecture's name
         * being its trunk's quantisation rather than its cache. Naming it here makes that coverage
         * visible; the fingerprint below is therefore `qwen4exp_fp8`'s exact value, since the two
         * declare one graph, and the test after this table says by name what that graph holds. */
        add("qwen4exp_fp8, bf16 KV", [this](RadBuilder* b) {
                auto c = q4_ctx(0, 1);
                c.kv_dtype = RAD_BF16;
                return qwen4exp_fp8::declare(b, &q4, &c); },
            qwen4exp_fp8::step, 3, {}, 0x0f402d93ca192adbull);
        add("lfm2_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return lfm2_bf16::declare(b, &lfm, &c); },
            lfm2_bf16::step, 0, {}, 0x5bce917918dbd478ull);
        add("gemma4_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return gemma4_bf16::declare(b, &g4, &c); },
            gemma4_bf16::step, 0, {}, 0x2ba2f4943031468full);
        add("gemma4_bf16, w4a8", [this, sc, g4w4](RadBuilder* b) {
                g4w4(b); auto c = sc(1, 0); return gemma4_bf16::declare(b, &g4, &c); },
            gemma4_bf16::step, 0, {}, 0x2e4c6e3235dd3802ull);
        add("gemma4_bf16, w4a8, two ranks", [this, sc, g4w4](RadBuilder* b) {
                g4w4(b); auto c = sc(2, 0); return gemma4_bf16::declare(b, &g4, &c); },
            nullptr, 0, {}, 0x3a160b13b3a80d4eull);
        add("gemma4_bf16, w4a8, fused qkv", [this, sc, g4qkv](RadBuilder* b) {
                g4qkv(b); auto c = sc(1, 0); return gemma4_bf16::declare(b, &g4, &c); },
            gemma4_bf16::step, 0, {}, 0x3913aca59733d0e2ull);
        add("gemma4_bf16, w4a8, fused qkv, two ranks", [this, sc, g4qkv](RadBuilder* b) {
                g4qkv(b); auto c = sc(2, 0); return gemma4_bf16::declare(b, &g4, &c); },
            nullptr, 0, {}, 0x7be4844610b039d8ull);
        add("gemma4", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return gemma4::declare(b, &g4m, &c); },
            gemma4::step, 0, {}, 0xfe8442fba92e5d6dull);
        add("gemma4 + DFlash", [this, sc](RadBuilder* b) {
                auto c = sc(1, 15); return gemma4::declare(b, &g4df, &c); },
            gemma4::step, 15, {}, 0x5d420566673e5a19ull);
        add("embgemma2", [this](RadBuilder* b) {
                auto c = eg2_ctx(); return embgemma2::declare(b, eg2.meta(), &c); },
            embgemma2::step, 0, {}, 0xac72c705bc97d2b3ull);
    }
};

ArchCases& arch_cases() {
    static ArchCases c;
    return c;
}

/* The engine steps a model runs: decode, a prefill chunk, a mixed step, and with a drafter a
 * verify followed by its passes, each run on its own contexts. */
std::vector<std::vector<std::unique_ptr<RadCtx>>> model_steps(RadBuilder& b, StepFn step,
                                                               int max_spec) {
    std::vector<std::vector<std::unique_ptr<RadCtx>>> out;
    for (const auto& k : { kit(b, { 1, 1, 1 }, {}), kit(b, {}, { 24 }), kit(b, { 1 }, { 6 }) })
        out.push_back(run_passes(b, step, { k.get() }));
    if (b.drafter.kind == RAD_DRAFT_NONE) return out;
    const auto ver = kit(b, std::vector<int>(3, 1 + max_spec), {});
    std::vector<std::unique_ptr<Kit>> ps;
    if (b.drafter.kind == RAD_DRAFT_SERIAL) {
        ps.push_back(history_pass(b, *ver));
        for (int r = 2; r <= b.drafter.depth; ++r) ps.push_back(round_pass(b, 3, r));
    } else {
        ps.push_back(context_pass(b, *ver));
        ps.push_back(query_pass(b, 3, b.drafter.block, b.drafter.proposal_pitch));
    }
    std::vector<const Kit*> all = { ver.get() };
    for (const auto& p : ps) all.push_back(p.get());
    out.push_back(run_passes(b, step, all));
    return out;
}

std::string joined(const std::set<std::string>& s) {
    std::string r;
    for (const auto& x : s) r += (r.empty() ? "" : " ") + x;
    return r.empty() ? std::string("(none)") : r;
}

}  /* namespace */

/* For every architecture, declaring with only libref's vocabulary either asks for nothing else and
 * then issues only libref's ops (a CPU-only engine serves it), or asks for exactly the pinned
 * device-only ops libref lacks. A new fused op declared without a chain, or a block that starts
 * calling a device-only op, fails here rather than at a CPU-only engine's startup, where a
 * request for an op no library declares stops the declare. */
TEST(each_architecture_declares_on_librefs_vocabulary_or_names_its_device_only_ops) {
    const std::set<std::string>& vocab = libref_vocab();
    if (vocab.empty()) {
        fprintf(stderr, "    SKIP: libref.so not found under $RADIANCE_HOME/kernels\n");
        return;
    }
    for (const ArchCase& ac : arch_cases().list) {
        RadBuilder b;
        b.vocab = &vocab;
        const int rc = stderr_of_rc([&] { return ac.declare(&b); });
        const std::set<std::string> asked(b.asked_undeclared.begin(), b.asked_undeclared.end());
        std::set<std::string> want;
        for (const auto& op : ac.pinned) {
            if (vocab.count(op))
                fprintf(stderr, "    note: %s: libref now declares '%s'; drop it from the pin\n",
                        ac.name.c_str(), op.c_str());
            else
                want.insert(op);
        }
        if (asked != want)
            fprintf(stderr, "    %s asks for: %s\n    pinned: %s\n", ac.name.c_str(),
                    joined(asked).c_str(), joined(want).c_str());
        CHECK(asked == want);
        if (!asked.empty()) continue;
        CHECK_EQ(rc, RAD_OK);
        if (rc != RAD_OK || !ac.step) continue;
        for (const auto& v : model_steps(b, ac.step, ac.max_spec)) {
            CHECK_PASSES(b, v);
            for (const auto& c : v)
                for (const auto& op : issued_ops(b, *c))
                    if (!vocab.count(op)) {
                        fprintf(stderr, "    %s issues '%s'\n", ac.name.c_str(), op.c_str());
                        CHECK(vocab.count(op));
                    }
        }
    }
}

/* With every op declared, as an engine with libr4d sees it, each architecture declares its pinned
 * graph: the conventional-vocabulary fallbacks are taken only where a fused op is not declared or
 * does not resolve, so they leave these values alone. A snapshot (the fingerprint of the recorded
 * declare), since nothing independent describes a whole graph. */
TEST(with_every_op_declared_each_architecture_declares_its_unchanged_graph) {
    for (const ArchCase& ac : arch_cases().list) {
        RadBuilder b;
        const int rc = stderr_of_rc([&] { return ac.declare(&b); });
        CHECK_EQ(rc, RAD_OK);
        const uint64_t h = graph_hash(b);
        if (h != ac.graph)
            fprintf(stderr, "    %s: graph 0x%016llx, pinned 0x%016llx\n", ac.name.c_str(),
                    (unsigned long long)h, (unsigned long long)ac.graph);
        CHECK_EQ(h, ac.graph);
    }
}

/* What the bf16 KV pin above is worth: a fingerprint that moves says nothing about which ops it
 * covered. The full-attention group must be bf16, and every layer that declares its cache store
 * must have both forms at that dtype -- the streamed store and the prologue that folds it in, which
 * is the one bf16 issues. With the store's row gone from the vocabulary the bf16
 * declare loses both and the hash above moves; this names them, so a bf16 streamed path that
 * silently disappears fails as itself rather than as a number. */
TEST(the_bf16_kv_qwen4exp_graph_declares_the_streamed_store_and_its_fold) {
    const ArchCase* ac = nullptr;
    for (const ArchCase& c : arch_cases().list)
        if (c.name == "qwen4exp_fp8, bf16 KV") ac = &c;
    REQUIRE(ac != nullptr);
    RadBuilder b;
    REQUIRE_EQ(ac->declare(&b), RAD_OK);
    int group = 0, store = 0, folded = 0, hosted = 0;
    for (const auto& g : b.kv_groups)
        if (g.name.find("kv_attn") != std::string::npos) {
            CHECK_EQ((int)g.d.dtype, (int)RAD_BF16);
            ++group;
        }
    for (const auto& o : b.ops) {
        const bool stream = o.op == "kv_store_streamed";
        const bool fold   = o.op == "qk_norm_rope_gate_kv_store_streamed";
        if (o.op == "kv_store") ++hosted;
        if (!stream && !fold) continue;
        for (const auto& p : o.p)
            if (p.key == "kv_dtype") CHECK_EQ(p.sval, std::string("bf16"));
        ++(stream ? store : folded);
    }
    CHECK_EQ(group, 1);
    CHECK_EQ(hosted, 3);              /* two full-attention layers and the MTP head's */
    CHECK_EQ(store, hosted);          /* each one's store has a streamed form ... */
    CHECK_EQ(folded, store);          /* ... and a folded form beside it */
}

/* Where gate_quant_fp8 is not declared, or declared and not resolved, the fp8 gated attention runs
 * the chain it fuses: the per-head sigmoid into a gate buffer of the layer's own, the multiply
 * into the attention output, and quant_act_fp8 into the codes the output projection reads, the
 * three buffers the fused op writes. */
TEST(the_fp8_output_gate_runs_its_chain_where_gate_quant_fp8_is_not_served) {
    MetaKV mk = q35fp8_meta();
    for (int how = 0; how < 2; ++how) {
        RadBuilder b;
        if (how == 0) b.undeclared.insert("gate_quant_fp8");
        else          b.refuse.insert("gate_quant_fp8");
        RadBuildCtx ctx = build_ctx();
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(count_resolved(b, "gate_quant_fp8"), 0);
        CHECK_EQ(count_op(b, "sigmoid"), 2);
        CHECK_EQ(count_op(b, "mul"), 2);
        const rad_buf ax = (rad_buf)buf_named(b, "attn_out");
        const rad_buf aq = (rad_buf)buf_named(b, "attn_out.q");
        const rad_buf as = (rad_buf)buf_named(b, "attn_out.s");
        REQUIRE(ax && aq && as);
        CHECK(find_b(b, "blk.3.attn_gate") != nullptr);
        CHECK(find_b(b, "blk.7.attn_gate") != nullptr);
        const auto k = kit(b, { 1, 1 }, {});
        const auto v = run_passes(b, qwen35_fp8::step, { k.get() });
        CHECK_PASSES(b, v);
        /* attention, then gate, multiply and quantiser, then the out projection, per layer */
        const std::vector<std::string> ops = issued_ops(b, *v[0]);
        int chains = 0;
        for (size_t i = 0; i + 4 < ops.size(); ++i)
            if (ops[i] == "attn_paged") {
                CHECK_EQ(ops[i + 1], std::string("sigmoid"));
                CHECK_EQ(ops[i + 2], std::string("mul"));
                CHECK_EQ(ops[i + 3], std::string("quant_act_fp8"));
                CHECK_EQ(ops[i + 4], std::string("gemm_nt_q"));
                const RecIssue& q = v[0]->issues[i + 3];
                CHECK_EQ(q.opd[0].handle, ax);
                CHECK_EQ(q.opd[1].handle, aq);
                CHECK_EQ(q.opd[2].handle, as);
                ++chains;
            }
        CHECK_EQ(chains, 2);
    }
}

/* Under an int8 trunk the routed arm of rotated experts is int8 too. The connection's read writes
 * the experts' rotated codes as int8 beside its plain int8 codes (`moe_hr.q8`), the gated product
 * is gated_had_quant_i8g into `moe_eff.q8`, and every grouped GEMM is its format's int8 row at one
 * activation scale a 128; no E4M3 plane or quantiser of the arm is declared. */
TEST(qwen4exp_feeds_its_routed_arm_int8_under_an_int8_trunk) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    for (const char* t : { "attn_qg.weight", "qsa_qk.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up_shexp", "ffn_down_shexp" })
        b.encs.push_back({ t, i8 });
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    CHECK(qwen4exp_fp8::g_model[0].arm8);
    const rad_buf hr = (rad_buf)buf_named(b, "moe_hr.q8"), eff = (rad_buf)buf_named(b, "moe_eff.q8");
    REQUIRE(hr != 0);
    REQUIRE(eff != 0);
    int gu = 0, dn = 0, reads_rot = 0;
    for (const RecOp& o : b.ops) {
        if (o.op == "hc_read" && pint(&o, "rotate") == 1) reads_rot += in(o.writes, hr);
        if (o.op.rfind("moe_gemm_q", 0) != 0) continue;
        CHECK_EQ(pstr(&o, "dtype"), std::string("w4a8i8h"));
        CHECK_EQ(pint(&o, "a_group"), 128);
        const bool token = pstr(&o, "a_order") == "token";
        CHECK(in(o.reads, token ? hr : eff));
        (token ? gu : dn) += 1;
    }
    CHECK(gu > 0);
    CHECK_EQ(gu, dn);
    CHECK_EQ(reads_rot, gu);
    CHECK_EQ(count_op(b, "gated_had_quant_i8g"), dn);
    for (const char* op : { "gated_had_quant_fp8", "had_quant_act_fp8", "had_quant_act_i8g" })
        CHECK_EQ(count_op(b, op), 0);
    for (const char* a : { "moe_hr.q", "moe_eff.q" }) CHECK(find_b(b, a) == nullptr);
}

/* Under an int8 routed arm the forms with an int8 row read one input, so each layer and projection
 * takes its own experts' encoding: here layer 1's experts are rotated int8 and layer 2's down
 * projection alone, beside four-bit everywhere else. Each grouped GEMM is declared at its own
 * dtype, and an int8 expert's codes view is int8 beside the four-bit ones'. */
TEST(qwen4exp_takes_each_layers_expert_encoding_under_an_int8_routed_arm) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    RadEncoding e8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(e8.transform, "fwht128");
    const RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    RadBuilder b;
    b.encs = { { "blk.1.ffn_gate_up_exps", e8 }, { "blk.1.ffn_down_exps", e8 },
               { "blk.2.ffn_down_exps", e8 }, { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    for (const char* t : { "attn_qg.weight", "qsa_qk.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up_shexp", "ffn_down_shexp" })
        b.encs.push_back({ t, i8 });
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    CHECK(qwen4exp_fp8::g_model[0].arm8);
    int n8 = 0, n4 = 0;
    for (const RecOp& o : b.ops) {
        if (o.op.rfind("moe_gemm_q", 0) != 0) continue;
        REQUIRE(!o.w.empty());
        const RecWeight& w0 = b.weights[(size_t)o.w[0] - 1];
        const bool down = pstr(&o, "a_order") == "sorted";
        const bool want8 = w0.name.rfind("blk.1.", 0) == 0 || (down && w0.name.rfind("blk.2.", 0) == 0);
        CHECK_EQ(pstr(&o, "dtype"), std::string(want8 ? "i8a8h" : "w4a8i8h"));
        CHECK_EQ(w0.d.dtype, want8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_I4);
        (want8 ? n8 : n4) += 1;
    }
    CHECK_EQ(n8, 3);
    CHECK_EQ(n4, 2 * 9 - 3);
    /* An expert whose encoding is not its layer's is still refused. */
    RadBuilder b2;
    b2.encs = { { "blk.1.ffn_gate_up_exps.3.", e8 }, { "ffn_gate_up_exps", w4 },
                { "ffn_down_exps", w4 } };
    for (const char* t : { "attn_qg.weight", "qsa_qk.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up_shexp", "ffn_down_shexp" })
        b2.encs.push_back({ t, i8 });
    RadBuildCtx ctx2 = q4_ctx(0, 1);
    CHECK_EQ(qwen4exp_fp8::declare(&b2, &meta, &ctx2), RAD_E_DTYPE);
}

/* --calib-dir over an int8 trunk records two inputs a layer from the codes its GEMMs read (the
 * mixer's input and its output projection's), named by the container's weights, at one rank; over
 * two ranks the codes are a rank's part and the declare refuses. */
TEST(qwen4exp_records_the_trunks_inputs_from_its_int8_codes_at_one_rank) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    auto encs = [&](RadBuilder& b) {
        b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
        for (const char* t : { "attn_qg.weight", "qsa_qk.weight", "attn_k.weight", "attn_v.weight",
                               "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                               "ffn_gate_up_shexp", "ffn_down_shexp" })
            b.encs.push_back({ t, i8 });
    };
    RadBuilder b;
    encs(b);
    RadBuildCtx ctx = q4_ctx(0, 1);
    ctx.calib_dir = "/nonexistent/calib";
    ctx.calib_tokens = 4096;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    const auto& m = qwen4exp_fp8::g_model[0];
    REQUIRE(m.calib.on);
    CHECK_EQ(m.calib.sites.size(), 2 * m.layers.size());
    int grams = 0;
    for (const RecOp& o : b.ops)
        if (o.op == "gram_accum") {
            CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8"));
            ++grams;
        }
    CHECK_EQ(grams, (int)(2 * m.layers.size()));
    for (size_t l = 0; l < m.layers.size(); ++l) {
        const auto& lay = m.layers[l];
        REQUIRE(lay.cs_in >= 0);
        REQUIRE(lay.cs_o >= 0);
        const std::string L = "blk." + std::to_string(l);
        CHECK_EQ(m.calib.sites[(size_t)lay.cs_in].linears,
                 lay.full ? L + ".attn_qg," + L + ".attn_k," + L + ".attn_v" : L + ".ssm_inz");
        CHECK_EQ(m.calib.sites[(size_t)lay.cs_o].linears, L + (lay.full ? ".attn_output" : ".ssm_out"));
        CHECK_EQ(m.calib.sites[(size_t)lay.cs_in].domain, 0u);
    }
    /* A calibration run issues its taps in the serial step. */
    RadBuilder b2;
    encs(b2);
    RadBuildCtx ctx2 = q4_ctx(0, 2);
    ctx2.calib_dir = "/nonexistent/calib";
    ctx2.calib_tokens = 4096;
    CHECK_EQ(qwen4exp_fp8::declare(&b2, &meta, &ctx2), RAD_E_UNSUPPORTED);
}

/* A trunk stored rotated (int8 /fwht128) and the routed experts read one pair: each connection's
 * read writes the block input's rotated codes as `x`'s own int8 pair and its plain codes into a
 * pair of its own that nothing else touches, no `moe_hr` copy is declared, the trunk's projections
 * are i8a8h, and the routed gate_up reads `x.q8`. */
TEST(qwen4exp_on_a_rotated_trunk_shares_the_block_inputs_rotated_codes) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    RadEncoding r8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(r8.transform, "fwht128");
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    for (const char* t : { "attn_qg.weight", "qsa_qk.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up_shexp", "ffn_down_shexp" })
        b.encs.push_back({ t, r8 });
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    CHECK(qwen4exp_fp8::g_model[0].trunk_rot);
    CHECK(qwen4exp_fp8::g_model[0].arm8);
    const rad_buf xq = (rad_buf)buf_named(b, "x.q8");
    REQUIRE(xq != 0);
    CHECK(find_b(b, "moe_hr.q8") == nullptr);
    int reads = 0, gu = 0;
    for (const RecOp& o : b.ops) {
        if (o.op == "hc_read" && pint(&o, "group") == 128) {
            ++reads;
            CHECK_EQ(pint(&o, "rotate"), 1);
            /* x, inj, the plain pair, then the rotated pair: `x`'s own. */
            REQUIRE_EQ(o.writes.size(), (size_t)6);
            CHECK_EQ(o.writes[4], xq);
            CHECK(o.writes[2] != xq);
            CHECK(b.bufs[(size_t)o.writes[2] - 1].name.find(".plain.q8") != std::string::npos);
        }
        if (o.op == "gemm_nt_q") CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
        if (o.op.rfind("moe_gemm_q", 0) == 0 && pstr(&o, "a_order") == "token") {
            ++gu;
            CHECK(in(o.reads, xq));
        }
    }
    CHECK_EQ(reads, 2 * 9);
    CHECK_EQ(gu, 9);
    /* No unrotated quantiser, and the only rotating one is the delta net's output on the rows its
     * decode fold does not write. */
    CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
    const rad_buf go = (rad_buf)buf_named(b, "gdn_o");
    for (const RecOp& o : b.ops)
        if (o.op == "had_quant_act_i8g") CHECK(in(o.reads, go));
    const auto k = kit(b, { 1, 1 }, { 6 });
    const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
    CHECK_PASSES(b, v);
}

/* On a rotated trunk the sparse attention's merge gates and quantises too: one
 * attn_paged_gate_had_quant a full-attention layer for a chunk after a long history, writing the
 * out projection's rotated int8 pair, and no attn_paged_gate_quant, whose codes are unrotated. */
TEST(qwen4exp_on_a_rotated_trunk_gates_and_quantises_in_the_sparse_merge) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    RadEncoding r8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(r8.transform, "fwht128");
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    for (const char* t : { "attn_qg.weight", "qsa_qk.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up_shexp", "ffn_down_shexp" })
        b.encs.push_back({ t, r8 });
    RadBuildCtx ctx = q4_ctx(0, 1, 4);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    REQUIRE(qwen4exp_fp8::g_model[0].trunk_rot);
    CHECK_EQ(count_op(b, "attn_paged_gate_quant"), 0);
    const rad_buf oq = (rad_buf)buf_named(b, "attn_out.q8");
    REQUIRE(oq != 0);
    const auto k = kit(b, {}, { 1024 }, 9000);
    const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
    CHECK_PASSES(b, v);
    const auto at = issues_of(b, *v[0], "attn_paged_gate_had_quant");
    REQUIRE_EQ(at.size(), (size_t)2);                    /* two full-attention layers */
    for (const RecIssue* i : at) {
        CHECK_EQ(i->n, 1);
        CHECK_EQ(i->opd[0].rows, 1024);
        const RecOp* o = b.op_of(i->op);
        REQUIRE(o != nullptr);
        CHECK(in(o->writes, oq));
    }
    CHECK(issues_of(b, *v[0], "gate_had_quant_i8g").empty());
}

/* Where gated_had_quant_fp8 is not served, a four-bit routed down projection's input is made by
 * silu_mul into the bf16 plane and had_quant_act_fp8 from it, into the rotated codes the fused op
 * writes. */
TEST(the_rotated_expert_gate_runs_its_chain_where_gated_had_quant_fp8_is_not_served) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    const RadEncoding w4 = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                rad_enc_copy_str(e.transform, "fwht128"); return e; }();
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    b.undeclared.insert("gated_had_quant_fp8");
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    CHECK(b.asked_undeclared.empty());
    CHECK_EQ(count_op(b, "gated_had_quant_fp8"), 0);
    /* One a routed layer (eight trunk layers and the MTP head), from the product into the down
     * projection's codes. */
    const rad_buf eff = (rad_buf)buf_named(b, "moe_eff"), effq = (rad_buf)buf_named(b, "moe_eff.q");
    int chained = 0;
    for (const auto& o : b.ops)
        if (o.op == "had_quant_act_fp8" && in(o.reads, eff)) {
            ++chained;
            CHECK_EQ(pint(&o, "group"), 128);
            CHECK(in(o.writes, effq));
        }
    CHECK_EQ(chained, 9);
    const auto k = kit(b, { 1, 1 }, { 6 });
    const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
    CHECK_PASSES(b, v);
    const std::vector<std::string> ops = issued_ops(b, *v[0]);
    int pairs = 0;
    for (size_t i = 0; i + 1 < ops.size(); ++i)
        pairs += ops[i] == "silu_mul" && ops[i + 1] == "had_quant_act_fp8";
    CHECK(pairs >= 8);
}

/* What each plugin's probe tells the engine before declare: a drafter's depth (an MTP head's
 * operating point, a block drafter's trained block, fixed), an encoder's pooling window and audio
 * budget, a generator's widest pass. */
TEST(the_probes_size_what_declare_needs) {
    {   /* the routed and dense fp8 plugins share the MTP head's depth of 3, not fixed */
        MetaKV moe = q35moe_meta();
        RadArchProbe p{};
        CHECK_OK(qwen35moe_fp8::probe(moe.meta(), &p));
        CHECK_EQ(p.draft_depth, 3);
        CHECK_EQ(p.draft_depth_fixed, 0);
        MetaKV none = q35moe_meta();
        none.drop("qwen35.nextn_predict_layers");
        RadArchProbe q{};
        CHECK_OK(qwen35moe_fp8::probe(none.meta(), &q));
        CHECK_EQ(q.draft_depth, 0);
    }
    {   /* the dense llama build reads DSpark's trained block like llama_fp8 */
        MetaKV d = with_dspark(llama_meta(""));
        RadArchProbe p{};
        CHECK_OK(llama_dense_fp8::probe(d.meta(), &p));
        CHECK_EQ(p.draft_depth, 7);
        CHECK_EQ(p.draft_depth_fixed, 1);
    }
    {   /* the vision tower pools 3 x 3; 30 s of audio at a 10 ms hop is 3000 frames, whole rows of
         * four */
        MetaKV e = eg2_meta();
        RadArchProbe p{};
        CHECK_OK(embgemma2::probe(e.meta(), &p));
        CHECK_EQ(p.enc_merge, 9);
        CHECK_EQ(p.audio_frames, 3000);
    }
    {   /* two entries (an image and its guidance) of the largest image's 16800 latent rows; ten
         * references of 1024 x 1024 pixels at 16 x 16 a patch, with a tenth for rounding */
        MetaKV q = qimg_meta();
        RadArchProbe p{};
        CHECK_OK(qwenimage21::probe(q.meta(), &p));
        CHECK_EQ(p.gen_rows, 2 * 16800);
        CHECK_EQ(p.gen_patches, 10 * 4096 * 11 / 10);
    }
    {   /* Anima: an image and its guidance at the largest size, each with its 512 context rows;
         * no reference patches */
        MetaKV a = anima_meta("aesthetic");
        RadArchProbe p{};
        CHECK_OK(anima::probe(a.meta(), &p));
        CHECK_EQ(p.gen_rows, 2 * (9216 + 512));
        CHECK_EQ(p.gen_patches, 0);
    }
}

/* ==================================================================== rad_arch.h's helpers */

/* The vocabulary split, swept: every rank's slice starts where the previous ends, the slices sum
 * to the vocabulary, a split that divides is even, and one that does not cuts only at whole
 * 128-row tiles (a block-fp8 lm_head's tile), the remainder going to the last rank with rows. A
 * rank can be left empty (301 rows over four ranks: 128, 128, 45, 0), and geom_from refuses that
 * world rather than declare a zero-row head. */
TEST(the_vocabulary_split_tiles_the_vocabulary) {
    int bad = 0;
    for (int64_t total : { 1, 7, 127, 128, 129, 300, 301, 1000, 32000, 151669, 248320, 262144 })
        for (int world = 1; world <= 8; ++world) {
            int64_t at = 0;
            for (int r = 0; r < world; ++r) {
                const int64_t lo = rad::arch::shard_begin(total, world, r);
                const int64_t n = rad::arch::shard_extent(total, world, r);
                const bool ok = lo == at && n >= 0 && lo + n <= total &&
                                (total % world == 0 ? n == total / world
                                                    : (n % 128 == 0 || lo + n == total));
                if (!ok) {
                    ++bad;
                    fprintf(stderr, "    total %lld world %d rank %d: [%lld, +%lld)\n",
                            (long long)total, world, r, (long long)lo, (long long)n);
                }
                at = lo + n;
            }
            if (at != total) ++bad;
        }
    CHECK_EQ(bad, 0);
    CHECK_EQ(rad::arch::shard_extent(301, 4, 2), 45);
    CHECK_EQ(rad::arch::shard_extent(301, 4, 3), 0);
    CHECK_EQ(rad::arch::shard_extent(300, 4, 3), 75);
    CHECK_EQ(rad::arch::shard_extent(248320, 3, 2), 82688);
    rad::arch::Geom g{};
    RadModelMeta m = lfm2_meta();
    m.n_vocab = 301;
    m.n_head = 4; m.n_head_kv = 4;
    RadBuildCtx c = build_ctx(3, 4);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::geom_from(g, &m, &c, "bf16", RAD_BF16); }),
             RAD_E_INVAL);
}

/* A layer schedule: names map to full (1) or linear (0), an unknown name is refused by its
 * index, a list shorter than the trunk is refused, and entries past the trunk (an MTP block's)
 * are not compared. */
TEST(the_layer_schedule_readers) {
    std::vector<int> k;
    CHECK_OK(rad::arch::parse_layer_kinds("t", "[\"a\", \"b\", \"a\"]", { "b" }, "a", &k));
    CHECK(k == (std::vector<int>{ 0, 1, 0 }));
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::parse_layer_kinds("t", "a c", { "b" }, "a", &k); }),
             RAD_E_UNSUPPORTED);
    CHECK(rad::arch::is_full_attention(3, 4) && rad::arch::is_full_attention(7, 4));
    CHECK(!rad::arch::is_full_attention(4, 4) && !rad::arch::is_full_attention(3, 0));
    const char* lin = "linear_attention", * ful = "full_attention";
    const std::string four = std::string(lin) + " " + lin + " " + lin + " " + ful;
    CHECK_OK(rad::arch::check_layer_schedule("t", four.c_str(), 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", (four + " " + ful).c_str(), 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", nullptr, 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", "", 4, 4, { ful }));
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::check_layer_schedule("t", four.c_str(), 8, 4, { ful }); }),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::check_layer_schedule("t", four.c_str(), 4, 2, { ful }); }),
             RAD_E_UNSUPPORTED);
}

/* A drafter's taps: out-of-range ids are dropped, not clamped; parsing stops at the first token
 * that is not a number; at most `max` are kept. */
TEST(parse_layer_ids_keeps_the_trunks_layers_in_order) {
    int out[4] = {};
    CHECK_EQ(rad::arch::parse_layer_ids("1 10 20 30 39", 40, out, 4), 4);
    CHECK(out[0] == 1 && out[1] == 10 && out[2] == 20 && out[3] == 30);
    CHECK_EQ(rad::arch::parse_layer_ids("[3, 64, -1, 5]", 40, out, 4), 2);
    CHECK(out[0] == 3 && out[1] == 5);
    CHECK_EQ(rad::arch::parse_layer_ids("2 x 5", 40, out, 4), 1);
    CHECK_EQ(rad::arch::parse_layer_ids(nullptr, 40, out, 4), 0);
    CHECK_EQ(rad::arch::parse_layer_ids("", 40, out, 4), 0);
}

/* The remainder order is a permutation of the ranks for every rotation: attention-zero ranks
 * take places 0.. first, then the attention ranks rotated by `parity`. */
TEST(remainder_place_orders_every_rank_once) {
    for (int world : { 1, 2, 3, 4, 6 })
        for (int attn_world : { world, world - 1 }) {
            if (attn_world < 1) continue;
            rad::arch::Geom g{};
            g.world = world;
            g.attn_world = attn_world;
            for (int parity = 0; parity < world; ++parity) {
                std::set<int64_t> seen;
                for (int r = 0; r < world; ++r) {
                    const int64_t p = rad::arch::remainder_place(g, r, parity);
                    seen.insert(p);
                    if (r >= attn_world) CHECK_EQ(p, (int64_t)(r - attn_world));
                }
                CHECK_EQ((int)seen.size(), world);
                CHECK_EQ(*seen.begin(), 0);
                CHECK_EQ(*seen.rbegin(), (int64_t)(world - 1));
            }
        }
}

/* divide_or_fail refuses rather than rounds, a world of zero included; batch_split takes a pure
 * step's split from its phase and a mixed step's from its fields. */
TEST(divide_or_fail_and_batch_split) {
    int64_t q = -1;
    CHECK_OK(rad::arch::divide_or_fail(48, 4, "x", &q));
    CHECK_EQ(q, 12);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::divide_or_fail(48, 5, "x", &q); }), RAD_E_INVAL);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::divide_or_fail(48, 0, "x", &q); }), RAD_E_INVAL);
    RadBatch b{};
    b.n_seq = 5; b.n_tok = 40; b.n_seq_decode = 2; b.n_tok_decode = 7;
    int64_t s = -1, t = -1;
    b.phase = RAD_PHASE_PREFILL; rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 0 && t == 0);
    b.phase = RAD_PHASE_DECODE;  rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 5 && t == 40);
    b.phase = RAD_PHASE_MIXED;   rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 2 && t == 7);
}

/* Gemma 4's assistant head keeps no cache of its own: its history pass is one row per drafting
 * sequence, gathered from the trunk's verify rows, and each later round one row per sequence. The
 * trunk's verify and the head's passes, linted as the one engine step they are, at one and two
 * ranks. */
TEST(gemma4_assistant_head_engine_step_lints_clean) {
    RadModelMeta meta = gemma4_draft_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = spec_ctx(rank, world, 3);
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            auto his = kit(b, { 1, 1, 1, 1 }, {});
            his->b.draft_pass = -1;
            his->out = { 3, 7, 11, 15 };               /* each sequence's last verify row */
            his->b.out_ids = his->out.data();
            his->b.n_out = 4;
            his->dout = { 0, 1, 2, 3 };
            his->b.draft_out_ids = his->dout.data();
            his->b.n_draft_out = 4;
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, gemma4_bf16::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, rank, world);
            CHECK_PASSES(b, v);
        }
}

/* gemma4_bf16 asks for its fused prologue and int8 junctions only where a library declares them:
 * on libref's and libavx's vocabulary a w4a8 (compressed-tensors) model declares the unfused
 * chains and asks for no op outside it, at one rank and at two, issuing what it issues when the
 * fused ops are declared and do not resolve. With them declared nothing changes (the graph
 * fingerprints above). */
TEST(gemma4_bf16_without_the_fused_ops_declares_the_chains_and_asks_for_no_undeclared_op) {
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    const std::set<std::string> fused = { "qkv_norm_rope", "qkv_norm_rope_kv_store", "rmsnorm_quant_i8",
                                          "gelu_mul_quant_i8", "ar_rmsnorm_quant_i8" };
    for (int world : { 1, 2 }) {
        std::vector<std::string> seq[2];
        for (int undeclared = 0; undeclared < 2; ++undeclared) {
            RadBuilder b;
            b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
                       { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
            (undeclared ? b.undeclared : b.refuse) = fused;
            RadModelMeta meta = gemma4_meta();
            RadBuildCtx ctx = build_ctx(0, world);
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            CHECK(b.asked_undeclared.empty());
            for (const auto& op : fused) CHECK_EQ(count_resolved(b, op), 0);
            const auto k = kit(b, { 1, 1, 1, 1 }, {});
            const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, 0, world);
            CHECK_PASSES(b, v);
            seq[undeclared] = issued_ops(b, *v[0]);
        }
        CHECK_EQ(join(seq[1]), join(seq[0]));
        CHECK(count_issued(seq[1], "quant_act_i8") > 0);
    }
}

/* ==================================================================== qwen3asr */

/* Qwen3-ASR 1.7B's shape, its decoder and its audio tower each cut to TWO layers. */
const char* k_asr_keys[] = {
    "audio_config.model_type", "audio_config.d_model", "audio_config.encoder_attention_heads",
    "audio_config.encoder_ffn_dim", "audio_config.downsample_hidden_size",
    "audio_config.num_mel_bins", "audio_config.encoder_layers", "audio_config.n_window",
    "audio_config.n_window_infer", "audio_config.output_dim", "audio_config.num_key_value_heads",
    "audio_config.activation_function", "audio_token_id",
};
const char* k_asr_vals[] = { "qwen3_asr_encoder", "1024", "16", "4096", "480", "128", "2", "50",
                             "800", "2048", "16", "gelu", "151676" };

RadModelMeta qwen3asr_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen3asr";
    m.name        = "test-qwen3-asr-1.7b";
    m.quant       = "bf16";
    m.n_layers    = 2;
    m.n_embd      = 2048;
    m.n_head      = 16;
    m.n_head_kv   = 8;
    m.head_dim    = 128;
    m.n_ff        = 6144;
    m.n_vocab     = 151936;
    m.n_ctx_train = 65536;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_asr_keys) / sizeof(k_asr_keys[0]));
    m.kv_key      = k_asr_keys;
    m.kv_val      = k_asr_vals;
    return m;
}

/* The decoder is plain Qwen3 and the tower is declared as the audio encoder: GQA 2 paged attention
 * with NeoX rotary over the whole head and a tied head; the two 480-channel convolutions as unfold
 * plus GEMM against the checkpoint's kernels read as matrices; positions a second at a time and
 * attention in blocks of 104 rows read at cu_div 8. */
TEST(qwen3asr_declares_a_qwen3_decoder_and_its_audio_tower) {
    RadModelMeta meta = qwen3asr_meta();
    RadArchProbe pb{};
    CHECK_OK(qwen3asr_bf16::probe(&meta, &pb));
    CHECK_EQ(pb.audio_frames, (int64_t)3120);

    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = pb.audio_frames;
    CHECK_OK(qwen3asr_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.audio_encoder_name, std::string("audio"));
    CHECK_EQ(b.audio_encoder.modalities, (uint32_t)RAD_MM_AUDIO);
    CHECK_EQ(b.audio_encoder.patch_dim, 128);
    CHECK_EQ(b.audio_encoder.merge, 8);
    CHECK_EQ(b.audio_encoder.max_patches, 3120);
    CHECK_EQ(b.audio_encoder.n_embd, 2048);

    for (const auto& o : b.ops)
        if (o.op == "attn_paged") {
            CHECK_EQ(pint(&o, "gqa"), 2);
            CHECK_EQ(pint(&o, "head_dim"), 128);
            CHECK_EQ(pint(&o, "causal"), 1);
        }
    int rope = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope") {
            ++rope;
            CHECK_EQ(pstr(&o, "mode"), std::string("neox"));
            CHECK_EQ(pint(&o, "rotary_dim"), 128);
        }
    CHECK_EQ(rope, 4);
    const RecMap* head = find_map(b, "output.weight");
    REQUIRE(head != nullptr);
    REQUIRE_EQ(head->src.size(), (size_t)2);
    CHECK_EQ(head->src[1], std::string("model.language_model.embed_tokens.weight"));

    CHECK_EQ(count_op(b, "conv2d"), 1);
    CHECK_EQ(count_op(b, "conv2d_unfold"), 3);
    CHECK_EQ(count_op(b, "seg_sinusoid"), 1);
    const RecWeight* c2 = find_w(b, "a.conv2.w");
    REQUIRE(c2 != nullptr);
    CHECK_EQ(c2->d.rank, 2u);
    CHECK_EQ(c2->d.shape[0], 480);
    CHECK_EQ(c2->d.shape[1], 4320);
    const RecMap* qkv = find_map(b, "a.blk.1.qkv.w");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)3);
    CHECK_EQ(qkv->src[2], std::string("model.audio_tower.layers.1.self_attn.v_proj.weight"));
    int dense = 0;
    for (const auto& o : b.ops)
        if (o.op == "attn_dense") {
            ++dense;
            CHECK_EQ(pint(&o, "block"), 104);
            CHECK_EQ(pint(&o, "cu_div"), 8);
            CHECK_EQ(pint(&o, "head_dim"), 64);
        }
    CHECK_EQ(dense, 2);
    for (const auto& o : b.ops)
        if (o.op == "conv2d_unfold" || o.op == "conv2d")
            CHECK_EQ(pint(&o, "t_last"), pint(&o, "kh") == 3 ? 1 : 0);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* How it hears: the clip's markers and a transcription turn with every slot. */
    const RadAudioFormat* f = qwen3asr_bf16::audio_format(&meta);
    REQUIRE(f != nullptr);
    CHECK_EQ(f->struct_size, (uint32_t)sizeof(RadAudioFormat));
    CHECK_EQ(std::string(f->audio_open), std::string("<|audio_start|>"));
    CHECK(std::string(f->transcribe_prompt).find("{audio}") != std::string::npos);
    CHECK(std::string(f->transcribe_prompt).find("{context}") != std::string::npos);
    CHECK_EQ(f->n_languages, 30u);
}

/* An audio pass runs the tower, in order, and reads the core's planes: the first convolution and the
 * unfolds take plane 1 as their mask, the positions and the attention the pass's clip bounds. */
TEST(a_qwen3asr_audio_pass_issues_the_tower_and_nothing_else) {
    RadModelMeta meta = qwen3asr_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = 3120;
    CHECK_OK(qwen3asr_bf16::declare(&b, &meta, &ctx));

    /* One clip of two whole seconds: 208 staged frames, 26 rows. */
    static uint16_t frames[208 * 128];
    static int32_t  coord[4 * 208];
    static const int32_t cu[] = { 0, 208 };
    RadBatch batch{};
    batch.enc         = RAD_MM_AUDIO;
    batch.enc_n_patch = 208;
    batch.enc_pixels  = frames;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 1;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 208;
    RadCtx c; c.b = &b; c.rank = 0; c.world = 1; c.batch = &batch;
    qwen3asr_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());

    std::vector<std::string> want = { "conv2d", "conv2d_unfold", "gemm_nt_bias", "conv2d_unfold",
                                      "gemm_nt_bias", "conv2d_unfold", "gemm_nt", "seg_sinusoid",
                                      "add" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    const RecIssue* c1 = issue_of(b, c, "conv2d");
    REQUIRE(c1 != nullptr);
    CHECK_EQ(c1->n, 208);
    CHECK(c1->opd[3].raw == coord + 208);
    /* The second convolution's GEMM runs over 52 time rows of 32 positions. */
    const RecIssue* g2 = issue_of(b, c, "gemm_nt_bias");
    REQUIRE(g2 != nullptr);
    CHECK_EQ(g2->n, 52 * 32);
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    CHECK_EQ(at->n, 26);
    CHECK(at->opd[3].raw == cu);
    CHECK_EQ(c.issues.back().n, 26);
}

/* What this tower does not carry is refused by name. */
TEST(qwen3asr_refuses_a_tower_it_does_not_serve) {
    static const char* vals[] = { "qwen3_asr_encoder", "1024", "16", "4096", "480", "128", "2",
                                  "50", "800", "2048", "16", "silu", "151676" };
    RadModelMeta meta = qwen3asr_meta();
    meta.kv_val = vals;
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = 3120;
    CHECK_EQ(qwen3asr_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    /* No audio budget: the decoder alone, and no tower. */
    RadModelMeta ok = qwen3asr_meta();
    RadBuilder bt;
    RadBuildCtx text = build_ctx();
    CHECK_OK(qwen3asr_bf16::declare(&bt, &ok, &text));
    CHECK(bt.audio_encoder_name.empty());
    CHECK_EQ(count_op(bt, "conv2d"), 0);
}

/* A control vector's projection is declared for the layers whose scale is nonzero, and issued only
 * on a pass where some row takes the vector (RadBatch::control_vector): with no such row every
 * row would be skipped, so the pass is the one a deployment without a vector runs. */
TEST(the_control_vector_projection_is_issued_only_when_a_row_takes_it) {
    using namespace rad::arch;
    RadBuilder b;
    Geom g{};
    g.n_layer = 4; g.n_embd = 64; g.max_tok = 16;
    std::vector<float> dir((size_t)(4 * 64), 0.0f);
    float scales[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
    const RadControlVector cv{ dir.data(), scales, 64, 4 };
    RadBufDecl hd{};
    hd.dtype = RAD_BF16; hd.rank = 2; hd.shape[0] = 16; hd.shape[1] = 4 * 64;
    const rad_buf h = rad_decl_buffer(&b, "h", &hd);
    const rad_buf mask = HcProject::declare_mask(&b, "on", g, &cv);
    REQUIRE(mask != 0);
    HcProject p[4];
    for (int l = 0; l < 4; ++l) CHECK_OK(p[l].declare(&b, "proj", g, l, &cv, mask, h, 4));
    CHECK(!p[0].steered());
    CHECK(p[1].steered());
    CHECK(!p[2].steered());
    CHECK(p[3].steered());
    CHECK_EQ(count_op(b, "hc_project"), 2);

    RadBatch none{};
    none.n_tok = 3;
    RadCtx c{ .b = &b, .batch = &none };
    for (const HcProject& x : p) x.step(&c, &none, 0, 3);
    CHECK_EQ(c.issues.size(), (size_t)0);
    RadBatch some = none;
    some.control_vector = 1;
    c.batch = &some;
    for (const HcProject& x : p) x.step(&c, &some, 0, 3);
    CHECK_EQ(c.issues.size(), (size_t)2);

    /* No vector: no mask, no op. */
    RadBuilder b0;
    CHECK_EQ(HcProject::declare_mask(&b0, "on", g, nullptr), (rad_buf)0);
    HcProject q;
    CHECK_OK(q.declare(&b0, "proj", g, 1, nullptr, 0, h, 4));
    CHECK(!q.steered());
}

RAD_TEST_MAIN()

/* Without --prefill-span a 4096 budget declares and steps the pinned graph and launches: for
 * Flash-Next and the other plugins built on the same fp8 blocks, at one, two and four ranks, taken
 * from the plugins as they were before the span. */
TEST(a_4096_budget_without_a_span_is_the_pinned_graph_and_steps) {
    MetaKV q4s = q4_small(), q35 = q35fp8_meta(), q35moe = q35moe_meta();
    struct Case {
        const char* name;
        int world, rank;
        uint64_t graph, steps;
    };
    const Case cases[] = {
        /* Flash-Next's hashes. The fingerprint's decode step, verify, history pass and round are
         * below the indexer's exactness bound, where the indexer issues no query, score or select
         * and a one-row sequence attends over its block table (docs/QSA.md, "Below the bound").
         * The decode step's issue order is in them too (qwen4exp_fp8.cpp, kVerifyRows;
         * a_flash_next_decode_step_orders_its_small_launches pins it): in the graphs, the
         * indexer's within-step buffers kept out of the planner's sharing and, past one rank,
         * the router's weight read and each read's verify-height twin; in the steps, k and v in
         * front of q|gate, the indexer's keys behind the attention's launches and, at the verify,
         * the shared down projection in front of the routed one. Prefill and mixed steps, past
         * the bound, issue everything a layer in declared order. */
        { "qwen4exp_fp8", 1, 0, 0xe733184739f30fd2ull, 0x816cd78a65a92598ull },
        { "qwen4exp_fp8", 2, 0, 0x92921730d6df8e8full, 0x3b5931abdde2195dull },
        { "qwen4exp_fp8", 2, 1, 0x9c9352c2e075cb44ull, 0x7cb5889253d852cdull },
        { "qwen4exp_fp8", 4, 0, 0x74b4bf4e70c955e5ull, 0x4292bbe719b32ed1ull },
        { "qwen4exp_fp8", 4, 3, 0xc36df9e24f272283ull, 0x9c08be223ae002dbull },
        { "qwen35_fp8", 1, 0, 0x0f08ba5027cba71aull, 0xa7ec3cf6252cd441ull },
        { "qwen35_fp8", 2, 1, 0xe7780f241685a18bull, 0xdd5d65ef158c4136ull },
        { "qwen35moe_fp8", 1, 0, 0xb1e6173bd121dc2aull, 0x2b5989b2a6dbc7dfull },
        { "qwen35moe_fp8", 2, 0, 0x46d061a0de9a8933ull, 0x4a597c66b5868eeeull },
    };
    for (const Case& cs : cases) {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(cs.rank, cs.world, 3);
        ctx.max_tok = 4096;
        ctx.max_seqs = 8;
        ctx.max_ctx = 200000;
        StepFn step = nullptr;
        int rc = RAD_E_INVAL;
        if (!std::strcmp(cs.name, "qwen4exp_fp8")) {
            b.encs = { { "draft_head", u2_head() } };
            rc = stderr_of_rc([&] { return qwen4exp_fp8::declare(&b, q4s.meta(), &ctx); });
            step = qwen4exp_fp8::step;
        } else if (!std::strcmp(cs.name, "qwen35_fp8")) {
            rc = stderr_of_rc([&] { return qwen35_fp8::declare(&b, q35.meta(), &ctx); });
            step = qwen35_fp8::step;
        } else {
            rc = stderr_of_rc([&] { return qwen35moe_fp8::declare(&b, q35moe.meta(), &ctx); });
            step = qwen35moe_fp8::step;
        }
        REQUIRE_EQ(rc, RAD_OK);
        const uint64_t g = graph_hash(b);
        const uint64_t s = pinned_steps(b, step, cs.rank, cs.world);
        fprintf(stderr, "    pin: %s world %d rank %d: graph 0x%016llxull, steps 0x%016llxull\n",
                cs.name, cs.world, cs.rank, (unsigned long long)g, (unsigned long long)s);
        CHECK_EQ(g, cs.graph);
        CHECK_EQ(s, cs.steps);
    }
}

/* qwen35_w8a8 at two ranks where libr4d serves the `_had_quant_i8g` norms (a_h.q8_fed): the norms write
 * rotated int8 codes, which the pipelined prefill's boundary norms (NormQuantFP8::step_plain) do not issue,
 * so the prefill stays serial there: the step whole, nothing on the second lane, no step failure. Where
 * the forms are not served each projection quantises its own input and the pipeline runs as on
 * qwen35_fp8. A pipeline over the int8 codes would hand the int8 op the boundary norm's E4M3 codes,
 * which libr4d refuses, failing every such step. */
TEST(qwen35_w8a8_with_int8_codes_from_the_norms_prefills_serially) {
    MetaKV mk = with_dflash2(q35w8a8_meta());
    for (bool fused : { true, false })
        for (int rank = 0; rank < 2; ++rank) {
            RadBuilder b;
            if (!fused)
                for (const char* op : kW8a8Fused) b.undeclared.insert(op);
            w8a8_encs(&b);
            RadBuildCtx ctx = spec_ctx(rank, 2, 7);
            ctx.max_tok = 2048;
            REQUIRE_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                       RAD_OK);
            const qwen35_w8a8::Model& m = qwen35_w8a8::g_model[rank];
            CHECK_EQ(m.a_h.q8_fed, fused);
            CHECK_EQ(m.pipe_on, !fused);
            const auto pf = kit(b, {}, { 2048 }, 0);
            const auto v = run_passes(b, qwen35_w8a8::step, { pf.get() }, rank, 2);
            CHECK_PASSES(b, v);
            const RadCtx& c = *v[0];
            CHECK(c.step_fail.empty());
            int lane1 = 0;
            for (const RecIssue& i : c.issues) lane1 += i.lane == 1;
            if (fused) CHECK_EQ(lane1, 0);
            else CHECK(lane1 > 0);
        }
}

/* A Qwen3.5/3.8 prefill step of one sequence at two ranks runs in two pieces of 1024 rows
 * (step_pipelined): every collective is on the second lane, the layers' all-reduces a piece at a
 * time at `overlap` 1, the embedding's whole (standalone, or folded into layer 0's norm where the
 * wire is exact, as this geometry's is); no other fused all-reduce + norm is issued; the delta net
 * and the attention run twice, each piece over its own bounds; the DFlash2 taps are copied a piece
 * at a time. A prompt's first chunk and a later one alike. One rank runs serially. */
TEST(qwen35_a_wide_prefill_runs_in_two_pieces_with_its_boundaries_on_the_second_lane) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank)
            for (int32_t pctx : { 0, 4096 }) {
                RadBuilder b;
                RadBuildCtx ctx = spec_ctx(rank, world, 7);
                ctx.max_tok = 2048;
                REQUIRE_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&b, mk.meta(), &ctx); }),
                           RAD_OK);
                const qwen35_fp8::Model& m = qwen35_fp8::g_model[rank];
                CHECK_EQ(m.pipe_on, world == 2);
                CHECK(m.n_taps > 0);
                const auto pf = kit(b, {}, { 2048 }, pctx);
                const auto v = run_passes(b, qwen35_fp8::step, { pf.get() }, rank, world);
                CHECK_PASSES(b, v);
                const RadCtx& c = *v[0];
                CHECK(c.step_fail.empty());
                const int64_t E = m.g.n_embd, piece = 1024 * E;
                int ar = 0, ar_lane0 = 0, ar_piece = 0, ar_overlap = 0, arn = 0, arn_lane0 = 0;
                int conv = 0, attn = 0, wide_attn = 0, taps = 0, wide_taps = 0, gdn = 0;
                for (const auto& l : m.layers) gdn += !l.full;
                const int full = (int)m.layers.size() - gdn;
                for (const RecIssue& i : c.issues) {
                    const RecOp* o = b.op_of(i.op);
                    if (!o) continue;
                    if (o->op == "all_reduce") {
                        ++ar;
                        ar_lane0 += i.lane == 0;
                        ar_piece += i.n == piece;
                        ar_overlap += pint(o, "overlap") == 1;
                    }
                    if (o->op == "ar_rmsnorm_quant_fp8") { ++arn; arn_lane0 += i.lane == 0; }
                    if (o->op == "gdn_conv_prep") ++conv;
                    if (o->op == "attn_paged") { ++attn; wide_attn += i.n > 1024; }
                    if (o->op == "cast") { ++taps; wide_taps += i.n > 1024; }
                }
                const int pieces = world == 2 ? 2 : 1;
                CHECK(gdn > 0 && full > 0);
                CHECK_EQ(conv, pieces * gdn);
                CHECK_EQ(attn, pieces * full);
                CHECK_EQ(taps, pieces * m.n_taps);
                if (world == 1) {
                    CHECK_EQ(ar + arn, 0);
                    CHECK(c.joins.empty());
                    CHECK_EQ(wide_attn, attn);
                    continue;
                }
                CHECK_EQ(ar_lane0 + arn_lane0, 0);
                /* Two blocks a layer, two pieces a block, and the embedding's whole. */
                CHECK_EQ(ar + arn, 4 * (int)m.layers.size() + 1);
                CHECK(arn <= 1);
                CHECK_EQ(ar_piece, 4 * (int)m.layers.size());
                CHECK_EQ(ar_overlap, 4 * (int)m.layers.size());
                CHECK_EQ(wide_attn, 0);
                CHECK_EQ(wide_taps, 0);
                CHECK(!c.joins.empty());
            }
}

/* Past two ranks the pipelined prefill is off: a 2048-row step of one sequence at four and eight
 * ranks runs whole, with nothing on the second lane. */
static void n_rank_serial_case(int world, bool gemma) {
    for (int rank = 0; rank < world; ++rank) {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(rank, world, 7);
        ctx.max_tok = 2048;
        MetaKV mk = gemma ? gemma3_dflash2_meta() : with_dflash2(q35fp8_meta());
        if (gemma) b.encs = { { "blk.", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) } };
        const int rc = stderr_of_rc([&] {
            return gemma ? gemma3::declare(&b, mk.meta(), &ctx) : qwen35_fp8::declare(&b, mk.meta(), &ctx);
        });
        REQUIRE_EQ(rc, RAD_OK);
        CHECK(!(gemma ? gemma3::g_model[rank].pipe_on : qwen35_fp8::g_model[rank].pipe_on));
        const auto pf = kit(b, {}, { 2048 }, 0);
        const auto v = run_passes(b, gemma ? gemma3::step : qwen35_fp8::step, { pf.get() }, rank, world);
        CHECK_PASSES(b, v);
        const RadCtx& c = *v[0];
        CHECK(c.step_fail.empty());
        CHECK(c.joins.empty());
        int lane1 = 0;
        for (const RecIssue& i : c.issues) lane1 += i.lane == 1;
        CHECK_EQ(lane1, 0);
    }
}
TEST(qwen35_four_ranks_prefill_serially) { n_rank_serial_case(4, false); }
TEST(gemma3_four_ranks_prefill_serially) { n_rank_serial_case(4, true); }
TEST(qwen35_eight_ranks_prefill_serially) { n_rank_serial_case(8, false); }
TEST(gemma3_eight_ranks_prefill_serially) { n_rank_serial_case(8, true); }

/* Under a span a Flash-Next prefill of two intervals runs a pass at a time. At every world its
 * connection reads take a pass each and the indexer scores its pass-tall plane in windows, the MTP
 * head's pass included. At four ranks the step is pipelined (step_pipelined): every layer's
 * all-reduce carries one pass's rows on the second lane at `overlap` 1 and the delta net runs each
 * piece over its own bounds, while the embedding's collective stays whole, on the second lane at
 * no overlap. At two ranks the gated residual's fused write takes the layers' all-reduces and the
 * step runs on one lane. */
TEST(flashnext_a_spanned_prefill_runs_a_pass_at_a_time_with_its_all_reduces_beside_it) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.encs = { { "draft_head", u2_head() } };
            RadBuildCtx ctx = q4_ctx(rank, world);
            ctx.max_tok = 4096;
            ctx.max_mixed_tok = 2048;
            REQUIRE_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&b, &meta, &ctx); }),
                       RAD_OK);
            qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
            CHECK(m.span_on);
            CHECK_EQ(m.pipe_on, world > 2);
            CHECK_EQ(m.qsacfg.plane_rows, 2048);
            const auto pf = kit(b, {}, { 4096 }, 4096);
            const auto his = history_pass(b, *pf);
            const auto v = run_passes(b, qwen4exp_fp8::step, { pf.get(), his.get() }, rank, world);
            CHECK_PASSES(b, v);
            const RadCtx& c = *v[0];
            CHECK(c.step_fail.empty());
            /* The MTP head's pass over the same rows reads and scores a pass at a time too. */
            CHECK(v[1]->step_fail.empty());
            for (const RecIssue& i : v[1]->issues)
                if (const RecOp* o = b.op_of(i.op); o && (o->op == "hc_read" || o->op == "qsa_score"))
                    CHECK(i.n <= 2048);
            const int64_t piece = 2048 * m.g.n_embd;
            int reads = 0, wide_reads = 0, ar = 0, ar_lane0 = 0, ar_piece = 0, ar_overlap = 0;
            int score = 0, wide_score = 0, conv = 0, wide_conv = 0;
            for (const RecIssue& i : c.issues) {
                const RecOp* o = b.op_of(i.op);
                if (!o) continue;
                if (o->op == "hc_read") { ++reads; wide_reads += i.n > 2048; }
                if (o->op == "qsa_score") { ++score; wide_score += i.n > 2048; }
                if (o->op == "gdn_conv_prep") { ++conv; wide_conv += i.n > 2048; }
                if (o->op == "all_reduce") {
                    ++ar;
                    ar_lane0 += i.lane == 0;
                    ar_piece += i.n == piece;
                    ar_overlap += pint(o, "overlap") == 1;
                }
            }
            CHECK(reads > 0);
            CHECK_EQ(wide_reads, 0);
            CHECK(score > 0);
            CHECK_EQ(wide_score, 0);
            CHECK(conv > 0);
            /* This fake runtime has no engine teardown between model redeclarations. */
            rad_dev_free(m.pipe_cu_h, RAD_MEM_HOST_PINNED);
            m.pipe_cu_h = nullptr;
            m.pipe_cu_tried = false;
            if (world < 4) {
                CHECK_EQ(ar, world > 1 ? 1 : 0);
                CHECK_EQ(ar_lane0, ar);
                CHECK(c.joins.empty());
                continue;
            }
            CHECK_EQ(wide_conv, 0);
            CHECK(ar > 2);
            CHECK_EQ(ar_lane0, 0);
            CHECK_EQ(ar_piece, ar - 1);
            CHECK_EQ(ar_overlap, ar - 1);
            CHECK(!c.joins.empty());
        }
}

/* --prefill-span-pass step (RadBuildCtx::span_pass_step): where a spanned step runs serially, at
 * one and two ranks, the routed block takes the whole step in one pass -- one router launch a
 * routed block instead of one a piece, over planes a step tall -- while the connection reads and
 * the indexer's windows stay a piece. At four ranks the step is pipelined and keeps its pieces, so
 * the flag changes neither the graph nor the launches there. Off, the span is a pass a piece:
 * the graph and the step's launches are those of a context that never heard of it. */
TEST(flashnext_span_pass_step_takes_a_spanned_step_in_one_routed_pass) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            struct Run {
                uint64_t graph = 0;
                int routers = 0, wide_routers = 0, wide_reads = 0, wide_score = 0, issues = 0;
                int64_t rows = 0, piece = 0, sorted = 0;
                bool pass_step = false, pipe = false;
            } run[2];
            for (int on = 0; on < 2; ++on) {
                RadBuilder b;
                b.encs = { { "draft_head", u2_head() } };
                RadBuildCtx ctx = q4_ctx(rank, world);
                ctx.max_tok = 4096;
                ctx.max_mixed_tok = 2048;
                ctx.span_pass_step = on;
                REQUIRE_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&b, &meta, &ctx); }),
                           RAD_OK);
                qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
                Run& r = run[on];
                r.graph = graph_hash(b);
                r.pass_step = m.pass_step;
                r.pipe = m.pipe_on;
                r.rows = m.moecfg.rows;
                r.piece = m.piece_rows;
                for (const RecBuf& rb : b.bufs)
                    if (rb.name == "moe_sorted") r.sorted = rb.d.shape[0];
                CHECK(m.span_on);
                CHECK_EQ(m.qsacfg.plane_rows, 2048);
                const auto pf = kit(b, {}, { 4096 }, 4096);
                const auto v = run_passes(b, qwen4exp_fp8::step, { pf.get() }, rank, world);
                CHECK_PASSES(b, v);
                const RadCtx& c = *v[0];
                CHECK(c.step_fail.empty());
                r.issues = (int)c.issues.size();
                for (const RecIssue& i : c.issues) {
                    const RecOp* o = b.op_of(i.op);
                    if (!o) continue;
                    if (o->op == "router_topk" || o->op == "router_topk_scatter") {
                        ++r.routers;
                        r.wide_routers += i.n > 2048;
                    }
                    if (o->op == "hc_read") r.wide_reads += i.n > 2048;
                    if (o->op == "qsa_score") r.wide_score += i.n > 2048;
                }
                /* This fake runtime has no engine teardown between model redeclarations. */
                rad_dev_free(m.pipe_cu_h, RAD_MEM_HOST_PINNED);
                m.pipe_cu_h = nullptr;
                m.pipe_cu_tried = false;
            }
            /* Off: pieces, whatever the world. */
            CHECK(!run[0].pass_step);
            CHECK_EQ(run[0].rows, 2048);
            CHECK_EQ(run[0].piece, 2048);
            CHECK_EQ(run[0].wide_routers, 0);
            CHECK(run[0].routers > 0);
            CHECK_EQ(run[0].routers % 2, 0);
            CHECK_EQ(run[1].piece, 2048);
            CHECK_EQ(run[1].wide_reads, 0);
            CHECK_EQ(run[1].wide_score, 0);
            if (world > 2) {
                /* Pipelined: the flag is declined, and nothing moves. */
                CHECK(run[1].pipe);
                CHECK(!run[1].pass_step);
                CHECK_EQ(run[1].rows, 2048);
                CHECK_EQ(run[1].graph, run[0].graph);
                CHECK_EQ(run[1].issues, run[0].issues);
                CHECK_EQ(run[1].routers, run[0].routers);
                continue;
            }
            CHECK(run[1].pass_step);
            CHECK_EQ(run[1].rows, 4096);
            /* One pass a routed block where there were two, each over the whole step. */
            CHECK_EQ(run[1].routers * 2, run[0].routers);
            CHECK_EQ(run[1].wide_routers, run[1].routers);
            /* The planes with a row per (token, slot) are a step tall. */
            CHECK_EQ(run[1].sorted, 2 * run[0].sorted);
            CHECK(run[1].issues < run[0].issues);
        }
}

/* ============================ unlimitedocr: the DeepEncoder batch ============================= */

namespace {

const char* k_uocr_keys[] = {
    "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "n_shared_experts",
    "first_k_dense_replace", "scoring_func", "routed_scaling_factor", "topk_group", "n_group",
    "use_mla", "norm_topk_prob", "sliding_window_size",
};
const char* k_uocr_vals[] = { "64", "6", "896", "2", "1", "softmax", "1.0", "1", "1", "0", "0",
                              "128" };

RadModelMeta uocr_meta() {
    RadModelMeta m{};
    m.arch_id     = "unlimitedocr";
    m.name        = "test-unlimitedocr";
    m.quant       = "bf16";
    m.n_layers    = 12;
    m.n_embd      = 1280;
    m.n_head      = 10;
    m.n_head_kv   = 10;
    m.head_dim    = 128;
    m.n_ff        = 6848;
    m.n_vocab     = 129280;
    m.n_ctx_train = 32768;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 10000.0f;
    m.n_kv        = (int)(sizeof(k_uocr_keys) / sizeof(k_uocr_keys[0]));
    m.kv_key      = k_uocr_keys;
    m.kv_val      = k_uocr_vals;
    return m;
}

}  /* namespace */

/* One encode pass of two 640 tile lines (four tiles): the tower batches every GEMM over the whole
 * pass, and the segment-bounded ops carry one cu entry a tile. */
TEST(unlimitedocr_batches_the_tower_across_tile_segments) {
    RadBuilder b;
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    b.encs = { { "ffn_gate_up_exps", plain }, { "ffn_down_exps", plain } };
    RadModelMeta meta = uocr_meta();
    RadBuildCtx ctx = build_ctx();
    ctx.max_enc_patches = 16384;
    CHECK_OK(unlimitedocr_bf16::declare(&b, &meta, &ctx));
    /* A pass runs one of the two towers, so the core must not mix grids in one. */
    CHECK_EQ(b.encoder.modalities, (uint32_t)RAD_MM_IMAGE);
    CHECK_EQ(b.encoder.one_grid, 1);

    const int64_t q = 40, nseg = 2, n_patch = nseg * (2 * q + 4) * q;   /* two lines of two tiles */
    std::vector<int32_t> sh((size_t)nseg, (int32_t)q), sw((size_t)nseg, (int32_t)(2 * q + 4));
    std::vector<int32_t> cu = { 0, (int32_t)((2 * q + 4) * q), (int32_t)n_patch };
    std::vector<uint16_t> pix((size_t)n_patch * 768, 0);
    RadBatch batch{};
    batch.enc         = RAD_MM_IMAGE;
    batch.enc_n_patch = n_patch;
    batch.enc_pixels  = pix.data();
    batch.enc_n_seg   = nseg;
    batch.enc_cu      = cu.data();
    batch.enc_seg_h   = sh.data();
    batch.enc_seg_w   = sw.data();
    batch.enc_max_seg = (2 * q + 4) * q;
    RadCtx c{ .b = &b, .batch = &batch };
    unlimitedocr_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());
    CHECK_LINT(b, &c);

    /* The batch bound is the pass bound over this grid's tokens, capped at 8 (10 tiles at 40,
     * 4 at 64 for 16384 patches). */
    const auto& enc = unlimitedocr_bf16::g_model[0].enc;
    CHECK_EQ(enc.t40.Bmax, 8);
    CHECK_EQ(enc.t64.Bmax, 4);

    constexpr int64_t kTok40 = 1600, kTokP40 = 1764, kClip40 = 101;
    /* One patch-embed GEMM over all four tiles, not four of one. */
    const RecIssue* patch = issue_of(b, c, "gemm_nt_bias");
    REQUIRE(patch != nullptr);
    CHECK_EQ(patch->n, 4 * kTok40);
    /* SAM's first attention is layer 0's windowed path over the padded grid. */
    const RecIssue* sat = issue_of(b, c, "sam_attn");
    REQUIRE(sat != nullptr);
    CHECK_EQ(sat->n, 4 * kTokP40);
    REQUIRE(sat->opd.size() > 4);
    CHECK_EQ(sat->opd[3].rows, 5);
    /* Layer 2 is the first global block: no padding, four kSamTok-row segments. */
    const RecIssue* satg = issue_of(b, c, "sam_attn", 2);
    REQUIRE(satg != nullptr);
    CHECK_EQ(satg->n, 4 * kTok40);
    CHECK_EQ(satg->opd[3].rows, 5);
    /* CLIP's attention is one dense call over four kClipTok-row segments. */
    const RecIssue* cat = issue_of(b, c, "attn_dense");
    REQUIRE(cat != nullptr);
    CHECK_EQ(cat->n, 4 * kClip40);
    REQUIRE(cat->opd.size() > 4);
    CHECK_EQ(cat->opd[3].rows, 5);
    /* The unfold sees the four tiles as four kGrid-row segments. */
    const RecIssue* unf = issue_of(b, c, "conv2d_unfold");
    REQUIRE(unf != nullptr);
    CHECK_EQ(unf->opd[0].rows, 4 * q);

    /* The class row, the map cast, the fuse and the newline stay per tile: 4 / 4 / 4 / 2. */
    int cls = 0, cmap = 0, fuse = 0, nl = 0;
    for (const RecIssue& i : c.issues) {
        if (i.op == enc.t40.op_cls) ++cls;
        else if (i.op == enc.t40.op_cmap) ++cmap;
        else if (i.op == enc.t40.op_fuse_clip) ++fuse;
        else if (i.op == enc.t40.op_newline) ++nl;
    }
    CHECK_EQ(cls, 4);
    CHECK_EQ(cmap, 4);
    CHECK_EQ(fuse, 4);
    CHECK_EQ(nl, nseg);

    /* A one-tile pass is the same graph at B = 1. */
    std::vector<int32_t> sh1 = { (int32_t)q }, sw1 = { (int32_t)(q + 4) };
    std::vector<int32_t> cu1 = { 0, (int32_t)((q + 4) * q) };
    std::vector<uint16_t> pix1((size_t)((q + 4) * q) * 768, 0);
    RadBatch one{};
    one.enc         = RAD_MM_IMAGE;
    one.enc_n_patch = (q + 4) * q;
    one.enc_pixels  = pix1.data();
    one.enc_n_seg   = 1;
    one.enc_cu      = cu1.data();
    one.enc_seg_h   = sh1.data();
    one.enc_seg_w   = sw1.data();
    one.enc_max_seg = (q + 4) * q;
    RadCtx c1{ .b = &b, .batch = &one };
    unlimitedocr_bf16::step(&c1, &one);
    CHECK(c1.step_fail.empty());
    CHECK_LINT(b, &c1);
    const RecIssue* p1 = issue_of(b, c1, "gemm_nt_bias");
    REQUIRE(p1 != nullptr);
    CHECK_EQ(p1->n, kTok40);
    const RecIssue* s1 = issue_of(b, c1, "sam_attn");
    REQUIRE(s1 != nullptr);
    CHECK_EQ(s1->n, kTokP40);
    REQUIRE(s1->opd.size() > 4);
    CHECK_EQ(s1->opd[3].rows, 2);
    const RecIssue* c_1 = issue_of(b, c1, "attn_dense");
    REQUIRE(c_1 != nullptr);
    CHECK_EQ(c_1->n, kClip40);
}

/* ==================================================================== krea2 */

namespace {

/* Krea 2 with every network cut: a 36-layer conditioning language model of 256 (the last tap is
 * after layer 35), a one-block text fusion and DiT of 128-wide heads, and Wan 2.1's VAE. */
MetaKV krea2_meta() {
    RadModelMeta m{};
    m.arch_id = "krea2"; m.name = "test-krea2"; m.quant = "";
    m.n_layers = 36; m.n_embd = 256; m.n_head = 2; m.n_head_kv = 1; m.head_dim = 128;
    m.n_ff = 512; m.n_vocab = 151936; m.n_ctx_train = 32768;
    m.rms_eps = 1e-6f; m.rope_theta = 5e6f; m.rope_scale = 1.0f;
    MetaKV k(m);
    std::string mean, sd;
    for (int i = 0; i < 16; ++i) {
        mean += (i ? " " : "") + std::to_string(0.01 * i);
        sd += (i ? " " : "") + std::to_string(1.0 + 0.01 * i);
    }
    k.set("pipeline._class_name", "Krea2Pipeline")
     .set("transformer.text_hidden_dim", "256")
     .set("transformer.text_num_attention_heads", "2")
     .set("transformer.text_intermediate_size", "512")
     .set("transformer.num_layerwise_text_blocks", "1")
     .set("transformer.num_refiner_text_blocks", "1")
     .set("transformer.num_layers", "1")
     .set("transformer.attention_head_dim", "128")
     .set("transformer.num_attention_heads", "2")
     .set("transformer.num_key_value_heads", "2")
     .set("transformer.intermediate_size", "512")
     .set("transformer.axes_dims_rope", "32 48 48")
     .set("vae.latents_mean", mean.c_str())
     .set("vae.latents_std", sd.c_str());
    return k;
}

}  /* namespace */

TEST(krea2_reduces_the_denoiser_and_output_buffers_for_a_pixel_limit) {
    MetaKV mk = krea2_meta();
    RadArchProbe probe{};
    probe.gen_max_pixels = 512 * 512;
    REQUIRE_EQ(krea2::probe(mk.meta(), &probe), RAD_OK);
    CHECK_EQ(probe.gen_rows, 2 * (1024 + 512));

    RadBuilder b;
    RadBuildCtx ctx = qimg_ctx(0, 1);
    ctx.gen_max_pixels = probe.gen_max_pixels;
    ctx.max_tok = probe.gen_rows;
    REQUIRE_EQ(krea2::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.generator.max_tokens, 1024);
    CHECK_EQ(b.generator.default_width, 512);
    CHECK_EQ(b.generator.default_height, 512);
    CHECK_EQ(b.generator.max_refs, 0);
    REQUIRE(b.generator.image_out > 0);
    CHECK_EQ(b.bufs[b.generator.image_out - 1].d.shape[0], 512 * 512 * 3);
}

TEST(krea2_target_buffers_cover_multiple_bounded_images) {
    MetaKV mk = krea2_meta();
    RadArchProbe probe{};
    probe.gen_max_pixels = 256 * 256;
    REQUIRE_EQ(krea2::probe(mk.meta(), &probe), RAD_OK);
    for (const int entries : {1, 2, 4}) {
        RadBuilder b;
        RadBuildCtx ctx = qimg_ctx(0, 1);
        ctx.max_seqs = entries;
        ctx.max_tok = probe.gen_rows;
        ctx.gen_max_pixels = probe.gen_max_pixels;
        REQUIRE_EQ(krea2::declare(&b, mk.meta(), &ctx), RAD_OK);
        const auto* latent = find_b(b, "dit.lat");
        const auto* velocity = find_b(b, "dit.v");
        REQUIRE(latent && velocity);
        CHECK_EQ(latent->d.shape[0], entries * 256);
        CHECK_EQ(velocity->d.shape[0], entries * 256);
    }
}

/* An IQ8R container (every quantised linear at one scale a 256 of K): where the rotating twins are
 * served a DiT block's two modulated norms write the int8 codes in their own pass
 * (rms_mod_had_quant_i8g at the weights' group) and every attention's output gate writes the out
 * projection's (gate_had_quant_i8g, a token a row); where they are not, the plain norm and the
 * sigmoid and product are followed by had_quant_act_i8g. No op that writes E4M3 is declared. */
TEST(krea2_on_iq8r_takes_the_rotating_norm_and_gate_where_served) {
    MetaKV mk = krea2_meta();
    for (bool fused : { true, false }) {
        RadBuilder b;
        if (!fused) {
            b.undeclared.insert("rms_mod_had_quant_i8g");
            b.undeclared.insert("gate_had_quant_i8g");
        }
        for (const char* t : { "attn_qkv.weight", "attn_gate.weight", "attn_output.weight",
                               "ffn_gate_up.weight", "ffn_down.weight" })
            b.encs.push_back({ t, iq8r_enc() });
        RadBuildCtx ctx = qimg_ctx(0, 1);
        REQUIRE_EQ(krea2::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK(b.asked_undeclared.empty());
        for (const char* op : { "quant_act_fp8", "rms_mod_quant_fp8", "gate_quant_fp8",
                                "gated_quant_fp8" })
            CHECK_EQ(count_op(b, op), 0);
        /* One DiT block: two norms. Its gate and the two text fusion blocks': three gates. */
        CHECK_EQ(count_op(b, "rms_mod_had_quant_i8g"), fused ? 2 : 0);
        CHECK_EQ(count_op(b, "gate_had_quant_i8g"), fused ? 3 : 0);
        CHECK_EQ(count_op(b, "sigmoid"), fused ? 0 : 3);
        for (const RecOp& o : b.ops) {
            if (o.op == "rms_mod_had_quant_i8g") CHECK_EQ(pint(&o, "group"), 256);
            if (o.op == "gate_had_quant_i8g") {
                CHECK_EQ(pint(&o, "group"), 256);
                CHECK_EQ(pint(&o, "n"), 256);
                CHECK_EQ(pstr(&o, "act"), std::string("sigmoid"));
            }
        }
        for (const auto& buf : b.bufs) CHECK(buf.d.dtype != (uint32_t)RAD_F8E4M3);
        /* The text pass issues the fusion blocks' gates, a denoising pass the DiT block's norms
         * and gate. */
        const int want[2][2] = { { 0, 2 }, { 2, 1 } };
        int p = 0;
        for (const auto& k : { gen_pass(b, RAD_GEN_PASS_COND, 12),
                               gen_pass(b, RAD_GEN_PASS_DENOISE, 16) }) {
            RadCtx c{ .b = &b, .batch = &k->b, .rank = 0, .world = 1 };
            krea2::step(&c, &k->b);
            CHECK(c.step_fail.empty());
            const auto issued = issued_ops(b, c);
            CHECK_EQ(count_issued(issued, "rms_mod_had_quant_i8g"), fused ? want[p][0] : 0);
            CHECK_EQ(count_issued(issued, "gate_had_quant_i8g"), fused ? want[p][1] : 0);
            CHECK_EQ(count_issued(issued, "sigmoid"), fused ? 0 : want[p][1]);
            ++p;
        }
    }
}

/* `rebalance` (RAD_GEN_COND_REBALANCE, declared in the generator's cond_opts with its presets):
 * each of the twelve taps is scaled into te.h2 by its preset's gain (row 0 ComfyUI's
 * ConditioningKrea2Rebalance at its defaults, multiplier 4 times 1, 1, 1, 1, 1, 1, 1, 2.5, 5, 1.1,
 * 4, 1; rows 1 to 3 balanced, detail, subtle), f32 and rounded once, and the tap's scatter reads
 * te.h2. Without it, or with a preset row the plugin does not know, the pass issues no scale and
 * scatters te.x. */
TEST(krea2_rebalance_scales_each_tap_and_is_off_without_the_option) {
    MetaKV mk = krea2_meta();
    /* RAD_GEN_COND_REBALANCE_PRESET rows 0 (the default), 1 balanced, 2 detail, 3 subtle */
    const double gains[4][12] = {
        { 4, 4, 4, 4, 4, 4, 4, 10, 20, 4.4, 16, 4 },
        { 0.44109, 0.44109, 0.44109, 0.44109, 0.44109, 0.44109, 0.44109, 1.1027, 2.2055, 0.4852,
          1.7644, 0.44109 },
        { 0.28562, 0.28562, 0.32132, 0.32132, 0.35702, 0.35702, 0.42842, 1.0711, 2.1421, 0.53553,
          1.7851, 0.42842 },
        { 0.84759, 0.84759, 0.84759, 0.84759, 0.84759, 0.84759, 0.84759, 1.2714, 1.6952, 0.84759,
          1.2714, 0.84759 },
    };
    RadArchProbe probe{};
    REQUIRE_EQ(krea2::probe(mk.meta(), &probe), RAD_OK);
    CHECK_EQ(probe.gen_rows, (int64_t)2 * (16384 + 512));
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = qimg_ctx(rank, world);
            REQUIRE_EQ(stderr_of_rc([&] { return krea2::declare(&b, mk.meta(), &ctx); }), RAD_OK);
            CHECK_EQ(b.generator.cond_opts,
                     (uint32_t)(RAD_GEN_COND_REBALANCE | RAD_GEN_COND_REBALANCE_PRESET));
            const RecBuf* bx = find_b(b, "te.x");
            const RecBuf* bh = find_b(b, "te.h2");
            const RecBuf* bt = find_b(b, "te.taps");
            REQUIRE(bx && bh && bt);
            const rad_buf x = (rad_buf)(bx - b.bufs.data() + 1);
            const rad_buf h2 = (rad_buf)(bh - b.bufs.data() + 1);
            const rad_buf taps = (rad_buf)(bt - b.bufs.data() + 1);
            /* -1: no option; 0 to 3 the presets; 4 a row the plugin does not know */
            for (int preset : { -1, 0, 1, 2, 3, 4 }) {
                const bool rebalance = preset >= 0 && preset < 4;
                auto k = gen_pass(b, RAD_GEN_PASS_COND, 12);
                k->g.cond_opts = preset >= 0 ? RAD_GEN_COND_REBALANCE |
                                                   (preset << RAD_GEN_COND_REBALANCE_PRESET_SHIFT)
                                             : 0;
                RadCtx c{ .b = &b, .batch = &k->b, .rank = rank, .world = world };
                krea2::step(&c, &k->b);
                CHECK(c.step_fail.empty());
                CHECK_EQ(count_issued(issued_ops(b, c), "chan_affine"), rebalance ? 12 : 0);
                int tap = 0;
                for (size_t i = 0; i < c.issues.size(); ++i) {
                    const RecOp* o = b.op_of(c.issues[i].op);
                    if (!o || o->op != "scatter_rows" || c.issues[i].opd.size() != 3 ||
                        c.issues[i].opd[2].handle != taps)
                        continue;
                    REQUIRE(tap < 12);
                    CHECK_EQ(c.issues[i].opd[0].handle, rebalance ? h2 : x);
                    if (rebalance) {
                        REQUIRE(i > 0);
                        const RecIssue& gi = c.issues[i - 1];
                        const RecOp* g = b.op_of(gi.op);
                        REQUIRE(g && g->op == "chan_affine");
                        CHECK_EQ(gi.opd[0].handle, x);
                        CHECK_EQ(gi.opd[1].handle, h2);
                        CHECK_EQ(pint(g, "n"), 64LL);
                        CHECK_EQ(pint(g, "round"), 0LL);
                        CHECK_EQ(pstr(g, "mode"), std::string("fma"));
                        /* 64 copies of the tap's gain, and 64 zeros */
                        auto list = [](const std::string& s) {
                            std::vector<double> v;
                            std::istringstream in(s);
                            for (double d; in >> d;) v.push_back(d);
                            return v;
                        };
                        const std::vector<double> mul = list(pstr(g, "mul")), add = list(pstr(g, "add"));
                        CHECK_EQ(mul.size(), (size_t)64);
                        CHECK_EQ(add.size(), (size_t)64);
                        for (double v : mul) CHECK_EQ(v, gains[preset][tap]);
                        for (double v : add) CHECK_EQ(v, 0.0);
                    }
                    ++tap;
                }
                CHECK_EQ(tap, 12);
            }
        }
}

/* The plain-bf16 plugins of the small models ask for chained launches with a drafter and without
 * one (RadArchProbe::chain_pays): every kernel of their decode step has a chained form. */
TEST(the_plain_bf16_plugins_ask_for_chained_launches) {
    {
        RadModelMeta meta = qwen35_meta();
        RadArchProbe p{};
        CHECK_OK(qwen35_bf16::probe(&meta, &p));
        CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);
    }
    {
        RadModelMeta meta = lfm2_meta();
        RadArchProbe p{};
        CHECK_OK(lfm2_bf16::probe(&meta, &p));
        CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);
    }
    {
        RadModelMeta meta = uocr_meta();
        RadArchProbe p{};
        CHECK_OK(unlimitedocr_bf16::probe(&meta, &p));
        CHECK_EQ(p.chain_pays, (int)RAD_CHAIN_ALWAYS);
    }
}

TEST(krea_pixel_limits_preserve_the_default_canvas_and_reject_unusable_declarations) {
    MetaKV meta = krea2_meta();
    RadArchProbe probe{};
    REQUIRE_EQ(krea2::probe(meta.meta(), &probe), RAD_OK);
    RadBuildCtx ctx = qimg_ctx(0, 1);
    ctx.max_tok = probe.gen_rows;
    ctx.max_seqs = 16;
    RadBuilder builder;
    REQUIRE_EQ(krea2::declare(&builder, meta.meta(), &ctx), RAD_OK);
    CHECK_EQ(builder.generator.default_width, 1024);
    CHECK_EQ(builder.generator.default_height, 1024);
    const auto* latent = find_b(builder, "dit.lat");
    REQUIRE(latent);
    CHECK_EQ(latent->d.shape[0], ctx.max_tok);
    ctx.gen_max_pixels = 255;
    RadBuilder too_small;
    CHECK_EQ(krea2::declare(&too_small, meta.meta(), &ctx), RAD_E_INVAL);
    ctx.gen_max_pixels = 256 * 256;
    ctx.max_seqs = 0;
    RadBuilder no_entries;
    CHECK_EQ(krea2::declare(&no_entries, meta.meta(), &ctx), RAD_E_INVAL);
}
