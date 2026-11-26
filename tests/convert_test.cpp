/* convert_test.cpp -- rad-convert end to end on a tiny synthetic checkpoint (tests/tiny_model.h).
 *
 * The binary runs as a child on the host backend; the containers it writes are read back with
 * core's reader and decoder. The checkpoint is a two-layer LLaMA served by llama_dense_fp8, whose
 * recipe makes every linear rotated int8 and keeps the norms, the embedding and the head as the
 * checkpoint holds them, so one conversion writes a quantised and a copied weight of every kind.
 */
#include "rad_test.h"
#include "gguf_writer.h"
#include "tiny_model.h"
#include "e2e_config.h"

#include "format/encoding.h"
#include "format/radfile.h"
#include "gen_config.h"
#include "text/tokenizer.h"

#include <sys/stat.h>

#include <cstddef>
#include <map>

using namespace rad;

namespace {

std::vector<std::string> convert_env(const std::vector<std::string>& more = {}) {
    std::vector<std::string> e = { "ROCR_VISIBLE_DEVICES=99", "OMP_WAIT_POLICY=passive",
                                   "OMP_NUM_THREADS=2",
                                   std::string("RADIANCE_HOME=") + RAD_E2E_HOME };
    e.insert(e.end(), more.begin(), more.end());
    return e;
}

/* rad-convert `ckpt` with `args`: its exit status, and its output in `log`. */
int convert(const std::string& ckpt, const std::vector<std::string>& args, std::string* log,
            const std::vector<std::string>& env = {}) {
    std::vector<std::string> a = { RAD_E2E_CONVERT, ckpt };
    a.insert(a.end(), args.begin(), args.end());
    std::string out;
    const int st = tiny::run(a, convert_env(env), &out);
    if (log) *log = out;
    return st;
}

bool exists(const std::string& p) {
    struct stat st {};
    return stat(p.c_str(), &st) == 0;
}

bool copy_file(const std::string& from, const std::string& to) {
    std::error_code ec;
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
    return !ec;
}

/* The checkpoint, its recipe and one conversion of it, shared by every case. */
struct World {
    tiny::TempDir dir{ "convert" };
    std::string ckpt, recipe, base;
    std::map<std::string, tiny::Tensor> src;   /* the checkpoint's tensors by name */
    std::string log;
    bool ok = false;

    World() {
        if (!dir.ok()) return;
        ckpt = dir / "llama";
        recipe = dir / "i8.recipe";
        base = dir / "base.rad";
        if (mkdir(ckpt.c_str(), 0755) != 0 || !tiny::write_llama(ckpt, {}) ||
            !tiny::write_text(recipe, tiny::kLlamaI8Recipe))
            return;
        for (tiny::Tensor& t : tiny::read_safetensors(ckpt + "/model.safetensors"))
            src[t.name] = std::move(t);
        ok = !src.empty() && convert(ckpt, { "-o", base, "--recipe", recipe }, &log) == 0;
        if (!ok) fprintf(stderr, "    the base conversion failed:\n%s\n", log.c_str());
    }
};

World& world() {
    static World w;
    return w;
}

/* A weight's planes, decoded to f32 [rows, cols] by the core's decoder. */
std::vector<float> decode(const RadFile& f, const RadFileEntry& e) {
    const int64_t cols = e.shape[e.rank - 1];
    int64_t rows = 1;
    for (uint32_t k = 0; k + 1 < e.rank; ++k) rows *= e.shape[k];
    const void* planes[RAD_ENC_MAX_PLANES] = {};
    for (uint32_t k = 0; k < e.n_planes; ++k) planes[k] = f.data(f.plane(e, (int)k));
    EncodedView v;
    enc_view(f.encoding(e), rows, cols, planes, &v);
    std::vector<float> out((size_t)(rows * cols));
    if (enc_decode_rows(v, 0, rows, out.data()) != RAD_OK) out.clear();
    return out;
}

/* The bytes of an entry's planes, concatenated. */
std::string entry_bytes(const RadFile& f, const RadFileEntry& e) {
    std::string s;
    for (uint32_t k = 0; k < e.n_planes; ++k) {
        const RadFilePlane& p = f.plane(e, (int)k);
        s.append((const char*)f.data(p), (size_t)p.bytes);
    }
    return s;
}

/* Every weight whose planes differ between two containers. */
std::vector<std::string> differing_weights(const RadFile& a, const RadFile& b) {
    std::vector<std::string> out;
    for (int64_t i = 0; i < a.entry_count(); ++i) {
        const RadFileEntry& e = a.entry(i);
        const RadFileEntry* o = b.find(a.str(e.name));
        if (!o || entry_bytes(a, e) != entry_bytes(b, *o)) out.push_back(a.str(e.name));
    }
    return out;
}

}  /* namespace */

/* ================================================================== what a conversion writes */

namespace {

struct VisionExtension {
    tiny::TempDir dir{"add-vision"};
    std::string base = dir / "text.rad", vision = dir / "vision.safetensors", log;
    std::vector<tiny::Tensor> tower;
    rad::Json config;
    bool ok = false;

    VisionExtension() {
        const std::string text_dir = dir / "text", draft_dir = dir / "draft";
        tiny::Gemma4 s;
        s.layers = 2;
        s.hidden = 256;  // current int8 DFlash prefill tiles require 256-row projections
        s.vocab = 384;
        if (!dir.ok() || mkdir(text_dir.c_str(), 0755) || mkdir(draft_dir.c_str(), 0755) ||
            !tiny::write_gemma4(text_dir, s)) return;
        config = rad::Json::parse(tiny::read_text(text_dir + "/config.json"));
        auto& text = config["text_config"];
        text["enable_moe_block"] = true;
        text["num_experts"] = 4;
        text["top_k_experts"] = 2;
        text["moe_intermediate_size"] = 64;
        auto ts = tiny::read_safetensors(text_dir + "/model.safetensors");
        tiny::Rng r(46);
        auto add = [&](const std::string& name, std::vector<int64_t> shape) {
            ts.push_back(tiny::noise(name, std::move(shape), r, 0.25f));
        };
        for (int l = 0; l < s.layers; ++l) {
            const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
            add(p + "experts.gate_up_proj", {4, 128, s.hidden});
            add(p + "experts.down_proj", {4, s.hidden, 64});
            add(p + "router.proj.weight", {4, s.hidden});
            add(p + "router.scale", {s.hidden});
            add(p + "router.per_expert_scale", {4});
            for (const char* n : {"pre_feedforward_layernorm_2.weight",
                                  "post_feedforward_layernorm_1.weight",
                                  "post_feedforward_layernorm_2.weight"})
                add(p + n, {s.hidden});
        }
        if (!tiny::write_checkpoint(text_dir, config, ts, {"<image>", "<video>", "<boi>", "<eoi>"}))
            return;

        rad::Json draft = {
            {"architectures", {"DFlashDraftModel"}}, {"model_type", "qwen3"},
            {"hidden_size", 256}, {"vocab_size", s.vocab}, {"num_hidden_layers", 1},
            {"num_attention_heads", 1}, {"num_key_value_heads", 1}, {"head_dim", 256},
            {"intermediate_size", 256}, {"rms_norm_eps", 1e-6}, {"rope_theta", 10000},
            {"block_size", 4}, {"dflash_config", {{"mask_token_id", 4}, {"target_layer_ids", {1}}}},
        };
        ts.clear();
        for (const char* n : {"fc.weight", "layers.0.self_attn.q_proj.weight",
                              "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
                              "layers.0.self_attn.o_proj.weight", "layers.0.mlp.gate_proj.weight",
                              "layers.0.mlp.up_proj.weight", "layers.0.mlp.down_proj.weight"})
            add(n, {256, 256});
        for (const char* n : {"hidden_norm.weight", "norm.weight", "layers.0.input_layernorm.weight",
                              "layers.0.post_attention_layernorm.weight",
                              "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.k_norm.weight"})
            add(n, {256});
        if (!tiny::write_checkpoint(draft_dir, draft, ts) ||
            !tiny::write_text(dir / "chat.jinja", "{{ messages[0].content }}")) return;
        // Current libref lacks the rotated-expert i8g schemas. Keep this host
        // fixture's experts BF16 while retaining real two-plane int8 DFlash weights.
        std::string draft_rules;
        std::istringstream rules(tiny::read_text(RAD_E2E_RECIPES "/gemma4-26b-a4b-df-i8.recipe"));
        for (std::string line; std::getline(rules, line);)
            if (line.starts_with("dflash.")) draft_rules += line + "\n";
        const std::string draft_recipe = dir / "draft-i8.recipe";
        if (draft_rules.empty() || !tiny::write_text(draft_recipe, draft_rules)) return;
        // The tiny Gemma head dimensions exercise the reference kernels.
        const int rc = convert(text_dir, {"-o", base, "--draft-model", draft_dir, "--kernels", "libref",
                               "--recipe", draft_recipe,
                               "--chat-template", dir / "chat.jinja",
                               "--tokenizer2", text_dir + "/tokenizer.json",
                               "--set", "test.base_metadata=preserved"}, &log);
        if (rc != 0) { std::fprintf(stderr, "%s\n", log.c_str()); return; }

        config["text_config"]["use_bidirectional_attention"] = "vision";
        config["vision_config"] = {
            {"model_type", "gemma4_vision"}, {"hidden_size", 16}, {"num_attention_heads", 4},
            {"head_dim", 4}, {"intermediate_size", 32}, {"num_hidden_layers", 1},
            {"patch_size", 2}, {"pooling_kernel_size", 3}, {"position_embedding_size", 64},
            {"hidden_activation", "gelu_pytorch_tanh"}, {"standardize", true},
            {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 100}}}
        };
        config["image_token_id"] = 259; config["video_token_id"] = 260;
        config["boi_token_id"] = 261; config["eoi_token_id"] = 262;
        const std::string v = "model.vision_tower.", l = v + "encoder.layers.0.";
        ts.clear();
        add(v + "patch_embedder.input_proj.weight", {16, 12});
        add(v + "patch_embedder.position_embedding_table", {2, 64, 16});
        add("model.embed_vision.embedding_projection.weight", {s.hidden, 16});
        add(v + "std_bias", {16});
        add(v + "std_scale", {16});
        for (const char* n : {"input_layernorm.weight", "post_attention_layernorm.weight",
                              "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"})
            add(l + n, {16});
        for (const char* n : {"q_proj", "k_proj", "v_proj", "o_proj"})
            add(l + "self_attn." + n + ".linear.weight", {16, 16});
        for (const char* n : {"q_norm", "k_norm"}) add(l + "self_attn." + n + ".weight", {4});
        add(l + "mlp.gate_proj.linear.weight", {32, 16});
        add(l + "mlp.up_proj.linear.weight", {32, 16});
        add(l + "mlp.down_proj.linear.weight", {16, 32});
        tower = std::move(ts);
        ok = tiny::write_text(dir / "config.json", config.dump()) &&
             tiny::write_text(dir / "processor_config.json",
                              R"({"image_processor":{"max_soft_tokens":140}})") &&
             tiny::write_safetensors(vision, tower);
    }

    int repack(const std::string& source, const std::string& out, std::string* log,
               const std::vector<std::string>& options = {}) const {
        std::vector<std::string> args = {"--add-vision", base, "-o", out, "--kernels", "libref"};
        args.insert(args.end(), options.begin(), options.end());
        return convert(source, args, log);
    }
};

VisionExtension& vision_extension() {
    static VisionExtension fixture;
    return fixture;
}

}  /* namespace */

TEST(add_vision_copies_every_text_expert_and_dflash_plane_and_adds_only_the_tower) {
    const auto& f = vision_extension();
    REQUIRE(f.ok);
    const std::string out = f.dir / "with-vision.rad";
    std::string log;
    int rc = f.repack(f.vision, out, &log, {"--plan-only"});
    if (rc) std::fprintf(stderr, "%s\n", log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(!exists(out));
    rc = f.repack(f.vision, out, &log);
    if (rc) std::fprintf(stderr, "%s\n", log.c_str());
    REQUIRE_EQ(rc, 0);

    RadFile old, result;
    REQUIRE_EQ(old.open(f.base.c_str()), RAD_OK);
    REQUIRE_EQ(result.open(out.c_str()), RAD_OK);
    CHECK_EQ(result.entry_count(), old.entry_count() + 16);
    for (int64_t i = 0; i < old.entry_count(); ++i) {
        const auto& e = old.entry(i);
        const auto* n = result.find(old.str(e.name));
        REQUIRE(n != nullptr);
        CHECK(rad_enc_equal(&old.encoding(e), &result.encoding(*n)));
        CHECK_EQ(e.rank, n->rank);
        CHECK(std::equal(e.shape, e.shape + RAD_MAX_RANK, n->shape));
        CHECK_EQ(e.layer, n->layer);
        CHECK_EQ(e.expert, n->expert);
        CHECK_EQ(e.slot, n->slot);
        CHECK_EQ(std::string(old.str(e.quantizer)), std::string(result.str(n->quantizer)));
        CHECK_EQ(std::string(old.str(e.options)), std::string(result.str(n->options)));
        REQUIRE_EQ(e.n_planes, n->n_planes);
        for (uint32_t k = 0; k < e.n_planes; ++k) {
            const auto& a = old.plane(e, (int)k);
            const auto& b = result.plane(*n, (int)k);
            REQUIRE_EQ(a.bytes, b.bytes);
            CHECK(std::memcmp(old.data(a), result.data(b), a.bytes) == 0);
        }
    }
    REQUIRE(old.find("blk.0.ffn_gate_up_exps.0.weight"));
    REQUIRE(old.find("dflash.draft_head.weight"));
    CHECK_EQ(old.find("blk.0.ffn_gate_up_exps.0.weight")->n_planes, 1u);
    CHECK_EQ(old.find("dflash.draft_head.weight")->n_planes, 2u);
    for (int64_t i = 0; i < old.meta_count(); ++i) {
        const auto& a = old.meta_at(i);
        const auto& b = result.meta_at(i);
        CHECK_EQ(std::string(old.str(a.key)), std::string(result.str(b.key)));
        REQUIRE_EQ(a.type, b.type);
        if (a.type == RAD_P_STR) CHECK_EQ(std::string(old.str(a.v.s)), std::string(result.str(b.v.s)));
        else if (a.type == RAD_P_INT) CHECK_EQ(a.v.i, b.v.i);
        else CHECK_EQ(a.v.f, b.v.f);
    }
    CHECK_EQ(std::string(old.str(old.header().recipe)), std::string(result.str(result.header().recipe)));
    CHECK_EQ(std::string(result.get_s("radiance.encoder", "")), std::string("vision"));
    CHECK_EQ(std::string(result.get_s("processor.image_processor.max_soft_tokens", "")),
             std::string("140"));
    CHECK_EQ(std::string(result.get_s("text_config.use_bidirectional_attention", "")),
             std::string("vision"));
    for (bool aux : {false, true}) {
        VocabBuild a, b;
        const auto av = aux ? old.aux_vocab() : old.vocab();
        const auto bv = aux ? result.aux_vocab() : result.vocab();
        REQUIRE_EQ(vocab_deserialize(old.base(), old.size(), av.h, a), RAD_OK);
        REQUIRE_EQ(vocab_deserialize(result.base(), result.size(), bv.h, b), RAD_OK);
        CHECK(a.tokens == b.tokens);
        CHECK(a.types == b.types);
        CHECK(a.scores == b.scores);
        CHECK(a.merges == b.merges);
        CHECK_EQ(a.chat_template, b.chat_template);
        CHECK_EQ(a.kind, b.kind);
        CHECK_EQ(a.ignore_merges, b.ignore_merges);
        CHECK(std::tie(a.bos, a.eos, a.eot, a.pad, a.unk, a.sep, a.add_bos, a.add_eos) ==
              std::tie(b.bos, b.eos, b.eot, b.pad, b.unk, b.sep, b.add_bos, b.add_eos));
        REQUIRE_EQ(a.steps.size(), b.steps.size());
        for (size_t i = 0; i < a.steps.size(); ++i) {
            const auto& x = a.steps[i];
            const auto& y = b.steps[i];
            CHECK(std::tie(x.kind, x.flags, x.arg0, x.arg1, x.iarg) ==
                  std::tie(y.kind, y.flags, y.arg0, y.arg1, y.iarg));
        }
    }
    for (const auto& [name, index] : {std::pair{"v.embed.weight", 2}, {"v.std_bias", 3}, {"v.std_scale", 4}}) {
        const auto* e = result.find(name);
        REQUIRE(e != nullptr);
        const auto& expected = f.tower[(size_t)index].bytes;
        REQUIRE_EQ(result.plane(*e, 0).bytes, expected.size());
        CHECK(std::memcmp(result.data(result.plane(*e, 0)), expected.data(), expected.size()) == 0);
    }

    auto source_bytes = [&](const std::string& name) {
        auto it = std::find_if(f.tower.begin(), f.tower.end(),
                               [&](const tiny::Tensor& t) { return t.name == name; });
        CHECK(it != f.tower.end());
        if (it == f.tower.end()) return std::string{};
        return std::string(reinterpret_cast<const char*>(it->bytes.data()), it->bytes.size());
    };
    auto check_matrix = [&](const char* name, int64_t rows, int64_t cols, const std::string& bytes) {
        const auto* e = result.find(name);
        REQUIRE(e != nullptr);
        const RadEncoding bf16 = rad_enc_plain(RAD_BF16);
        CHECK(rad_enc_equal(&result.encoding(*e), &bf16));
        REQUIRE_EQ(e->n_planes, 1u);
        CHECK_EQ(e->rank, 2u);
        CHECK_EQ(e->shape[0], rows);
        CHECK_EQ(e->shape[1], cols);
        const auto& plane = result.plane(*e, 0);
        REQUIRE_EQ(plane.bytes, bytes.size());
        CHECK(std::memcmp(result.data(plane), bytes.data(), bytes.size()) == 0);
    };
    /* The source table is [2, 64, 16] with X then Y. Fused matrices stack rows in
     * Q/K/V and gate/up order. Derive the expected BF16 bits directly from the fixture. */
    const std::string pos = source_bytes("model.vision_tower.patch_embedder.position_embedding_table");
    constexpr size_t axis_bytes = 64 * 16 * sizeof(uint16_t);
    REQUIRE_EQ(pos.size(), 2 * axis_bytes);
    check_matrix("v.pos_x.weight", 64, 16, pos.substr(0, axis_bytes));
    check_matrix("v.pos_y.weight", 64, 16, pos.substr(axis_bytes));
    const std::string layer = "model.vision_tower.encoder.layers.0.";
    check_matrix("v.blk.0.qkv", 48, 16,
                 source_bytes(layer + "self_attn.q_proj.linear.weight") +
                 source_bytes(layer + "self_attn.k_proj.linear.weight") +
                 source_bytes(layer + "self_attn.v_proj.linear.weight"));
    check_matrix("v.blk.0.gate_up", 64, 16,
                 source_bytes(layer + "mlp.gate_proj.linear.weight") +
                 source_bytes(layer + "mlp.up_proj.linear.weight"));

    /* Auto media must redeclare the encoder from the finished container. */
    rc = tiny::run(radtest::config_argv({RAD_E2E_SERVER, "--model", out, "--kernels", "libref", "--debug-graph"}),
                   convert_env(), &log);
    if (rc) std::fprintf(stderr, "%s\n", log.c_str());
    REQUIRE_EQ(rc, 0);
    CHECK(log.find("vision tower:") != std::string::npos);
    CHECK(log.find("the DFlash drafter is declared") != std::string::npos);
}

TEST(add_vision_refuses_invalid_sources_and_never_overwrites_a_destination) {
    const auto& f = vision_extension();
    REQUIRE(f.ok);
    const std::string out = f.dir / "refused.rad";
    const std::string before = tiny::read_text(f.base);
    std::string log;
    CHECK(f.repack(f.vision, f.base, &log) != 0);
    CHECK(log.find("requires a new output path") != std::string::npos);
    CHECK(tiny::read_text(f.base) == before);
    REQUIRE(tiny::write_text(out, "existing"));
    CHECK(f.repack(f.vision, out, &log) != 0);
    CHECK_EQ(tiny::read_text(out), std::string("existing"));

    const std::string bad = f.dir / "bad.safetensors", absent = f.dir / "absent.rad";
    auto ts = f.tower;
    ts.erase(ts.begin() + 4); /* missing standardization scale */
    REQUIRE(tiny::write_safetensors(bad, ts));
    CHECK(f.repack(bad, absent, &log) != 0);
    CHECK(log.find("std_scale") != std::string::npos);
    CHECK(!exists(absent));

    ts = f.tower;
    ts[2].shape = {128, 32};  /* same byte size, incompatible projector */
    REQUIRE(tiny::write_safetensors(bad, ts));
    CHECK(f.repack(bad, absent, &log) != 0);
    CHECK(log.find("v.embed.weight") != std::string::npos);
    CHECK(!exists(absent));

    const std::string bad_config = f.dir / "bad-config";
    REQUIRE_EQ(mkdir(bad_config.c_str(), 0755), 0);
    auto config = f.config;
    config["image_token_id"] = 300;  /* outside the base's tokenizer */
    REQUIRE(tiny::write_text(bad_config + "/config.json", config.dump()));
    CHECK(f.repack(f.vision, absent, &log, {"--config", bad_config}) != 0);
    CHECK(log.find("outside the base vocabulary") != std::string::npos);
    CHECK(!exists(absent));
}

/* A recipe weight is rotated int8 over the rows its name map joins, in the map's order: attn_qkv
 * is q, then k, then v, and ffn_gate_up is gate then up. A row's 128 columns under one scale
 * decode to their source within a twentieth of the source's RMS there: the codes' step is the
 * rotated group's largest magnitude over 127, about a hundredth of its RMS in error. A part out of
 * order, or a group's scale applied to another group, is off by the source's own magnitude. */
TEST(a_recipe_weight_decodes_to_its_source_rows_in_name_map_order) {
    World& w = world();
    REQUIRE(w.ok);
    RadFile f;
    REQUIRE_EQ(f.open(w.base.c_str()), RAD_OK);
    const struct { const char* weight; std::vector<const char*> parts; } kJoins[] = {
        { "blk.0.attn_qkv.weight", { "model.layers.0.self_attn.q_proj.weight",
                                     "model.layers.0.self_attn.k_proj.weight",
                                     "model.layers.0.self_attn.v_proj.weight" } },
        { "blk.1.ffn_gate_up.weight", { "model.layers.1.mlp.gate_proj.weight",
                                        "model.layers.1.mlp.up_proj.weight" } },
        { "blk.1.ffn_down.weight", { "model.layers.1.mlp.down_proj.weight" } },
        { "blk.0.attn_output.weight", { "model.layers.0.self_attn.o_proj.weight" } },
    };
    for (const auto& j : kJoins) {
        const RadFileEntry* e = f.find(j.weight);
        REQUIRE(e != nullptr);
        CHECK_EQ(enc_name(f.encoding(*e)), std::string("i8*bf16[1x128]/fwht128"));
        CHECK_EQ(std::string(f.str(e->quantizer)), std::string("rtn"));
        const std::vector<float> got = decode(f, *e);
        REQUIRE(!got.empty());
        const int64_t rows = e->shape[0], cols = e->shape[1];
        std::vector<float> want;
        for (const char* part : j.parts) {
            const tiny::Tensor& t = w.src.at(part);
            REQUIRE_EQ(t.shape[1], cols);
            for (int64_t i = 0; i < t.shape[0] * cols; ++i) {
                uint16_t h;
                std::memcpy(&h, t.bytes.data() + 2 * i, 2);
                want.push_back(rad_bf16_to_f32(h));
            }
        }
        REQUIRE_EQ((int64_t)want.size(), rows * cols);
        int64_t bad = 0;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t c0 = 0; c0 < cols; c0 += 128) {
                double sw = 0, se = 0;
                for (int64_t c = c0; c < std::min(cols, c0 + 128); ++c) {
                    const size_t i = (size_t)(r * cols + c);
                    sw += (double)want[i] * want[i];
                    se += ((double)got[i] - want[i]) * ((double)got[i] - want[i]);
                }
                if (!(std::sqrt(se) <= 0.05 * std::sqrt(sw) + 1e-6)) ++bad;
            }
        if (bad)
            fprintf(stderr, "    %s: %lld scale groups outside the bound\n", j.weight,
                    (long long)bad);
        CHECK_EQ(bad, (int64_t)0);
    }
}

/* A weight no rule matches keeps the checkpoint's encoding, byte for byte. */
TEST(a_weight_no_rule_matches_is_the_checkpoints_bytes) {
    World& w = world();
    REQUIRE(w.ok);
    RadFile f;
    REQUIRE_EQ(f.open(w.base.c_str()), RAD_OK);
    const std::pair<const char*, const char*> kCopies[] = {
        { "token_embd.weight", "model.embed_tokens.weight" },
        { "output.weight", "lm_head.weight" },
        { "output_norm.weight", "model.norm.weight" },
        { "blk.1.attn_norm.weight", "model.layers.1.input_layernorm.weight" },
        { "blk.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight" },
    };
    for (const auto& [name, from] : kCopies) {
        const RadFileEntry* e = f.find(name);
        REQUIRE(e != nullptr);
        CHECK_EQ(enc_name(f.encoding(*e)), std::string("bf16"));
        CHECK_EQ(std::string(f.str(e->quantizer)), std::string(""));
        const tiny::Tensor& t = w.src.at(from);
        CHECK(entry_bytes(f, *e) == std::string(t.bytes.begin(), t.bytes.end()));
    }
}

/* The header names the architecture and the recipe, the metadata carries config.json's
 * dimensions under RadModelMeta's names, and the provenance names the checkpoint without the
 * absolute path of the machine that converted it. */
TEST(the_header_and_metadata_describe_the_checkpoint) {
    World& w = world();
    REQUIRE(w.ok);
    RadFile f;
    REQUIRE_EQ(f.open(w.base.c_str()), RAD_OK);
    const RadFileHeader& h = f.header();
    CHECK_EQ(std::string(f.str(h.arch_id)), std::string("llama"));
    CHECK_EQ(std::string(f.str(h.quant)), std::string(""));
    const std::string recipe = f.str(h.recipe);
    for (const char* rule : { "blk.*.attn_qkv.weight", "blk.*.attn_output.weight",
                              "blk.*.ffn_gate_up.weight", "blk.*.ffn_down.weight" })
        CHECK(has(recipe, rule));
    const std::string by = f.str(h.created_by);
    CHECK(has(by, "rad-convert"));
    CHECK(!has(by, w.dir.path));
    const tiny::Llama s{};
    const std::pair<const char*, int64_t> kDims[] = {
        { "n_layers", s.layers }, { "n_embd", s.hidden }, { "n_head", s.heads },
        { "n_head_kv", s.kv_heads }, { "head_dim", s.head_dim }, { "n_ff", s.ffn },
        { "n_vocab", s.vocab }, { "n_ctx_train", s.ctx } };
    for (const auto& [key, want] : kDims) {
        int64_t v = -1;
        CHECK(f.get_i(key, &v));
        CHECK_EQ(v, want);
    }
    CHECK_EQ(f.entry_count(), (int64_t)15);
    CHECK(f.vocab());
}

/* Weights are quantised in parallel pieces; the bytes must not depend on how many threads. */
TEST(a_conversion_is_the_same_bytes_at_any_thread_count) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_t");
    for (const char* n : { "1", "7" }) {
        const std::string out = d / (std::string("t") + n + ".rad");
        REQUIRE_EQ(convert(w.ckpt, { "-o", out, "--recipe", w.recipe }, nullptr,
                           { std::string("OMP_NUM_THREADS=") + n }), 0);
        CHECK(tiny::read_text(out) == tiny::read_text(w.base));
    }
}

/* ================================================================== --reuse and --changed */

/* --reuse copies every weight converted the same way, which for an unchanged checkpoint is all
 * of them: the result is the same file. */
TEST(reuse_of_an_unchanged_checkpoint_writes_the_same_file) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_r");
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "r.rad", "--recipe", w.recipe, "--reuse", w.base },
                       &log), 0);
    CHECK(has(log, "--reuse: 15 weight(s)"));
    CHECK(tiny::read_text(d / "r.rad") == tiny::read_text(w.base));
}

TEST(reuse_only_copies_matching_planes_and_refuses_fresh_conversion) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_reuse_only");
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "ok.rad", "--recipe", w.recipe, "--reuse", w.base,
                                  "--reuse-only" }, &log), 0);
    CHECK(has(log, "--reuse: 15 weight(s)"));
    CHECK(tiny::read_text(d / "ok.rad") == tiny::read_text(w.base));
    REQUIRE(tiny::write_text(d / "changed.recipe",
                             std::string("blk.0.attn_qkv.weight rtn codes=i8 group=128 scale=bf16 "
                                         "transform=fwht128 clamp=sym rule=search\n") +
                                 tiny::kLlamaI8Recipe));
    CHECK(convert(w.ckpt, { "-o", d / "bad.rad", "--recipe", d / "changed.recipe",
                            "--reuse", w.base, "--reuse-only" }, &log) != 0);
    CHECK(has(log, "--reuse-only: no matching source"));
    CHECK_EQ(convert(w.ckpt, { "-o", d / "missing.rad", "--recipe", w.recipe, "--reuse-only" },
                     &log), 2);
    CHECK_EQ(convert(w.ckpt, { "--recipe", w.recipe, "--reuse", w.base, "--reuse-only",
                               "--in-place" }, &log), 2);
    CHECK(tiny::read_text(d / "ok.rad") == tiny::read_text(w.base));
}

/* --reuse matches on how a weight was converted, not on its source bytes, so a changed tensor is
 * copied stale unless --changed names it; named, exactly the weights that read it are converted
 * again. */
TEST(changed_reconverts_exactly_the_weights_that_read_a_changed_tensor) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_c");
    const std::string ck = d / "llama";
    REQUIRE(mkdir(ck.c_str(), 0755) == 0);
    for (const char* f : { "config.json", "tokenizer.json" })
        REQUIRE(copy_file(w.ckpt + "/" + f, ck + "/" + f));
    std::vector<tiny::Tensor> ts = tiny::read_safetensors(w.ckpt + "/model.safetensors");
    for (tiny::Tensor& t : ts)
        if (t.name == "model.layers.0.mlp.down_proj.weight")
            for (size_t i = 0; i < t.bytes.size(); i += 2) t.bytes[i + 1] ^= 0x80;   /* negate */
    REQUIRE(tiny::write_safetensors(ck + "/model.safetensors", ts));
    REQUIRE(tiny::write_text(d / "changed",
                             "# the fine-tuned tensor\nmodel.layers.0.mlp.down_proj.weight\n"));

    RadFile base;
    REQUIRE_EQ(base.open(w.base.c_str()), RAD_OK);
    {
        REQUIRE_EQ(convert(ck, { "-o", d / "stale.rad", "--recipe", w.recipe, "--reuse", w.base },
                           nullptr), 0);
        RadFile f;
        REQUIRE_EQ(f.open((d / "stale.rad").c_str()), RAD_OK);
        CHECK(differing_weights(base, f).empty());
    }
    std::string log;
    REQUIRE_EQ(convert(ck, { "-o", d / "fresh.rad", "--recipe", w.recipe, "--reuse", w.base,
                             "--changed", d / "changed" }, &log), 0);
    RadFile f;
    REQUIRE_EQ(f.open((d / "fresh.rad").c_str()), RAD_OK);
    const std::vector<std::string> diff = differing_weights(base, f);
    REQUIRE_EQ(diff.size(), (size_t)1);
    CHECK(!diff.empty() && diff[0] == "blk.0.ffn_down.weight");
    /* And it is the negated weight, not merely different bytes. */
    const RadFileEntry* a = base.find("blk.0.ffn_down.weight");
    const RadFileEntry* b = f.find("blk.0.ffn_down.weight");
    REQUIRE(a && b);
    const std::vector<float> va = decode(base, *a), vb = decode(f, *b);
    REQUIRE(!va.empty() && va.size() == vb.size());
    int64_t neg = 0;
    for (size_t i = 0; i < va.size(); ++i) neg += va[i] == -vb[i];
    CHECK_EQ(neg, (int64_t)va.size());
}

/* --changed is a list for --reuse: without it, or naming a tensor no weight reads (a typo would
 * otherwise copy the stale weight it meant), the run stops before writing anything. */
TEST(changed_is_refused_without_reuse_or_for_a_tensor_nothing_reads) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_cr");
    REQUIRE(tiny::write_text(d / "unread", "model.layers.0.mlp.nope.weight\n"));
    REQUIRE(tiny::write_text(d / "real", "model.layers.0.mlp.down_proj.weight\n"));
    const std::vector<std::vector<std::string>> kRuns = {
        { "-o", d / "a.rad", "--recipe", w.recipe, "--changed", d / "real" },
        { "-o", d / "b.rad", "--recipe", w.recipe, "--reuse", w.base, "--changed", d / "unread" },
    };
    for (const auto& args : kRuns) {
        std::string log;
        CHECK(convert(w.ckpt, args, &log) != 0);
        CHECK(!exists(args[1]));
        CHECK(!exists(args[1] + ".blob.tmp"));
    }
}

/* ================================================================== --lora */

namespace {

/* A PEFT adapter file: `<module>.lora_A.weight` [r,K] of 0.25 and `<module>.lora_B.weight`
 * [N,r] of 0.5 -- so B @ A is 0.25 * r everywhere -- and, when `alpha` is given, a rank-0
 * `<module>.alpha` scalar. */
bool write_lora(const std::string& path, const std::string& module, int64_t N, int64_t K,
                int64_t r, const float* alpha) {
    std::vector<tiny::Tensor> ts;
    tiny::Tensor A{ module + ".lora_A.weight", "F32", { r, K }, {} };
    A.bytes.resize((size_t)(r * K) * 4);
    for (int64_t i = 0; i < r * K; ++i) rad_store_f32(A.bytes.data(), RAD_F32, i, 0.25f);
    tiny::Tensor B{ module + ".lora_B.weight", "F32", { N, r }, {} };
    B.bytes.resize((size_t)(N * r) * 4);
    for (int64_t i = 0; i < N * r; ++i) rad_store_f32(B.bytes.data(), RAD_F32, i, 0.5f);
    ts.push_back(std::move(A));
    ts.push_back(std::move(B));
    if (alpha) {
        tiny::Tensor s{ module + ".alpha", "F32", {}, std::vector<uint8_t>(4, 0) };
        rad_store_f32(s.bytes.data(), RAD_F32, 0, *alpha);
        ts.push_back(std::move(s));
    }
    return tiny::write_safetensors(path, ts);
}

/* A copy of the world's checkpoint with each named tensor raised everywhere by its delta, in
 * the same f32 arithmetic add_lora does. */
bool copy_ckpt_bumped(World& w, const std::string& ck,
                      const std::vector<std::pair<std::string, float>>& bumps) {
    if (mkdir(ck.c_str(), 0755) != 0) return false;
    for (const char* f : { "config.json", "tokenizer.json" })
        if (!copy_file(w.ckpt + "/" + f, ck + "/" + f)) return false;
    std::vector<tiny::Tensor> ts = tiny::read_safetensors(w.ckpt + "/model.safetensors");
    size_t hits = 0;
    for (tiny::Tensor& t : ts)
        for (const std::pair<std::string, float>& b : bumps)
            if (t.name == b.first) {
                ++hits;
                for (size_t i = 0; i < t.bytes.size() / 2; ++i)
                    rad_store_f32(t.bytes.data(), RAD_BF16, (int64_t)i,
                                  rad_load_f32(t.bytes.data(), RAD_BF16, (int64_t)i) + b.second);
            }
    return hits == bumps.size() && tiny::write_safetensors(ck + "/model.safetensors", ts);
}

bool copy_ckpt_raised(World& w, const std::string& ck, const std::string& target, float delta) {
    return copy_ckpt_bumped(w, ck, { { target, delta } });
}

}  /* namespace */

/* --lora merges W + B @ A into the tensor the module names, and the merge marks that tensor
 * changed: --reuse converts the weights reading it and copies the rest untouched. The merged
 * container is byte-identical to one converted from a hand-raised checkpoint -- same f32
 * arithmetic, same recipe. */
TEST(lora_merges_into_its_target_and_reconverts_the_weights_that_read_it) {
    World& w = world();
    REQUIRE(w.ok);
    const std::string target = "model.layers.0.mlp.down_proj.weight";
    const auto it = w.src.find(target);
    REQUIRE(it != w.src.end());
    REQUIRE_EQ(it->second.dtype, std::string("BF16"));
    REQUIRE_EQ(it->second.shape.size(), (size_t)2);
    const int64_t N = it->second.shape[0], K = it->second.shape[1];

    /* PEFT names the module, without the target's `.weight`. */
    const std::string mod = target.substr(0, target.size() - 7);
    tiny::TempDir d("convert_l");
    REQUIRE(write_lora(d / "lora.safetensors", mod, N, K, 2, nullptr));
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "merged.rad", "--recipe", w.recipe, "--reuse", w.base,
                                 "--lora", d / "lora.safetensors" }, &log), 0);
    CHECK(has(log, "merged at strength 1"));
    RadFile m;
    REQUIRE_EQ(m.open((d / "merged.rad").c_str()), RAD_OK);
    RadFile base;
    REQUIRE_EQ(base.open(w.base.c_str()), RAD_OK);
    const std::vector<std::string> diff = differing_weights(base, m);
    REQUIRE_EQ(diff.size(), (size_t)1);
    CHECK(!diff.empty() && diff[0] == "blk.0.ffn_down.weight");

    REQUIRE(copy_ckpt_raised(w, d / "ref", target, 0.5f * 0.25f * 2.f));
    REQUIRE_EQ(convert(d / "ref", { "-o", d / "ref.rad", "--recipe", w.recipe }, nullptr), 0);
    RadFile rf;
    REQUIRE_EQ(rf.open((d / "ref.rad").c_str()), RAD_OK);
    const RadFileEntry* a = m.find("blk.0.ffn_down.weight");
    const RadFileEntry* b = rf.find("blk.0.ffn_down.weight");
    REQUIRE(a && b);
    CHECK(entry_bytes(m, *a) == entry_bytes(rf, *b));
}

/* A module spelled under a wrapper prefix resolves to the checkpoint's tensor, and an .alpha
 * states the scale: alpha 4 at rank 2 doubles the merge. */
TEST(lora_strips_wrapper_prefixes_and_scales_by_alpha) {
    World& w = world();
    REQUIRE(w.ok);
    const std::string target = "model.layers.0.mlp.down_proj.weight";
    const auto it = w.src.find(target);
    REQUIRE(it != w.src.end());
    const int64_t N = it->second.shape[0], K = it->second.shape[1];

    tiny::TempDir d("convert_lp");
    const std::string mod = target.substr(0, target.size() - 7);
    const float alpha = 4.f;
    REQUIRE(write_lora(d / "lora.safetensors", "diffusion_model." + mod, N, K, 2, &alpha));
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "merged.rad", "--recipe", w.recipe,
                                 "--lora", d / "lora.safetensors" }, &log), 0);
    REQUIRE(copy_ckpt_raised(w, d / "ref", target, 0.5f * 0.25f * 2.f * 2.f));
    REQUIRE_EQ(convert(d / "ref", { "-o", d / "ref.rad", "--recipe", w.recipe }, nullptr), 0);
    RadFile m, rf;
    REQUIRE_EQ(m.open((d / "merged.rad").c_str()), RAD_OK);
    REQUIRE_EQ(rf.open((d / "ref.rad").c_str()), RAD_OK);
    const RadFileEntry* a = m.find("blk.0.ffn_down.weight");
    const RadFileEntry* b = rf.find("blk.0.ffn_down.weight");
    REQUIRE(a && b);
    CHECK(entry_bytes(m, *a) == entry_bytes(rf, *b));
    /* And --lora-strength scales the whole merge: 0.5 of an alpha/r of 2 is a scale of 1. */
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "half.rad", "--recipe", w.recipe,
                                 "--lora", d / "lora.safetensors", "--lora-strength", "0.5" },
                       nullptr), 0);
    REQUIRE(copy_ckpt_raised(w, d / "ref_half", target, 0.5f * 0.25f * 2.f));
    REQUIRE_EQ(convert(d / "ref_half", { "-o", d / "ref_half.rad", "--recipe", w.recipe },
                       nullptr), 0);
    RadFile h, rh;
    REQUIRE_EQ(h.open((d / "half.rad").c_str()), RAD_OK);
    REQUIRE_EQ(rh.open((d / "ref_half.rad").c_str()), RAD_OK);
    const RadFileEntry* c = h.find("blk.0.ffn_down.weight");
    const RadFileEntry* e = rh.find("blk.0.ffn_down.weight");
    REQUIRE(c && e);
    CHECK(entry_bytes(h, *c) == entry_bytes(rh, *e));
}

/* A module the checkpoint does not ship is refused by name; A and B that do not fit the target
 * are refused with the shapes; --lora with --in-place is refused where --changed is. */
TEST(lora_refusals_name_the_module_the_shapes_or_the_flag) {
    World& w = world();
    REQUIRE(w.ok);
    const std::string target = "model.layers.0.mlp.down_proj.weight";
    const auto it = w.src.find(target);
    REQUIRE(it != w.src.end());
    const int64_t N = it->second.shape[0], K = it->second.shape[1];
    tiny::TempDir d("convert_lf");

    std::string log;
    REQUIRE(write_lora(d / "nope.safetensors", "model.layers.0.mlp.nope_proj", N, K, 2, nullptr));
    CHECK(convert(w.ckpt, { "-o", d / "a.rad", "--recipe", w.recipe,
                            "--lora", d / "nope.safetensors" }, &log) != 0);
    CHECK(has(log, "nope_proj"));
    CHECK(!exists(d / "a.rad"));

    {
        std::vector<tiny::Tensor> ts;
        const std::string mod = target.substr(0, target.size() - 7);
        tiny::Tensor A{ mod + ".lora_A.weight", "F32", { 2, K + 3 }, {} };
        A.bytes.assign((size_t)(2 * (K + 3)) * 4, 0);
        tiny::Tensor B{ mod + ".lora_B.weight", "F32", { N + 1, 2 }, {} };
        B.bytes.assign((size_t)((N + 1) * 2) * 4, 0);
        ts.push_back(std::move(A));
        ts.push_back(std::move(B));
        REQUIRE(tiny::write_safetensors(d / "bad.safetensors", ts));
    }
    CHECK(convert(w.ckpt, { "-o", d / "b.rad", "--recipe", w.recipe,
                            "--lora", d / "bad.safetensors" }, &log) != 0);
    CHECK(has(log, "does not fit"));
    CHECK(!exists(d / "b.rad"));

    CHECK(convert(w.ckpt, { "--recipe", w.recipe, "--reuse", w.base, "--in-place",
                            "--lora", d / "nope.safetensors" }, &log) != 0);
    CHECK(has(log, "--lora merges into the source"));
}

/* A --lora-map rule lands one fused module on several checkpoint tensors: B's rows split
 * along the targets in the order listed, each taking its own row count, and every target
 * joins the changed set. */
TEST(lora_map_fuses_one_module_into_several_targets_by_row_window) {
    World& w = world();
    REQUIRE(w.ok);
    const std::string q = "model.layers.0.self_attn.q_proj.weight";
    const std::string k = "model.layers.0.self_attn.k_proj.weight";
    const std::string v = "model.layers.0.self_attn.v_proj.weight";
    const auto qq = w.src.find(q), qk = w.src.find(k), qv = w.src.find(v);
    REQUIRE(qq != w.src.end() && qk != w.src.end() && qv != w.src.end());
    const int64_t Nq = qq->second.shape[0], Nk = qk->second.shape[0], Nv = qv->second.shape[0],
                  K = qq->second.shape[1];
    REQUIRE(qk->second.shape[1] == K && qv->second.shape[1] == K);
    REQUIRE(Nq != Nk);  /* an asymmetric split can only come from the targets' own row counts */

    tiny::TempDir d("convert_lm");
    REQUIRE(tiny::write_text(d / "map.json",
                             R"({"strip": ["diffusion_model."], "modules": {"layers.*.attn.qkv": [
                                   "model.layers.*.self_attn.q_proj.weight",
                                   "model.layers.*.self_attn.k_proj.weight",
                                   "model.layers.*.self_attn.v_proj.weight"]}})"));
    /* A [2,K] of 0.25; B's three windows read 0.5, 1.0 and 2.0 -- so with no .alpha the
     * merge must raise q, k and v by 0.25, 0.5 and 1.0 (r * b * 0.25 each). No other
     * attribution of B's rows -- one size for all, or the windows in another order --
     * produces those three deltas on those three tensors. */
    const int64_t r = 2, N = Nq + Nk + Nv;
    std::vector<tiny::Tensor> ts;
    tiny::Tensor A{ "diffusion_model.layers.0.attn.qkv.lora_A.weight", "F32", { r, K }, {} };
    A.bytes.resize((size_t)(r * K) * 4);
    for (int64_t i = 0; i < r * K; ++i) rad_store_f32(A.bytes.data(), RAD_F32, i, 0.25f);
    tiny::Tensor B{ "diffusion_model.layers.0.attn.qkv.lora_B.weight", "F32", { N, r }, {} };
    B.bytes.resize((size_t)(N * r) * 4);
    for (int64_t n = 0; n < N; ++n) {
        const float b = n < Nq ? 0.5f : n < Nq + Nk ? 1.0f : 2.0f;
        for (int64_t j = 0; j < r; ++j) rad_store_f32(B.bytes.data(), RAD_F32, n * r + j, b);
    }
    ts.push_back(std::move(A));
    ts.push_back(std::move(B));
    REQUIRE(tiny::write_safetensors(d / "qkv.safetensors", ts));

    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "merged.rad", "--recipe", w.recipe, "--reuse", w.base,
                                 "--lora", d / "qkv.safetensors", "--lora-map", d / "map.json" },
                       &log), 0);
    CHECK(has(log, "merged at strength 1"));
    RadFile m, base;
    REQUIRE_EQ(m.open((d / "merged.rad").c_str()), RAD_OK);
    REQUIRE_EQ(base.open(w.base.c_str()), RAD_OK);
    const std::vector<std::string> diff = differing_weights(base, m);
    CHECK_EQ(diff.size(), (size_t)1);  /* all three merged, and only the weight reading them */
    CHECK(!diff.empty() && diff[0] == "blk.0.attn_qkv.weight");

    REQUIRE(copy_ckpt_bumped(w, d / "ref", { { q, 0.25f }, { k, 0.5f }, { v, 1.0f } }));
    REQUIRE_EQ(convert(d / "ref", { "-o", d / "ref.rad", "--recipe", w.recipe }, nullptr), 0);
    RadFile rf;
    REQUIRE_EQ(rf.open((d / "ref.rad").c_str()), RAD_OK);
    const RadFileEntry* a = m.find("blk.0.attn_qkv.weight");
    const RadFileEntry* b = rf.find("blk.0.attn_qkv.weight");
    REQUIRE(a && b);
    CHECK(entry_bytes(m, *a) == entry_bytes(rf, *b));
}

/* A rule whose target the checkpoint lacks is a refusal, not a miss; a B whose rows do not
 * sum to its targets' rows is a refusal; --lora-map without --lora is a refusal. */
TEST(lora_map_refusals_name_the_target_the_sums_or_the_flag) {
    World& w = world();
    REQUIRE(w.ok);
    const int64_t Nq = w.src["model.layers.0.self_attn.q_proj.weight"].shape[0];
    const int64_t Nk = w.src["model.layers.0.self_attn.k_proj.weight"].shape[0];
    const int64_t K = w.src["model.layers.0.self_attn.q_proj.weight"].shape[1];
    tiny::TempDir d("convert_lmr");

    /* One pair of the fused module's name, B of any row count. */
    const std::string mod = "diffusion_model.layers.0.attn.qkv";
    const auto pair = [&](int64_t rows) {
        std::vector<tiny::Tensor> ts;
        tiny::Tensor A{ mod + ".lora_A.weight", "F32", { 2, K }, {} };
        A.bytes.assign((size_t)(2 * K) * 4, 0);
        tiny::Tensor B{ mod + ".lora_B.weight", "F32", { rows, 2 }, {} };
        B.bytes.assign((size_t)(rows * 2) * 4, 0);
        ts.push_back(std::move(A));
        ts.push_back(std::move(B));
        return tiny::write_safetensors(d / "p.safetensors", ts);
    };

    std::string log;
    REQUIRE(tiny::write_text(d / "ghost.json",
                             R"({"strip": ["diffusion_model."], "modules": {"layers.*.attn.qkv": [
                                   "model.layers.*.self_attn.q_proj.weight",
                                   "model.layers.*.self_attn.ghost_proj.weight"]}})"));
    REQUIRE(pair(Nq + Nk));
    CHECK(convert(w.ckpt, { "-o", d / "a.rad", "--recipe", w.recipe, "--lora", d / "p.safetensors",
                            "--lora-map", d / "ghost.json" }, &log) != 0);
    CHECK(has(log, "ghost_proj"));
    CHECK(has(log, "does not ship"));
    CHECK(!exists(d / "a.rad"));

    REQUIRE(tiny::write_text(d / "ok.json",
                             R"({"strip": ["diffusion_model."], "modules": {"layers.*.attn.qkv": [
                                   "model.layers.*.self_attn.q_proj.weight",
                                   "model.layers.*.self_attn.k_proj.weight"]}})"));
    REQUIRE(pair(Nq + Nk + 1));  /* one row past the two targets */
    CHECK(convert(w.ckpt, { "-o", d / "b.rad", "--recipe", w.recipe, "--lora", d / "p.safetensors",
                            "--lora-map", d / "ok.json" }, &log) != 0);
    CHECK(has(log, "rows sum to"));
    CHECK(!exists(d / "b.rad"));

    CHECK(convert(w.ckpt, { "-o", d / "c.rad", "--recipe", w.recipe,
                            "--lora-map", d / "ok.json" }, &log) != 0);
    CHECK(has(log, "--lora-map renames"));
    CHECK(!exists(d / "c.rad"));
}

/* A checkpoint tensor no declared weight reads is not written (a tower or head the plugin does not
 * know), and the run names it; a tensor beside one that is read (its scale) is not named, nor is
 * anything when every tensor is read. */
TEST(a_checkpoint_tensor_no_weight_reads_is_named) {
    World& w = world();
    REQUIRE(w.ok);
    CHECK(w.log.find("read by no declared weight") == std::string::npos);
    tiny::TempDir d("convert_u");
    const std::string ck = d / "llama";
    REQUIRE(mkdir(ck.c_str(), 0755) == 0);
    for (const char* f : { "config.json", "tokenizer.json" })
        REQUIRE(copy_file(w.ckpt + "/" + f, ck + "/" + f));
    std::vector<tiny::Tensor> ts = tiny::read_safetensors(w.ckpt + "/model.safetensors");
    const size_t n = ts.size();
    ts.push_back({ "model.vision_embedder.patch_dense.weight", "BF16", { 4, 4 },
                   std::vector<uint8_t>(32, 0) });
    ts.push_back({ "model.layers.0.mlp.down_proj.weight_scale", "BF16", { 1 },
                   std::vector<uint8_t>(2, 0) });
    REQUIRE(tiny::write_safetensors(ck + "/model.safetensors", ts));
    std::string log;
    REQUIRE_EQ(convert(ck, { "-o", d / "u.rad", "--recipe", w.recipe }, &log), 0);
    const std::string want = "1 of " + std::to_string(n + 2) + " checkpoint tensors are read by no "
                             "declared weight and are not written: model.vision_embedder.patch_dense.weight";
    if (log.find(want) == std::string::npos) fprintf(stderr, "    log:\n%s\n", log.c_str());
    CHECK(log.find(want) != std::string::npos);
    CHECK(log.find("down_proj.weight_scale") == std::string::npos);
}

/* ================================================================== --in-place */

/* Extending a container with nothing new keeps every weight at its offset, and the .pre-append
 * file (the old header and length) restores the original byte for byte. */
TEST(in_place_keeps_every_weight_where_it_was_and_can_be_undone) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_ip");
    const std::string ip = d / "ip.rad";
    REQUIRE(copy_file(w.base, ip));
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "--recipe", w.recipe, "--reuse", ip, "--in-place" }, &log), 0);
    CHECK(has(log, "15 weight(s)"));
    RadFile a, b;
    REQUIRE_EQ(a.open(w.base.c_str()), RAD_OK);
    REQUIRE_EQ(b.open(ip.c_str()), RAD_OK);
    REQUIRE_EQ(a.entry_count(), b.entry_count());
    for (int64_t i = 0; i < a.entry_count(); ++i) {
        const RadFileEntry* e = b.find(a.str(a.entry(i).name));
        REQUIRE(e != nullptr);
        CHECK_EQ(e->offset, a.entry(i).offset);
    }
    b.close();

    const std::string bak = tiny::read_text(ip + ".pre-append");
    REQUIRE_EQ(bak.size(), sizeof(RadFileHeader) + sizeof(int64_t));
    int64_t len = 0;
    std::memcpy(&len, bak.data() + sizeof(RadFileHeader), sizeof len);
    std::string file = tiny::read_text(ip);
    REQUIRE(len > 0 && (size_t)len <= file.size());
    file.replace(0, sizeof(RadFileHeader), bak.data(), sizeof(RadFileHeader));
    file.resize((size_t)len);
    CHECK(file == tiny::read_text(w.base));
}

/* A weight the container holds in another movement group than the plugin now declares (a plugin
 * that regrouped it after the conversion) is kept in place, its entry taking the declared group:
 * the group places no bytes of a dense weight. */
TEST(in_place_keeps_a_weight_the_plugin_has_regrouped) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_ipg");
    const std::string ip = d / "ip.rad";
    REQUIRE(copy_file(w.base, ip));
    RadFile a;
    REQUIRE_EQ(a.open(w.base.c_str()), RAD_OK);
    REQUIRE(a.entry_count() > 0);
    const RadFileEntry& e0 = a.entry(0);
    REQUIRE(e0.expert < 0);
    const std::string name = a.str(e0.name);
    const int32_t layer = e0.layer;
    const uint64_t off = e0.offset;
    {
        std::string file = tiny::read_text(ip);
        const size_t at = (size_t)a.header().dir_off + offsetof(RadFileEntry, layer);
        const int32_t other = layer + 7;
        std::memcpy(&file[at], &other, sizeof other);
        REQUIRE(tiny::write_text(ip, file));
    }
    a.close();
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "--recipe", w.recipe, "--reuse", ip, "--in-place" }, &log), 0);
    RadFile b;
    REQUIRE_EQ(b.open(ip.c_str()), RAD_OK);
    const RadFileEntry* e = b.find(name.c_str());
    REQUIRE(e != nullptr);
    CHECK_EQ(e->layer, layer);
    CHECK_EQ(e->offset, off);
}

/* In place, a rule that would write a weight differently is refused unless --replace names the
 * weight, and the file is left as it was. Without --in-place, -o may not name the --reuse
 * container, whose weights the run reads while it truncates the output. */
TEST(in_place_refuses_to_change_a_weight_and_reuse_never_truncates_its_source) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_ipr");
    const std::string ip = d / "ip.rad";
    REQUIRE(copy_file(w.base, ip));
    REQUIRE(tiny::write_text(d / "search.recipe",
                             std::string("blk.0.attn_qkv.weight rtn codes=i8 group=128 scale=bf16 "
                                         "transform=fwht128 clamp=sym rule=search\n") +
                                 tiny::kLlamaI8Recipe));
    std::string log;
    CHECK(convert(w.ckpt, { "--recipe", d / "search.recipe", "--reuse", ip, "--in-place" },
                  &log) != 0);
    CHECK(has(log, "Name it with --replace"));
    CHECK(tiny::read_text(ip) == tiny::read_text(w.base));
    CHECK_EQ(convert(w.ckpt, { "-o", ip, "--recipe", w.recipe, "--reuse", ip }, &log), 2);
    CHECK(tiny::read_text(ip) == tiny::read_text(w.base));
}

/* --replace: the named weight is written again past the container's end under its new rule, equal
 * to a conversion to a new file; every other weight stays at its offset, and the old planes are
 * reported as unreachable. A glob no weight matches, or --replace without --in-place, is refused. */
TEST(in_place_replace_writes_the_named_weight_again_past_the_end) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_ipx");
    const std::string ip = d / "ip.rad", fresh = d / "fresh.rad";
    REQUIRE(copy_file(w.base, ip));
    const char* kName = "blk.0.attn_qkv.weight";
    REQUIRE(tiny::write_text(d / "search.recipe",
                             std::string(kName) + " rtn codes=i8 group=128 scale=bf16 "
                                                  "transform=fwht128 clamp=sym rule=search\n" +
                                 tiny::kLlamaI8Recipe));
    std::string log;
    CHECK(convert(w.ckpt, { "--recipe", w.recipe, "--reuse", ip, "--in-place",
                            "--replace", "blk.9.nope" }, &log) != 0);
    CHECK(has(log, "matches no weight"));
    CHECK(tiny::read_text(ip) == tiny::read_text(w.base));
    CHECK_EQ(convert(w.ckpt, { "-o", fresh, "--recipe", d / "search.recipe", "--reuse", w.base,
                               "--replace", kName }, &log), 2);

    REQUIRE_EQ(convert(w.ckpt, { "--recipe", d / "search.recipe", "--reuse", ip, "--in-place",
                                 "--replace", kName }, &log), 0);
    CHECK(has(log, "1 of the added weight(s) replace"));
    REQUIRE_EQ(convert(w.ckpt, { "-o", fresh, "--recipe", d / "search.recipe" }, &log), 0);
    RadFile a, b, c;
    REQUIRE_EQ(a.open(w.base.c_str()), RAD_OK);
    REQUIRE_EQ(b.open(ip.c_str()), RAD_OK);
    REQUIRE_EQ(c.open(fresh.c_str()), RAD_OK);
    REQUIRE_EQ(a.entry_count(), b.entry_count());
    for (int64_t i = 0; i < a.entry_count(); ++i) {
        const std::string name = a.str(a.entry(i).name);
        const RadFileEntry* e = b.find(name.c_str());
        const RadFileEntry* f = c.find(name.c_str());
        REQUIRE(e != nullptr);
        REQUIRE(f != nullptr);
        if (name == kName) {
            CHECK(e->offset >= (uint64_t)a.size());
            CHECK(has(b.str(e->options), "rule=search"));
        } else {
            CHECK_EQ(e->offset, a.entry(i).offset);
        }
        REQUIRE_EQ(e->n_planes, f->n_planes);
        for (uint32_t k = 0; k < e->n_planes; ++k) {
            const RadFilePlane &pe = b.plane(*e, (int)k), &pf = c.plane(*f, (int)k);
            REQUIRE_EQ(pe.bytes, pf.bytes);
            CHECK(std::memcmp(b.data(pe), c.data(pf), (size_t)pe.bytes) == 0);
        }
    }
}

/* ================================================================== refusals and options */

/* --plan-only declares and plans every weight, and writes nothing. */
TEST(plan_only_writes_nothing) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_p");
    std::string log;
    CHECK_EQ(convert(w.ckpt, { "-o", d / "p.rad", "--recipe", w.recipe, "--plan-only" }, &log), 0);
    CHECK(has(log, "nothing was written"));
    CHECK(!exists(d / "p.rad"));
}

/* Inkling 2's routed experts as the release stores them: e2m1 codes, an e4m3 scale per 16 of K and an
 * f32 level per expert, named by hf_quant_config.json's nvfp4 parameter group. The plugin serves no
 * NVFP4 expert, so a recipe that keeps them as stored is refused by name before anything is
 * written; data/recipes/inkling2-mini-bf16.recipe's rules (the experts dequantised and cast to
 * bf16, the calibration container) write, and the writer checks each expert's slot order. */
TEST(an_inkling2_nvfp4_expert_checkpoint_converts_to_the_bf16_calibration_form) {
    tiny::TempDir d("convert_nv");
    REQUIRE(d.ok());
    const std::string ckpt = d / "inkling";
    REQUIRE(mkdir(ckpt.c_str(), 0755) == 0);
    /* a hidden and a ffn width of 128: the stored int8 and the NVFP4 int8 activations are 128-wide */
    tiny::Inkling2 s;
    s.hidden = 128;
    s.ffn = 128;
    REQUIRE(tiny::write_inkling2(ckpt, s));

    std::vector<tiny::Tensor> ts = tiny::read_safetensors(ckpt + "/model.safetensors");
    REQUIRE(!ts.empty());
    const int64_t E = s.experts, H = s.hidden, F = s.ffn;
    /* e4m3 bytes 0.5, 1.0 and 2.0 in turn, and e2m1 codes that exercise every nibble */
    const uint8_t e4m3[3] = { 0x30, 0x38, 0x40 };
    auto nvfp4 = [&](const std::string& base, int64_t N, int64_t K, float amax) {
        tiny::Tensor w{ base + "_weight", "U8", { E, N, K / 2 }, {} };
        for (int64_t i = 0; i < E * N * K / 2; ++i) w.bytes.push_back((uint8_t)((i * 37 + 11) & 0xff));
        tiny::Tensor sc{ base + "_weight_scale", "F8_E4M3", { E, N, K / 16 }, {} };
        for (int64_t i = 0; i < E * N * K / 16; ++i) sc.bytes.push_back(e4m3[i % 3]);
        tiny::Tensor a{ base + "_weight_amax", "F32", { E }, std::vector<uint8_t>((size_t)E * 4) };
        for (int64_t e = 0; e < E; ++e) {
            const float v = amax * (float)(e + 1);
            std::memcpy(a.bytes.data() + e * 4, &v, 4);
        }
        tiny::Tensor dn{ base + "_weight_outer_scale_denominator", "F32", { 1 }, std::vector<uint8_t>(4) };
        const float denom = 2.0f;
        std::memcpy(dn.bytes.data(), &denom, 4);
        return std::vector<tiny::Tensor>{ w, sc, a, dn };
    };
    std::vector<tiny::Tensor> out;
    std::vector<std::string> names;
    for (tiny::Tensor& t : ts) {
        const bool expert = t.name.find(".mlp.experts.w13_weight") != std::string::npos ||
                            t.name.find(".mlp.experts.w2_weight") != std::string::npos;
        if (!expert) out.push_back(std::move(t));
    }
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".mlp.experts.";
        for (const tiny::Tensor& t : nvfp4(p + "w13", 2 * F, H, 3.0f)) out.push_back(t);
        for (const tiny::Tensor& t : nvfp4(p + "w2", H, F, 6.0f)) out.push_back(t);
        names.push_back(p + "w13_weight");
        names.push_back(p + "w2_weight");
    }
    REQUIRE(tiny::write_safetensors(ckpt + "/model.safetensors", out));

    std::string groups = "{\"model\":{\"parameter_groups\":[{\"compute_dtype\":\"nvfp4\",\"tensors\":[";
    for (size_t i = 0; i < names.size(); ++i)
        groups += (i ? ",\"" : "\"") + names[i] + "\"";
    groups += "]}]}}";
    REQUIRE(tiny::write_text(ckpt + "/hf_quant_config.json", groups));
    const std::string keep = d / "keep.recipe";
    REQUIRE(tiny::write_text(keep, "# every routed expert is kept as stored\n"));
    std::string log;
    CHECK(convert(ckpt, { "-o", d / "nv.rad", "--recipe", keep }, &log) != 0);
    CHECK(has(log, "blk.0.e0.gu"));
    CHECK(has(log, "data/recipes/inkling2-mini-bf16.recipe"));
    CHECK(!exists(d / "nv.rad"));

    const std::string bf16 = d / "bf16.recipe";
    REQUIRE(tiny::write_text(bf16, "blk.*.e*.gu cast dtype=bf16\nblk.*.e*.dn cast dtype=bf16\n"));
    // a full write, not plan-only: the writer checks that each expert's slots run 0..n-1 in order
    CHECK_EQ(convert(ckpt, { "-o", d / "bf.rad", "--recipe", bf16 }, &log), 0);
    CHECK(exists(d / "bf.rad"));
}

/* A recipe or flag that cannot be honoured stops the run with status 2 before any file exists:
 * an unknown quantiser, option, or non-integer value, a recipe that cannot be read, an unset
 * $VARIABLE, a --set without `=`, and a KV dtype that is not bf16 or fp8. */
TEST(a_recipe_or_flag_that_cannot_be_honoured_is_refused_before_writing) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_x");
    const char* kRule = "blk.*.attn_qkv.weight ";
    const std::pair<const char*, std::string> kRecipes[] = {
        { "quantiser", std::string(kRule) + "nope codes=fp8_e4m3" },
        { "option", std::string(kRule) + "rtn codez=fp8_e4m3 block=128x128 scale=bf16" },
        { "integer", std::string(kRule) + "rtn codes=i4 group=abc scale=f16" },
        { "variable", std::string(kRule) + "rtn codes=$RAD_CONVERT_TEST_UNSET block=128x128" },
    };
    int k = 0;
    for (const auto& [what, text] : kRecipes) {
        const std::string r = d / (std::string(what) + ".recipe");
        const std::string out = d / ("x" + std::to_string(k++));
        REQUIRE(tiny::write_text(r, text + "\n"));
        std::string log;
        const int st = convert(w.ckpt, { "-o", out, "--recipe", r }, &log,
                               { "RAD_CONVERT_TEST_UNSET" });
        if (st != 2) fprintf(stderr, "    %s: status %d\n%s\n", what, st, log.c_str());
        CHECK_EQ(st, 2);
        CHECK(!exists(out));
    }
    const std::vector<std::vector<std::string>> kFlags = {
        { "--recipe", d / "absent.recipe" },
        { "--recipe", w.recipe, "--set", "foo" },
        { "--recipe", w.recipe, "--kv-cache-dtype", "auto" },
    };
    for (const auto& args : kFlags) {
        std::vector<std::string> a = { "-o", d / "f.rad" };
        a.insert(a.end(), args.begin(), args.end());
        CHECK_EQ(convert(w.ckpt, a, nullptr), 2);
        CHECK(!exists(d / "f.rad"));
    }
}

/* A rule that declines a GGUF block tensor stops the run rather than keep the tensor's planes: the
 * tiny LLaMA as a GGUF with its first attn_norm in Q8_0, under an `rtn` rule, which declines a
 * rank-1 tensor. */
TEST(a_rule_that_declines_a_gguf_block_tensor_is_an_error) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_gguf");
    const tiny::Llama s;
    const std::vector<ggufw::GKv> kvs = {
        { "general.architecture", 8, 0, { "llama" } },
        { "llama.block_count", 4, (double)s.layers, {} },
        { "llama.embedding_length", 4, (double)s.hidden, {} },
        { "llama.attention.head_count", 4, (double)s.heads, {} },
        { "llama.attention.head_count_kv", 4, (double)s.kv_heads, {} },
        { "llama.attention.key_length", 4, (double)s.head_dim, {} },
        { "llama.feed_forward_length", 4, (double)s.ffn, {} },
        { "llama.context_length", 4, (double)s.ctx, {} },
        { "llama.vocab_size", 4, (double)s.vocab, {} },
    };
    std::vector<ggufw::GTensor> ts;
    for (const auto& [name, t] : w.src) {
        ggufw::GTensor g{ name, t.dtype == "F32" ? 0u : 30u, {}, t.bytes };
        g.ne.assign(t.shape.rbegin(), t.shape.rend());
        if (name == "model.layers.0.input_layernorm.weight") {
            g.type = 8;
            g.bytes.clear();
            for (uint64_t b = 0; b < g.ne[0] / 32; ++b) {
                g.bytes.insert(g.bytes.end(), { 0x00, 0x3C });   /* d = 1 in f16 */
                g.bytes.insert(g.bytes.end(), 32, 1);
            }
        }
        ts.push_back(std::move(g));
    }
    REQUIRE(ggufw::write_gguf(d / "llama.gguf", kvs, ts));
    std::string log;
    CHECK(convert(d / "llama.gguf",
                  { "-o", d / "g.rad", "--recipe", w.recipe, "--quant",
                    "blk.0.attn_norm.weight=rtn:codes=i8,group=32,scale=bf16" },
                  &log) != 0);
    CHECK(!exists(d / "g.rad"));
}

/* ================================================================== an int4 release */

namespace {

/* compressed-tensors' value of one packed linear of a tiny int4 release (tiny_model.h,
 * write_llama_ct_int4): (q - zero_point) * scale, f32 [N, K]. */
std::vector<float> ct_values(const std::map<std::string, tiny::Tensor>& src, const std::string& b) {
    const tiny::Tensor& p = src.at(b + ".weight_packed");
    const tiny::Tensor& s = src.at(b + ".weight_scale");
    const auto z = src.find(b + ".weight_zero_point");
    const int64_t N = p.shape[0], K = p.shape[1] * 8, G = s.shape[1];
    std::vector<float> v((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            uint32_t w;
            std::memcpy(&w, p.bytes.data() + (n * (K / 8) + k / 8) * 4, 4);
            const int q = (int)((w >> (4 * (k % 8))) & 15) - 8;
            int zp = 0;
            if (z != src.end()) {
                std::memcpy(&w, z->second.bytes.data() + ((n / 8) * G + k / 128) * 4, 4);
                zp = (int)((w >> (4 * (n % 8))) & 15) - 8;
            }
            v[(size_t)(n * K + k)] =
                (float)(q - zp) * rad_load_f32(s.bytes.data(), RAD_BF16, n * G + k / 128);
        }
    return v;
}

}  /* namespace */

/* An int4 W4A16 release (compressed-tensors, asymmetric with zero points, and symmetric with
 * act-order static) converts with the import rule into the 4-bit GEMMs' form,
 * `u4*f16[1x128]-u4[1x128]`, each joined weight decoding to compressed-tensors' (q - zp) * scale
 * bit for bit, its rule recorded, and the run saying every shipped value is exact. Declaring it
 * resolves the cast, the f16 4-bit GEMM, the int8 quantiser and the int8 4-bit GEMM on the host
 * libraries. Without the rule the plugin refuses the shipped form by name. */
TEST(an_int4_release_imports_exactly_through_the_import_rule) {
    for (const bool asym : { true, false }) {
        tiny::TempDir d(asym ? "convert_i4a" : "convert_i4s");
        REQUIRE(d.ok());
        const std::string ck = d / "llama-int4", recipe = d / "int4.recipe", out = d / "i4.rad";
        REQUIRE_EQ(mkdir(ck.c_str(), 0755), 0);
        REQUIRE(tiny::write_llama_ct_int4(ck, {}, asym));
        REQUIRE(tiny::write_text(recipe, tiny::kInt4ImportRecipe));
        std::map<std::string, tiny::Tensor> src;
        for (tiny::Tensor& t : tiny::read_safetensors(ck + "/model.safetensors"))
            src[t.name] = std::move(t);
        std::string log;
        const int st = convert(ck, { "-o", out, "--recipe", recipe }, &log);
        if (st != 0) fprintf(stderr, "%s\n", log.c_str());
        REQUIRE_EQ(st, 0);
        CHECK(has(log, "every shipped value exact"));
        RadFile f;
        REQUIRE_EQ(f.open(out.c_str()), RAD_OK);
        const struct { const char* weight; std::vector<std::string> parts; } kJoins[] = {
            { "blk.0.attn_qkv.weight", { "model.layers.0.self_attn.q_proj",
                                         "model.layers.0.self_attn.k_proj",
                                         "model.layers.0.self_attn.v_proj" } },
            { "blk.1.ffn_gate_up.weight", { "model.layers.1.mlp.gate_proj",
                                            "model.layers.1.mlp.up_proj" } },
            { "blk.1.ffn_down.weight", { "model.layers.1.mlp.down_proj" } },
            { "blk.0.attn_output.weight", { "model.layers.0.self_attn.o_proj" } },
        };
        for (const auto& j : kJoins) {
            const RadFileEntry* e = f.find(j.weight);
            REQUIRE(e != nullptr);
            CHECK_EQ(enc_name(f.encoding(*e)), std::string("u4*f16[1x128]-u4[1x128]"));
            CHECK_EQ(std::string(f.str(e->quantizer)), std::string("import"));
            CHECK_EQ(std::string(f.str(e->options)), std::string("codes=u4,scale=f16,zero=u4"));
            std::vector<float> want;
            for (const std::string& part : j.parts) {
                const std::vector<float> v = ct_values(src, part);
                want.insert(want.end(), v.begin(), v.end());
            }
            const std::vector<float> got = decode(f, *e);
            REQUIRE_EQ(got.size(), want.size());
            int64_t bad = 0;
            for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
            CHECK_EQ(bad, (int64_t)0);
        }
        /* The head the release leaves bf16 stays so. */
        const RadFileEntry* head = f.find("output.weight");
        REQUIRE(head != nullptr);
        CHECK_EQ(enc_name(f.encoding(*head)), std::string("bf16"));

        std::string log2;
        CHECK(convert(ck, { "-o", d / "none.rad" }, &log2) != 0);
        CHECK(has(log2, asym ? "u4*bf16[1x128]-u4[1x128]" : "i4*bf16[1x128]"));
        CHECK(has(log2, "import rule"));
    }
}

/* A rule that matches nothing is almost always a typo, so it is named; the run goes on. */
TEST(a_rule_that_matches_no_weight_is_warned_about) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_w");
    REQUIRE(tiny::write_text(d / "r", std::string("blk.*.attn_qvk.weight rtn codes=fp8_e4m3 "
                                                  "block=128x128 scale=bf16\n") +
                                          tiny::kLlamaI8Recipe));
    std::string log;
    CHECK_EQ(convert(w.ckpt, { "-o", d / "w.rad", "--recipe", d / "r" }, &log), 0);
    CHECK(has(log, "'blk.*.attn_qvk.weight rtn' matches no weight"));
}

/* --set writes a key into the container, where declare and the engine both read it; setting the
 * quantization_config makes it the header's descriptor, which selects the plugin. */
TEST(set_writes_a_metadata_key_and_can_state_the_quantisation) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_s");
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "s.rad", "--recipe", w.recipe, "--set", "foo.bar=7",
                                 "--set", "quantization_config.quant_method=fp8",
                                 "--set", "quantization_config.fmt=e4m3" }, nullptr), 0);
    RadFile f;
    REQUIRE_EQ(f.open((d / "s.rad").c_str()), RAD_OK);
    const char* v = f.get_s("foo.bar");
    int64_t i = -1;
    CHECK((v && std::string(v) == "7") || (f.get_i("foo.bar", &i) && i == 7));
    CHECK_EQ(std::string(f.str(f.header().quant)), std::string("fp8_e4m3"));
}

/* generation_config.json beside the checkpoint travels as `generation.*` keys, which is where
 * the engine reads a container's sampling defaults. */
TEST(generation_config_travels_as_generation_keys) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_g");
    const std::string ck = d / "llama";
    REQUIRE(mkdir(ck.c_str(), 0755) == 0);
    for (const char* f : { "config.json", "tokenizer.json", "model.safetensors" })
        REQUIRE(copy_file(w.ckpt + "/" + f, ck + "/" + f));
    REQUIRE(tiny::write_text(
        ck + "/generation_config.json",
        R"({"temperature": 0.6, "top_k": 20, "top_p": 0.95, "do_sample": true})"));
    REQUIRE_EQ(convert(ck, { "-o", d / "g.rad", "--recipe", w.recipe }, nullptr), 0);
    RadFile f;
    REQUIRE_EQ(f.open((d / "g.rad").c_str()), RAD_OK);
    std::vector<std::string> keep;
    GenSampling g;
    CHECK(gen_sampling_from_kv(
        [&](const char* k) -> const char* {
            const std::string key = std::string("generation.") + k;
            if (const char* s = f.get_s(key.c_str())) return s;
            int64_t i;
            double x;
            if (f.get_i(key.c_str(), &i)) keep.push_back(std::to_string(i));
            else if (f.get_f(key.c_str(), &x)) keep.push_back(std::to_string(x));
            else return nullptr;
            return keep.back().c_str();
        },
        &g));
    CHECK_NEAR(g.temp, 0.6, 1e-6);
    CHECK_EQ(g.top_k, 20);
    CHECK_NEAR(g.top_p, 0.95, 1e-6);
}

/* Declare runs at the KV dtype the engine will use; the int8 cache resolves on the host kernels
 * too. */
TEST(an_int8_kv_cache_declares) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_k");
    CHECK_EQ(convert(w.ckpt, { "-o", d / "k.rad", "--recipe", w.recipe, "--kv-cache-dtype", "i8" },
                     nullptr), 0);
    CHECK(exists(d / "k.rad"));
}

/* --chat-template bakes a template into the container's vocab section, for a release that ships
 * none (DeepSeek's). Without the flag the container carries what the checkpoint carried. A missing
 * file is refused, not silently skipped. */
TEST(chat_template_flag_bakes_the_template) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_ct");
    const std::string tmpl = d / "chat_template.jinja";
    REQUIRE(tiny::write_text(tmpl, "{%- raw %}{{ messages }}{%- endraw %}\n\n"));
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "with.rad", "--recipe", w.recipe,
                                 "--chat-template", tmpl }, &log), 0);
    RadFile f;
    REQUIRE_EQ(f.open((d / "with.rad").c_str()), RAD_OK);
    VocabBuild vb;
    REQUIRE_EQ(vocab_deserialize(f.base(), (uint64_t)f.size(), f.vocab().h, vb), RAD_OK);
    CHECK_EQ(vb.chat_template, std::string("{%- raw %}{{ messages }}{%- endraw %}"));

    std::string bad;
    CHECK_EQ(convert(w.ckpt, { "-o", d / "bad.rad", "--recipe", w.recipe,
                               "--chat-template", d / "absent.jinja" }, &bad), 1);
}

/* A convert stopped after one large weight keeps its journal and a resume finishes the rest
 * without rewriting it; the resumed container is byte-identical to an uninterrupted one. */
TEST(a_checkpointed_write_resumes_and_keeps_its_finished_weights) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_r");
    const std::string part = d / "r.rad";
    std::string log;
    /* --large-f32-bytes 1: every weight is a large one, so the journal records each. */
    REQUIRE_EQ(convert(w.ckpt, { "-o", part, "--recipe", w.recipe, "--large-f32-bytes", "1",
                                 "--checkpoint-at", "1" }, &log), 3);
    CHECK(exists(part + ".journal"));
    std::string rlog;
    REQUIRE_EQ(convert(w.ckpt, { "-o", part, "--recipe", w.recipe, "--large-f32-bytes", "1", "--resume" },
                       &rlog), 0);
    CHECK(rlog.find("resume") != std::string::npos);
    CHECK(!exists(part + ".journal"));
    RadFile a, b;
    REQUIRE_EQ(a.open(w.base.c_str()), RAD_OK);
    REQUIRE_EQ(b.open(part.c_str()), RAD_OK);
    const std::vector<std::string> diff = differing_weights(a, b);
    CHECK(diff.empty());
    if (!diff.empty()) fprintf(stderr, "    resumed differs at %s\n", diff[0].c_str());
}

RAD_TEST_MAIN()
