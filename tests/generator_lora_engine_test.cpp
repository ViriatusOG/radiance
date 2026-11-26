/* Public Engine API over a tiny synthetic generator on the host backend. No GPU or downloads. */
#include "rad_test.h"
#include "tiny_model.h"
#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <utility>

namespace {

// The plugin registry is process-wide. Both tests use one isolated home, kept alive until exit,
// so unrelated stale plugins are excluded without changing the registry's identity between cases.
const std::string& generator_test_home() {
    static tiny::TempDir directory("generator-lora-home");
    static const std::string home = [&]() -> std::string {
        if (!directory.ok()) return {};
        const std::string path = directory / "home";
        std::error_code error;
        for (const char* sub : {"kernels", "architectures", "quantizers"}) {
            std::filesystem::create_directories(path + "/" + sub, error);
            if (error) return {};
        }
        for (const char* file : {"kernels/libref.so", "architectures/qwenimage21.so", "quantizers/libquant.so"}) {
            if (!std::filesystem::copy_file(std::string(RAD_E2E_HOME) + "/" + file,
                                            path + "/" + file, error) || error) return {};
        }
        return path;
    }();
    return home;
}

} // namespace

TEST(discovered_replacements_and_explicit_reload_keep_in_flight_combinations) {
    tiny::TempDir dir("generator-lora-engine");
    REQUIRE(dir.ok());
    const std::string pipeline = dir / "pipeline", library = dir / "loras";
    REQUIRE_EQ(::mkdir(pipeline.c_str(), 0700), 0);
    REQUIRE_EQ(::mkdir(library.c_str(), 0700), 0);
    const std::string home = generator_test_home();
    REQUIRE(!home.empty());
    tiny::QwenImage shape;
    shape.te_hidden = 128;
    shape.te_head_dim = 32;
    shape.te_ffn = 256;
    REQUIRE(tiny::write_qwen_image(pipeline, shape));
    const std::string model = dir / "tiny.rad";
    std::string conversion_log;
    const int conversion_status = tiny::run({RAD_E2E_CONVERT, pipeline, "-o", model, "--kernels", "libref"},
        {std::string("RADIANCE_HOME=") + home, "ROCR_VISIBLE_DEVICES=99"},
        &conversion_log, 60);
    if (conversion_status != 0) std::fprintf(stderr, "conversion failed (%d):\n%s\n", conversion_status, conversion_log.c_str());
    REQUIRE_EQ(conversion_status, 0);
    const std::string path = library + "/local.safetensors", replacement = library + "/replacement";
    auto write_adapter = [&](const std::string& to, float gain) {
        tiny::Rng random(1);
        const std::string key = "transformer.transformer_blocks.0.attn.to_q";
        return tiny::write_safetensors(to, {
            tiny::noise(key + ".lora_A.weight", {1, 128}, random, 0, gain, "F32"),
            tiny::noise(key + ".lora_B.weight", {128, 1}, random, 0, 1, "F32")});
    };
    REQUIRE(write_adapter(path, 1));
    rad::Config config;
    config.model = model;
    config.radiance_home = home;
    config.kernel_hierarchy = {"libref"};
    config.accept_reference_kernels = true;
    config.max_seqs = 2;
    config.max_ctx = 2048;
    config.image_gen_max_pixels = 64 * 64;
    config.prefix_cache = false;
    config.lora.dir = library;
    config.lora.max_rank = 2;
    config.lora.max_loras = 3; // Each held revision remains live; no synthetic epoch advance.
    rad::Engine engine(std::move(config));
    REQUIRE_EQ(engine.start(), RAD_OK);
    int32_t first_slot = -1, second_slot = -1, reloaded_slot = -1;
    std::shared_ptr<void> first_hold, second_hold, reloaded_hold;
    std::string note, why;
    const std::vector<rad::Engine::LoraUse> uses = {{"local", 1.0}};
    REQUIRE_EQ(engine.gen_lora_acquire(uses, &first_slot, &first_hold, &note, &why), RAD_OK);
    REQUIRE(first_slot >= 0);
    REQUIRE(first_hold.use_count() > 0);
    REQUIRE(write_adapter(replacement, 2));
    REQUIRE_EQ(::rename(replacement.c_str(), path.c_str()), 0);
    REQUIRE_EQ(engine.gen_lora_acquire(uses, &second_slot, &second_hold, &note, &why), RAD_OK);
    CHECK(second_slot != first_slot);
    CHECK(first_hold.use_count() > 0);
    int held = 0;
    REQUIRE_EQ(engine.lora_unload("local", &why, &held), RAD_OK);
    CHECK_EQ(held, 2);
    auto names = engine.gen_lora_names();
    CHECK(std::find(names.begin(), names.end(), "local") == names.end());
    std::shared_ptr<void> rejected;
    int32_t rejected_slot = -1;
    CHECK_EQ(engine.gen_lora_acquire(uses, &rejected_slot, &rejected, &note, &why), RAD_E_NOTFOUND);
    REQUIRE_EQ(engine.lora_load("local", path, &why), RAD_OK);
    REQUIRE_EQ(engine.gen_lora_acquire(uses, &reloaded_slot, &reloaded_hold, &note, &why), RAD_OK);
    CHECK(reloaded_slot != first_slot);
    CHECK(reloaded_slot != second_slot);
    names = engine.gen_lora_names();
    CHECK(std::find(names.begin(), names.end(), "local") != names.end());
    CHECK(first_hold.use_count() > 0);
    CHECK(second_hold.use_count() > 0);
}

TEST(validation_keeps_the_only_held_slot_and_rejects_present_incompatible_files) {
    tiny::TempDir dir("generator-lora-engine");
    REQUIRE(dir.ok());
    const std::string pipeline = dir / "pipeline", library = dir / "loras";
    REQUIRE_EQ(::mkdir(pipeline.c_str(), 0700), 0);
    REQUIRE_EQ(::mkdir(library.c_str(), 0700), 0);
    const std::string home = generator_test_home();
    REQUIRE(!home.empty());
    tiny::QwenImage shape;
    shape.te_hidden = 128;
    shape.te_head_dim = 32;
    shape.te_ffn = 256;
    REQUIRE(tiny::write_qwen_image(pipeline, shape));
    const std::string model = dir / "tiny.rad";
    std::string conversion_log;
    const int conversion_status = tiny::run({RAD_E2E_CONVERT, pipeline, "-o", model, "--kernels", "libref"},
        {std::string("RADIANCE_HOME=") + home, "ROCR_VISIBLE_DEVICES=99"},
        &conversion_log, 60);
    if (conversion_status != 0) std::fprintf(stderr, "conversion failed (%d):\n%s\n", conversion_status, conversion_log.c_str());
    REQUIRE_EQ(conversion_status, 0);
    const std::string path = library + "/local.safetensors", replacement = library + "/replacement";
    auto write_adapter = [&](const std::string& to, float gain) {
        tiny::Rng random(1);
        const std::string key = "transformer.transformer_blocks.0.attn.to_q";
        return tiny::write_safetensors(to, {
            tiny::noise(key + ".lora_A.weight", {1, 128}, random, 0, gain, "F32"),
            tiny::noise(key + ".lora_B.weight", {128, 1}, random, 0, 1, "F32")});
    };
    REQUIRE(write_adapter(path, 1));
    rad::Config config;
    config.model = model;
    config.radiance_home = home;
    config.kernel_hierarchy = {"libref"};
    config.accept_reference_kernels = true;
    config.max_seqs = 2;
    config.max_ctx = 2048;
    config.image_gen_max_pixels = 64 * 64;
    config.prefix_cache = false;
    config.lora.dir = library;
    config.lora.max_rank = 2;
    config.lora.max_loras = 1;
    rad::Engine engine(std::move(config));
    REQUIRE_EQ(engine.start(), RAD_OK);
    const std::string other = library + "/other.safetensors";
    REQUIRE(write_adapter(other, 2));
    const std::string incompatible = library + "/bias.safetensors";
    tiny::Rng bias_random(9);
    REQUIRE(tiny::write_safetensors(incompatible, {
        tiny::noise("transformer.not_a_module.diff_b", {1}, bias_random, 0, 1, "F32")}));
    int32_t first_slot = -1;
    std::shared_ptr<void> first_hold;
    std::string note, why;
    const std::vector<rad::Engine::LoraUse> first = {{"local", 1.0}};
    REQUIRE_EQ(engine.gen_lora_acquire(first, &first_slot, &first_hold, &note, &why), RAD_OK);
    REQUIRE(first_slot >= 0);
    REQUIRE(first_hold.use_count() > 0);
    CHECK_EQ(engine.gen_lora_validate({{"other", 1.0}}, &why), RAD_OK);
    CHECK(first_hold.use_count() > 0);
    CHECK_EQ(engine.gen_lora_validate({{"bias", 1.0}}, &why), RAD_E_INVAL);
    CHECK(why.find("unsupported bias target") != std::string::npos);
    CHECK_EQ(engine.gen_lora_validate({{"missing", 1.0}}, &why), RAD_E_NOTFOUND);
    std::shared_ptr<void> reacquired, rejected;
    int32_t same_slot = -1, rejected_slot = -1;
    REQUIRE_EQ(engine.gen_lora_acquire(first, &same_slot, &reacquired, &note, &why), RAD_OK);
    CHECK_EQ(same_slot, first_slot);
    CHECK_EQ(engine.gen_lora_acquire({{"other", 1.0}}, &rejected_slot, &rejected, &note, &why), RAD_E_FULL);
    CHECK(!rejected);
    CHECK(first_hold.use_count() > 0);
}

int main(int argc, char** argv) {
    ::setenv("ROCR_VISIBLE_DEVICES", "99", 1);
    ::setenv("RADIANCE_HOST_VRAM_MIB", "2048", 1);
    ::setenv("OMP_NUM_THREADS", "2", 1);
    return radtest::run_all(argc, argv);
}
