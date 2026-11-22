/* The Engine's actual sealing policy must not treat runtime adapter storage as base weights. */
#include "rad_test.h"
#include "engine_priv.h"
#include "format/radfile.h"
#include <unistd.h>
namespace rad {
struct EngineBaseWeightsTestAccess {
    static int prepare(Engine& engine, const std::string& path) {
        EngineScope scope(engine.owner_, "base-weight-refill-test");
        auto rank = std::make_unique<Engine::Rank>();
        CtxDesc desc; desc.program = &rank->program;
        RAD_TRY(rank->ctx.init(desc));
        RAD_TRY(rank->pools.configure(engine.cfg_, 0));
        engine.ranks_.push_back(std::move(rank));
        engine.file_ = std::make_unique<RadFile>();
        return engine.file_->open(path.c_str());
    }
    static void seal(Engine& engine) { engine.seal_base_weights(); }
    static void* base(Engine& engine) { return engine.ranks_[0]->pools.weights().base(); }
};
}
TEST(runtime_lora_keeps_the_engine_base_pool_container_refillable) {
    if (!rad::device_is_host() || !rad::VMem::supported()) {
        radtest::skip("needs the host virtual-memory backend"); return;
    }
    const auto path = radtest::tmp_path("engine-base-refill.rad");
    struct Remove { const std::string& path; ~Remove() { std::remove(path.c_str()); } } remove{path};
    rad::RadWriter writer;
    REQUIRE_EQ(writer.begin(path.c_str()), RAD_OK);
    writer.set_arch("radtest");
    rad::RadWriter::Weight weight;
    weight.name = "base"; weight.enc = rad_enc_plain(RAD_BF16);
    weight.rank = 2; weight.shape[0] = 32; weight.shape[1] = 1;
    const uint16_t values[32] = {};
    const void* planes[] = {values};
    REQUIRE(writer.add_weight(weight, planes) >= 0);
    REQUIRE_EQ(writer.finish(), RAD_OK);
    rad::Config config;
    config.lora.max_loras = 1; config.lora.max_rank = 4;
    config.vram_weights_mib = 64; config.vram_kv_mib = 1;
    REQUIRE(config.lora.enabled());
    rad::Engine engine(config);
    REQUIRE_EQ(rad::EngineBaseWeightsTestAccess::prepare(engine, path), RAD_OK);
    rad::EngineScope owner(engine.owner(), "base-weight-refill-test");
    auto* base = rad::EngineBaseWeightsTestAccess::base(engine);
    REQUIRE(base != nullptr);
    std::memset(base, 0, (size_t)(64ll << 20));
    void* adapter = rad_dev_alloc(64ll << 20, RAD_MEM_DEVICE);
    REQUIRE(adapter != nullptr);
    struct Free { void* p; ~Free() { rad_dev_free(p, RAD_MEM_DEVICE); } } free{adapter};
    std::memset(adapter, 0x5a, (size_t)(64ll << 20));
    rad::EngineBaseWeightsTestAccess::seal(engine);
    rad::PageOut request;
    rad::PageResult result;
    REQUIRE_EQ(rad::dev_page_out(engine.owner(), request, &result), RAD_OK);
    CHECK(result.file >= (64ll << 20));
    CHECK(result.host >= (64ll << 20));
    CHECK(result.host < (128ll << 20));
    std::vector<rad::PageSpan> refill;
    REQUIRE_EQ(rad::dev_page_in(engine.owner(), &refill), RAD_OK);
    REQUIRE_EQ(refill.size(), size_t(1));
    CHECK(refill[0].p == base);
    CHECK_EQ(static_cast<uint8_t*>(adapter)[0], 0x5a);
    CHECK_EQ(static_cast<uint8_t*>(adapter)[(64ll << 20)-1], 0x5a);
}
RAD_TEST_MAIN()
