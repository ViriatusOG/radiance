/* Include the private Fleet implementation so refusal tests exercise its real callers without
 * opening a listener, loading a model, or adding a production-only control API. */
#include "rad_test.h"
#include "engine_priv.h"
#include "../core/fleet/fleet.cpp"
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace rad {
struct EngineMemoryTestAccess {
    static RadStream init(Engine& e) {
        auto rank = std::make_unique<Engine::Rank>();
        CtxDesc desc; desc.program = &rank->program;
        if (rank->ctx.init(desc) < 0) return nullptr;
        auto stream = rank->ctx.stream();
        e.ranks_.push_back(std::move(rank));
        return stream;
    }
};
}
namespace rad::fleet {
namespace {
struct FleetMemoryTestAccess {
    static void exercise() {
        const std::string original = "models:\n  primary:\n    model: /m/primary.rad\n  secondary:\n    model: /m/secondary.rad\n";
        FleetConfig config;
        REQUIRE_EQ(parse(original, "guard.yaml", &config), RAD_OK);
        const std::string path = radtest::tmp_path("fleet-guard.yaml");
        std::ofstream(path).close();   /* the file a reload reads: empty until a step writes it */
        REQUIRE(std::filesystem::exists(path));
        config.path = path;
        Fleet fleet(config);
        for (const auto& entry : config.models)
            fleet.front_.add(entry.name, entry.aliases, 1000, false);
        auto* model = fleet.find("secondary");
        REQUIRE(model != nullptr);
        model->engine = std::make_shared<Engine>(model->entry.cfg);
        model->claim = 8192; model->charge = 4096;
        model->resident = true;
        model->entry.park.ram = ParkSpec::Ram::Never;
        fleet.front_.set_parked("secondary", true);
        Engine* before = model->engine.get();
        RadStream stream = EngineMemoryTestAccess::init(*before);
        REQUIRE(stream != nullptr);
        void* gate = nullptr;
        REQUIRE_EQ(rad_stream_gate(stream, &gate), RAD_OK);
        /* Open the gate even if a later check fails: destructors must not see the injected fault. */
        struct Cleanup {
            void*& gate;
            ~Cleanup() { if (gate) rad_gate_open(gate); }
        } cleanup{gate};
        CHECK_EQ(before->prepare_unload(), RAD_E_STATE);
        CHECK(!fleet.unload(*model, 0, "test refused unload"));
        CHECK(!fleet.take_from(*model, {}, "test refused eviction"));
        auto unchanged = [&] {
            CHECK(fleet.find("secondary") == model);
            CHECK(model->engine.get() == before);
            CHECK_EQ(model->claim, 8192);
            CHECK_EQ(model->charge, 4096);
            CHECK(model->resident);
            CHECK_EQ(model->unloads, 0);
            CHECK(fleet.front_.activity("secondary").state == server::Front::State::Parked);
        };
        unchanged();
        auto write = [&](const std::string& yaml) { std::ofstream file(path); file << yaml; };
        write("models:\n  primary:\n    model: /m/primary.rad\n");
        CHECK(!fleet.reload().empty());
        unchanged();
        write(original + "    max-num-seqs: 2\n");
        const auto old_args = model->entry.args;
        CHECK(!fleet.reload().empty());
        CHECK(model->entry.args == old_args);
        unchanged();
        CHECK_OK(rad_gate_open(gate)); gate = nullptr;
        CHECK(fleet.unload(*model, 0, "test recovered unload"));
        CHECK(!model->engine);
        CHECK(!model->resident);
        CHECK_EQ(model->unloads, 1);
        CHECK(fleet.unload(*model, 0, "idempotent unload"));
        CHECK_EQ(model->unloads, 1);
        write("models:\n  primary:\n    model: /m/primary.rad\n");
        CHECK(fleet.reload().empty());
        CHECK(fleet.find("secondary") == nullptr);
    }
};
} // namespace
} // namespace rad::fleet
TEST(refused_unload_preserves_fleet_ownership_and_reload_configuration) {
    rad::fleet::FleetMemoryTestAccess::exercise();
}
RAD_TEST_MAIN()
