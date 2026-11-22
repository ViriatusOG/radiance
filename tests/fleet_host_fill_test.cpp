/* Real Front leases defer optional Fleet host fill; releasing a lease resumes dispatch. */
#include "rad_test.h"
#include "engine_priv.h"
#include "server/server.h"
#include "../core/fleet/fleet.cpp"
namespace rad::fleet { namespace {
struct FleetHostFillTestAccess {
    static void fill_waits_for_serving() {
        if (!device_is_host() || !VMem::supported()) {
            radtest::skip("needs the host virtual-memory backend"); return;
        }
        FleetConfig config;
        REQUIRE_EQ(parse("models:\n  primary:\n    model: /m/primary.rad\n  secondary:\n    model: /m/secondary.rad\n",
                         "fill-guard.yaml", &config), RAD_OK);
        Fleet fleet(config);
        for (const auto& entry : config.models)
            fleet.front_.add(entry.name, {}, 1000, false);
        server::Server primary(server::Deps{}, server::ServerOptions{});
        fleet.front_.attach("primary", {}, &primary);
        auto* model = fleet.find("secondary");
        REQUIRE(model != nullptr);
        model->engine = std::make_shared<Engine>(model->entry.cfg);
        model->parked = true;
        fleet.front_.set_parked("secondary", true);
        EngineScope owner(model->engine->owner(), "fill-policy-test");
        constexpr int64_t bytes = (int64_t)64 << 20;
        void* page = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
        REQUIRE(page != nullptr);
        struct Cleanup {
            Fleet::Model* model; void* page;
            ~Cleanup() {
                if (model->worker.joinable()) model->worker.join();
                rad_dev_free(page, RAD_MEM_DEVICE);
            }
        } cleanup{model, page};
        dev_page_seal(page);
        dev_page_refilled(page);
        PageOut request;
        PageResult result;
        CHECK_OK(dev_page_out(model->engine->owner(), request, &result));
        const int64_t pending = dev_page_fillable(model->engine->owner());
        REQUIRE(pending >= bytes);
        auto response = fleet.front_.hold_active_job(&primary);
        REQUIRE(bool(response));
        CHECK_EQ(fleet.front_.activity("primary").leases, 1);
        CHECK(fleet.quiet()); // lifecycle-quiet can still have an active serving lease
        fleet.fill_tick();
        CHECK(!model->worker.joinable());
        CHECK_EQ(dev_page_fillable(model->engine->owner()), pending);
        response.reset();
        CHECK_EQ(fleet.front_.activity("primary").leases, 0);
        fleet.fill_tick();
        /* The fixture has no container to refill. Worker ownership, rather than a transient
         * kFill value or completed bytes, proves that idle warming was dispatched again. */
        CHECK(model->worker.joinable());
        if (model->worker.joinable()) model->worker.join();
    }
};
}}
TEST(automatic_host_fill_waits_for_active_responses_then_resumes_when_idle) {
    rad::fleet::FleetHostFillTestAccess::fill_waits_for_serving();
}
RAD_TEST_MAIN()
