/* Exercise the pager's actual private stream set without exporting test controls in the ABI. */
#include "rad_test.h"
#include "../core/device/paging.cpp"
#include <sys/wait.h>
#include <unistd.h>
TEST(a_failed_pager_fence_still_visits_every_copy_stream) {
    rad::Streams streams;
    RadStream first = nullptr, second = nullptr;
    REQUIRE_EQ(rad_stream_create(&first, 0), RAD_OK);
    REQUIRE_EQ(rad_stream_create(&second, 0), RAD_OK);
    streams.s.emplace(0, first);
    /* Two logical devices on the one-device CPU backend. The invalid second binding must not
     * suppress the fence on its valid, independently created host stream. */
    streams.s.emplace(1, second);
    void* gate = nullptr;
    REQUIRE_EQ(rad_stream_gate(first, &gate), RAD_OK);
    const int64_t before = rad::rad_dev_host_blocks();
    CHECK_EQ(streams.sync_all(), RAD_E_STATE);
    CHECK_EQ(rad::rad_dev_host_blocks() - before, 2);
    CHECK_OK(rad_gate_open(gate));
    CHECK_OK(rad_dev_set(0));
}
TEST(a_failed_device_selection_cannot_publish_a_pager_stream) {
    rad::Streams streams;
    CHECK(streams.on(rad_dev_count()) == nullptr);
    CHECK(streams.s.empty());
    CHECK_OK(rad_dev_set(0));
}
TEST(a_read_error_cannot_hide_an_unproven_readback_fence) {
    if (!rad::device_is_host()) { radtest::skip("host backend only"); return; }
    const pid_t child = fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        rad::Streams streams;
        RadStream stream = streams.on(0);
        void* gate = nullptr;
        if (!stream || rad_stream_gate(stream, &gate) < 0) _exit(98);
        rad::Page page;
        page.span = 4096;
        page.device = 0;
        page.file = "/radiance-missing-readback-test-file";
        page.ch.resize(1);
        page.ch[0].on_disk = true;
        (void)rad::read_back({ { nullptr, &page, 0 } }, streams);
        _exit(99); // the failed fence must terminate before error cleanup can recycle anything
    }
    int status = 0;
    REQUIRE_EQ(waitpid(child, &status, 0), child);
    REQUIRE(WIFEXITED(status));
    CHECK_EQ(WEXITSTATUS(status), 70);
}
RAD_TEST_MAIN()
