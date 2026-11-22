/* Exercise the actual loader ring with local completion-failure injection. No serving hooks. */
#include "rad_test.h"
#include "engine_priv.h"
#include <sys/wait.h>
#include <unistd.h>

namespace {
enum Fault { None, Create, Record, Wait, Copy, Fence, CopyFence };
Fault fault = None;
int fences = 0;
int event_create(RadEvent* event) {
    return fault == Create ? RAD_E_DEVICE : rad_event_create(event);
}
int event_record(RadEvent event, RadStream stream) {
    const int status = rad_event_record(event, stream);
    return fault == Record ? RAD_E_DEVICE : status; // failure after submission
}
int event_sync(RadEvent event) {
    return fault == Wait || (fault == Fence || fault == CopyFence) ? RAD_E_DEVICE : rad_event_sync(event);
}
int stream_sync(RadStream stream) {
    ++fences;
    return (fault == Fence || fault == CopyFence) ? RAD_E_DEVICE : rad_stream_sync(stream);
}
int copy_async(void* dst, const void* src, int64_t bytes, RadStream stream) {
    const int status = rad_memcpy_async(dst, src, bytes, stream);
    return (fault == Copy || fault == CopyFence) ? RAD_E_DEVICE : status; // partial submission cannot be undone safely
}
}
#define rad_event_create event_create
#define rad_event_record event_record
#define rad_event_sync event_sync
#define rad_stream_sync stream_sync
#define rad_memcpy_async copy_async
#define RAD_LOADER_INTERNALS_TEST 1
#include "../core/engine_bringup.cpp"
#undef RAD_LOADER_INTERNALS_TEST
#undef rad_event_create
#undef rad_event_record
#undef rad_event_sync
#undef rad_stream_sync
#undef rad_memcpy_async

TEST(failed_completion_events_drain_before_resolving_ring_ownership) {
    if (!rad::device_is_host()) { radtest::skip("host backend only"); return; }
    for (Fault mode : { Create, Record, Wait, Copy }) {
        fault = None;
        fences = 0;
        rad::loader_test::PinRing ring;
        ring.start(1, true);
        auto piece = ring.take(4096);
        REQUIRE(bool(piece));
        std::memset(piece.p, 0x5a, 4096);
        std::vector<uint8_t> destination(4096, 0);
        if (mode == Copy) fault = mode;
        CHECK_EQ(ring.send(piece, destination.data(), 0, 4096, 0), mode == Copy ? RAD_E_DEVICE : RAD_OK);
        fault = mode;
        const int given = ring.give(piece);
        CHECK_EQ(given, (mode == Wait || mode == Copy) ? RAD_OK : RAD_E_DEVICE);
        CHECK(!piece);
        CHECK_EQ(ring.drain(), RAD_E_DEVICE);
        CHECK_EQ(fences, 1);
        CHECK(std::all_of(destination.begin(), destination.end(), [](uint8_t b) { return b == 0x5a; }));
        fault = None;
    }
}
TEST(ambiguous_upload_or_failed_fallback_fence_cannot_recycle_the_ring) {
    if (!rad::device_is_host()) { radtest::skip("host backend only"); return; }
    for (Fault mode : { CopyFence, Fence }) {
        const pid_t child = fork();
        REQUIRE(child >= 0);
        if (child == 0) {
            fault = None;
            rad::loader_test::PinRing ring;
            ring.start(1, true);
            auto piece = ring.take(4096);
            if (!piece) _exit(98);
            std::vector<uint8_t> destination(4096);
            if (mode == CopyFence) fault = mode;
            (void)ring.send(piece, destination.data(), 0, 4096, 0);
            (void)ring.give(piece);
            fault = mode;
            (void)ring.drain();
            _exit(99);
        }
        int status = 0;
        REQUIRE_EQ(waitpid(child, &status, 0), child);
        REQUIRE(WIFEXITED(status));
        CHECK_EQ(WEXITSTATUS(status), 70);
    }
}
RAD_TEST_MAIN()
