/* device_test.cpp -- the device layer against its own contract.
 *
 * The ordering tests matter most. A backend whose "async" calls execute inline lets every ordering
 * bug above it (a missing rad_event_wait, a staging buffer reused before the mover's copy landed)
 * pass on the host backend and corrupt weights on a card. The event tests make out-of-order
 * execution produce wrong bytes, not just odd timestamps, and run each race several times.
 */
#include "rad_test.h"
#include "host_kargs_main.h"

#include "rad_internal.h"
#include "device/aql.h"
#include "device/device.h"
#include "device/directio.h"
#include "sched/advance.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using rad::DirectFile;
using rad::RAD_DIO_READ;
using rad::RAD_DIO_WRITE;
using rad::RAD_DIO_CREATE;
using rad::RAD_DIO_TRUNC;
using rad::RAD_DIO_ALLOW_BUFFERED;

namespace {

struct Stream {
    RadStream s = nullptr;
    explicit Stream(int hi = 0) { rad_stream_create(&s, hi); }
    ~Stream() { if (s) rad_stream_destroy(s); }
};

struct Event {
    RadEvent e = nullptr;
    Event() { rad_event_create(&e); }
    ~Event() { if (e) rad_event_destroy(e); }
};

/* A rad_dev_alloc allocation of `kind`, freed when it leaves scope (an early return included).
 * It converts to its pointer, so it reads like one. */
template <class T> struct DevBuf {
    T* p;
    int kind;
    DevBuf(int64_t bytes, int k) : p((T*)rad_dev_alloc(bytes, k)), kind(k) {}
    ~DevBuf() { if (p) rad_dev_free(p, kind); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    operator T*() const { return p; }
    T* operator->() const { return p; }
};

/* Deterministic filler, so a torn copy shows up as a mismatch at a known index. */
void fill(uint8_t* p, int64_t n, uint32_t seed) {
    uint32_t x = seed | 1u;
    for (int64_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (uint8_t)(x >> 24);
    }
}

/* False, with the case marked skipped, unless the backend has virtual memory management, and (with
 * `host_visible`) the elastic memory is host memory a test can read back. */
bool need_vmem(bool host_visible = false) {
    if (rad::VMem::supported() && (!host_visible || rad::device_is_host())) return true;
    radtest::skip("%s", host_visible ? "needs host-visible elastic memory"
                                     : "no virtual memory management on this backend");
    return false;
}

/* False, with the case marked skipped, on a card. */
bool need_host_backend() {
    if (rad::device_is_host()) return true;
    radtest::skip("host backend only");
    return false;
}

std::string temp_path(const char* tag) {
    /* The working directory (ctest's build directory), not /tmp: /tmp is usually tmpfs, which
     * refuses O_DIRECT, so only the buffered fallback would be exercised. */
    return rad::fmt("./rad_%s_test_%d.bin", tag, (int)getpid());
}

}  /* namespace */

/* ------------------------------------------------------------------ identity */

TEST(backend_reports_itself) {
    const char* name = rad::device_backend_name();
    CHECK(name && *name);
    CHECK_EQ(rad::device_is_host(), std::strcmp(name, "host") == 0);

    const int n = rad_dev_count();
    CHECK(n >= 1);

    RadDeviceProps p{};
    CHECK_OK(rad_dev_props(0, &p));
    CHECK(p.name[0] != '\0');
    CHECK(p.arch[0] != '\0');
    CHECK(p.vram_bytes > 0);
    CHECK(p.vram_free >= 0);
    CHECK(p.n_cu >= 1);
    CHECK(p.warp_size >= 1);
    CHECK_EQ(p.device_id, 0);
    if (rad::device_is_host()) {
        CHECK_EQ(p.is_host_backend, 1);
        CHECK_EQ(std::string(p.arch), std::string("host"));
    }
    /* Out of range is refused, not clamped: clamping a bad rank index would put two ranks on one
     * card. */
    CHECK(rad_dev_props(n, &p) < 0);
    CHECK(rad_dev_set(-1) < 0);
    CHECK_OK(rad_dev_set(0));
}

/* Chained launches are on only where the engine says the model's plugin measured a gain with its
 * drafter (rad_dev_chain_infer) and the backend is AQL, which writes the record a chained form
 * reads: on the host backend a plugin's ask leaves them off. Ends with them off again. */
TEST(chained_launches_follow_the_plugins_ask_on_the_aql_backend_only) {
    const bool aql = std::strcmp(rad::device_backend_name(), "aql") == 0;
    rad_dev_chain_infer(0);
    CHECK_EQ(rad_dev_chain_enabled(), 0);
    rad_dev_chain_infer(1);
    CHECK_EQ(rad_dev_chain_enabled(), aql ? 1 : 0);
    rad_dev_chain_infer(0);
    CHECK_EQ(rad_dev_chain_enabled(), 0);
}

TEST(host_vram_budget_comes_from_the_environment) {
    /* A real card reports its own VRAM. */
    if (!need_host_backend()) return;
    const char* old = std::getenv("RADIANCE_HOST_VRAM_MIB");
    const std::string saved = old ? old : "";

    setenv("RADIANCE_HOST_VRAM_MIB", "777", 1);
    RadDeviceProps p{};
    CHECK_OK(rad_dev_props(0, &p));
    CHECK_EQ(p.vram_bytes, (int64_t)777 << 20);

    unsetenv("RADIANCE_HOST_VRAM_MIB");
    CHECK_OK(rad_dev_props(0, &p));
    /* The default derives from physical RAM, so only a plausible lower bound can be checked. */
    CHECK(p.vram_bytes > (int64_t)64 << 20);

    if (!saved.empty()) setenv("RADIANCE_HOST_VRAM_MIB", saved.c_str(), 1);
}

/* ------------------------------------------------------------------ allocation */

TEST(alloc_free_roundtrip) {
    const int kinds[] = { RAD_MEM_DEVICE, RAD_MEM_HOST_PINNED, RAD_MEM_HOST, RAD_MEM_HOST_MAPPED };
    const int64_t n = 64 * 1024 + 17;   /* deliberately not a page multiple */

    for (int k : kinds) {
        void* p = rad_dev_alloc(n, k);
        CHECK(p != nullptr);
        if (!p) continue;

        /* Host-visible kinds are writable from here; a device allocation on a real card is not. */
        if (rad::device_is_host() || k != RAD_MEM_DEVICE) {
            auto* b = (uint8_t*)p;
            fill(b, n, (uint32_t)k + 5u);
            uint8_t expect[64];
            fill(expect, 64, (uint32_t)k + 5u);
            CHECK_EQ(std::memcmp(b, expect, 64), 0);
        }
        if (k == RAD_MEM_HOST_MAPPED) {
            /* The zero-copy site of spec §5.1: one allocation, addressable from both sides. */
            CHECK(rad_dev_host_ptr(p) != nullptr);
            CHECK(rad_dev_device_ptr(p) != nullptr);
            /* Invariant: both views are the same address on every backend. Ctx::bind_buffers
             * stores one base pointer per buffer, and the host activation arena is
             * RAD_MEM_HOST_MAPPED so a host-site op and a device kernel share it. HIP's unified
             * addressing guarantees this; where it fails, Ctx refuses to start. */
            CHECK_EQ(rad_dev_host_ptr(p), rad_dev_device_ptr(p));
            CHECK_EQ(rad_dev_host_ptr(p), p);
        }
        /* Host-visible allocations are page-aligned, so a pinned buffer can be an O_DIRECT target
         * with no bounce (directio.h). A device allocation only needs RAD_ALIGN_SUB, the
         * container's sub-block. */
        CHECK_EQ((uintptr_t)p % (k == RAD_MEM_DEVICE && !rad::device_is_host() ? 256u : 4096u), 0u);
        rad_dev_free(p, k);
    }

    CHECK(rad_dev_alloc(0, RAD_MEM_HOST) == nullptr);
    CHECK(rad_dev_alloc(-1, RAD_MEM_HOST) == nullptr);
    CHECK(rad_dev_alloc(16, 99) == nullptr);
    rad_dev_free(nullptr, RAD_MEM_HOST);   /* must not fault */
    CHECK(rad_dev_host_ptr(nullptr) == nullptr);
}

/* A pinned range larger than one pin is locked in pieces (core/device/aql.cpp, pin_pieces),
 * invisibly above the device layer: one base, every byte at the same address on the card, and
 * copies and fills across a piece boundary land whole. A fill of pageable memory is refused, so the
 * fill also checks no piece is treated as pageable. 2.5 GiB is a single-pin size an Intel VT-d host
 * refused; the crossed boundaries are whole GiBs. Under HSA_USE_SVM=0 (ctest's device_test_nosvm)
 * the range is one host-pool allocation and the same checks apply. */
TEST(a_pinned_range_larger_than_one_pin_is_one_range_on_the_card) {
    const int64_t gib = (int64_t)1 << 30;
    const int64_t n = 5 * (gib / 2) + 4096 + 17;
    const int64_t w = 8 << 20;
    Stream st;
    DevBuf<uint8_t> h(n, RAD_MEM_HOST_MAPPED);
    DevBuf<uint8_t> d(w, RAD_MEM_DEVICE);
    CHECK(h != nullptr);
    CHECK(d != nullptr);
    if (!h || !d) return;
    /* The host backend resolves only an allocation's base; interior addresses are checked on a
     * card only. */
    CHECK_EQ(rad_dev_device_ptr(h), (void*)h);
    if (!rad::device_is_host())
        for (int64_t off : { gib - 1, gib, 2 * gib - 4096, 2 * gib, n - 1 })
            CHECK_EQ(rad_dev_device_ptr(h + off), (void*)(h + off));

    std::string expect((size_t)w, '\0');
    for (int64_t b : { gib, 2 * gib }) {
        uint8_t* span = h + b - w / 2;
        fill(span, w, (uint32_t)(b >> 20));
        std::memcpy(expect.data(), span, (size_t)w);
        CHECK_OK(rad_memcpy_async(d, span, w, st.s));
        CHECK_OK(rad_stream_sync(st.s));
        std::memset(span, 0, (size_t)w);
        CHECK_OK(rad_memcpy_async(span, d, w, st.s));
        CHECK_OK(rad_stream_sync(st.s));
        CHECK_EQ(std::memcmp(span, expect.data(), (size_t)w), 0);
        CHECK_OK(rad_memset_async(span, 0xA5, w, st.s));
        CHECK_OK(rad_stream_sync(st.s));
        CHECK(span[0] == 0xA5 && span[w / 2 - 1] == 0xA5 && span[w / 2] == 0xA5 && span[w - 1] == 0xA5);
    }
}

/* A device allocation the card cannot take is an error, even after a queue was destroyed. ROCm
 * 7.2's runtime handles a driver-refused VRAM allocation by walking every queue the process ever
 * created, destroyed ones included, and segfaults; the backend refuses such a request itself
 * (core/device/aql.cpp, "the room check"). The test destroys a stream first, then asks for the
 * whole card: more than the driver has left, but not more than one allocation may be, so without
 * the check it would reach the driver. */
TEST(an_allocation_the_card_cannot_take_is_refused_after_a_queue_was_destroyed) {
    if (rad::device_is_host()) SKIP("a card only");
    { Stream st; CHECK(st.s != nullptr); }
    RadDeviceProps p{};
    CHECK_OK(rad_dev_props(0, &p));
    void* d = rad_dev_alloc(p.vram_bytes, RAD_MEM_DEVICE);
    CHECK(d == nullptr);
    const char* why = rad_dev_last_error();
    CHECK(std::strstr(why, "on card 0: the driver has") != nullptr);
    CHECK(std::strstr(why, "this process") != nullptr);
    if (d) rad_dev_free(d, RAD_MEM_DEVICE);

    /* The card still serves an allocation it can take. */
    void* ok = rad_dev_alloc(8 << 20, RAD_MEM_DEVICE);
    CHECK(ok != nullptr);
    rad_dev_free(ok, RAD_MEM_DEVICE);
}

TEST(device_allocation_moves_vram_free) {
    /* On a card, other processes move it too. */
    if (!need_host_backend()) return;
    RadDeviceProps before{}, during{};
    CHECK_OK(rad_dev_props(0, &before));
    const int64_t n = 32 << 20;
    void* p = rad_dev_alloc(n, RAD_MEM_DEVICE);
    CHECK(p != nullptr);
    CHECK_OK(rad_dev_props(0, &during));
    /* The host backend must track vram_free, or a plan that overcommits a real card would pass
     * here. */
    CHECK_EQ(before.vram_free - during.vram_free, n);
    rad_dev_free(p, RAD_MEM_DEVICE);
    CHECK_OK(rad_dev_props(0, &during));
    CHECK_EQ(before.vram_free, during.vram_free);
}

/* ------------------------------------------------------------------ transfers */

TEST(memcpy_async_then_stream_sync) {
    const int64_t n = 4 << 20;
    Stream st;
    CHECK(st.s != nullptr);

    DevBuf<uint8_t> src(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dst(n, RAD_MEM_HOST);
    CHECK(src && dst);
    if (!src || !dst) return;

    fill(src, n, 0xC0FFEEu);
    std::memset(dst, 0, (size_t)n);

    CHECK_OK(rad_memcpy_async(dst, src, n, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ(std::memcmp(dst, src, (size_t)n), 0);

    /* Zero bytes is a no-op, not an error, so callers need no branch for an empty span. */
    CHECK_OK(rad_memcpy_async(dst, src, 0, st.s));
}

/* A list copy hands many copies to the card's DMA engine at once, so errors appear at segment
 * boundaries: a copy offset, short, or overrunning its neighbour. Dozens of copies, 16 B to ~200
 * KiB, sources out of order, destinations separated by sentinel gaps, odd lengths and unaligned
 * sources; the result is read back by an ordinary copy on the same stream, which must wait for the
 * list. */
TEST(memcpy_list_async_copies_every_segment_and_nothing_else) {
    const int nseg = 77;
    const int64_t gap = 48, slab = 16 << 20;
    Stream st;
    DevBuf<uint8_t> src(slab, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dev(slab, RAD_MEM_DEVICE);
    DevBuf<uint8_t> back(slab, RAD_MEM_HOST_PINNED);
    CHECK(src && dev && back);
    if (!src || !dev || !back) return;
    fill(src, slab, 0x5EC5u);

    std::vector<RadCopySeg> segs;
    std::vector<int64_t> s_off, d_off, len;
    int64_t d = 0, s = slab;
    for (int i = 0; i < nseg; ++i) {
        int64_t n = 16 * (1 + (int64_t)((i * 2654435761u) % 13000));   /* 16 B .. ~200 KiB */
        if (i % 19 == 7) n += 5;                                       /* an odd length */
        s -= n + 64;                                                   /* sources backwards */
        if (i % 23 == 11) s -= 4;                                      /* an unaligned source */
        s_off.push_back(s);
        d_off.push_back(d);
        len.push_back(n);
        d = (d + n + gap + 15) & ~(int64_t)15;
    }
    CHECK(s > 0 && d < slab);
    CHECK_OK(rad_memset_async(dev, 0xA7, slab, st.s));
    for (int i = 0; i < nseg; ++i) segs.push_back(RadCopySeg{src + s_off[i], dev + d_off[i], len[i]});
    segs.push_back(RadCopySeg{src, dev, 0});                           /* nothing to copy */
    CHECK_OK(rad_memcpy_list_async(segs.data(), (int)segs.size(), 0, st.s));
    CHECK_OK(rad_memcpy_async(back, dev, slab, st.s));
    CHECK_OK(rad_stream_sync(st.s));

    int bad = 0, smashed = 0;
    for (int i = 0; i < nseg; ++i) {
        if (std::memcmp(back + d_off[i], src + s_off[i], (size_t)len[i]) != 0) ++bad;
        const int64_t end = i + 1 < nseg ? d_off[i + 1] : d;
        for (int64_t j = d_off[i] + len[i]; j < end; ++j)
            if (back[j] != 0xA7) { ++smashed; break; }
    }
    CHECK_EQ(bad, 0);
    CHECK_EQ(smashed, 0);
}

/* The DMA queue's ring wraps. Thousands of small-copy lists write several rings' worth of packets,
 * so lists straddle the wrap and wait behind a full ring; every copy must still land in place.
 * Every other list yields, so the gate's checks wrap too. */
TEST(memcpy_list_async_survives_its_ring_wrapping) {
    const int lists = 1500, per = 64;
    const int64_t each = 256, total = (int64_t)lists * per * each;
    Stream st;
    DevBuf<uint8_t> src(total, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dev(total, RAD_MEM_DEVICE);
    DevBuf<uint8_t> back(total, RAD_MEM_HOST_PINNED);
    CHECK(src && dev && back);
    if (!src || !dev || !back) return;
    fill(src, total, 0x3A11u);
    CHECK_OK(rad_memset_async(dev, 0, total, st.s));
    std::vector<RadCopySeg> segs((size_t)per);
    for (int k = 0; k < lists; ++k) {
        for (int i = 0; i < per; ++i) {
            const int64_t at = ((int64_t)k * per + i) * each;
            segs[(size_t)i] = RadCopySeg{src + (total - each - at), dev + at, each};   /* mirrored */
        }
        CHECK_OK(rad_memcpy_list_async(segs.data(), per, k & 1 ? RAD_COPY_YIELD : 0, st.s));
    }
    CHECK_OK(rad_memcpy_async(back, dev, total, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    int64_t bad = 0;
    for (int64_t at = 0; at < total; at += each)
        bad += std::memcmp(back + at, src + (total - each - at), (size_t)each) != 0;
    CHECK_EQ(bad, (int64_t)0);
}

/* List copies are coherent with the L2 both ways. A buffer small enough to stay in L2 is read by a
 * kernel, overwritten by a list, and read again on the same stream: the second read must see the
 * list's bytes, not stale lines. Conversely, a list copying to the host must read bytes a kernel
 * just wrote, still dirty in L2. Twelve rounds with new bytes; the inbound list yields, which
 * splits its copy into pieces that must join exactly. */
TEST(memcpy_list_async_is_ordered_against_the_l2_both_ways) {
    const int64_t n = 4 << 20;
    Stream st;
    DevBuf<uint8_t> src(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> back(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> a(n, RAD_MEM_DEVICE);
    DevBuf<uint8_t> b(n, RAD_MEM_DEVICE);
    CHECK(src && back && a && b);
    if (!src || !back || !a || !b) return;
    int stale = 0, lost = 0;
    for (int r = 0; r < 12; ++r) {
        fill(src, n, 0x1000u + (uint32_t)r);
        CHECK_OK(rad_memset_async(a, 0x40 + r, n, st.s));
        CHECK_OK(rad_memcpy_async(b, a, n, st.s));                  /* a's lines into the L2 */
        const RadCopySeg in{src, a, n};
        CHECK_OK(rad_memcpy_list_async(&in, 1, RAD_COPY_YIELD, st.s));   /* in pieces */
        CHECK_OK(rad_memcpy_async(b, a, n, st.s));                  /* must read the list's bytes */
        CHECK_OK(rad_memcpy_async(back, b, n, st.s));
        CHECK_OK(rad_stream_sync(st.s));
        stale += std::memcmp(back, src, (size_t)n) != 0;

        CHECK_OK(rad_memset_async(b, 0x80 + r, n, st.s));          /* dirty in the L2 */
        const RadCopySeg out{b, back, n};
        CHECK_OK(rad_memcpy_list_async(&out, 1, 0, st.s));
        CHECK_OK(rad_stream_sync(st.s));
        for (int64_t i = 0; i < n; i += 4093)
            if (back[i] != (uint8_t)(0x80 + r)) { ++lost; break; }
    }
    CHECK_EQ(stale, 0);
    CHECK_EQ(lost, 0);
}

/* A held link stops a yielding list until released. Stream `c` holds the link and waits on a long
 * chain of fills on `slow` before releasing, so a yielding list on `t` behind the hold finishes
 * only after the chain. Without the gate the 64 MiB list (a few ms) would finish well before the
 * fills (tens of ms). A backend without a DMA engine holds nothing and accepts a second hold. */
TEST(link_hold_stops_a_yielding_list_until_released) {
    const int64_t n = 64 << 20, big = 256 << 20;
    Stream c, t, slow;
    Event held, chain;
    CHECK_OK(rad_memcpy_list_async(nullptr, 0, 0, t.s));   /* the card's DMA queue, if it has one */
    CHECK_OK(rad_link_hold(c.s, 1));
    if (rad_link_hold(c.s, 1) == RAD_OK) return;   /* nothing to hold on this backend */
    (void)rad_dev_last_error();
    DevBuf<uint8_t> src(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dst(n, RAD_MEM_DEVICE);
    DevBuf<uint8_t> back(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> fat(big, RAD_MEM_DEVICE);
    CHECK(src && dst && back && fat);
    if (!src || !dst || !back || !fat) { (void)rad_link_hold(c.s, 0); return; }
    fill(src, n, 0x6A7Eu);
    for (int k = 0; k < 40; ++k) CHECK_OK(rad_memset_async(fat, k, big, slow.s));
    CHECK_OK(rad_event_record(chain.e, slow.s));
    CHECK_OK(rad_event_record(held.e, c.s));
    CHECK_OK(rad_event_wait(c.s, chain.e));
    CHECK_OK(rad_link_hold(c.s, 0));
    CHECK_EQ(rad_link_hold(c.s, 0), RAD_E_STATE);   /* a release must follow a hold */
    (void)rad_dev_last_error();

    CHECK_OK(rad_event_wait(t.s, held.e));
    const RadCopySeg seg{src, dst, n};
    CHECK_OK(rad_memcpy_list_async(&seg, 1, RAD_COPY_YIELD, t.s));
    CHECK_OK(rad_stream_sync(t.s));
    CHECK_EQ(rad_event_query(chain.e), 1);
    CHECK_OK(rad_memcpy_async(back, dst, n, t.s));
    CHECK_OK(rad_stream_sync(t.s));
    CHECK(std::memcmp(back, src, (size_t)n) == 0);
    CHECK_OK(rad_stream_sync(c.s));
    CHECK_OK(rad_stream_sync(slow.s));
}

/* A hold ended by the work it covered: a collective that knows when its peers' bytes have landed
 * decrements the gate word (rad_link_gate) itself, and the stream records mode 2, so no barrier
 * follows. The host plays the collective here: the word is odd while held and even once ended, a
 * yielding list behind it runs, and a second end is refused since nothing is held. */
TEST(link_gate_word_ends_a_hold_from_the_work_it_covers) {
    const int64_t n = 4 << 20;
    Stream c, t;
    CHECK_OK(rad_memcpy_list_async(nullptr, 0, 0, t.s));   /* the card's DMA queue, if it has one */
    volatile long long* g = (volatile long long*)rad_link_gate(c.s);
    if (!g) return;                                  /* nothing to hold on this backend */
    CHECK_OK(rad_link_hold(c.s, 1));
    CHECK_OK(rad_stream_sync(c.s));
    CHECK_EQ((int)(*g & 1), 1);
    *g = *g - 1;                                     /* what the launch does once its bytes land */
    CHECK_OK(rad_link_hold(c.s, 2));
    CHECK_EQ((int)(*g & 1), 0);
    CHECK_EQ(rad_link_hold(c.s, 2), RAD_E_STATE);
    (void)rad_dev_last_error();

    DevBuf<uint8_t> src(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dst(n, RAD_MEM_DEVICE);
    DevBuf<uint8_t> back(n, RAD_MEM_HOST_PINNED);
    CHECK(src && dst && back);
    if (!src || !dst || !back) return;
    fill(src, n, 0x9A7Eu);
    const RadCopySeg seg{src, dst, n};
    CHECK_OK(rad_memcpy_list_async(&seg, 1, RAD_COPY_YIELD, t.s));
    CHECK_OK(rad_memcpy_async(back, dst, n, t.s));
    CHECK_OK(rad_stream_sync(t.s));
    CHECK(std::memcmp(back, src, (size_t)n) == 0);
}

TEST(memset_async) {
    const int64_t n = 1 << 20;
    Stream st;
    /* Pinned, not RAD_MEM_HOST: rad_memset_async needs a device-accessible destination, and
     * hipMemsetAsync refuses pageable memory. The host backend cannot reproduce that refusal, so a
     * pageable target here would pass code that fails on a card. See core/device/device.h. */
    DevBuf<uint8_t> p(n, RAD_MEM_HOST_PINNED);
    CHECK(p != nullptr);
    if (!p) return;

    std::memset(p, 0, (size_t)n);
    CHECK_OK(rad_memset_async(p, 0x5A, n, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    bool all = true;
    for (int64_t i = 0; i < n; ++i) if (p[i] != 0x5A) { all = false; break; }
    CHECK(all);

    /* Only the low byte of `value` is written, matching hipMemsetAsync and memset. */
    CHECK_OK(rad_memset_async(p, 0x1FF, n, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ((int)p[0], 0xFF);
    CHECK_EQ((int)p[n - 1], 0xFF);
}

TEST(memset_of_pageable_host_memory_is_refused_on_a_card) {
    /* The one documented asymmetry between backends. On a card the destination must be
     * device-accessible and RAD_MEM_HOST is not; on the host backend every kind is the same
     * memory, so nothing is refused. */
    Stream st;
    DevBuf<uint8_t> p(4096, RAD_MEM_HOST);
    CHECK(p != nullptr);
    if (!p) return;
    int s = rad_memset_async(p, 0, 4096, st.s);
    if (rad::device_is_host()) {
        CHECK_OK(s);
        CHECK_OK(rad_stream_sync(st.s));
    } else {
        CHECK_EQ(s, RAD_E_UNSUPPORTED);
        CHECK(std::strstr(rad_dev_last_error(), "device-accessible") != nullptr);
    }
}

TEST(memcpy_2d_respects_the_pitch) {
    /* A KV block's rows are strided by the pool's pitch, so the padding between rows belongs to
     * someone else and must survive the copy. */
    const int64_t W = 48, H = 7, DP = 64, SP = 55;
    Stream st;
    DevBuf<uint8_t> s(SP * H, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> d(DP * H, RAD_MEM_HOST_PINNED);
    CHECK(s && d);
    if (!s || !d) return;

    fill(s, SP * H, 0xBEEF);
    CHECK_OK(rad_memset_async(d, 0xCC, DP * H, st.s));
    CHECK_OK(rad_memcpy_2d_async(d, DP, s, SP, W, H, st.s));
    CHECK_OK(rad_stream_sync(st.s));

    bool rows_ok = true, pad_ok = true;
    for (int64_t r = 0; r < H; ++r) {
        if (std::memcmp(d + r * DP, s + r * SP, (size_t)W) != 0) rows_ok = false;
        for (int64_t c = W; c < DP; ++c) if (d[r * DP + c] != 0xCC) pad_ok = false;
    }
    CHECK(rows_ok);
    CHECK(pad_ok);

    /* A pitch shorter than the row is a caller bug and is refused, not clamped. */
    CHECK(rad_memcpy_2d_async(d, W - 1, s, SP, W, H, st.s) < 0);
}

/* Three laps of the kernel argument ring with no host wait. A backend that keeps kernel arguments
 * in host-rewritten memory must make each slot's new contents visible to its kernel; a stale slot
 * silently reruns the previous lap's arguments. Each copy moves a distinct 64-byte piece, so a
 * stale argument misplaces a piece and the comparison finds it. */
TEST(twelve_thousand_copies_each_land_where_their_own_arguments_say) {
    const int64_t piece = 64, n = 12000, bytes = piece * n;
    Stream st;
    DevBuf<uint8_t> host(bytes, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> back(bytes, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> src(bytes, RAD_MEM_DEVICE);
    DevBuf<uint8_t> dst(bytes, RAD_MEM_DEVICE);
    CHECK(host && back && src && dst);
    if (!host || !back || !src || !dst) return;

    fill(host, bytes, 0xA11CE);
    CHECK_OK(rad_memcpy_async(src, host, bytes, st.s));
    /* Piece i of dst comes from piece (i * 7919) mod n of src, a permutation since 7919 is prime
     * and does not divide n, so no two copies share arguments. */
    for (int64_t i = 0; i < n; ++i)
        CHECK_OK(rad_memcpy_async(dst + i * piece, src + ((i * 7919) % n) * piece, piece, st.s));
    CHECK_OK(rad_memcpy_async(back, dst, bytes, st.s));
    CHECK_OK(rad_stream_sync(st.s));

    int64_t wrong = 0, first = -1;
    for (int64_t i = 0; i < n; ++i)
        if (std::memcmp(back + i * piece, host + ((i * 7919) % n) * piece, (size_t)piece) != 0) {
            if (first < 0) first = i;
            ++wrong;
        }
    CHECK_EQ(wrong, (int64_t)0);
    if (wrong) std::fprintf(stderr, "    first misplaced piece: %lld\n", (long long)first);
}

/* ------------------------------------------------------------------ ordering */

TEST(event_on_one_stream_orders_work_on_another) {
    const int64_t n = 4 << 20;
    const int     chain = 8;    /* copies stream A does before it records */
    const int     rounds = 3;   /* the race, run more than once */

    Stream a, b;
    CHECK(a.s && b.s);

    DevBuf<uint8_t> src(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> mid(n, RAD_MEM_DEVICE);
    DevBuf<uint8_t> dst(n, RAD_MEM_HOST);
    CHECK(src && mid && dst);
    if (!src || !mid || !dst) return;

    for (int round = 0; round < rounds; ++round) {
        Event e_a, e_b;
        CHECK(e_a.e && e_b.e);

        fill(src, n, 0x51ED0000u + (uint32_t)round);

        /* Poison the intermediate from a synchronised point, so a read that jumps the queue sees
         * 0xEE rather than a stale copy of the right answer. */
        CHECK_OK(rad_memset_async(mid, 0xEE, n, b.s));
        CHECK_OK(rad_stream_sync(b.s));

        /* Enough work on A that an unordered reader on B lands in the middle of it. */
        CHECK_OK(rad_memset_async(mid, 0x00, n, a.s));
        for (int i = 0; i < chain; ++i) CHECK_OK(rad_memcpy_async(mid, src, n, a.s));
        CHECK_OK(rad_event_record(e_a.e, a.s));

        /* B waits on the event, not the host (no rad_stream_sync(a)): spec §5.4 requires
         * transfers without host synchronisation. */
        CHECK_OK(rad_event_wait(b.s, e_a.e));
        CHECK_OK(rad_memcpy_async(dst, mid, n, b.s));
        CHECK_OK(rad_event_record(e_b.e, b.s));

        CHECK_OK(rad_stream_sync(b.s));
        CHECK_EQ(std::memcmp(dst, src, (size_t)n), 0);

        /* Each timestamp is taken when its stream reaches the record, so B's precedes A's only if
         * the wait did nothing. */
        float ms = -1.0f;
        CHECK_OK(rad_event_elapsed_ms(e_a.e, e_b.e, &ms));
        CHECK(ms >= 0.0f);

        CHECK_OK(rad_stream_sync(a.s));
        CHECK_EQ(rad_event_query(e_a.e), 1);
        CHECK_EQ(rad_event_query(e_b.e), 1);
    }
}

/* Two streams joining each other hundreds of times, the host far ahead and one stream lagging, as
 * a model's two compute lanes meet every layer. A backend that reuses an event's completion object
 * before a wait on its previous record is reached turns that wait into one on a later record,
 * behind a join the lagging stream has not reached: a deadlock (this test would hang). The bytes
 * are checked too, since the joins also order the copies. */
TEST(two_streams_joining_each_other_hundreds_of_times_neither_deadlock_nor_reorder) {
    const int64_t big = 1 << 20, small = 4096;
    const int     joins = 400;
    Stream a, b;
    Event ea, eb;
    DevBuf<uint8_t> d0(big, RAD_MEM_DEVICE);
    DevBuf<uint8_t> d1(big, RAD_MEM_DEVICE);
    DevBuf<uint8_t> h(big, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> r(small, RAD_MEM_HOST_PINNED);
    CHECK(d0 && d1 && h && r);
    if (!d0 || !d1 || !h || !r) return;
    fill(h, big, 0x5EED);
    CHECK_OK(rad_memcpy_async(d0, h, big, a.s));
    for (int i = 0; i < joins; ++i) {
        /* b lags: 1 MiB per round against a's 4 KiB. */
        CHECK_OK(rad_event_record(ea.e, a.s));
        CHECK_OK(rad_event_wait(b.s, ea.e));
        CHECK_OK(rad_memcpy_async(d1, d0, big, b.s));
        CHECK_OK(rad_event_record(eb.e, b.s));
        CHECK_OK(rad_event_wait(a.s, eb.e));
        CHECK_OK(rad_memcpy_async(d0, d1, small, a.s));
    }
    CHECK_OK(rad_memcpy_async(r, d0, small, a.s));
    CHECK_OK(rad_stream_sync(a.s));
    CHECK_OK(rad_stream_sync(b.s));
    CHECK_EQ(std::memcmp(r, h, (size_t)small), 0);
}

TEST(an_event_destroyed_while_queued_work_names_it_still_orders_that_work) {
    /* Teardown (the mover's included) may destroy events whose record or wait has not run yet,
     * as HIP allows. Queued work must behave as if the handle were alive: the wait waits, the copy
     * behind it sees the bytes, and neither stream wedges. Events are churned between rounds so
     * memory freed too early is reused and overwritten. */
    const int64_t n = 4 << 20;
    Stream a, b;
    CHECK(a.s && b.s);
    DevBuf<uint8_t> src(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> mid(n, RAD_MEM_DEVICE);
    DevBuf<uint8_t> dst(n, RAD_MEM_HOST);
    CHECK(src && mid && dst);
    if (!src || !mid || !dst) return;

    for (int round = 0; round < 4; ++round) {
        fill(src, n, 0xDE570000u + (uint32_t)round);
        CHECK_OK(rad_memset_async(mid, 0xEE, n, b.s));
        CHECK_OK(rad_stream_sync(b.s));

        RadEvent ga = nullptr, eb = nullptr;
        CHECK_OK(rad_event_create(&ga));
        CHECK_OK(rad_event_create(&eb));
        for (int i = 0; i < 8; ++i) CHECK_OK(rad_memcpy_async(mid, src, n, a.s));
        CHECK_OK(rad_event_record(ga, a.s));
        CHECK_OK(rad_event_wait(b.s, ga));
        CHECK_OK(rad_memcpy_async(dst, mid, n, b.s));
        CHECK_OK(rad_event_record(eb, b.s));
        rad_event_destroy(eb);
        rad_event_destroy(ga);
        for (int k = 0; k < 4; ++k) {
            Event churn;
            CHECK_OK(rad_event_record(churn.e, b.s));
        }

        CHECK_OK(rad_stream_sync(b.s));
        CHECK_OK(rad_stream_sync(a.s));
        CHECK_EQ(std::memcmp(dst, src, (size_t)n), 0);
    }
}

TEST(waiting_on_an_unrecorded_event_is_a_no_op) {
    /* The mover relies on this: a weight with no transfer in flight has an unrecorded event, and
     * an error here would force a branch on the issue path. */
    Stream st;
    Event e;
    CHECK_EQ(rad_event_query(e.e), 1);
    CHECK_OK(rad_event_wait(st.s, e.e));
    CHECK_OK(rad_stream_sync(st.s));
}

/* An advance runs at its place in stream order. The sampled token it accepts is the last thing a
 * step copies into place, and the advance is issued behind that copy without a host wait. On the
 * host backend each stream is a queue with a worker thread, so an advance run on the issuing
 * thread would read the previous token. A long chain of copies precedes the token's, so an advance
 * that jumps the queue reads the poison. */
TEST(an_advance_reads_the_token_a_copy_issued_before_it_delivers) {
    const int64_t big = 4 << 20;
    const int     chain = 16;
    Stream q;
    const RadStream s = q.s;
    REQUIRE(s != nullptr);
    DevBuf<uint8_t> blob_src(big, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> blob_dst(big, RAD_MEM_DEVICE);
    DevBuf<char> slab(4096, RAD_MEM_DEVICE);
    DevBuf<int32_t> src(64, RAD_MEM_DEVICE);
    DevBuf<rad::StepSlot> st((int64_t)sizeof(rad::StepSlot), RAD_MEM_DEVICE);
    DevBuf<int32_t> stage(4096, RAD_MEM_HOST_PINNED);
    REQUIRE(blob_src && blob_dst && slab && src && st && stage);
    std::memset(blob_src, 0x5A, (size_t)big);

    /* One device row that sampled row 0 with no drafts, at context 10, slot 0. */
    rad::AdvanceArgs a;
    a.mode = rad::ADV_ACCEPT;
    a.n_seq = 1;
    a.slab = slab;
    a.row_off = 0;   a.row_pitch = 1;
    a.tok_off = 16;  a.pos_off = 24; a.ctx_off = 32; a.acc_off = 40; a.out_off = 48;
    a.sid_off = 56;  a.cu_off = 64;  a.q_off = 72;   a.grp_off = 80;
    a.st = st;
    a.src = src;
    const int32_t layout[20] = { 1, 0, 0, 0,  0, 0,  0, 0,  10, 0,  0, 0,  0, 0,  0, 0,  0, 0,  1, 0 };
    std::memcpy(stage, layout, sizeof(layout));
    for (int round = 0; round < 3; ++round) {
        stage[64] = 7;                       /* the poison: what src holds before the copy */
        stage[65] = 4242 + round;            /* the token the copy delivers */
        REQUIRE(rad_memcpy_async(slab, stage, sizeof(layout), s) >= 0);
        REQUIRE(rad_memcpy_async(src, stage + 64, 4, s) >= 0);
        REQUIRE(rad_memset_async(st, 0, (int64_t)sizeof(rad::StepSlot), s) >= 0);
        REQUIRE(rad_stream_sync(s) >= 0);

        for (int i = 0; i < chain; ++i) REQUIRE(rad_memcpy_async(blob_dst, blob_src, big, s) >= 0);
        REQUIRE(rad_memcpy_async(src, stage + 65, 4, s) >= 0);
        CHECK_OK(rad::step_advance(a, s));
        rad::StepSlot got{};
        REQUIRE(rad_memcpy_async(stage + 128, st, (int64_t)sizeof(rad::StepSlot), s) >= 0);
        REQUIRE(rad_stream_sync(s) >= 0);
        std::memcpy(&got, stage + 128, sizeof(rad::StepSlot));
        CHECK_EQ(got.tok[0], 4242 + round);
        CHECK_EQ(got.ctx, 11);
        CHECK_EQ(got.n_acc, 1);
    }
}

TEST(the_null_stream_is_refused) {
    /* HIP's stream 0 serialises against every other blocking stream in the process, defeating
     * async transfers. Both backends refuse it, so the host backend cannot hide its use. */
    uint8_t a[16] = {}, b[16] = {};
    CHECK(rad_memcpy_async(a, b, sizeof a, nullptr) < 0);
    CHECK(std::strlen(rad_dev_last_error()) > 0);
    /* Drained by reading, per the header. */
    CHECK_EQ(std::strlen(rad_dev_last_error()), 0u);

    CHECK(rad_memset_async(a, 0, sizeof a, nullptr) < 0);
    CHECK(rad_stream_sync(nullptr) < 0);
}

/* ------------------------------------------------------------------ the file tier */

TEST(directfile_write_then_read_roundtrip) {
    const std::string path = temp_path("dio");
    const int64_t A = DirectFile::alignment();
    const int64_t body = 3 * A, tail = 123;

    DirectFile f;
    CHECK_OK(f.open(path.c_str(), RAD_DIO_READ | RAD_DIO_WRITE | RAD_DIO_CREATE | RAD_DIO_TRUNC |
                                  RAD_DIO_ALLOW_BUFFERED));
    if (!f.is_open()) return;
    if (!f.direct())
        fprintf(stderr, "       (note: %s has no O_DIRECT; the buffered fallback is under test)\n",
                path.c_str());
    CHECK_EQ(f.size(), (int64_t)0);

    /* The mover reads into pinned buffers, so the aligned path is tested with one: page-aligned
     * address, offset and length give one pread with no intermediate copy. */
    DevBuf<uint8_t> w(body, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> r(body, RAD_MEM_HOST_PINNED);
    CHECK(w && r);
    if (!w || !r) { f.close(); ::unlink(path.c_str()); return; }
    fill(w, body, 0xD1EC7);

    CHECK_OK(f.write_at(w, 0, body));
    CHECK_EQ(f.size(), body);

    /* The misaligned tail: an unaligned length from an unaligned stack buffer. This is the read
     * path's only bounce, needed because O_DIRECT returns EINVAL otherwise. */
    uint8_t t[tail];
    fill(t, tail, 0x7A11);
    CHECK_OK(f.write_at(t, body, tail));
    CHECK_EQ(f.size(), body + tail);

    std::memset(r, 0, (size_t)body);
    CHECK_OK(f.read_at(r, 0, body));
    CHECK_EQ(std::memcmp(r, w, (size_t)body), 0);

    uint8_t t2[tail];
    std::memset(t2, 0, sizeof t2);
    CHECK_OK(f.read_at(t2, body, tail));
    CHECK_EQ(std::memcmp(t2, t, sizeof t2), 0);

    /* A slice that is aligned at neither end and straddles two blocks. */
    uint8_t slice[2000];
    CHECK_OK(f.read_at(slice, 1000, sizeof slice));
    CHECK_EQ(std::memcmp(slice, w + 1000, sizeof slice), 0);

    /* One byte past the end is RAD_E_IO, not a short read: read_at fills exactly what was asked
     * or fails, so callers on the step path need not check a byte count. */
    CHECK(f.read_at(t2, body, tail + 1) < 0);
    CHECK(f.read_at(r, body + tail + A, A) < 0);

    /* Reopen without O_TRUNC: the bytes are on disk, not in this handle. */
    f.close();
    CHECK_OK(f.open(path.c_str(), RAD_DIO_READ | RAD_DIO_ALLOW_BUFFERED));
    CHECK_EQ(f.size(), body + tail);
    std::memset(r, 0, (size_t)body);
    CHECK_OK(f.read_at(r, 0, body));
    CHECK_EQ(std::memcmp(r, w, (size_t)body), 0);
    f.close();

    CHECK(f.read_at(r, 0, A) < 0);          /* closed: RAD_E_STATE, not a crash */
    CHECK_EQ(f.size(), (int64_t)-1);
    ::unlink(path.c_str());
}

TEST(directfile_refuses_a_missing_file) {
    DirectFile f;
    CHECK(f.open("/nonexistent/rad/dio/path.bin", RAD_DIO_READ | RAD_DIO_ALLOW_BUFFERED) < 0);
    CHECK(!f.is_open());
    CHECK(f.open(nullptr, RAD_DIO_READ) < 0);
}

/* ------------------------------------------------------------------ elastic memory
 *
 * Elastic pools let two pools trade VRAM (the KV cache returns what no session holds and the
 * expert slab takes it). The trade is safe only if an address never moves, a granule still in use
 * is never unmapped, and a range reported as committed is actually backed. Each fails silently:
 * a kernel reads another group's blocks, or a fault hits a previously valid address, or the first
 * write past the old boundary faults under load.
 *
 * The risk is all interval arithmetic, so it is tested on the host backend, where a decommitted
 * page can also be shown gone by reading it back as zero.
 */

TEST(elastic_memory_takes_address_space_and_no_pages) {
    if (!need_vmem()) return;
    const int64_t g = rad::VMem::granularity();
    CHECK(g > 0);
    CHECK_EQ(g % 4096, 0);              /* a granule is whole pages */

    rad::VMem v;
    CHECK_OK(v.reserve(8 * g));
    CHECK(v.base() != nullptr);
    CHECK_EQ(v.size(), 8 * g);
    CHECK_EQ(v.committed(), 0);         /* reserved is not allocated */
    CHECK_EQ((int64_t)((uintptr_t)v.base() % (uintptr_t)g), 0);

    /* A size that is not a whole number of granules rounds up; otherwise the trailing partial
     * granule could never be committed. */
    rad::VMem odd;
    CHECK_OK(odd.reserve(g + 1));
    CHECK_EQ(odd.size(), 2 * g);

    CHECK_EQ(v.reserve(0), RAD_E_INVAL);
    CHECK_EQ(v.size(), 0);              /* a refused reserve releases what it held */
}

TEST(a_committed_range_is_backed_and_keeps_its_address) {
    if (!need_vmem(/*host_visible=*/true)) return;
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(8 * g));
    void* const addr = v.base();

    CHECK_OK(v.commit(0, 2 * g));
    CHECK_EQ(v.committed(), 2 * g);
    CHECK(v.is_committed(0, 2 * g));
    CHECK(!v.is_committed(0, 3 * g));

    fill((uint8_t*)v.base(), 2 * g, 0x5eedu);
    uint8_t expect[128];
    fill(expect, sizeof expect, 0x5eedu);
    CHECK_EQ(std::memcmp(v.base(), expect, sizeof expect), 0);

    /* Growing at the far end keeps existing data and the base, from which every block index the
     * pool handed out is an offset. */
    CHECK_OK(v.commit(6 * g, 2 * g));
    CHECK_EQ(v.committed(), 4 * g);
    CHECK_EQ(v.base(), addr);
    CHECK_EQ(std::memcmp(v.base(), expect, sizeof expect), 0);
}

TEST(commit_rounds_outward_and_decommit_rounds_inward) {
    if (!need_vmem()) return;
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(4 * g));

    /* Committing one byte commits its whole granule. */
    CHECK_OK(v.commit(g / 2, 1));
    CHECK_EQ(v.committed(), g);
    CHECK(v.is_committed(0, g));

    /* A decommit releases only whole granules; the rest of a partly covered granule is still
     * owned. */
    CHECK_OK(v.decommit(g / 2, 1));
    CHECK_EQ(v.committed(), g);
    CHECK_OK(v.decommit(0, g - 1));
    CHECK_EQ(v.committed(), g);
    CHECK_OK(v.decommit(0, g));
    CHECK_EQ(v.committed(), 0);
}

TEST(the_commit_quantum_is_the_pool_alignment_on_every_backend) {
    if (!need_vmem()) return;
    /* Both backends use RAD_ALIGN_POOL (which every pool is aligned to) as the granule, not the
     * driver's minimum (one reports a single host page), so the commit arithmetic checked without
     * a card is what a card performs, and growth avoids a driver call per page. */
    const int64_t g = rad::VMem::granularity();
    CHECK(g >= (int64_t)RAD_ALIGN_POOL);
    CHECK_EQ(g % (int64_t)RAD_ALIGN_POOL, 0);
}

TEST(committing_over_a_commit_maps_only_the_gap) {
    if (!need_vmem()) return;
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(8 * g));

    /* A growing pool commits [0,n) each time n changes. The driver refuses overlapping maps, so
     * commit must map only the gap; a pool growing twice depends on it. */
    CHECK_OK(v.commit(0, 2 * g));
    CHECK_OK(v.commit(0, 5 * g));
    CHECK_EQ(v.committed(), 5 * g);
    CHECK_OK(v.commit(0, 5 * g));
    CHECK_EQ(v.committed(), 5 * g);

    /* Two islands and then the bridge between them. */
    rad::VMem w;
    CHECK_OK(w.reserve(8 * g));
    CHECK_OK(w.commit(0, g));
    CHECK_OK(w.commit(4 * g, g));
    CHECK_EQ(w.committed(), 2 * g);
    CHECK_OK(w.commit(0, 5 * g));
    CHECK_EQ(w.committed(), 5 * g);
    CHECK(w.is_committed(0, 5 * g));

    /* Past the end clamps rather than failing: the reserved range is the maximum. */
    CHECK_OK(w.commit(7 * g, 4 * g));
    CHECK_EQ(w.committed(), 6 * g);
}

TEST(decommitting_the_middle_leaves_the_ends) {
    if (!need_vmem()) return;
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(8 * g));
    CHECK_OK(v.commit(0, 4 * g));
    CHECK_OK(v.decommit(g, g));

    CHECK_EQ(v.committed(), 3 * g);
    CHECK(v.is_committed(0, g));
    CHECK(!v.is_committed(g, g));
    CHECK(v.is_committed(2 * g, 2 * g));
    /* A span across the hole is not committed, which is what a caller checks before handing a
     * range to a kernel. */
    CHECK(!v.is_committed(0, 4 * g));

    /* Refilling the hole merges the extents, so the extent list does not grow without bound. */
    CHECK_OK(v.commit(g, g));
    CHECK_EQ(v.committed(), 4 * g);
    CHECK(v.is_committed(0, 4 * g));
}

TEST(a_recommitted_range_comes_back_zeroed) {
    if (!need_vmem(/*host_visible=*/true)) return;
    const int64_t g = rad::VMem::granularity();
    rad::VMem v;
    CHECK_OK(v.reserve(4 * g));
    CHECK_OK(v.commit(0, 2 * g));

    fill((uint8_t*)v.base(), 2 * g, 0xabcdu);
    CHECK((( uint8_t*)v.base())[g] != 0);

    /* Decommitted pages are actually released: recommitting returns zeroes, not the old bytes. A
     * decommit that only forgot the range would pass every test above. */
    CHECK_OK(v.decommit(0, 2 * g));
    CHECK_EQ(v.committed(), 0);
    CHECK_OK(v.commit(0, 2 * g));
    const uint8_t* b = (const uint8_t*)v.base();
    for (int64_t i = 0; i < 2 * g; i += 4093) CHECK_EQ((int)b[i], 0);
}

TEST(elastic_memory_is_accounted_and_released_whole) {
    if (!need_vmem()) return;
    const int64_t g = rad::VMem::granularity();
    const int64_t before = rad::rad_dev_vmem_committed();
    {
        rad::VMem v;
        CHECK_OK(v.reserve(8 * g));
        CHECK_OK(v.commit(0, 3 * g));
        CHECK_EQ(rad::rad_dev_vmem_committed() - before, 3 * g);
        CHECK_OK(v.decommit(0, g));
        CHECK_EQ(rad::rad_dev_vmem_committed() - before, 2 * g);
        /* Destruction releases what is still committed. Freeing a range with pages in it leaks
         * them and on some drivers fails the address free, so release() fixes the order. */
    }
    CHECK_EQ(rad::rad_dev_vmem_committed(), before);
}

/* ------------------------------------------------------------------ the ranged copy */

/* Every step's staging upload and the block-table entries that changed go through one ranged copy
 * per rank, so a wrong range offset is a wrong block id: another sequence's KV with fluent output.
 * The copy is issued behind a chain that rewrites the source, so it must read the chain's result,
 * and a range may overwrite part of the prefix after it is copied. The expected bytes are composed
 * on the host from the source's final contents. */
TEST(a_ranged_copy_lands_its_prefix_then_each_range_in_order) {
    const int64_t n = 8192, chain_bytes = 1 << 20;
    Stream st;
    DevBuf<uint8_t> src(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dst(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> next(n, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> big_a(chain_bytes, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> big_b(chain_bytes, RAD_MEM_DEVICE);
    DevBuf<RadCopyRange> list(8 * (int64_t)sizeof(RadCopyRange), RAD_MEM_HOST_PINNED);
    REQUIRE(src && dst && next && big_a && big_b && list);
    fill(src, n, 0x1234);
    fill(next, n, 0x5678);
    std::memset(big_a, 0x11, (size_t)chain_bytes);
    CHECK_OK(rad_memset_async(dst, 0xEE, n, st.s));
    CHECK_OK(rad_stream_sync(st.s));

    /* Five ranges: two adjacent at the destination, one written over the prefix, one from the
     * start of the source and one into the last bytes. */
    const RadCopyRange r[5] = {
        { 1024, 512, 64 }, { 2048, 576, 128 }, { 3000, 128, 32 }, { 0, 4000, 256 }, { 4, 8184, 8 },
    };
    std::memcpy(list, r, sizeof r);
    /* A chain ahead of the copy, ending with the source's final contents. */
    for (int i = 0; i < 8; ++i) CHECK_OK(rad_memcpy_async(big_b, big_a, chain_bytes, st.s));
    CHECK_OK(rad_memcpy_async(src, next, n, st.s));
    CHECK_OK(rad_memcpy_ranges_async(dst, src, 256, list, 5, st.s));
    CHECK_OK(rad_stream_sync(st.s));

    std::vector<uint8_t> want((size_t)n, 0xEE);
    std::memcpy(want.data(), next, 256);
    for (const RadCopyRange& x : r)
        std::memcpy(want.data() + x.dst, next + x.src, (size_t)x.bytes);
    CHECK_EQ(std::memcmp(dst, want.data(), (size_t)n), 0);

    /* Nothing to copy is not an error and moves nothing. */
    CHECK_OK(rad_memcpy_ranges_async(dst, src, 0, nullptr, 0, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ(std::memcmp(dst, want.data(), (size_t)n), 0);
    CHECK_EQ(rad_memcpy_ranges_async(dst, src, 0, nullptr, 2, st.s), RAD_E_INVAL);
    CHECK_EQ(rad_memcpy_ranges_async(dst, src, -16, list, 1, st.s), RAD_E_INVAL);
}

/* The card's copy kernel moves 16-byte pieces: the prefix is a multiple of 16 and both bases are
 * 16-byte aligned. Both backends refuse a call outside that, so a caller bug fails on the host
 * suite and not first on a card. */
TEST(a_ranged_copy_refuses_what_the_cards_kernel_cannot_copy) {
    Stream st;
    DevBuf<uint8_t> src(4096, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dst(4096, RAD_MEM_HOST_PINNED);
    DevBuf<RadCopyRange> list((int64_t)sizeof(RadCopyRange), RAD_MEM_HOST_PINNED);
    REQUIRE(src && dst && list);
    *list.p = RadCopyRange{ 0, 0, 16 };
    CHECK_EQ(rad_memcpy_ranges_async(dst, src, 24, list, 1, st.s), RAD_E_INVAL);
    CHECK_EQ(rad_memcpy_ranges_async(dst + 4, src, 16, list, 1, st.s), RAD_E_UNSUPPORTED);
    CHECK_EQ(rad_memcpy_ranges_async(dst, src + 8, 16, list, 1, st.s), RAD_E_UNSUPPORTED);
    CHECK_OK(rad_memcpy_ranges_async(dst + 16, src + 32, 16, list, 1, st.s));
    CHECK_OK(rad_stream_sync(st.s));
}

/* ------------------------------------------------------------------ stream contract extras */

/* rad_stream_sync waits for the work submitted before the call, not for an empty queue: the step
 * thread syncs while another thread keeps issuing to the same stream, and a sync that waited for
 * the queue to drain would not return while the issuing continues. A pusher keeps the stream's
 * ring full: each pair is a 64 KiB copy and a copy of the pair's number into `last`. Every sync
 * must return while the pusher is still running, with every pair issued before it landed. */
TEST(stream_sync_returns_while_another_thread_keeps_issuing) {
    const int64_t piece = 64 << 10;
    const int64_t slots = 4096;          /* more pairs than the ring holds, so a slot is retired
                                          * before the pusher writes it again */
    Stream st;
    DevBuf<uint8_t> a(piece, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> b(piece, RAD_MEM_HOST_PINNED);
    DevBuf<uint32_t> seq(slots * 4, RAD_MEM_HOST_PINNED);
    DevBuf<uint32_t> last(64, RAD_MEM_HOST_PINNED);
    REQUIRE(a && b && seq && last);
    std::memset(a, 1, (size_t)piece);
    last[0] = 0;
    std::atomic<bool> stop{ false }, running{ true };
    std::atomic<int64_t> pushed{ 0 };
    std::thread pusher([&] {
        /* Bounded, so a sync that waits for an empty queue fails the checks below instead of
         * hanging: the pusher gives up after 100k pairs. */
        for (int64_t k = 1; !stop.load() && k <= 100000; ++k) {
            seq[k % slots] = (uint32_t)k;
            if (rad_memcpy_async(b, a, piece, st.s) < 0) break;
            if (rad_memcpy_async(last, seq + k % slots, 4, st.s) < 0) break;
            pushed.store(k);
        }
        running.store(false);
    });
    while (pushed.load() < 2048 && running.load()) std::this_thread::yield();
    int while_running = 0, behind = 0;
    for (int i = 0; i < 10; ++i) {
        const int64_t before = pushed.load();
        CHECK_OK(rad_stream_sync(st.s));
        while_running += running.load();
        behind += (int64_t)*(volatile uint32_t*)last.p < before;
    }
    stop.store(true);
    pusher.join();
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ(while_running, 10);
    CHECK_EQ(behind, 0);
}

/* Only an event made for host waits can be waited on by the host; a local or plain event raises
 * no interrupt on a card, so a host sync on it is refused on both backends rather than spinning. */
TEST(a_host_sync_on_an_event_not_made_for_it_is_refused) {
    Stream st;
    RadEvent local = nullptr, plain = nullptr, host = nullptr;
    CHECK_OK(rad_event_create_local(&local));
    CHECK_OK(rad_event_create_as(&plain, 0));
    CHECK_OK(rad_event_create(&host));
    for (RadEvent e : { local, plain, host }) CHECK_OK(rad_event_record(e, st.s));
    CHECK_EQ(rad_event_sync(local), RAD_E_INVAL);
    CHECK_EQ(rad_event_sync(plain), RAD_E_INVAL);
    CHECK_OK(rad_event_sync(host));
    CHECK_OK(rad_stream_sync(st.s));
    /* Timing needs both events recorded. */
    RadEvent never = nullptr;
    CHECK_OK(rad_event_create(&never));
    float ms = 0;
    CHECK(rad_event_elapsed_ms(host, never, &ms) < 0);
    CHECK(rad_event_elapsed_ms(never, host, &ms) < 0);
    if (rad::device_is_host()) CHECK_EQ(rad_event_elapsed_ms(never, host, &ms), RAD_E_STATE);
    for (RadEvent e : { local, plain, host, never }) rad_event_destroy(e);
}

/* ------------------------------------------------------------------ gates */

/* Work queued behind a closed gate must not run until the host opens it, while work ahead of the
 * gate runs as usual: rad-kbench queues a whole timed batch behind one so the device never waits
 * for submission. The check behind the gate is bounded: a gate that does not hold would let a 2 KiB
 * copy land well inside 20 ms. */
TEST(nothing_behind_a_closed_gate_runs_until_it_opens) {
    Stream st;
    DevBuf<uint8_t> src(4096, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dst(4096, RAD_MEM_HOST_PINNED);
    REQUIRE(src && dst);
    fill(src, 4096, 7);
    std::memset(dst, 0, 4096);
    Event before, after;
    CHECK_OK(rad_memcpy_async(dst, src, 2048, st.s));
    CHECK_OK(rad_event_record(before.e, st.s));
    void* gate = nullptr;
    REQUIRE_EQ(rad_stream_gate(st.s, &gate), RAD_OK);
    REQUIRE(gate != nullptr);
    CHECK_OK(rad_memcpy_async(dst + 2048, src + 2048, 2048, st.s));
    CHECK_OK(rad_event_record(after.e, st.s));
    CHECK_OK(rad_event_sync(before.e));
    CHECK_EQ(std::memcmp(dst, src, 2048), 0);
    const auto t0 = std::chrono::steady_clock::now();
    int ran = 0;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(20))
        ran |= rad_event_query(after.e) != 0;
    CHECK_EQ(ran, 0);
    int64_t untouched = 0;
    for (int i = 2048; i < 4096; ++i) untouched += ((volatile uint8_t*)dst.p)[i] == 0;
    CHECK_EQ(untouched, 2048ll);
    CHECK_OK(rad_gate_open(gate));
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ(rad_event_query(after.e), 1);
    CHECK_EQ(std::memcmp(dst, src, 4096), 0);
}

/* A ride a prefetch DMA list polls stays out of the signal pool until the engine has read it,
 * however early the caller returns it. The pool reuses a signal at 0 that no queue packet names,
 * and a DMA poll is none: a mark handed out again and set to 1 before the engine polls it would
 * hold the list until the next use completes, which can sit behind a wait for this very list. The
 * engine is held on an earlier list whose mark completes only once a gate opens, so the list
 * under test has not polled its mark when the mark's dispatch has run and it is returned. */
TEST(a_dma_lists_mark_stays_out_of_the_pool_until_the_engine_has_read_it) {
    if (rad::device_is_host()) SKIP("a card only");
    if (!rad_xp_supported(0)) SKIP("no prefetch DMA queue on this card");
    Stream held, st;
    DevBuf<uint8_t> a(256, RAD_MEM_DEVICE), b(256, RAD_MEM_DEVICE);
    DevBuf<uint32_t> word(64, RAD_MEM_HOST_PINNED);
    REQUIRE(a && b && word);
    *(volatile uint32_t*)word.p = 0;

    void* gate = nullptr;
    REQUIRE_EQ(rad_stream_gate(held.s, &gate), RAD_OK);
    void* first = nullptr;
    REQUIRE_EQ(rad_ride(held.s, &first), RAD_OK);
    CHECK_OK(rad_memset_async(a, 1, 256, held.s));
    REQUIRE_EQ(rad_xp_list(0, first, 0, nullptr, nullptr, nullptr, nullptr, 0, RAD_XP_NO_YIELD),
               RAD_OK);

    void* mark = nullptr;
    REQUIRE_EQ(rad_ride(st.s, &mark), RAD_OK);
    CHECK_OK(rad_memset_async(b, 2, 256, st.s));
    REQUIRE_EQ(rad_xp_list(0, mark, 0, nullptr, nullptr, nullptr, word, 7, RAD_XP_NO_YIELD),
               RAD_OK);
    CHECK_OK(rad_stream_sync(st.s));                 /* the mark is 0 now */
    rad_ride_release(mark, nullptr, 0);
    CHECK(!rad::aql::ride_reusable(mark));
    CHECK_EQ(*(volatile uint32_t*)word.p, 0u);

    CHECK_OK(rad_gate_open(gate));
    CHECK_OK(rad_stream_sync(held.s));
    const auto t0 = std::chrono::steady_clock::now();
    while (*(volatile uint32_t*)word.p != 7u &&
           std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5))
        std::this_thread::yield();
    CHECK_EQ(*(volatile uint32_t*)word.p, 7u);
    bool back = false;
    while (!back && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(6))
        back = rad::aql::ride_reusable(mark);
    CHECK(back);
    rad_ride_release(first, nullptr, 0);
}

/* A host wait on a stream at a closed gate could only end when the waiting thread opened it, so it
 * is refused with RAD_E_STATE instead of hanging, and works again once the gate is open. */
TEST(a_host_wait_on_a_gated_stream_is_refused_not_a_deadlock) {
    Stream st;
    void* gate = nullptr;
    REQUIRE_EQ(rad_stream_gate(st.s, &gate), RAD_OK);
    CHECK_EQ(rad_stream_sync(st.s), RAD_E_STATE);
    CHECK(std::strstr(rad_dev_last_error(), "closed gate") != nullptr);
    CHECK_OK(rad_gate_open(gate));
    CHECK_OK(rad_stream_sync(st.s));
}

/* Behind a closed gate the queue only fills. Submissions up to the room succeed and the one after
 * is refused rather than waiting for room that cannot appear; opening the gate runs every accepted
 * submission in order. The ring holds 1024 tasks, the gate's own among them until the worker takes
 * it, so 1023 or 1024 fit. */
TEST(a_full_gated_queue_refuses_the_next_submission_instead_of_waiting) {
    if (!need_host_backend()) return;
    Stream st;
    DevBuf<uint8_t> a(64, RAD_MEM_DEVICE);
    REQUIRE(a);
    void* gate = nullptr;
    REQUIRE_EQ(rad_stream_gate(st.s, &gate), RAD_OK);
    CHECK(rad_gate_room(st.s) >= 1023);
    int ok = 0, refused = 0;
    for (int i = 0; i < 2048 && !refused; ++i) {
        const int rc = rad_memset_async(a, (ok + 1) & 0xff, 64, st.s);
        if (rc == RAD_OK) {
            ++ok;
        } else {
            CHECK_EQ(rc, RAD_E_STATE);
            ++refused;
        }
    }
    CHECK_EQ(refused, 1);
    CHECK(ok == 1023 || ok == 1024);
    CHECK_EQ(rad_gate_room(st.s), 0ll);
    CHECK_OK(rad_gate_open(gate));
    CHECK_OK(rad_stream_sync(st.s));
    CHECK_EQ((int)a[0], ok & 0xff);
    CHECK_EQ((int)a[63], ok & 0xff);
}

/* The point of a gate: two events around work queued behind it time the work, not the host. The
 * host stalls 20 ms between the two records; without a gate that stall is inside the interval,
 * with one it is not. Timing is the behaviour here, and the margins are an order of magnitude. */
TEST(events_behind_a_gate_leave_the_hosts_submission_time_out) {
    Stream st;
    DevBuf<uint8_t> a(4096, RAD_MEM_DEVICE);
    REQUIRE(a);
    Event e0, e1;
    float open_ms = 0, gated_ms = 0;
    for (const bool gated : { false, true }) {
        void* gate = nullptr;
        if (gated) REQUIRE_EQ(rad_stream_gate(st.s, &gate), RAD_OK);
        CHECK_OK(rad_event_record(e0.e, st.s));
        CHECK_OK(rad_memset_async(a, 1, 4096, st.s));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK_OK(rad_memset_async(a, 2, 4096, st.s));
        CHECK_OK(rad_event_record(e1.e, st.s));
        if (gated) CHECK_OK(rad_gate_open(gate));
        CHECK_OK(rad_stream_sync(st.s));
        CHECK_OK(rad_event_elapsed_ms(e0.e, e1.e, gated ? &gated_ms : &open_ms));
    }
    CHECK(open_ms >= 15.0f);
    CHECK(gated_ms < 10.0f);
}

/* A pitched copy is ordered behind the copies before it like any other, and either pitch shorter
 * than the row is refused, not clamped. */
TEST(a_pitched_copy_reads_what_the_copies_before_it_wrote) {
    const int64_t W = 40, H = 9, SP = 64, DP = 48, big = 1 << 20;
    Stream st;
    DevBuf<uint8_t> src(SP * H, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> next(SP * H, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> dst(DP * H, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> big_a(big, RAD_MEM_HOST_PINNED);
    DevBuf<uint8_t> big_b(big, RAD_MEM_DEVICE);
    REQUIRE(src && next && dst && big_a && big_b);
    fill(src, SP * H, 1);
    fill(next, SP * H, 2);
    std::memset(big_a, 0, (size_t)big);
    CHECK_OK(rad_memset_async(dst, 0xCC, DP * H, st.s));
    for (int i = 0; i < 16; ++i) CHECK_OK(rad_memcpy_async(big_b, big_a, big, st.s));
    CHECK_OK(rad_memcpy_async(src, next, SP * H, st.s));
    CHECK_OK(rad_memcpy_2d_async(dst, DP, src, SP, W, H, st.s));
    CHECK_OK(rad_stream_sync(st.s));
    int64_t wrong = 0;
    for (int64_t r = 0; r < H; ++r)
        for (int64_t c = 0; c < DP; ++c)
            wrong += dst[r * DP + c] != (c < W ? next[r * SP + c] : 0xCC);
    CHECK_EQ(wrong, 0ll);
    CHECK(rad_memcpy_2d_async(dst, DP, src, W - 1, W, H, st.s) < 0);
    CHECK(rad_memcpy_2d_async(dst, W - 1, src, SP, W, H, st.s) < 0);
    CHECK_OK(rad_memcpy_2d_async(dst, DP, src, SP, 0, H, st.s));     /* empty: nothing to do */
}

/* Freeing memory the device layer did not allocate would corrupt the heap; the host backend
 * refuses it with a warning and keeps its accounting. */
TEST(freeing_a_pointer_the_device_layer_did_not_allocate_is_ignored) {
    if (!need_host_backend()) return;
    int64_t used0 = 0, used1 = 0;
    CHECK_OK(rad::dev_vram_used(0, &used0));
    uint8_t on_stack[64];
    const std::string err = stderr_of([&] { rad_dev_free(on_stack, RAD_MEM_DEVICE); });
    CHECK(has(err, "not allocated by the device layer"));
    CHECK_OK(rad::dev_vram_used(0, &used1));
    CHECK_EQ(used1, used0);
}

/* ------------------------------------------------------------------ code objects */

/* The AQL backend picks the card's code object out of a kernel library's fat binary and reads
 * each kernel's argument layout from its metadata. This decides whether a library loads at all
 * on a given card, and a misread argument offset faults every launch. The readers are host code,
 * checked here against byte images built to the formats' specifications. */
namespace {

/* A clang offload bundle: magic, entry count, then per entry (offset, size, triple length,
 * triple), then the payloads. */
std::vector<unsigned char> offload_bundle(const std::vector<std::pair<std::string, std::string>>& e) {
    std::vector<unsigned char> b;
    auto u64 = [&b](uint64_t v) {
        for (int k = 0; k < 8; ++k) b.push_back((unsigned char)(v >> (8 * k)));
    };
    const char magic[] = "__CLANG_OFFLOAD_BUNDLE__";
    b.insert(b.end(), magic, magic + 24);
    u64(e.size());
    uint64_t off = 32;
    for (const auto& x : e) off += 24 + x.first.size();
    for (const auto& x : e) {
        u64(off);
        u64(x.second.size());
        u64(x.first.size());
        b.insert(b.end(), x.first.begin(), x.first.end());
        off += x.second.size();
    }
    for (const auto& x : e) b.insert(b.end(), x.second.begin(), x.second.end());
    return b;
}

/* The payload `bundle_code_object` picked, or "" when it refused. */
std::string pick(const std::vector<unsigned char>& b, const std::vector<std::string>& isas,
                 std::string* why = nullptr) {
    const void* out = nullptr;
    size_t size = 0;
    std::string w;
    const bool ok = rad::aql::bundle_code_object(b.data(), isas, &out, &size, &w);
    if (why) *why = w;
    if (!ok) return std::string();
    return std::string((const char*)out, size);
}

/* A minimal msgpack writer: the forms code object metadata uses. */
struct MsgPack {
    std::vector<unsigned char> b;
    void map(uint32_t n) { b.push_back((unsigned char)(0x80 | n)); }
    void arr(uint32_t n) { b.push_back((unsigned char)(0x90 | n)); }
    void str(const std::string& s) {
        if (s.size() < 32) b.push_back((unsigned char)(0xa0 | s.size()));
        else { b.push_back(0xd9); b.push_back((unsigned char)s.size()); }
        b.insert(b.end(), s.begin(), s.end());
    }
    void uint(uint64_t v) {
        if (v < 128) { b.push_back((unsigned char)v); return; }
        const int nb = v < 0x10000 ? 2 : v < 0x100000000ull ? 4 : 8;
        b.push_back(nb == 2 ? 0xcd : nb == 4 ? 0xce : 0xcf);
        for (int k = nb - 1; k >= 0; --k) b.push_back((unsigned char)(v >> (8 * k)));
    }
    void boolean(bool v) { b.push_back(v ? 0xc3 : 0xc2); }
};

/* An ELF64 file with one PT_NOTE program header whose note is NT_AMDGPU_METADATA over `desc`.
 * `cut` drops bytes from the end of the file. */
std::vector<unsigned char> elf_with_note(const std::vector<unsigned char>& desc, size_t cut = 0) {
    std::vector<unsigned char> f(64 + 56, 0);
    f[0] = 0x7f; f[1] = 'E'; f[2] = 'L'; f[3] = 'F'; f[4] = 2; f[5] = 1;
    auto put = [&f](size_t at, uint64_t v, int nb) {
        for (int k = 0; k < nb; ++k) f[at + (size_t)k] = (unsigned char)(v >> (8 * k));
    };
    put(32, 64, 8);                                  /* e_phoff */
    put(54, 56, 2);                                  /* e_phentsize */
    put(56, 1, 2);                                   /* e_phnum */
    const size_t note = f.size();
    std::vector<unsigned char> n(12, 0);
    const uint32_t namesz = 7, descsz = (uint32_t)desc.size(), type = 32;
    std::memcpy(n.data(), &namesz, 4);
    std::memcpy(n.data() + 4, &descsz, 4);
    std::memcpy(n.data() + 8, &type, 4);
    const char name[8] = "AMDGPU";
    n.insert(n.end(), name, name + 8);
    n.insert(n.end(), desc.begin(), desc.end());
    while (n.size() % 4) n.push_back(0);
    put(64, 4, 4);                                   /* p_type = PT_NOTE */
    put(64 + 8, note, 8);                            /* p_offset */
    put(64 + 32, n.size(), 8);                       /* p_filesz */
    f.insert(f.end(), n.begin(), n.end());
    f.resize(f.size() - cut);
    return f;
}

/* Metadata for two kernels: one with an explicit scalar, a buffer, a hidden block count and the
 * hostcall buffer this backend does not provide; one with no arguments and a long symbol. */
std::vector<unsigned char> two_kernels() {
    MsgPack m;
    m.map(2);
    m.str("amdhsa.version"); m.arr(2); m.uint(1); m.uint(2);
    m.str("amdhsa.kernels"); m.arr(2);
    m.map(6);
    m.str(".symbol"); m.str("scale.kd");
    m.str(".kernarg_segment_size"); m.uint(280);
    m.str(".group_segment_fixed_size"); m.uint(70000);
    m.str(".private_segment_fixed_size"); m.uint(16);
    m.str(".uses_dynamic_stack"); m.boolean(false);
    m.str(".args"); m.arr(4);
    const struct { uint32_t off, size; const char* kind; } args[4] = {
        { 0, 4, "by_value" }, { 8, 8, "global_buffer" }, { 16, 4, "hidden_block_count_x" },
        { 24, 8, "hidden_hostcall_buffer" },
    };
    for (const auto& a : args) {
        m.map(3);
        m.str(".offset"); m.uint(a.off);
        m.str(".size"); m.uint(a.size);
        m.str(".value_kind"); m.str(a.kind);
    }
    m.map(3);
    m.str(".symbol"); m.str("a_kernel_whose_symbol_is_longer_than_31.kd");
    m.str(".uses_dynamic_stack"); m.boolean(true);
    m.str(".kernarg_segment_size"); m.uint(0);
    return m.b;
}

}  /* namespace */

TEST(the_cards_exact_code_object_wins_and_its_family_build_is_the_fallback) {
    const auto b = offload_bundle({
        { "host-x86_64-unknown-linux-gnu-", "HOST" },
        { "hipv4-amdgcn-amd-amdhsa--gfx1201", "EXACT" },
        { "hipv4-amdgcn-amd-amdhsa--gfx12-generic", "FAMILY" },
        { "hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-", "PINNED" },
        { "hipv4-amdgcn-amd-amdhsa--gfx942", "ANYMODE" },
        { "hipv4-amdgcn-amd-amdhsa--gfx1100", "" },          /* an empty entry is skipped */
    });
    const std::string pre = "amdgcn-amd-amdhsa--";
    CHECK_EQ(pick(b, { pre + "gfx1201", pre + "gfx12-generic" }), std::string("EXACT"));
    CHECK_EQ(pick(b, { pre + "gfx1200", pre + "gfx12-generic" }), std::string("FAMILY"));
    /* A code object pinned to the card's mode beats one that runs in either mode, and one pinned
     * to another mode never runs. */
    CHECK_EQ(pick(b, { pre + "gfx942:sramecc+:xnack-" }), std::string("PINNED"));
    CHECK_EQ(pick(b, { pre + "gfx942:sramecc-:xnack-" }), std::string("ANYMODE"));
    CHECK_EQ(pick(b, { "gfx942:sramecc+:xnack-" }), std::string("PINNED"));   /* no prefix */
    std::string why;
    CHECK_EQ(pick(b, { pre + "gfx1100" }, &why), std::string());
    CHECK(has(why, "gfx1201") && has(why, "gfx942:sramecc+:xnack-") && has(why, "this card runs [gfx1100]"));
    /* A card without the mode a code object pins cannot run it. */
    const auto pinned = offload_bundle({ { "hip-amdgcn-amd-amdhsa--gfx90a:xnack-", "X" } });
    CHECK_EQ(pick(pinned, { pre + "gfx90a" }, &why), std::string());
    CHECK_EQ(pick(pinned, { pre + "gfx90a:xnack-" }), std::string("X"));
}

TEST(a_bundle_that_is_not_one_is_refused_by_name) {
    const std::string pre = "amdgcn-amd-amdhsa--";
    std::string why;
    std::vector<unsigned char> ccob(64, 0);
    std::memcpy(ccob.data(), "CCOB", 4);
    CHECK_EQ(pick(ccob, { pre + "gfx1201" }, &why), std::string());
    CHECK(has(why, "compressed"));
    std::vector<unsigned char> junk(64, 'x');
    CHECK_EQ(pick(junk, { pre + "gfx1201" }, &why), std::string());
    CHECK(has(why, "does not start with"));
    auto none = offload_bundle({});
    CHECK_EQ(pick(none, { pre + "gfx1201" }, &why), std::string());
    CHECK(has(why, "entry count"));
    auto many = offload_bundle({ { "hip-amdgcn-amd-amdhsa--gfx1201", "A" } });
    const uint64_t n = 4097;
    std::memcpy(many.data() + 24, &n, 8);
    CHECK_EQ(pick(many, { pre + "gfx1201" }, &why), std::string());
    CHECK(has(why, "entry count"));
    auto triple = offload_bundle({ { "hip-amdgcn-amd-amdhsa--gfx1201", "A" } });
    const uint64_t tl = 5000;
    std::memcpy(triple.data() + 32 + 16, &tl, 8);
    CHECK_EQ(pick(triple, { pre + "gfx1201" }, &why), std::string());
    CHECK(has(why, "triple"));
    const void* out = (const void*)1;
    size_t size = 1;
    CHECK(!rad::aql::bundle_code_object(nullptr, { pre + "gfx1201" }, &out, &size, &why));
    CHECK(out == nullptr && size == 0);
}

TEST(kernel_argument_layouts_come_from_the_metadata_note) {
    const auto elf = elf_with_note(two_kernels());
    std::vector<rad::aql::KernelMeta> ks;
    std::string why;
    REQUIRE(rad::aql::code_object_kernels(elf.data(), elf.size(), &ks, &why));
    REQUIRE_EQ(ks.size(), 2u);
    const rad::aql::KernelMeta& k = ks[0];
    CHECK_EQ(k.symbol, std::string("scale.kd"));
    CHECK_EQ(k.kernarg_size, 280u);
    CHECK_EQ(k.group_size, 70000u);
    CHECK_EQ(k.private_size, 16u);
    CHECK(!k.dynamic_stack);
    REQUIRE_EQ(k.args.size(), 4u);
    using H = rad::aql::Hidden;
    const struct { uint32_t off, size; H h; } want[4] = {
        { 0, 4, H::None }, { 8, 8, H::None }, { 16, 4, H::BlockCountX }, { 24, 8, H::Unsupported },
    };
    for (size_t i = 0; i < 4; ++i) {
        CHECK_EQ(k.args[i].offset, want[i].off);
        CHECK_EQ(k.args[i].size, want[i].size);
        CHECK(k.args[i].hidden == want[i].h);
    }
    CHECK_EQ(k.args[3].kind, std::string("hidden_hostcall_buffer"));
    CHECK_EQ(ks[1].symbol, std::string("a_kernel_whose_symbol_is_longer_than_31.kd"));
    CHECK(ks[1].dynamic_stack);
    CHECK(ks[1].args.empty());
}

TEST(a_damaged_code_object_is_refused_not_misread) {
    std::vector<rad::aql::KernelMeta> ks;
    std::string why;
    /* The note's descriptor cut short: the metadata is not there. */
    auto cut = elf_with_note(two_kernels(), 40);
    CHECK(!rad::aql::code_object_kernels(cut.data(), cut.size(), &ks, &why));
    CHECK(has(why, "metadata"));
    /* The note intact but its msgpack truncated inside a map. */
    std::vector<unsigned char> half = two_kernels();
    half.resize(half.size() / 2);
    auto bad = elf_with_note(half);
    CHECK(!rad::aql::code_object_kernels(bad.data(), bad.size(), &ks, &why));
    CHECK(has(why, "malformed"));
    /* Not ELF64, and program headers past the end. */
    std::vector<unsigned char> elf = elf_with_note(two_kernels());
    elf[4] = 1;
    CHECK(!rad::aql::code_object_kernels(elf.data(), elf.size(), &ks, &why));
    CHECK(has(why, "64-bit ELF"));
    elf = elf_with_note(two_kernels());
    elf[56] = 200;
    CHECK(!rad::aql::code_object_kernels(elf.data(), elf.size(), &ks, &why));
    CHECK(has(why, "past its end"));
    /* Metadata without a kernel list. */
    MsgPack m;
    m.map(1);
    m.str("amdhsa.version"); m.arr(0);
    auto nolist = elf_with_note(m.b);
    CHECK(!rad::aql::code_object_kernels(nolist.data(), nolist.size(), &ks, &why));
    CHECK(has(why, "amdhsa.kernels"));
}

RAD_TEST_MAIN_HOST_KARGS()
