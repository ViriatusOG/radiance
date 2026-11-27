/* vmemprobe -- measures what a virtual-memory decommit actually returns to the card.
 *
 * The elastic KV pool assumes that unmapping a sub-range of a live reservation frees memory another
 * pool can commit. The API does not promise this and the pool cannot detect it (a failed commit
 * looks like a full pool).
 *
 * Each trial commits a range, releases it a different way, and reports the card's free counter
 * beside what hipMalloc can still reach; they disagree where the driver failed to credit memory
 * back.
 *
 * Build: hipcc -O1 --offload-arch=<gfx> -I/opt/rocm/include -L/opt/rocm/lib \
 *        -o vmemprobe vmemprobe.cpp
 * Run on an idle card: other allocations skew every reading. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static size_t G = 0;
static hipMemAllocationProp prop{};
static hipMemAccessDesc desc{};
static double freem() {
    size_t fr = 0, tot = 0;
    (void)hipMemGetInfo(&fr, &tot);
    return fr / 1048576.0;
}
static double reach(int cap_gib) {
    std::vector<void*> got; size_t n = 0; const size_t step = 128ull << 20;
    for (;;) { void* p = nullptr;
        if (hipMalloc(&p, step) != hipSuccess) break;
        got.push_back(p); n += step; if (n > (size_t)cap_gib << 30) break; }
    for (void* p : got) (void)hipFree(p);
    return n / 1048576.0;
}
/* keep_handle: hold the allocation handle past the map, and release it only at the end. */
static void trial(const char* what, bool keep_handle, bool piecewise, bool free_va) {
    const size_t bytes = (4ull << 30) / G * G;
    void* base = nullptr;
    if (hipMemAddressReserve(&base, bytes, G, nullptr, 0) != hipSuccess) {
        printf("%-34s reserve failed\n", what);
        return;
    }
    const double f0 = freem();
    hipMemGenericAllocationHandle_t h{};
    /* A failed trial gives back what it took, or later trials read less free memory. */
    if (hipMemCreate(&h, bytes, &prop, 0) != hipSuccess) {
        printf("%-34s create failed\n", what);
        (void)hipMemAddressFree(base, bytes);
        return;
    }
    if (hipMemMap(base, bytes, 0, h, 0) != hipSuccess) {
        printf("%-34s map failed\n", what);
        (void)hipMemRelease(h);
        (void)hipMemAddressFree(base, bytes);
        return;
    }
    if (!keep_handle) (void)hipMemRelease(h);
    (void)hipMemSetAccess(base, bytes, &desc, 1);
    const double f1 = freem();
    if (piecewise) {
        const size_t chunk = bytes / 8 / G * G;
        for (int i = 0; i < 8; ++i) (void)hipMemUnmap((char*)base + (size_t)i * chunk, chunk);
    } else {
        (void)hipMemUnmap(base, bytes);
    }
    if (keep_handle) (void)hipMemRelease(h);
    const double f2 = freem();
    if (free_va) (void)hipMemAddressFree(base, bytes);
    const double f3 = freem();
    printf("%-34s free: %8.1f -> %8.1f (mapped) -> %8.1f (let go) -> %8.1f (va freed)  "
           "reach %8.1f\n",
           what, f0, f1, f2, f3, reach(24));
    if (!free_va) (void)hipMemAddressFree(base, bytes);
}
/* The elastic pool's cycle: commit a sub-range of a long-lived reservation, unmap it, repeat. Free
 * memory that drops on every pass would leak under a varying workload. */
static void cycle(int n) {
    const size_t total = (8ull << 30) / G * G;
    const size_t part  = (2ull << 30) / G * G;
    void* base = nullptr;
    if (hipMemAddressReserve(&base, total, G, nullptr, 0) != hipSuccess) {
        printf("cycle: reserve failed\n");
        return;
    }
    printf("\ncycle: %.1f GiB reservation held, %.1f GiB mapped and unmapped inside it\n",
           total / 1073741824.0, part / 1073741824.0);
    for (int i = 0; i < n; ++i) {
        hipMemGenericAllocationHandle_t h{};
        if (hipMemCreate(&h, part, &prop, 0) != hipSuccess) {
            printf("  pass %d: create failed\n", i);
            break;
        }
        if (hipMemMap(base, part, 0, h, 0) != hipSuccess) {
            (void)hipMemRelease(h);
            printf("  pass %d: map failed\n", i);
            break;
        }
        (void)hipMemRelease(h);
        (void)hipMemSetAccess(base, part, &desc, 1);
        const double fm = freem();
        (void)hipMemUnmap(base, part);
        printf("  pass %d: mapped %8.1f  unmapped %8.1f MiB free\n", i, fm, freem());
    }
    (void)hipMemAddressFree(base, total);
    printf("cycle: %.1f MiB free once the reservation itself is freed\n\n", freem());
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    if (hipSetDevice(dev) != hipSuccess) { printf("no device %d\n", dev); return 1; }
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;
    desc.location = prop.location; desc.flags = hipMemAccessFlagsProtReadWrite;
    size_t gr = 0, gm = 0;
    (void)hipMemGetAllocationGranularity(&gr, &prop, hipMemAllocationGranularityRecommended);
    (void)hipMemGetAllocationGranularity(&gm, &prop, hipMemAllocationGranularityMinimum);
    G = gr ? gr : 4096;
    printf("granularity: recommended %zu, minimum %zu\n", gr, gm);
    printf("baseline reach %.1f MiB, free %.1f MiB\n\n", reach(24), freem());
    trial("release-then-unmap",           false, false, false);
    cycle(8);
    trial("release-then-unmap, va freed", false, false, true);
    trial("unmap-then-release",           true,  false, false);
    trial("piecewise unmap",              false, true,  false);
    trial("piecewise unmap, va freed",    false, true,  true);
    return 0;
}
