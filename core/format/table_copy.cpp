/* table_copy.cpp -- see table_copy.h. */
#include "table_copy.h"

#include "../rad_internal.h"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <thread>
#include <vector>

namespace rad {

int table_copy_check(const uint8_t* src, int64_t n, const uint8_t* buf, std::string* why) {
    if (!why || !src || !buf || n <= 0) return RAD_E_INVAL;
    const int64_t probe = std::min<int64_t>(n, 4096);
    bool same = std::memcmp(buf, src, (size_t)probe) == 0 &&
                std::memcmp(buf + n - probe, src + n - probe, (size_t)probe) == 0;
    for (int64_t o = 0; same && o + 4096 <= n; o += std::max<int64_t>(n / 64, 4096))
        same = std::memcmp(buf + o, src + o, 4096) == 0;
    if (!same) {
        *why = "the copy does not match the container's mapping";
        return RAD_E_IO;
    }
    return RAD_OK;
}

int table_copy_fill(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                    uint8_t* buf, std::string* why) {
    if (!why || !src || !buf || n <= 0) return RAD_E_INVAL;
    const int64_t huge = (int64_t)2 << 20;

    /* One reader per hardware thread up to 16, each reading its own contiguous span in 64 MiB
     * pieces: a drive serves 16 queued reads at full rate, and spans keep each reader
     * sequential. */
    const int nt = (int)std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    const int64_t piece = (int64_t)64 << 20;
    const int64_t span = ((n + nt - 1) / nt + huge - 1) & ~(huge - 1);
    std::atomic<int> err{RAD_OK};
    std::vector<std::thread> rd;
    for (int i = 0; i < nt; ++i) {
        const int64_t lo = (int64_t)i * span, hi = std::min(n, lo + span);
        if (lo >= hi) break;
        rd.emplace_back([&, lo, hi] {
            for (int64_t o = lo; o < hi && err.load() == RAD_OK; o += piece) {
                const int64_t b = std::min(piece, hi - o);
                const int rc = file.read_at(buf + o, b, foff + o);
                if (rc < 0) { err.store(rc); return; }
                file.drop_range(foff + o, b);
            }
        });
    }
    for (std::thread& t : rd) t.join();

    if (err.load() != RAD_OK) {
        *why = fmt("reading the container failed (%s)", rad_strerror(err.load()));
        return RAD_E_IO;
    }
    return table_copy_check(src, n, buf, why);
}

int table_copy_make(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                    TableCopy* out, std::string* why) {
    if (!out || !why || !src || n <= 0) return RAD_E_INVAL;
    *out = TableCopy{};
    const size_t huge = (size_t)2 << 20;
    const size_t sz = ((size_t)n + huge - 1) & ~(huge - 1);
    void* m = ::mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                     -1, 0);
    if (m == MAP_FAILED) {
        *why = fmt("mmap of %s: %s", humanb((int64_t)sz).c_str(), std::strerror(errno));
        return RAD_E_NOMEM;
    }
    (void)::madvise(m, sz, MADV_HUGEPAGE);
    if (::mlock2(m, sz, MLOCK_ONFAULT) != 0) {
        *why = fmt("mlock2 of %s: %s", humanb((int64_t)sz).c_str(), std::strerror(errno));
        ::munmap(m, sz);
        return RAD_E_NOMEM;
    }
    const int rc = table_copy_fill(file, src, n, foff, static_cast<uint8_t*>(m), why);
    if (rc != RAD_OK) {
        ::munlock(m, sz);
        ::munmap(m, sz);
        return rc;
    }
    out->base = m;
    out->bytes = sz;
    return RAD_OK;
}

void table_copy_free(TableCopy* c) {
    if (!c || !c->base) return;
    ::munlock(c->base, c->bytes);
    ::munmap(c->base, c->bytes);
    *c = TableCopy{};
}

}  /* namespace rad */
