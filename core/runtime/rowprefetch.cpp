/* rowprefetch.cpp -- see rowprefetch.h. */
#include "runtime/rowprefetch.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>

namespace rad {

bool row_gather_op(const std::string& op) {
    return op == "embed_lookup" || op == "embed_lookup_q" || op == "embed_lookup_planes" ||
           op == "embed_lookup_ggml";
}

int64_t row_prefetch_page() {
    static const int64_t pg = [] {
        const long v = ::sysconf(_SC_PAGESIZE);
        return v > 0 ? (int64_t)v : (int64_t)4096;
    }();
    return pg;
}

bool row_id_at(const RadTensor& ids, int64_t i, int64_t* out) {
    /* Row-major over the shape, through the strides (elements). */
    int64_t off = i;
    if (ids.rank >= 1) {
        off = 0;
        int64_t r = i;
        for (int d = (int)ids.rank - 1; d >= 0; --d) {
            const int64_t s = ids.shape[d] > 0 ? ids.shape[d] : 1;
            off += (r % s) * ids.stride[d];
            r /= s;
        }
    }
    switch (ids.dtype) {
        case RAD_I32: *out = (int64_t)((const int32_t*)ids.data)[off]; return true;
        case RAD_U32: *out = (int64_t)((const uint32_t*)ids.data)[off]; return true;
        case RAD_I64: *out = ((const int64_t*)ids.data)[off]; return true;
        case RAD_I16: *out = (int64_t)((const int16_t*)ids.data)[off]; return true;
        default:      *out = 0; return false;
    }
}

void row_runs_merge(std::vector<PageRun>* runs) {
    std::vector<PageRun>& v = *runs;
    if (v.size() < 2) return;
    std::sort(v.begin(), v.end());
    size_t w = 0;
    for (size_t r = 1; r < v.size(); ++r) {
        if (v[r].first <= v[w].second) {
            if (v[r].second > v[w].second) v[w].second = v[r].second;
        } else {
            v[++w] = v[r];
        }
    }
    v.resize(w + 1);
}

int64_t row_prefetch(const RowPrefetch& rp, const RadTensor& ids, int64_t m, const RadTensor* t,
                     RowPrefetchScratch* s) {
    return row_page_runs(rp, ids, m, t, row_prefetch_page(), s, [](const PageRun& r) {
        /* The table is read-only and mapped; a failure (a range the kernel will not read ahead)
         * leaves the read to the fault, as without the hint. */
        (void)::madvise((void*)r.first, (size_t)(r.second - r.first), MADV_WILLNEED);
    });
}

/* ------------------------------------------------------------------ beside the kernel */

bool row_pages_resident(const RowPrefetch& rp, const RadTensor& ids, int64_t m, const RadTensor* t) {
    if (rp.n <= 0 || m <= 0 || !ids.data) return false;
    const int64_t pg = row_prefetch_page();
    const uintptr_t mask = ~(uintptr_t)(pg - 1);
    const int64_t S = m < kRowAheadSample ? m : kRowAheadSample;
    for (int64_t j = 0; j < S; ++j) {
        int64_t id;
        if (!row_id_at(ids, (2 * j + 1) * m / (2 * S), &id)) return false;
        id -= rp.voff;
        if (id < 0 || id >= rp.rows) continue;
        for (int k = 0; k < rp.n; ++k) {
            const uint8_t* base = (const uint8_t*)t[rp.opd[k]].data;
            if (!base || rp.row_bytes[k] <= 0) continue;
            const uintptr_t a = (uintptr_t)(base + id * rp.row_bytes[k]);
            /* The row's first and last page. */
            for (const uintptr_t p : { a & mask, (a + (uintptr_t)rp.row_bytes[k] - 1) & mask }) {
                unsigned char v = 0;
                if (::mincore((void*)p, (size_t)pg, &v) != 0 || !(v & 1)) return false;
            }
        }
    }
    return true;
}

RowPrefetchWorker::~RowPrefetchWorker() {
    if (!th_.joinable()) return;
    stop_.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_.notify_all();
    th_.join();
}

void RowPrefetchWorker::start(const RowPrefetch& rp, const RadTensor& ids, int64_t m,
                              const RadTensor* t) {
    /* finish() ran for the previous job, so the helper is idle: the job is ours to write. */
    rp_ = rp;
    rp_.n = 0;
    for (int k = 0; k < rp.n && k < kRowTablesMax; ++k) {
        t_[k] = t[rp.opd[k]];
        rp_.opd[k] = (int8_t)k;
        rp_.row_bytes[k] = rp.row_bytes[k];
        ++rp_.n;
    }
    ids_ = ids;
    m_ = m;
    pages_ = 0;
    stop_.store(false, std::memory_order_relaxed);
    busy_.store(true, std::memory_order_release);
    if (!th_.joinable()) {
        s_.reserve(4096);
        th_ = std::thread([this] { loop(); });
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        have_ = true;
    }
    cv_.notify_one();
}

void RowPrefetchWorker::loop() {
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return have_ || quit_; });
            if (quit_) return;
            have_ = false;
        }
        const int64_t pages = row_page_runs(rp_, ids_, m_, t_, row_prefetch_page(), &s_,
                                            [&](const PageRun& r) {
            if (stop_.load(std::memory_order_relaxed)) return false;
            (void)::madvise((void*)r.first, (size_t)(r.second - r.first), MADV_WILLNEED);
            return true;
        });
        pages_ = pages > 0 ? pages : 0;
        /* Release: pages_ is published, and the ids are no longer read. */
        busy_.store(false, std::memory_order_release);
    }
}

int64_t RowPrefetchWorker::finish() {
    stop_.store(true, std::memory_order_relaxed);
    {
        /* Not picked up yet (the kernel was quicker than the wake-up): take it back. */
        std::lock_guard<std::mutex> lk(mu_);
        if (have_) {
            have_ = false;
            busy_.store(false, std::memory_order_release);
            return 0;
        }
    }
    /* The helper checks stop_ before every hint, so this waits at most one madvise plus the walk
     * to the next flush of the scratch (a few thousand ids of page arithmetic). */
    while (busy_.load(std::memory_order_acquire)) std::this_thread::yield();
    return pages_;
}

}  /* namespace rad */
