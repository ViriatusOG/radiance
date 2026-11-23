#pragma once

/* Two dynamic converter job queues. GPU workers prefer device-backed jobs, host workers prefer
 * everything else, and either can steal after its preferred queue is empty. This reserves capacity
 * to feed the cards while CPU-only GPTQ, checkpoint copies and error measurement overlap it. */
#include <atomic>
#include <cstddef>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rad {

template <class F>
int convert_run_queues(const std::vector<size_t>& gpu, const std::vector<size_t>& host,
                       int gpu_workers, F run) {
    constexpr size_t none = (size_t)-1;
    std::atomic<size_t> ng{0}, nh{0};
    std::atomic<int> failed{0};
    auto take = [&](bool device) -> size_t {
        const auto& q = device ? gpu : host;
        const size_t n = device ? ng.fetch_add(1, std::memory_order_relaxed)
                                : nh.fetch_add(1, std::memory_order_relaxed);
        return n < q.size() ? q[n] : none;
    };
#pragma omp parallel
    {
#ifdef _OPENMP
        const int team = omp_get_num_threads();
        const int reserved = team > 1 ? (gpu_workers < team ? gpu_workers : team - 1) : 1;
        const bool device_first = omp_get_thread_num() < reserved;
#else
        const bool device_first = true;
        (void)gpu_workers;
#endif
        while (failed.load(std::memory_order_relaxed) == 0) {
            size_t job = take(device_first);
            if (job == none) job = take(!device_first);
            if (job == none) break;
            const int rc = run(job);
            if (rc < 0) {
                int ok = 0;
                failed.compare_exchange_strong(ok, rc);
            }
        }
    }
    return failed.load();
}

}  /* namespace rad */
