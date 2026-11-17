#include "rad_test.h"
#include "../tools/convert_work_queue.h"

#include <atomic>
#include <vector>

using namespace rad;

TEST(split_queues_execute_every_job_exactly_once) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(8);
#endif
    constexpr size_t N = 2000;
    std::vector<size_t> device, host;
    for (size_t i = 0; i < N; ++i) (i % 2 ? device : host).push_back(i);
    std::vector<std::atomic<int>> seen(N);
    for (auto& n : seen) n.store(0);
    const int rc = convert_run_queues(device, host, 3, [&](size_t i) {
        if (i >= N) return -10;
        seen[i].fetch_add(1, std::memory_order_relaxed);
        return 0;
    });
    CHECK_EQ(rc, 0);
    for (const auto& n : seen) CHECK_EQ(n.load(), 1);
}

TEST(split_queues_steal_from_either_empty_side) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(6);
#endif
    const std::vector<size_t> many = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const std::vector<size_t> empty;
    for (bool only_device : { false, true }) {
        std::atomic<int> seen{0};
        const int rc = convert_run_queues(only_device ? many : empty, only_device ? empty : many,
                                          2, [&](size_t) { ++seen; return 0; });
        CHECK_EQ(rc, 0);
        CHECK_EQ(seen.load(), (int)many.size());
    }
    CHECK_EQ(convert_run_queues(empty, empty, 2, [](size_t) { return 0; }), 0);
}

TEST(split_queues_propagate_worker_failure) {
    const std::vector<size_t> device = { 42 }, host;
    CHECK_EQ(convert_run_queues(device, host, 1, [](size_t) { return -17; }), -17);
}

RAD_TEST_MAIN()
