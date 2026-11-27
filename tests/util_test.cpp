/* util_test.cpp -- core/util/rad_util.cpp, which every other translation unit links against,
 * core/util/stallwatch.cpp, which reports where a phase that has gone quiet is waiting, and
 * core/util/rowpool.cpp, which spreads a loop's independent rows over threads kept for the
 * process.
 *
 * `Geometry` and `constraint_holds` decide which kernel runs. A geometry that hands a kernel
 * dangling parameter pointers, or a constraint that holds when it should not, does not fail: it
 * selects a different implementation that produces plausible output. The formatters are checked
 * because a size rounded the wrong way makes reports disagree with each other.
 *
 * No device, no model, no core: one translation unit and its header.
 */
#include "rad_test.h"

#include "rad_internal.h"
#include "util/rowpool.h"
#include "util/stallwatch.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>
#include <string>
#include <thread>
#include <vector>

using namespace rad;

/* ================================================================== small helpers */

TEST(align_up_rounds_up_and_leaves_an_exact_multiple_alone) {
    CHECK_EQ(align_up(0, 256), 0ll);
    CHECK_EQ(align_up(1, 256), 256ll);
    CHECK_EQ(align_up(255, 256), 256ll);
    CHECK_EQ(align_up(256, 256), 256ll);       /* already aligned: not bumped to the next unit */
    CHECK_EQ(align_up(257, 256), 512ll);
    CHECK_EQ(align_up(12345, 1), 12345ll);     /* an alignment of one is the identity */
    /* The sizes this is actually used on are pool extents, which are large. */
    CHECK_EQ(align_up(1073741825ll, 4096), 1073745920ll);
}

TEST(humanb_changes_unit_at_the_boundary_and_not_before) {
    CHECK_EQ(humanb(0), std::string("0 B"));
    CHECK_EQ(humanb(1), std::string("1 B"));
    /* 1023 is still bytes. Every size in every report is read against the others, so the unit
     * must not switch early. */
    CHECK_EQ(humanb(1023), std::string("1023 B"));
    CHECK_EQ(humanb(1024), std::string("1.00 KiB"));
    CHECK_EQ(humanb(1536), std::string("1.50 KiB"));
    CHECK_EQ(humanb(1048576), std::string("1.00 MiB"));
    CHECK_EQ(humanb(1073741824ll), std::string("1.00 GiB"));
    CHECK_EQ(humanb(1099511627776ll), std::string("1.00 TiB"));
    /* The ladder stops at TiB rather than inventing a unit. */
    CHECK_EQ(humanb(1125899906842624ll), std::string("1024.00 TiB"));
}

TEST(fmt_sizes_its_own_buffer) {
    CHECK_EQ(fmt("%d", 7), std::string("7"));
    CHECK_EQ(fmt("%s=%lld", "M", 64ll), std::string("M=64"));
    CHECK_EQ(fmt("%s", ""), std::string(""));
    /* Longer than any plausible fixed buffer. fmt measures with vsnprintf first, so there is no
     * cap; a regression to a stack buffer would truncate here, and fmt formats whole selection
     * tables. */
    const std::string big = fmt("%s", std::string(10000, 'x').c_str());
    REQUIRE_EQ(big.size(), (size_t)10000);
    CHECK_EQ(big.front(), 'x');
    CHECK_EQ(big.back(), 'x');
}

TEST(rad_mono_ns_moves_forward_and_is_not_the_wall_clock) {
    const uint64_t a = rad_mono_ns();
    CHECK(a > 0);
    uint64_t b = a;
    /* Spin rather than sleep: the property is only that it advances. */
    for (int i = 0; i < 1000000 && b == a; ++i) b = rad_mono_ns();
    CHECK(b >= a);
    /* Not a Unix timestamp: a monotonic clock counts from an arbitrary epoch (usually boot), so
     * comparing it against time() would mix two origins. */
    CHECK(a < 1000000000000000000ull);
}

/* split_ws also splits on commas, which RAD_C_IN depends on: a kernel writing
 * RAD_CIN("dtype", "bf16,fp8") and one writing "bf16 fp8" mean the same set. */
TEST(split_ws_takes_spaces_tabs_and_commas) {
    auto v = split_ws("a b c");
    REQUIRE_EQ(v.size(), 3u);
    CHECK_EQ(std::string(v[0]), std::string("a"));
    CHECK_EQ(std::string(v[2]), std::string("c"));

    v = split_ws("bf16,fp8");
    REQUIRE_EQ(v.size(), 2u);
    CHECK_EQ(std::string(v[1]), std::string("fp8"));

    v = split_ws("\ta ,, b\t,c  ");
    REQUIRE_EQ(v.size(), 3u);
    CHECK_EQ(std::string(v[0]), std::string("a"));
    CHECK_EQ(std::string(v[1]), std::string("b"));
    CHECK_EQ(std::string(v[2]), std::string("c"));

    /* An empty set has no tokens, not one empty token; otherwise a constraint naming no values
     * would hold for a geometry whose value is the empty string. */
    CHECK_EQ(split_ws("").size(), 0u);
    CHECK_EQ(split_ws("   ,, \t ").size(), 0u);
    CHECK_EQ(split_ws("one").size(), 1u);
}

TEST(the_log_level_is_readable_and_settable) {
    const Log saved = log_level();
    log_set_level(Log::Trace);
    CHECK((int)log_level() == (int)Log::Trace);
    log_set_level(Log::Error);
    CHECK((int)log_level() == (int)Log::Error);
    log_set_level(saved);
    CHECK((int)log_level() == (int)saved);
}

/* ================================================================== Geometry */

TEST(geometry_stores_by_kind_and_a_reader_of_the_wrong_kind_misses) {
    Geometry g;
    g.set_i("M", 64);
    g.set_s("dtype", "w4a8");
    g.set_f("eps", 1e-6);

    long long i = 0;
    CHECK(g.get_i("M", &i));
    CHECK_EQ(i, 64ll);
    CHECK_EQ(std::string(g.get_s("dtype")), std::string("w4a8"));
    double d = 0;
    CHECK(g.get_f("eps", &d));
    CHECK_NEAR(d, 1e-6, 1e-12);

    /* The kind is part of the match: get_i on a string or float slot misses, so a constraint on a
     * float key fails instead of comparing against a truncation. */
    CHECK(!g.get_i("dtype", &i));
    CHECK(!g.get_i("eps", &i));
    CHECK(!g.get_f("M", &d));
    CHECK(g.get_s("M") == nullptr);
    CHECK(g.get_s("missing") == nullptr);
    CHECK(!g.get_i("missing", &i));

    /* has() is kind-agnostic: it answers whether the key is carried at all. */
    CHECK(g.has("M")); CHECK(g.has("dtype")); CHECK(g.has("eps"));
    CHECK(!g.has("N"));

    /* A null out pointer is a membership test. */
    CHECK(g.get_i("M", nullptr));
    CHECK(!g.get_i("dtype", nullptr));
}

TEST(setting_a_key_again_replaces_its_value_and_its_kind) {
    Geometry g;
    g.set_i("x", 1);
    g.set_i("x", 2);
    long long i = 0;
    CHECK(g.get_i("x", &i));
    CHECK_EQ(i, 2ll);
    CHECK_EQ(g.n_params(), 1);            /* replaced, not appended */

    g.set_s("x", "str");
    CHECK(!g.get_i("x", &i));             /* it is a string now and nothing else */
    CHECK_EQ(std::string(g.get_s("x")), std::string("str"));
    CHECK_EQ(g.n_params(), 1);
}

TEST(the_param_view_is_rebuilt_after_a_mutation) {
    Geometry g;
    g.set_i("M", 8);
    CHECK_EQ(g.n_params(), 1);
    CHECK_EQ(g.params()[0].ival, 8ll);

    /* Every write must rebuild the view; a stale view hands a kernel the geometry it was resolved
     * against rather than the one it runs on. */
    g.set_i("M", 9);
    CHECK_EQ(g.params()[0].ival, 9ll);
    g.set_i("N", 4096);
    CHECK_EQ(g.n_params(), 2);

    /* A string param carries sval and an integer one a null sval, so a kernel can tell "not set"
     * from "the empty string". */
    g.set_s("dtype", "bf16");
    const RadParam* p = g.params();
    int seen = 0;
    for (int k = 0; k < g.n_params(); ++k) {
        if (std::string(p[k].key) == "dtype") {
            CHECK(p[k].sval != nullptr);
            CHECK_EQ(std::string(p[k].sval), std::string("bf16"));
            ++seen;
        } else {
            CHECK(p[k].sval == nullptr);
        }
    }
    CHECK_EQ(seen, 1);
}

/* A copy must own its strings. The view holds char* into the object's own storage, so a copy
 * that took the view along would hand a kernel pointers into the source, which often dies first
 * (every Resolved holds a Geometry and every Band holds two). */
TEST(a_copied_geometry_points_into_itself_and_not_at_the_source) {
    Geometry a;
    a.set_s("dtype", "w4a8");
    a.set_i("M", 64);
    const char* a_key = a.params()[0].key;

    Geometry b = a;
    CHECK_EQ(b.n_params(), 2);
    CHECK_EQ(std::string(b.params()[0].key), std::string("dtype"));
    CHECK(b.params()[0].key != a_key);             /* a different object's storage */
    CHECK(b.params()[0].sval != a.params()[0].sval);
    CHECK_EQ(std::string(b.params()[0].sval), std::string("w4a8"));
    CHECK(b.params()[0].sval == b.get_s("dtype"));  /* and into the copy's own string */

    /* And the two are independent afterwards, in both directions. */
    a.set_i("M", 1);
    long long i = 0;
    CHECK(b.get_i("M", &i));
    CHECK_EQ(i, 64ll);
    b.set_i("M", 2);
    CHECK(a.get_i("M", &i));
    CHECK_EQ(i, 1ll);

    /* Assignment is the same rule. */
    Geometry c;
    c.set_i("junk", 7);
    c = a;
    CHECK(!c.has("junk"));
    CHECK_EQ(c.n_params(), 2);
    CHECK(c.params()[0].key != a.params()[0].key);
}

TEST(a_moved_geometry_does_not_leave_the_view_pointing_at_a_stolen_buffer) {
    Geometry a;
    a.set_s("dtype", "fp8");
    a.set_i("K", 5120);

    Geometry b = std::move(a);
    CHECK_EQ(b.n_params(), 2);
    CHECK_EQ(std::string(b.params()[0].sval), std::string("fp8"));
    CHECK(b.params()[0].sval == b.get_s("dtype"));  /* the view went with the strings it names */
    CHECK_EQ(std::string(b.params()[0].key), std::string("dtype"));
    /* The moved-from object is empty rather than holding a view of a buffer it no longer owns. */
    CHECK_EQ(a.n_params(), 0);

    Geometry c;
    c.set_i("x", 1);
    c = std::move(b);
    CHECK_EQ(c.n_params(), 2);
    CHECK(!c.has("x"));
    CHECK_EQ(b.n_params(), 0);
}

/* Reading a geometry writes nothing, so one nobody is writing can be read from any number of
 * threads. A rank's sizing declares run side by side and all read the program's op geometries
 * (Engine::probe_arena_levels); a view built lazily on first read would be a concurrent write.
 * Each round reads a fresh, moved geometry (as every OpInfo's is) from eight threads released
 * together, and every reader must see the whole view and the same one. */
TEST(a_geometry_read_from_many_threads_at_once_is_not_written) {
    /* Eight threads kept for every round. Each round main publishes the geometry and all nine meet
     * at a barrier, whose completion releases the eight readers together; they meet again when
     * done. A barrier rather than a spin, so a loaded machine's scheduler is not fought. */
    constexpr int kThreads = 8, kRounds = 400;
    std::barrier sync(kThreads + 1);
    const Geometry* g = nullptr;
    int round = 0;
    std::vector<const RadParam*> seen(kThreads, nullptr);
    std::vector<int> ok(kThreads, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < kThreads; ++t)
        th.emplace_back([&, t] {
            for (int r = 0; r < kRounds; ++r) {
                sync.arrive_and_wait();
                const RadParam* p = g->params();
                const int n = g->n_params();
                bool good = n == 13 && std::string(p[0].key) == "dtype" && p[0].sval &&
                            std::string(p[0].sval).rfind("fp8_e4m3", 0) == 0;
                for (int k = 0; good && k < 12; ++k)
                    good = p[k + 1].kind == RAD_P_INT && p[k + 1].ival == k * 1000 + r &&
                           std::string(p[k + 1].key) == fmt("key_%d", k);
                seen[t] = p;
                ok[t] = good;
                sync.arrive_and_wait();
            }
        });
    int bad = 0;
    for (round = 0; round < kRounds; ++round) {
        Geometry src;
        src.set_s("dtype", "fp8_e4m3_with_a_name_past_the_short_string_buffer");
        for (int k = 0; k < 12; ++k) src.set_i(fmt("key_%d", k), k * 1000 + round);
        const Geometry moved = std::move(src);
        g = &moved;
        sync.arrive_and_wait();
        sync.arrive_and_wait();
        for (int t = 0; t < kThreads; ++t)
            if (!ok[t] || seen[t] != seen[0]) ++bad;
    }
    for (auto& x : th) x.join();
    CHECK_EQ(bad, 0);
}

TEST(geometry_str_prints_every_kind_in_declaration_order) {
    Geometry g;
    g.set_i("M", 64);
    g.set_i("N", 8192);
    g.set_s("dtype", "w4a8");
    CHECK_EQ(g.str(), std::string("M=64 N=8192 dtype=w4a8"));

    Geometry f;
    f.set_f("eps", 1e-6);
    CHECK_EQ(f.str(), std::string("eps=1e-06"));

    CHECK_EQ(Geometry{}.str(), std::string(""));
}

/* ================================================================== constraint matching */

static RadConstraint c_eq(const char* k, long long v)  { return RadConstraint RAD_CEQ(k, v); }
static RadConstraint c_le(const char* k, long long v)  { return RadConstraint RAD_CLE(k, v); }
static RadConstraint c_ge(const char* k, long long v)  { return RadConstraint RAD_CGE(k, v); }
static RadConstraint c_div(const char* k, long long v) { return RadConstraint RAD_CDIV(k, v); }
static RadConstraint c_in(const char* k, const char* s) { return RadConstraint RAD_CIN(k, s); }

TEST(the_four_integer_constraints_compare_the_way_they_read) {
    Geometry g;
    g.set_i("M", 64);

    CHECK(constraint_holds(c_eq("M", 64), g));
    CHECK(!constraint_holds(c_eq("M", 63), g));

    CHECK(constraint_holds(c_le("M", 64), g));      /* inclusive at the bound */
    CHECK(constraint_holds(c_le("M", 65), g));
    CHECK(!constraint_holds(c_le("M", 63), g));

    CHECK(constraint_holds(c_ge("M", 64), g));      /* inclusive at the bound */
    CHECK(constraint_holds(c_ge("M", 1), g));
    CHECK(!constraint_holds(c_ge("M", 65), g));

    CHECK(constraint_holds(c_div("M", 16), g));
    CHECK(constraint_holds(c_div("M", 1), g));
    CHECK(!constraint_holds(c_div("M", 5), g));
}

/* A divisor of zero is a kernel-set typo, not "divides everything". Answering true would route
 * every geometry to that row. */
TEST(a_zero_divisor_holds_for_nothing_rather_than_faulting) {
    Geometry g;
    g.set_i("M", 64);
    CHECK(!constraint_holds(c_div("M", 0), g));
    g.set_i("Z", 0);
    CHECK(!constraint_holds(c_div("Z", 0), g));
    CHECK(constraint_holds(c_div("Z", 16), g));     /* zero is divisible by anything non-zero */
}

/* A constraint on a key the geometry does not carry does not hold, so a row that constrains an
 * optional parameter serves only callers that state it. */
TEST(an_absent_key_fails_every_constraint) {
    Geometry g;
    g.set_i("M", 64);
    CHECK(!constraint_holds(c_eq("hops", 2), g));
    CHECK(!constraint_holds(c_le("hops", 99), g));
    CHECK(!constraint_holds(c_ge("hops", 0), g));
    CHECK(!constraint_holds(c_div("hops", 1), g));
    CHECK(!constraint_holds(c_in("hops", "1 2"), g));
}

/* A float parameter never satisfies a constraint, by design: a tolerance is not a predicate, and
 * a kernel that discriminates on eps has a bug. */
TEST(a_float_parameter_satisfies_no_constraint) {
    Geometry g;
    g.set_f("eps", 1e-6);
    g.set_f("scale", 1.0);
    CHECK(!constraint_holds(c_eq("scale", 1), g));
    CHECK(!constraint_holds(c_le("scale", 2), g));
    CHECK(!constraint_holds(c_ge("eps", 0), g));
    CHECK(!constraint_holds(c_div("scale", 1), g));
    CHECK(!constraint_holds(c_in("scale", "1"), g));
}

TEST(a_set_constraint_matches_a_string_and_an_integers_decimal_form) {
    Geometry g;
    g.set_s("dtype", "fp8");
    CHECK(constraint_holds(c_in("dtype", "bf16 fp8"), g));
    CHECK(constraint_holds(c_in("dtype", "bf16,fp8"), g));
    CHECK(!constraint_holds(c_in("dtype", "bf16 w4a8"), g));
    /* A prefix is not a member: matching "fp8" against "fp8e5m2" would route a kernel compiled
     * for one encoding to another. */
    CHECK(!constraint_holds(c_in("dtype", "fp8e5m2"), g));

    /* The integer form, which is what makes RAD_CIN("causal", "0 1") work. */
    Geometry i;
    i.set_i("causal", 1);
    CHECK(constraint_holds(c_in("causal", "0 1"), i));
    i.set_i("causal", 2);
    CHECK(!constraint_holds(c_in("causal", "0 1"), i));
    i.set_i("neg", -1);
    CHECK(constraint_holds(c_in("neg", "-1 0"), i));

    /* An empty or absent set matches nothing, including the empty string. */
    Geometry e;
    e.set_s("k", "");
    CHECK(!constraint_holds(c_in("k", ""), e));
    RadConstraint nullset = c_in("k", nullptr);
    CHECK(!constraint_holds(nullset, e));
}

TEST(an_unknown_constraint_operator_holds_for_nothing) {
    Geometry g;
    g.set_i("M", 64);
    RadConstraint c{};
    c.key = "M";
    c.op = 99;
    c.ival = 64;
    CHECK(!constraint_holds(c, g));
}

/* The miss column of the selection table (spec §16) says which kernel was rejected and why in
 * terms an operator can act on: the key, the bound and the value, and whether the value was
 * wrong or not supplied at all. */
TEST(constraint_why_names_the_key_the_bound_and_what_was_there) {
    Geometry g;
    g.set_i("M", 64);

    const std::string le = constraint_why(c_le("M", 32), g);
    CHECK(le.find("M") != std::string::npos);
    CHECK(le.find("<=") != std::string::npos);
    CHECK(le.find("32") != std::string::npos);
    CHECK(le.find("64") != std::string::npos);

    const std::string dv = constraint_why(c_div("M", 5), g);
    CHECK(dv.find("%") != std::string::npos);
    CHECK(dv.find("5") != std::string::npos);

    /* Absent reads differently from wrong, because the repair is different. */
    const std::string ab = constraint_why(c_ge("hops", 2), g);
    CHECK(ab.find("did not name") != std::string::npos);
    CHECK(ab.find("hops") != std::string::npos);

    Geometry s;
    s.set_s("dtype", "fp8");
    const std::string in = constraint_why(c_in("dtype", "bf16"), s);
    CHECK(in.find("bf16") != std::string::npos);
    CHECK(in.find("fp8") != std::string::npos);

    const std::string miss = constraint_why(c_in("dtype", "bf16"), g);
    CHECK(miss.find("<absent>") != std::string::npos);
}

/* ================================================================== StallWatch */

TEST(a_syscall_line_names_the_call_and_a_kfd_ioctl_by_its_name) {
    CHECK_EQ(stall_decode_syscall("running"), std::string("running"));
    CHECK_EQ(stall_decode_syscall("-1 0x7ffd1000 0x7f0012345678"), std::string("not in a system call"));
    CHECK(stall_decode_syscall("202 0x1 0x80 0x0 0x0 0x0 0x0 0x7ffd1000 0x0").rfind("futex", 0) == 0);
    /* A file that will not open -- a FIFO with no writer, a hung network mount -- is named too. */
    CHECK(stall_decode_syscall("257 0xffffff9c 0x5600 0x0 0x0 0x0 0x0 0x7ffd1000 0x0").rfind("openat", 0) == 0);
    /* An ioctl names the file its descriptor is open on and, for the KFD, the request by name:
     * _IOWR('K', 0x16, 40 bytes) is ALLOC_MEMORY_OF_GPU. */
    const int fd = open("/dev/null", O_RDONLY);
    CHECK(fd >= 0);
    const unsigned long req = (3ul << 30) | (40ul << 16) | ((unsigned long)'K' << 8) | 0x16;
    const std::string k = stall_decode_syscall(fmt("16 0x%x 0x%lx 0x0 0x0 0x0 0x0 0x0 0x0", fd, req));
    CHECK(k.find("ioctl(/dev/null, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU)") != std::string::npos);
    /* Another driver's request is spelt out rather than guessed at. */
    const unsigned long drm = (3ul << 30) | (32ul << 16) | ((unsigned long)'d' << 8) | 0x40;
    const std::string d = stall_decode_syscall(fmt("16 0x%x 0x%lx 0x0 0x0 0x0 0x0 0x0 0x0", fd, drm));
    CHECK(d.find("type 'd' number 0x40") != std::string::npos);
    close(fd);
}

/* Offsets are relative to the module's load base, which is what addr2line takes; that differs
 * from the file offset where the linker placed a segment away from its file position. */
TEST(an_address_is_named_by_its_module_and_its_offset_from_the_load_base) {
    const uintptr_t a = (uintptr_t)&stderr_of;
    Dl_info di{};
    CHECK(dladdr((void*)a, &di) != 0);
    CHECK_EQ(stall_module_offset(a), fmt("util_test+0x%lx", (unsigned long)(a - (uintptr_t)di.dli_fbase)));
}

/* A thread that waits on a lock with nothing logged is reported once the quiet has passed: what
 * it is in, and its stack through this binary's frames. The thread waits on a condition variable
 * until a helper sees the report's line logged, so it is in the wait whenever the watch looks and
 * the case ends as soon as the report is out. */
TEST(a_quiet_phase_names_the_wait_and_the_stack_of_the_thread_in_it) {
    const std::string out = stderr_of([] {
        std::mutex mu;
        std::condition_variable cv;
        bool reported = false;
        const uint64_t before = log_last_line_ns();
        std::thread watcher([&] {
            const uint64_t give_up = rad_mono_ns() + 10'000'000'000ull;
            while (log_last_line_ns() == before && rad_mono_ns() < give_up)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::lock_guard<std::mutex> g(mu);
            reported = true;
            cv.notify_all();
        });
        {
            std::unique_lock<std::mutex> lk(mu);
            StallWatch w("probe", 40);
            cv.wait(lk, [&] { return reported; });
        }
        watcher.join();
    });
    CHECK(out.find("probe: nothing has been logged for") != std::string::npos);
    CHECK(out.find("futex") != std::string::npos);
    CHECK(out.find("stack: ") != std::string::npos);
    CHECK(out.find("util_test+0x") != std::string::npos);
}

/* A phase that keeps logging is not quiet, however long it runs: logging every 5 ms for twice the
 * 150 ms quiet interval (a 30x margin on each gap) is never reported. */
TEST(a_phase_that_logs_is_never_reported) {
    const std::string out = stderr_of([] {
        StallWatch w("probe", 150);
        for (int i = 0; i < 60; ++i) {
            RAD_INFO("probe: step %d", i);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    CHECK(out.find("nothing has been logged") == std::string::npos);
    CHECK(out.find("probe: step 59") != std::string::npos);
}

/* Progress marked without a log line keeps a phase from being reported the same way. */
TEST(a_phase_that_marks_progress_is_never_reported) {
    const std::string out = stderr_of([] {
        StallWatch w("probe", 150);
        for (int i = 0; i < 60; ++i) {
            w.progress();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    CHECK(out.find("nothing has been logged") == std::string::npos);
}

/* ================================================================== the row pool */

/* Each row runs exactly once, whatever the width, and the call returns only after all of them. */
TEST(the_row_pool_runs_every_row_once) {
    for (int width : {1, 2, 8, 64}) {
        for (int64_t rows : {0, 1, 2, 7, 1000}) {
            std::vector<std::atomic<int>> hits((size_t)rows);
            row_pool_run(rows, width, [&](int64_t i) { hits[(size_t)i].fetch_add(1); });
            int bad = 0;
            for (auto& h : hits) bad += h.load() != 1;
            CHECK_EQ(bad, 0);
        }
    }
}

/* Under min_rows, or at width 1, the loop runs on the calling thread in order. */
TEST(a_short_or_narrow_loop_runs_inline_in_order) {
    const std::thread::id self = std::this_thread::get_id();
    std::vector<int64_t> order;
    bool elsewhere = false;
    row_pool_run(31, 8, [&](int64_t i) {
        order.push_back(i);
        elsewhere = elsewhere || std::this_thread::get_id() != self;
    }, 32);
    CHECK(!elsewhere);
    REQUIRE_EQ((int64_t)order.size(), (int64_t)31);
    bool sorted = true;
    for (size_t i = 0; i < order.size(); ++i) sorted = sorted && order[i] == (int64_t)i;
    CHECK(sorted);
    order.clear();
    row_pool_run(100, 1, [&](int64_t i) { order.push_back(i); });
    CHECK_EQ((int64_t)order.size(), (int64_t)100);
}

/* Callers on several threads at once each get all their rows, with workers shared between them,
 * and a slow row in one loop does not hold up another caller's. */
TEST(concurrent_callers_share_the_row_pool) {
    if (row_pool_workers() == 0) SKIP("the row pool has no worker threads on this machine");
    constexpr int kCallers = 6;
    constexpr int64_t kRows = 4000;
    std::vector<std::vector<std::atomic<int>>> hits(kCallers);
    for (auto& h : hits) h = std::vector<std::atomic<int>>((size_t)kRows);
    std::vector<std::thread> ts;
    for (int c = 0; c < kCallers; ++c)
        ts.emplace_back([&, c] {
            for (int rep = 0; rep < 20; ++rep)
                row_pool_run(kRows / 20, 8, [&](int64_t i) {
                    hits[(size_t)c][(size_t)(rep * (kRows / 20) + i)].fetch_add(1);
                });
        });
    for (auto& t : ts) t.join();
    int bad = 0;
    for (auto& h : hits)
        for (auto& x : h) bad += x.load() != 1;
    CHECK_EQ(bad, 0);

    /* Rows spread over more than one thread when workers exist. */
    std::mutex mu;
    std::vector<std::thread::id> seen;
    row_pool_run(256, 8, [&](int64_t) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        std::lock_guard<std::mutex> lk(mu);
        if (std::find(seen.begin(), seen.end(), std::this_thread::get_id()) == seen.end())
            seen.push_back(std::this_thread::get_id());
    });
    CHECK(seen.size() > 1);
}

RAD_TEST_MAIN()
