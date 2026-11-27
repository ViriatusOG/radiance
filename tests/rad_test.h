/* A minimal test harness: no framework, no dependency.
 *   TEST(name) { CHECK(cond); CHECK_EQ(a, b); }
 *   RAD_TEST_MAIN()
 *
 * Running a binary: <bin> [pattern...] [-pattern...] [--list] [--shard i/n]
 *   pattern       run only the cases whose name contains one of the patterns
 *   -pattern      leave out the cases whose name contains it
 *   --list        print the selected cases' names, in registration order, and run nothing
 *   --shard i/n   of the selected cases, run those whose index modulo n is i (0 <= i < n)
 * The environment:
 *   RAD_TEST_FILTER      more patterns in the same form, separated by spaces or commas
 *   RAD_TEST_LEVEL       2 runs the exhaustive sizes (scaled()); unset means 1
 *   RAD_TEST_STRICT      1: a missing in-tree artifact fails instead of skipping
 *   RAD_TEST_TMPDIR      where tmpdir() is made, before $TMPDIR and /tmp
 *   RAD_TEST_CPU_BUDGET  seconds of process CPU time the binary may spend; over it, it fails
 * tests/CMakeLists.txt sets the last three for every ctest entry.
 *   RAD_TEST_CPU_BUDGET_SCALE  a factor on that budget, for a host slower per thread than the
 *                        one the budgets were set on; unset or 0 is 1
 *
 * Exit status: 1 when a check failed (an exception escaping a case is a failed check), 77 when
 * nothing failed and no check ran, 2 for a command line it does not understand, else 0. 77 is
 * ctest's SKIP_RETURN_CODE: a binary that checked nothing has not passed. SKIP_REGULAR_EXPRESSION
 * is not used because CMake evaluates it before the return code, so a binary that printed a skip
 * line and then failed would be reported Skipped. A binary where some cases ran passes or fails on
 * those.
 *
 * Cases run one at a time on the main thread, in registration order (declaration order within a
 * file). CHECKs may run on other threads. A case that takes 0.1 s or more prints its time.
 */
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <cerrno>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace radtest {

struct Case { const char* name; std::function<void()> fn; };
inline std::vector<Case>& cases() { static std::vector<Case> v; return v; }
inline std::atomic<int>& failures() { static std::atomic<int> n{0}; return n; }
/* Assertions that ran; zero means every selected case skipped itself. */
inline std::atomic<int>& checks() { static std::atomic<int> n{0}; return n; }
inline const char*& current() { static const char* s = ""; return s; }
/* Set when the running case prints a skip line. */
inline std::atomic<bool>& case_skipped() { static std::atomic<bool> b{false}; return b; }
/* The binary's name, for the temporary directory. */
inline std::string& program() { static std::string s = "test"; return s; }

struct Reg { Reg(const char* n, std::function<void()> f) { cases().push_back({n, f}); } };

inline void fail(const char* file, int line, const std::string& what) {
    failures().fetch_add(1);
    fprintf(stderr, "  FAIL %s\n    %s:%d: %s\n", current(), file, line, what.c_str());
}

template <class A, class B>
inline std::string show(const A& a, const B& b) {
    char buf[512];
    if constexpr (std::is_convertible_v<A, std::string> && std::is_convertible_v<B, std::string>)
        snprintf(buf, sizeof buf, "\"%s\" vs \"%s\"",
                 std::string(a).c_str(), std::string(b).c_str());
    else if constexpr (std::is_floating_point_v<A> || std::is_floating_point_v<B>)
        snprintf(buf, sizeof buf, "%g vs %g", (double)a, (double)b);
    else
        snprintf(buf, sizeof buf, "%lld vs %lld", (long long)a, (long long)b);
    return buf;
}

/* The one spelling of a skip line, "  SKIP <case>: <reason>". SKIP() also leaves the case. */
[[gnu::format(printf, 1, 2)]] inline void skip(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    case_skipped() = true;
    fprintf(stderr, "  SKIP %s: %s\n", current(), buf);
}

inline bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

/* An in-tree artifact (a plugin this build makes, a checked-in fixture) is absent or does not load.
 * Under RAD_TEST_STRICT=1, which ctest sets, that is a broken build or a wrong RADIANCE_HOME and
 * fails the case; run by hand, the case skips. Model files and machine capabilities (a card,
 * io_uring, O_DIRECT, a media decoder) are not artifacts: those cases use SKIP. */
inline void artifact_missing(const char* file, int line, const std::string& what) {
    if (env_flag("RAD_TEST_STRICT"))
        fail(file, line, "missing in-tree artifact (RAD_TEST_STRICT is set): " + what);
    else
        skip("missing in-tree artifact: %s", what.c_str());
}

/* RAD_TEST_LEVEL, 1 when unset. */
inline int level() {
    const char* v = std::getenv("RAD_TEST_LEVEL");
    return v && *v ? std::atoi(v) : 1;
}

/* A loop's size: `normal` by default, `exhaustive` at RAD_TEST_LEVEL 2 and above. */
template <class T> inline T scaled(T normal, T exhaustive) {
    return level() >= 2 ? exhaustive : normal;
}

/* Set to the directory once tmpdir() has made it, for run_all to remove. */
inline std::string& made_tmpdir() { static std::string s; return s; }

/* A directory of this process's own, made on first use under $RAD_TEST_TMPDIR, else $TMPDIR,
 * else /tmp. Removed when the binary passes; kept, with its path printed, when it fails. Throws
 * when it cannot be made, which fails the case. */
inline const std::string& tmpdir() {
    static const std::string dir = [] {
        const char* base = std::getenv("RAD_TEST_TMPDIR");
        if (!base || !*base) base = std::getenv("TMPDIR");
        if (!base || !*base) base = "/tmp";
        ::mkdir(base, 0777);   /* RAD_TEST_TMPDIR may name a directory nothing has made yet */
        std::string path = std::string(base) + "/radtest-" + program() + "-XXXXXX";
        if (!::mkdtemp(path.data()))
            throw std::runtime_error("cannot make a temporary directory under " +
                                     std::string(base) + ": " + std::strerror(errno));
        made_tmpdir() = path;
        return path;
    }();
    return dir;
}

/* A path inside tmpdir(). */
inline std::string tmp_path(const std::string& leaf) { return tmpdir() + "/" + leaf; }

/* A new empty directory "<stem>.XXXXXX" inside tmpdir(), which goes with it. Throws when it cannot
 * be made, which fails the case, so a caller never writes into a path that is not its own. */
inline std::string make_tmp_dir(const std::string& stem) {
    std::string path = tmp_path(stem + ".XXXXXX");
    if (!::mkdtemp(path.data()))
        throw std::runtime_error("cannot make " + path + ": " + std::strerror(errno));
    return path;
}

struct Options {
    std::vector<std::string> include, exclude;
    int  shard = 0, shards = 1;
    bool list = false;
};

inline void add_pattern(Options& o, const std::string& p) {
    if (p.empty() || p == "-") return;
    if (p[0] == '-') o.exclude.push_back(p.substr(1));
    else             o.include.push_back(p);
}

/* False, with the reason printed, for a command line this harness does not understand. */
inline bool parse(int argc, char** argv, Options& o) {
    if (const char* f = std::getenv("RAD_TEST_FILTER")) {
        std::string word;
        for (const char* c = f;; ++c) {
            if (*c == '\0' || *c == ' ' || *c == ',' || *c == '\t' || *c == '\n') {
                add_pattern(o, word);
                word.clear();
                if (*c == '\0') break;
            } else {
                word += *c;
            }
        }
    }
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list") {
            o.list = true;
        } else if (a == "--shard" || a.rfind("--shard=", 0) == 0) {
            const std::string v = a == "--shard" ? (i + 1 < argc ? argv[++i] : "") : a.substr(8);
            int k = -1, n = 0;
            char tail = 0;
            if (std::sscanf(v.c_str(), "%d/%d%c", &k, &n, &tail) != 2 || n < 1 || k < 0 || k >= n) {
                fprintf(stderr, "--shard takes i/n with 0 <= i < n, not '%s'\n", v.c_str());
                return false;
            }
            o.shard = k;
            o.shards = n;
        } else if (a.rfind("--", 0) == 0) {
            fprintf(stderr, "unknown option %s\n"
                            "usage: %s [pattern...] [-pattern...] [--list] [--shard i/n]\n",
                    a.c_str(), argv[0]);
            return false;
        } else {
            add_pattern(o, a);
        }
    }
    return true;
}

inline bool selected(const char* name, const Options& o) {
    for (const std::string& x : o.exclude)
        if (std::strstr(name, x.c_str())) return false;
    if (o.include.empty()) return true;
    for (const std::string& x : o.include)
        if (std::strstr(name, x.c_str())) return true;
    return false;
}

inline double cpu_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

inline int run_all(int argc, char** argv) {
    if (argc > 0 && argv[0]) {
        const char* slash = std::strrchr(argv[0], '/');
        program() = slash ? slash + 1 : argv[0];
    }
    Options o;
    if (!parse(argc, argv, o)) return 2;

    std::vector<const Case*> run;
    size_t index = 0;
    for (const Case& c : cases())
        if (selected(c.name, o) && (int)(index++ % (size_t)o.shards) == o.shard) run.push_back(&c);
    if (o.list) {
        for (const Case* c : run) printf("%s\n", c->name);
        return 0;
    }

    using clock = std::chrono::steady_clock;
    struct Timed { const char* name; double s; };
    std::vector<Timed> times;
    int skipped = 0;
    const clock::time_point start = clock::now();
    for (const Case* c : run) {
        current() = c->name;
        case_skipped() = false;
        const int failed_before = failures();
        const int checked_before = checks();
        const clock::time_point t0 = clock::now();
        try {
            c->fn();
        } catch (const std::exception& e) {
            failures().fetch_add(1);
            fprintf(stderr, "  FAIL %s\n    exception: %s\n", c->name, e.what());
        } catch (...) {
            failures().fetch_add(1);
            fprintf(stderr, "  FAIL %s\n    exception of a type not derived from std::exception\n",
                    c->name);
        }
        const double s = std::chrono::duration<double>(clock::now() - t0).count();
        times.push_back({c->name, s});
        if (failures() != failed_before) continue;
        const bool skip_only = case_skipped() && checks() == checked_before;
        skipped += skip_only;
        if (s >= 0.1) fprintf(stderr, "  %s %s (%.2f s)\n", skip_only ? "skip" : "ok  ", c->name, s);
        else          fprintf(stderr, "  %s %s\n", skip_only ? "skip" : "ok  ", c->name);
    }
    const double wall = std::chrono::duration<double>(clock::now() - start).count();
    const double cpu = cpu_seconds();

    int rc = 0;
    if (run.empty())
        fprintf(stderr, "no case matches the filter\n");
    if (failures()) {
        fprintf(stderr, "%d failure(s)\n", failures().load());
        rc = 1;
    } else if (checks() == 0) {
        fprintf(stderr, "nothing was checked: every selected case guarded itself out\n");
        rc = 77;
    } else {
        char skips[32] = "";
        if (skipped) snprintf(skips, sizeof skips, " (%d skipped)", skipped);
        fprintf(stderr, "%d check(s) over %zu case(s)%s in %.2f s, %.2f s of CPU\n",
                checks().load(), run.size(), skips, wall, cpu);
    }
    if (times.size() > 1) {
        std::sort(times.begin(), times.end(), [](const Timed& a, const Timed& b) { return a.s > b.s; });
        std::string slowest;
        for (size_t i = 0; i < times.size() && i < 3 && times[i].s >= 0.1; ++i) {
            char buf[256];
            snprintf(buf, sizeof buf, "%s%s %.2f s", i ? ", " : "", times[i].name, times[i].s);
            slowest += buf;
        }
        if (!slowest.empty()) fprintf(stderr, "slowest: %s\n", slowest.c_str());
    }
    /* The suite's own speed: a binary that spends more CPU than its budget fails, naming the cost,
     * so a slow case or a spinning thread pool is caught when it lands. CPU time rather than wall
     * time, so a loaded machine does not fail a correct binary. */
    if (const char* b = std::getenv("RAD_TEST_CPU_BUDGET")) {
        double budget = std::atof(b);
        if (const char* s = std::getenv("RAD_TEST_CPU_BUDGET_SCALE"); s && std::atof(s) > 0)
            budget *= std::atof(s);
        if (budget > 0 && cpu > budget && rc != 1) {
            fprintf(stderr, "  FAIL the binary spent %.2f s of CPU, over its budget of %g s "
                            "(RAD_TEST_CPU_BUDGET)\n", cpu, budget);
            rc = 1;
        }
    }
    if (!made_tmpdir().empty()) {
        if (rc == 1) {
            fprintf(stderr, "kept %s\n", made_tmpdir().c_str());
        } else {
            std::error_code ec;
            std::filesystem::remove_all(made_tmpdir(), ec);
        }
    }
    return rc;
}

}  /* namespace radtest */

#define TEST(name)                                                                    \
    static void radtest_##name();                                                     \
    static ::radtest::Reg radtest_reg_##name(#name, radtest_##name);                  \
    static void radtest_##name()

#define CHECK(cond)                                                                   \
    do { ++::radtest::checks();                                                       \
         if (!(cond)) ::radtest::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); } while (0)

#define CHECK_EQ(a, b)                                                                \
    do { ++::radtest::checks();                                                       \
         auto _a = (a); auto _b = (b);                                                \
         if (!(_a == _b)) ::radtest::fail(__FILE__, __LINE__,                          \
             std::string("CHECK_EQ(" #a ", " #b "): ") + ::radtest::show(_a, _b));    \
    } while (0)

/* A CHECK whose failure makes the rest of the case meaningless (a size the next line indexes, a
 * pointer it dereferences). Reports like CHECK and then returns from the case, so the binary
 * does not segfault and lose every later case. Use CHECK where the case can continue. */
#define REQUIRE(cond)                                                                 \
    do { ++::radtest::checks();                                                       \
         if (!(cond)) { ::radtest::fail(__FILE__, __LINE__, "REQUIRE(" #cond ")");    \
                        return; } } while (0)

#define REQUIRE_EQ(a, b)                                                              \
    do { ++::radtest::checks();                                                       \
         auto _a = (a); auto _b = (b);                                                \
         if (!(_a == _b)) { ::radtest::fail(__FILE__, __LINE__,                        \
             std::string("REQUIRE_EQ(" #a ", " #b "): ") + ::radtest::show(_a, _b));  \
             return; } } while (0)

/* REQUIRE_EQ against RAD_OK: the status is ok, or the case leaves. */
#define REQUIRE_OK(expr) REQUIRE_EQ((expr), RAD_OK)

#define CHECK_NEAR(a, b, tol)                                                         \
    do { ++::radtest::checks();                                                       \
         double _a = (double)(a), _b = (double)(b);                                   \
         if (!(std::fabs(_a - _b) <= (tol))) ::radtest::fail(__FILE__, __LINE__,       \
             std::string("CHECK_NEAR(" #a ", " #b "): ") + ::radtest::show(_a, _b));  \
    } while (0)

#define CHECK_OK(expr)                                                                \
    do { ++::radtest::checks();                                                       \
         int _s = (expr);                                                             \
         if (_s < 0) ::radtest::fail(__FILE__, __LINE__,                               \
             std::string("CHECK_OK(" #expr "): ") + rad_strerror(_s)); } while (0)

/* Prints the skip line (radtest::skip) and leaves the case. */
#define SKIP(...)                                                                     \
    do { ::radtest::skip(__VA_ARGS__); return; } while (0)

/* Leaves the case through radtest::artifact_missing: a failure under RAD_TEST_STRICT, else a
 * skip. `what` names the artifact and where it was looked for. */
#define ARTIFACT_MISSING(what)                                                        \
    do { ::radtest::artifact_missing(__FILE__, __LINE__, (what)); return; } while (0)

/* Leaves the case through ARTIFACT_MISSING unless `path` is a readable file. */
#define NEED_ARTIFACT(path)                                                           \
    do { const std::string _p = (path);                                               \
         if (::access(_p.c_str(), R_OK) != 0) ARTIFACT_MISSING(_p); } while (0)

#define RAD_TEST_MAIN()                                                               \
    int main(int argc, char** argv) { return ::radtest::run_all(argc, argv); }

/* ------------------------------------------------------------------ helpers shared by the tests */

/* A substring test, so a CHECK reads as the claim it makes rather than as npos arithmetic. */
inline bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

/* The bytes of a file; empty when it cannot be read. */
inline std::string read_file(const std::string& path) {
    std::string out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

/* Writes `body` as the file at `path`. Throws when it cannot, which fails the case. */
inline void write_file(const std::string& path, const std::string& body) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path + ": " + std::strerror(errno));
    const bool ok = std::fwrite(body.data(), 1, body.size(), f) == body.size();
    if (std::fclose(f) != 0 || !ok) throw std::runtime_error("short write to " + path);
}

/* Everything written to stderr while `fn` runs. Capturing also keeps an expected refusal out of
 * the test output. A capture that cannot be set up is a failed check, and `fn` still runs. */
inline std::string stderr_of(const std::function<void()>& fn) {
    std::string out;
    FILE* tmp = std::tmpfile();
    CHECK(tmp != nullptr);
    std::fflush(stderr);
    const int saved = tmp ? dup(2) : -1;
    CHECK(saved >= 0);
    if (saved < 0) {
        if (tmp) std::fclose(tmp);
        fn();
        return out;
    }
    dup2(fileno(tmp), 2);
    fn();
    std::fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::rewind(tmp);
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, tmp)) > 0) out.append(buf, n);
    std::fclose(tmp);
    return out;
}

/* The radiance home holding the built plugins: $RAD_TESTPLUGIN_HOME for a hand-built run, else
 * $RADIANCE_HOME, which ctest sets. */
inline std::string home() {
    if (const char* d = std::getenv("RAD_TESTPLUGIN_HOME")) return d;
    const char* h = std::getenv("RADIANCE_HOME");
    return h ? h : "radiance_home";
}

/* A command line for config_parse, which takes `char**`: the strings are owned here, so they are
 * writable and outlive the call. argv[0] is "radiance", which the parser skips. */
struct Argv {
    std::vector<std::string> store;
    std::vector<char*>       argv;
    explicit Argv(std::vector<std::string> args) : store(std::move(args)) {
        store.insert(store.begin(), "radiance");
        for (std::string& s : store) argv.push_back(s.data());
    }
    int argc() const { return (int)argv.size(); }
};
