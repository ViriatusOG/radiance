/* kbench_timer.cpp -- rad-kbench's timing pass (kbench_timer.h). */
#include "kbench_timer.h"
#include "device/device.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <set>
#include <unistd.h>

namespace rad {
namespace kb {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kMiB = (int64_t)1 << 20;
/* Untimed launches between the gate and the timed batch: the first after the gate's L2 invalidate
 * and the keep-alive copy re-warms the activations, as the previous op leaves them in the
 * engine. */
constexpr int     kWarmUnits = 2;
/* The fewest batches a median is taken from (the fewest whose range covers the median with 95%
 * confidence), and its interval's target half-width. */
constexpr int     kMinReps = 6;
constexpr double  kCiPct = 1.0;
/* Batches repeated per case because a throttler came on. */
constexpr int     kMaxRetries = 3;
/* The keep-alive copies read the sweep source in pieces into one piece-sized ring, so they leave at
 * most this many dirty bytes in the caches for the batch to write back. */
constexpr int64_t kSweepPiece = kMiB;
/* The read probe's pieces: one workgroup of the ranged copy each. */
constexpr int64_t kReadPiece = 16 << 10;
/* Submissions kept free behind a gate for the event records. */
constexpr int64_t kRoomMargin = 8;
/* The device waited at the gate longer than this: the keep-alive copy was too short. */
constexpr double  kIdleUs = 20.0;
/* A host row is timed for at most this long. */
constexpr double  kHostBudgetS = 2.0;
/* A host batch without restores runs at least this long, so the clock reads are a small share. */
constexpr double  kHostBatchUs = 200.0;

/* Events by role. S and G bracket the gate (how long the device waited for the host), R0 and R1
 * the restore batch (each restore followed by a floor launch in the kernel's place), E0 and E1
 * the timed batch, B0 and B1 one bracketed launch. */
enum { kS = 0, kG, kR0, kR1, kE0, kE1, kB0, kB1 };

double ms_between(RadEvent a, RadEvent b) {
    float ms = 0;
    return rad_event_elapsed_ms(a, b, &ms) >= 0 ? (double)ms : -1.0;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return -1;
    return time_stats(std::move(v)).median;
}

/* The OpenMP threads a host library runs on, from the OpenMP runtime the libraries loaded; 0 when
 * none is loaded. */
int omp_threads() {
    for (const char* lib : { "libgomp.so.1", "libomp.so", "libomp.so.5", "libiomp5.so" }) {
        void* h = dlopen(lib, RTLD_NOW | RTLD_NOLOAD);
        if (!h) continue;
        auto f = (int (*)(void))dlsym(h, "omp_get_max_threads");
        const int n = f ? f() : 0;
        dlclose(h);
        if (n > 0) return n;
    }
    return 0;
}

std::string read_small(const std::string& path) {
    std::string s;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return s;
    char buf[256];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, f)) > 0 && s.size() < 4096) s.append(buf, got);
    std::fclose(f);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

/* How a case's operands rotated, for the verbose line: "" when nothing did. */
std::string rotation_of(int64_t R, size_t rotated, size_t tables) {
    if (!rotated) return std::string();
    std::string s = fmt("%lld cop%s", (long long)R, R == 1 ? "y" : "ies");
    if (tables) s += fmt(", %zu weight table(s) through pointer tables", tables);
    return s;
}

RadArgs args_of(const TimedCase& c, std::vector<RadTensor>& t) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = c.g.params();
    a.n_p = c.g.n_params();
    a.scratch = c.scratch;
    a.scratch_bytes = c.scratch_bytes;
    a.rank = 0;
    a.world_size = 1;
    return a;
}

/* Which operands rotate and how, shared by the device and host halves. */
struct RotPlan {
    std::vector<size_t> ops;      /* operand indices */
    int64_t R = 1;                /* copies */
};

RotPlan plan_rotation(const TimedCase& c, int64_t footprint) {
    RotPlan p;
    if (footprint <= 0) return p;
    const int64_t big = footprint / 32;
    int64_t per = 0;
    for (size_t k = 0; k < c.opd.size(); ++k) {
        const TimedOperand& d = c.opd[k];
        if (!d.t.data || d.bytes <= 0 || !rotated(d.role, d.bytes, big)) continue;
        p.ops.push_back(k);
        per += rotation_stride(d.bytes);
    }
    if (!p.ops.empty()) p.R = rotation_copies(per, footprint);
    return p;
}

/* A weight table rotated through pointer tables: stacked [n, ...] with equal entries. */
bool table_form(const TimedOperand& d, int64_t* n, int64_t* entry) {
    if (d.role != RAD_OPD_WTAB || d.t.rank < 2 || d.t.shape[0] <= 0 || d.t.stride[0] == 0)
        return false;
    *n = d.t.shape[0];
    if (d.bytes % *n) return false;
    *entry = d.bytes / *n;
    return true;
}

}  /* namespace */

/* ================================================================== the device half */

struct DeviceTimer::Prepared {
    TimedCase* c = nullptr;
    std::vector<RadTensor> t;
    struct Rot   { size_t k; uint8_t* arena; int64_t stride; };
    struct Tab   { size_t k; uint8_t* tables; int64_t n; int64_t count; };
    struct Store { void* dst; const void* src; int64_t bytes; };
    std::vector<Rot>   rot;
    std::vector<Tab>   tab;
    std::vector<Store> restores;
    std::vector<void*> frees;
    int64_t R = 1;
    int64_t next = 0;               /* the launch index, continued across batches */
    /* Measured behind the gate: submissions a unit (restores plus launch) and a restore set take,
     * and the room at the gate. */
    int64_t unit_subs = -1, restore_subs = 0, room = 0;
    bool    gated = true;
};

struct DeviceTimer::Rep {
    bool        ok = false;
    double      us = -1;            /* a launch, restores subtracted */
    double      unit_us = -1;       /* a unit, restores included: what sizes the next batch */
    double      restore_pct = -1;
    double      idle_us = 0;
    double      clock = -1;
    uint64_t    throttle = 0;       /* throttlers on that were off at idle */
    std::string why;
};

DeviceTimer::DeviceTimer(const TimerOpts& o, RadStream s, Phases* ph) : o_(o), s_(s), ph_(ph) {}

DeviceTimer::~DeviceTimer() {
    if (s_) (void)rad_stream_sync(s_);
    for (Held& h : held_) free_case(h);
    for (RadEvent& e : ev_) if (e) rad_event_destroy(e);
    for (void* p : { sweep_src_, sweep_dst_, word_ })
        if (p) rad_dev_free(p, RAD_MEM_DEVICE);
}

void DeviceTimer::free_case(Held& h) {
    for (void* p : h.c.frees) if (p) rad_dev_free(p, RAD_MEM_DEVICE);
    h.c.frees.clear();
}

int64_t DeviceTimer::add(TimedCase&& c) {
    int64_t extra = 0;
    for (const TimedOperand& d : c.opd) {
        if (d.role == RAD_OPD_INOUT && d.pristine && d.bytes > 0) extra += d.bytes;
        extra += (int64_t)d.variant.size();
    }
    if (!held_.empty() && held_bytes_ + c.held + extra > o_.budget) flush();

    Held h;
    h.id = next_id_++;
    h.pristine.assign(c.opd.size(), nullptr);
    std::string why;
    for (size_t k = 0; k < c.opd.size() && why.empty(); ++k) {
        TimedOperand& d = c.opd[k];
        if (d.role == RAD_OPD_INOUT && d.pristine && d.bytes > 0) {
            void* p = rad_dev_alloc(d.bytes, RAD_MEM_DEVICE);
            if (!p || rad_memcpy_async(p, d.pristine, d.bytes, s_) < 0) {
                why = "no device memory for a pristine copy of an in-place operand";
                if (p) rad_dev_free(p, RAD_MEM_DEVICE);
                break;
            }
            c.frees.push_back(p);
            h.pristine[k] = p;
        }
        d.pristine = nullptr;
        if (!d.variant.empty()) {
            void* p = rad_dev_alloc((int64_t)d.variant.size(), RAD_MEM_DEVICE);
            if (!p || rad_memcpy_async(p, d.variant.data(), (int64_t)d.variant.size(), s_) < 0) {
                why = "no device memory for a timing input";
                if (p) rad_dev_free(p, RAD_MEM_DEVICE);
                break;
            }
            c.frees.push_back(p);
            d.t.data = p;
            d.bytes = (int64_t)d.variant.size();
        }
    }
    /* The host bytes are the caller's and the variants are about to go. */
    if (rad_stream_sync(s_) < 0 && why.empty()) why = std::string("an upload failed: ") +
                                                      rad_dev_last_error();
    for (TimedOperand& d : c.opd) std::vector<uint8_t>().swap(d.variant);
    h.c = std::move(c);
    if (!why.empty()) {
        Timing t;
        t.note = why;
        results_[h.id] = t;
        free_case(h);
        return h.id;
    }
    h.c.held += extra;
    held_bytes_ += h.c.held;
    held_.push_back(std::move(h));
    return held_.back().id;
}

void DeviceTimer::drop(int64_t id) {
    for (auto it = held_.begin(); it != held_.end(); ++it) {
        if (it->id != id) continue;
        held_bytes_ -= it->c.held;
        (void)rad_stream_sync(s_);
        free_case(*it);
        held_.erase(it);
        break;
    }
    results_.erase(id);
}

const Timing* DeviceTimer::result(int64_t id) const {
    const auto it = results_.find(id);
    return it == results_.end() ? nullptr : &it->second;
}

double DeviceTimer::clock_now(uint64_t* throttle) {
    if (throttle) *throttle = 0;
    if (metrics_path_.empty()) return -1;
    uint8_t buf[512];
    const int fd = ::open(metrics_path_.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    const ssize_t n = ::read(fd, buf, sizeof buf);
    ::close(fd);
    GpuMetrics m;
    if (n <= 0 || !parse_gpu_metrics(buf, (size_t)n, &m)) return -1;
    if (throttle) *throttle = m.throttle;
    return m.gfxclk_mhz;
}

void DeviceTimer::sweep(int64_t bytes) {
    for (int64_t done = 0; done < bytes; done += kSweepPiece) {
        if (sweep_off_ + kSweepPiece > sweep_src_bytes_) sweep_off_ = 0;
        (void)rad_memcpy_async(sweep_dst_, (const uint8_t*)sweep_src_ + sweep_off_, kSweepPiece,
                               s_);
        sweep_off_ += kSweepPiece;
    }
}

void DeviceTimer::setup() {
    set_up_ = true;
    RadDeviceProps props{};
    const bool have_props = rad_dev_props(0, &props) >= 0;
    const bool host = have_props && props.is_host_backend;
    const int64_t llc = host ? host_llc_bytes() : 0;
    footprint_ = o_.footprint >= 0 ? o_.footprint : default_footprint(llc);
    scale_ = footprint_ > 0 ? footprint_ : default_footprint(llc);
    std::string fail;
    for (RadEvent& e : ev_)
        if (!e && rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT) < 0)
            fail = std::string("no events: ") + rad_dev_last_error();
    sweep_src_bytes_ = std::max(scale_, kSweepPiece);
    if (fail.empty()) {
        sweep_src_ = rad_dev_alloc(sweep_src_bytes_, RAD_MEM_DEVICE);
        sweep_dst_ = rad_dev_alloc(kSweepPiece, RAD_MEM_DEVICE);
        word_ = rad_dev_alloc(256, RAD_MEM_DEVICE);
        if (!sweep_src_ || !sweep_dst_ || !word_)
            fail = "no device memory for the keep-alive copy (" + humanb(sweep_src_bytes_) + ")";
    }
    if (fail.empty() && (rad_memset_async(sweep_src_, 0, sweep_src_bytes_, s_) < 0 ||
                         rad_stream_sync(s_) < 0))
        fail = std::string("the keep-alive source could not be written: ") + rad_dev_last_error();
    if (!fail.empty()) {
        withheld_ = fail;
        header_.push_back("timing: " + fail);
        return;
    }
    sweep_bytes_ = std::min<int64_t>(64 * kMiB, scale_);
    sweep_max_ = std::min<int64_t>(std::max<int64_t>(4 * scale_, 4 * kMiB), (int64_t)1 << 30);

    const std::string pci = dev_pci_address(0);
    if (!pci.empty()) {
        metrics_path_ = "/sys/bus/pci/devices/" + pci + "/gpu_metrics";
        if (clock_now(&idle_throttle_) < 0) metrics_path_.clear();
    }

    header_.push_back(fmt("device: %s (%s), the %s backend", have_props ? props.name : "?",
                          have_props ? props.arch : "?", device_backend_name()));
    if (footprint_ > 0)
        header_.push_back(fmt("cold: each launch reads its weights, weight tables and inputs of %s "
                              "or more from one of a set of copies spanning %s, so from memory; "
                              "smaller inputs stay warm, as the previous op leaves them; in-place "
                              "operands are restored before every launch and the restores' own "
                              "time subtracted", humanb(footprint_ / 32).c_str(),
                              humanb(footprint_).c_str()));
    else
        header_.push_back("warm: every launch rereads the same operands, so whatever fits in the "
                          "caches is read from them; in-place operands are restored before every "
                          "launch and the restores' own time subtracted");

    if (o_.warm_ms > 0) warm_up();
    uint64_t loaded = 0;
    warm_clock_ = clock_now(&loaded);
    if (o_.warm_ms > 0) load_throttle_ = loaded & ~idle_throttle_;
    calibrate();
    if (!metrics_path_.empty()) {
        const std::string idle = throttle_names(idle_throttle_);
        const std::string load = throttle_names(load_throttle_);
        header_.push_back(fmt("clock: %.0f MHz averaged graphics clock after a %d ms warm-up "
                              "(gpu_metrics)%s%s; a batch during which another throttler came on "
                              "is repeated", warm_clock_, o_.warm_ms,
                              idle.empty() ? "" : ("; on at idle, not counted: " + idle).c_str(),
                              load.empty() ? "" : ("; on under the warm-up's load, binding "
                                                   "throughout and not counted: " + load).c_str()));
    } else {
        header_.push_back("clock: not read (no gpu_metrics table for this device)");
    }
}

/* Sustained load for --bench-warm-ms: keep-alive copies and the held cases' own launches, one
 * round always queued behind the one the host waits for, so the card never idles meanwhile. */
void DeviceTimer::warm_up() {
    const auto t0 = Clock::now();
    size_t ci = 0;
    for (int64_t round = 0;
         std::chrono::duration<double, std::milli>(Clock::now() - t0).count() < o_.warm_ms;
         ++round) {
        sweep(scale_);
        if (!held_.empty()) {
            /* In-place operands restored as in the timed batches: a state applied over and over
             * can leave the range its kernel addresses. */
            Held& h = held_[ci++ % held_.size()];
            std::vector<RadTensor> t;
            for (const TimedOperand& d : h.c.opd) t.push_back(d.t);
            const RadArgs a = args_of(h.c, t);
            for (int i = 0; i < 8; ++i) {
                for (size_t k = 0; k < h.c.opd.size(); ++k)
                    if (h.pristine[k])
                        (void)rad_memcpy_async(h.c.opd[k].t.data, h.pristine[k], h.c.opd[k].bytes,
                                               s_);
                if (h.c.info->launch(&a, s_) < 0) break;
            }
        }
        (void)rad_event_record(ev_[kB0 + (round & 1)], s_);
        if (round > 0 && rad_event_sync(ev_[kB0 + ((round - 1) & 1)]) < 0) break;
    }
    (void)rad_stream_sync(s_);
    (void)rad_dev_last_error();
    last_busy_ = Clock::now();
}

/* Reads of `bytes` from src + i * bytes for i cycling over `copies`, as one ranged copy each
 * (`ranges`, device memory) whose pieces all land in the sweep's ring: the writes stay in the
 * cache, so the rate is the read stream's. GB/s of bytes read, the median of gated batches. */
double DeviceTimer::read_rate(const uint8_t* src, int64_t bytes, int64_t copies, const void* ranges,
                              int ranges_n) {
    std::vector<double> gbs;
    int64_t i = 0, n = 2;
    const double target = batch_target_us(floor_us_);
    auto read = [&]() {
        const uint8_t* from = src + (i++ % copies) * bytes;
        return rad_memcpy_ranges_async(sweep_dst_, from, 0, (const RadCopyRange*)ranges, ranges_n,
                                       s_);
    };
    for (int r = 0; r < 12; ++r) {
        void* gate = nullptr;
        sweep(sweep_bytes_);
        if (rad_stream_gate(s_, &gate) < 0) return -1;
        const int64_t room = rad_gate_room(s_) - kRoomMargin - kWarmUnits;
        int rc = RAD_OK;
        for (int w = 0; w < kWarmUnits && rc >= 0; ++w) rc = read();
        (void)rad_event_record(ev_[kE0], s_);
        for (int64_t j = 0; j < n && rc >= 0; ++j) rc = read();
        (void)rad_event_record(ev_[kE1], s_);
        (void)rad_gate_open(gate);
        if (rad_stream_sync(s_) < 0 || rc < 0) return -1;
        const double ms = ms_between(ev_[kE0], ev_[kE1]);
        if (ms <= 0) return -1;
        if (r > 0) gbs.push_back((double)bytes * (double)n / (ms * 1e-3) / 1e9);
        n = batch_launches(ms * 1e3 / (double)n, target, room, 0);
        if (timing_done(gbs, kMinReps, 10, kCiPct)) break;
    }
    return median_of(gbs);
}

void DeviceTimer::calibrate() {
    /* The launch floor: the cheapest kernel there is, the backend's own 4-byte fill. Back to back
     * is what a launch costs inside a batch; bracketed (alone between two events) is what it costs
     * timed one launch at a time. */
    {
        std::vector<double> b2b, bracketed;
        int64_t n = 256;
        for (int r = 0; r < 12 && (int)b2b.size() < 10; ++r) {
            void* gate = nullptr;
            sweep(sweep_bytes_);
            if (rad_stream_gate(s_, &gate) < 0) break;
            n = std::min<int64_t>(n, rad_gate_room(s_) - kRoomMargin - kWarmUnits);
            int rc = RAD_OK;
            for (int w = 0; w < kWarmUnits; ++w) rc |= rad_memset_async(word_, w, 4, s_);
            (void)rad_event_record(ev_[kE0], s_);
            for (int64_t j = 0; j < n; ++j) rc |= rad_memset_async(word_, (int)(j & 0xff), 4, s_);
            (void)rad_event_record(ev_[kE1], s_);
            (void)rad_event_record(ev_[kB0], s_);
            rc |= rad_memset_async(word_, 1, 4, s_);
            (void)rad_event_record(ev_[kB1], s_);
            (void)rad_gate_open(gate);
            if (rad_stream_sync(s_) < 0 || rc != RAD_OK) break;
            const double ms = ms_between(ev_[kE0], ev_[kE1]);
            const double one = ms_between(ev_[kB0], ev_[kB1]);
            if (ms > 0) b2b.push_back(ms * 1e3 / (double)n);
            if (one > 0) bracketed.push_back(one * 1e3);
            if (timing_done(b2b, kMinReps, 10, kCiPct) && (int)bracketed.size() >= kMinReps) break;
        }
        floor_us_ = median_of(b2b);
        floor_bracketed_us_ = median_of(bracketed);
        header_.push_back(fmt("launch floor: %.2f us a launch back to back in a batch, %.2f us "
                              "alone between two events (the backend's own 4-byte fill, the "
                              "cheapest kernel a tool can launch); a batch lasts at least %.0f us",
                              floor_us_, floor_bracketed_us_, batch_target_us(floor_us_)));
    }

    /* Streaming-read bandwidth at four sizes up to the footprint, warm (one buffer reread) and
     * cold (rotated over the footprint). The probes check that cold reads are cold, so warm
     * timing, which takes none, skips them. */
    if (footprint_ <= 0) {
        header_.push_back("read bandwidth: not probed (warm timing reads nothing cold)");
        if (floor_us_ <= 0) {
            withheld_ = "the launch floor could not be measured";
            header_.push_back("timings withheld: " + withheld_);
        }
        return;
    }
    std::vector<BwProbe> probes;
    int64_t last = 0;
    for (const int64_t div : { (int64_t)64, (int64_t)16, (int64_t)4, (int64_t)1 }) {
        const int64_t bytes = std::max<int64_t>(64 << 10, (scale_ / div) & ~(int64_t)0xFFFF);
        if (bytes == last) continue;
        last = bytes;
        BwProbe p;
        p.bytes = bytes;
        /* Pieces of 16 KiB, one workgroup each, cycling through the 1 MiB ring. */
        std::vector<RadCopyRange> pieces;
        for (int64_t off = 0; off < bytes; off += kReadPiece)
            pieces.push_back({ off, off % kSweepPiece, std::min(kReadPiece, bytes - off) });
        const int64_t list = (int64_t)(pieces.size() * sizeof(RadCopyRange));
        const int64_t R = rotation_copies(bytes, scale_);
        uint8_t* src = (uint8_t*)rad_dev_alloc(R * bytes, RAD_MEM_DEVICE);
        void* ranges = rad_dev_alloc(list, RAD_MEM_DEVICE);
        if (src && ranges && rad_memset_async(src, 1, R * bytes, s_) >= 0 &&
            rad_memcpy_async(ranges, pieces.data(), list, s_) >= 0 && rad_stream_sync(s_) >= 0) {
            p.warm_gbs = read_rate(src, bytes, 1, ranges, (int)pieces.size());
            p.cold_gbs = read_rate(src, bytes, R, ranges, (int)pieces.size());
        }
        if (src) rad_dev_free(src, RAD_MEM_DEVICE);
        if (ranges) rad_dev_free(ranges, RAD_MEM_DEVICE);
        probes.push_back(p);
    }
    std::string bw = "read bandwidth (ranged copies into a 1 MiB ring, so the writes stay in the "
                     "cache), warm / cold:";
    for (const BwProbe& p : probes)
        bw += fmt(" %s %.0f / %.0f GB/s;", humanb(p.bytes).c_str(), p.warm_gbs, p.cold_gbs);
    bw.pop_back();
    header_.push_back(bw);

    RadDeviceProps props{};
    const bool host = rad_dev_props(0, &props) >= 0 && props.is_host_backend;
    const std::string why = bandwidth_refusal(probes);
    if (!why.empty() && !host) {
        withheld_ = why;
        header_.push_back("timings withheld: " + why);
    } else if (!why.empty()) {
        header_.push_back("(host backend: no device caches to defeat, so the check does not "
                          "apply: " + why + ")");
    }
    if (floor_us_ <= 0 && withheld_.empty()) {
        withheld_ = "the launch floor could not be measured";
        header_.push_back("timings withheld: " + withheld_);
    }
}

bool DeviceTimer::prepare(Held& h, Prepared* p, std::string* why) {
    TimedCase& c = h.c;
    p->c = &c;
    for (const TimedOperand& d : c.opd) p->t.push_back(d.t);
    for (size_t k = 0; k < c.opd.size(); ++k)
        if (h.pristine[k])
            p->restores.push_back({ c.opd[k].t.data, h.pristine[k], c.opd[k].bytes });
    const RotPlan plan = plan_rotation(c, footprint_);
    p->R = plan.R;
    for (size_t k : plan.ops) {
        const TimedOperand& d = c.opd[k];
        const int64_t stride = rotation_stride(d.bytes);
        uint8_t* base = (uint8_t*)d.t.data;
        uint8_t* arena = nullptr;
        if (p->R > 1) {
            arena = (uint8_t*)rad_dev_alloc(p->R * stride, RAD_MEM_DEVICE);
            if (!arena) {
                *why = "no device memory for " + std::to_string(p->R) + " rotated copies of " +
                       humanb(d.bytes);
                return false;
            }
            p->frees.push_back(arena);
            /* The original into the first copy, then the filled prefix doubled. */
            int rc = rad_memcpy_async(arena, base, d.bytes, s_);
            for (int64_t have = 1; have < p->R && rc >= 0; have *= 2)
                rc = rad_memcpy_async(arena + have * stride, arena,
                                      std::min(have, p->R - have) * stride, s_);
            if (rc < 0) { *why = "the rotated copies could not be written"; return false; }
        }
        int64_t n = 0, entry = 0;
        if (table_form(d, &n, &entry)) {
            const int64_t count = table_count(p->R);
            std::vector<uint64_t> bases;
            for (int64_t r = 0; r < p->R; ++r)
                bases.push_back((uint64_t)(uintptr_t)(arena ? arena + r * stride : base));
            std::vector<uint64_t> ptr;
            ptr.reserve((size_t)(count * n));
            for (int64_t l = 0; l < count; ++l) table_pointers(bases, n, entry, c.seed, l, &ptr);
            uint8_t* tables = (uint8_t*)rad_dev_alloc(count * n * 8, RAD_MEM_DEVICE);
            if (!tables) {
                *why = "no device memory for the weight table's pointer tables";
                return false;
            }
            p->frees.push_back(tables);
            if (rad_memcpy_async(tables, ptr.data(), count * n * 8, s_) < 0 ||
                rad_stream_sync(s_) < 0) {
                *why = "the pointer tables could not be written";
                return false;
            }
            p->t[k].stride[0] = 0;
            p->tab.push_back({ k, tables, n, count });
        } else if (arena) {
            p->rot.push_back({ k, arena, stride });
        }
    }
    if (rad_stream_sync(s_) < 0) {
        *why = std::string("setup failed: ") + rad_dev_last_error();
        return false;
    }
    return true;
}

void DeviceTimer::release(Prepared* p) {
    (void)rad_stream_sync(s_);
    for (void* f : p->frees) rad_dev_free(f, RAD_MEM_DEVICE);
    p->frees.clear();
}

void DeviceTimer::unit(Prepared& p, bool restore, int* rc) {
    if (restore)
        for (const Prepared::Store& r : p.restores)
            if (rad_memcpy_async(r.dst, r.src, r.bytes, s_) < 0) { *rc = RAD_E_DEVICE; return; }
    const int64_t i = p.next++;
    for (const Prepared::Rot& r : p.rot) p.t[r.k].data = r.arena + (i % p.R) * r.stride;
    for (const Prepared::Tab& b : p.tab) p.t[b.k].data = b.tables + (i % b.count) * b.n * 8;
    const RadArgs a = args_of(*p.c, p.t);
    const int s = p.c->info->launch(&a, s_);
    if (s < 0) *rc = s;
}

DeviceTimer::Rep DeviceTimer::rep(Prepared& p, int64_t n, bool gated) {
    Rep r;
    void* gate = nullptr;
    int rc = RAD_OK;
    sweep(sweep_bytes_);
    if (gated) {
        (void)rad_event_record(ev_[kS], s_);
        if (rad_stream_gate(s_, &gate) < 0) {
            r.why = std::string("no gate: ") + rad_dev_last_error();
            return r;
        }
        (void)rad_event_record(ev_[kG], s_);
    }
    const int64_t room0 = gated ? rad_gate_room(s_) : 0;
    if (!p.restores.empty()) {
        /* The baseline the restores are subtracted with: each restore set followed, as in the
         * timed batch, by a dependent launch, here the floor's fill. Restores alone back to back
         * pipeline differently from a restore feeding a launch, and the difference came out under
         * the launch floor for kernels that are mostly restore. The fill's own cost, the floor,
         * is added back below. */
        (void)rad_event_record(ev_[kR0], s_);
        for (int64_t j = 0; j < n && rc >= 0; ++j) {
            for (const Prepared::Store& s : p.restores)
                if (rad_memcpy_async(s.dst, s.src, s.bytes, s_) < 0) rc = RAD_E_DEVICE;
            if (rc >= 0 && rad_memset_async(word_, (int)(j & 0xff), 4, s_) < 0) rc = RAD_E_DEVICE;
        }
        (void)rad_event_record(ev_[kR1], s_);
    }
    const int64_t room1 = gated ? rad_gate_room(s_) : 0;
    for (int w = 0; w < kWarmUnits && rc >= 0; ++w) unit(p, true, &rc);
    const int64_t room2 = gated ? rad_gate_room(s_) : 0;
    if (rc >= 0) (void)rad_event_record(ev_[kE0], s_);
    for (int64_t j = 0; j < n && rc >= 0; ++j) unit(p, true, &rc);
    if (rc >= 0) (void)rad_event_record(ev_[kE1], s_);
    if (gate) (void)rad_gate_open(gate);
    const int sync = rad_stream_sync(s_);
    if (rc < 0 || sync < 0) {
        const char* e = rad_dev_last_error();
        r.why = *e ? e : rad_strerror(rc < 0 ? rc : sync);
        return r;
    }
    if (gated) {
        /* Rounded up: on the host backend the worker may take the gate's own task between two
         * reads, which frees one slot. */
        const int64_t rs = room0 - room1 - (p.restores.empty() ? 0 : 2);
        p.restore_subs = p.restores.empty() || n <= 0 ? 0 : std::max<int64_t>(0, (rs + n - 1) / n);
        p.unit_subs = std::max<int64_t>(0, (room1 - room2 + kWarmUnits - 1) / kWarmUnits);
        p.room = room0;
    }
    const double tb = ms_between(ev_[kE0], ev_[kE1]);
    const double tr = p.restores.empty() ? 0.0 : ms_between(ev_[kR0], ev_[kR1]);
    if (tb < 0 || tr < 0) {
        r.why = "the events carry no timestamps";
        return r;
    }
    r.unit_us = tb * 1e3 / (double)n;
    r.us = (tb - tr) * 1e3 / (double)n + (p.restores.empty() ? 0.0 : floor_us_);
    /* The restores' share of the timed batch: the baseline's floor launches stand in for the
     * kernel, so they are not restore time. */
    r.restore_pct = p.restores.empty() || tb <= 0
                        ? -1
                        : 100.0 * std::max(0.0, tr - (double)n * floor_us_ * 1e-3) / tb;
    r.idle_us = gated ? std::max(0.0, ms_between(ev_[kS], ev_[kG]) * 1e3) : 0.0;
    uint64_t thr = 0;
    r.clock = clock_now(&thr);
    r.throttle = thr & ~(idle_throttle_ | load_throttle_);
    r.ok = true;
    return r;
}

void DeviceTimer::time_case(Held& h) {
    Timing out;
    Prepared p;
    std::string why;
    const auto t0 = Clock::now();
    if (!prepare(h, &p, &why)) {
        out.note = why;
        results_[h.id] = out;
        release(&p);
        free_case(h);
        return;
    }
    /* A pilot batch of two sizes the rest and measures what a unit takes behind the gate. A launch
     * that cannot run behind a gate (it waits on its own stream: a host-backend device row) is
     * timed without one, and says so. */
    const int64_t pilot = 2;
    Rep r = rep(p, pilot, true);
    if (!r.ok) {
        p.gated = false;
        out.note = r.why.find("closed gate") != std::string::npos
                       ? "ungated: the launch waits on its own stream, which a gate holds"
                       : "ungated: behind a gate, " + r.why;
        r = rep(p, pilot, false);
    }
    std::vector<double> us, clocks, restore;
    double est = r.unit_us;
    const double target = batch_target_us(floor_us_);
    const int max_reps = std::max(1, o_.max_reps);
    while (r.ok) {
        int64_t room = 0;
        if (p.gated && p.unit_subs > 0)
            room = (p.room - kRoomMargin - kWarmUnits * p.unit_subs) /
                   (p.unit_subs + p.restore_subs);
        const int64_t n = batch_launches(
            est, target, p.gated && p.unit_subs > 0 ? std::max<int64_t>(room, 1) : 0, o_.fixed);
        r = rep(p, n, p.gated);
        if (!r.ok) break;
        if (r.idle_us > kIdleUs && p.gated) sweep_bytes_ = std::min(2 * sweep_bytes_, sweep_max_);
        if (r.throttle) {
            if (out.retried < kMaxRetries) { ++out.retried; continue; }
            out.note += std::string(out.note.empty() ? "" : "; ") + "throttled (" +
                        throttle_names(r.throttle) + ") after " + std::to_string(kMaxRetries) +
                        " retries";
        }
        us.push_back(r.us);
        if (r.clock > 0) clocks.push_back(r.clock);
        if (r.restore_pct >= 0) restore.push_back(r.restore_pct);
        if (r.unit_us > 0) est = r.unit_us;
        if (timing_done(us, std::min(kMinReps, max_reps), max_reps, kCiPct)) break;
    }
    if (!r.ok)
        out.note += std::string(out.note.empty() ? "" : "; ") +
                    (us.empty() ? "timing failed: " : "stopped early: ") + r.why;
    out.st = time_stats(us);
    out.clock_mhz = median_of(clocks);
    out.restore_pct = median_of(restore);
    out.gated = p.gated && out.st.n > 0;
    out.rotation = rotation_of(p.R, p.rot.size() + p.tab.size(), p.tab.size());
    if (!p.restores.empty() && out.st.n > 0 && out.st.median <= 0) {
        out.note += std::string(out.note.empty() ? "" : "; ") +
                    "restore-dominated: the in-place restores took the whole batch";
    } else if (out.st.n > 0) {
        const std::string c = timing_caveat(out.st.median, floor_us_,
                                            p.restores.empty() ? -1.0 : out.restore_pct);
        if (!c.empty()) out.note += std::string(out.note.empty() ? "" : "; ") + c;
    }
    results_[h.id] = out;
    release(&p);
    free_case(h);
    if (o_.verbose)
        std::printf("  timed  %-60s %9.2f us  n=%d ci=%.1f%%  %s%s%s%s%.1f s\n",
                    h.c.what.c_str(), out.st.median, out.st.n, out.st.ci_pct,
                    out.gated ? "gated" : "ungated", out.rotation.empty() ? "" : ", ",
                    out.rotation.c_str(),
                    out.note.empty() ? "  " : ("  " + out.note + "  ").c_str(),
                    std::chrono::duration<double>(Clock::now() - t0).count());
}

void DeviceTimer::flush() {
    if (held_.empty()) return;
    PhaseTimer pt(ph_, PH_TIMING);
    const bool first = !set_up_;
    if (first) setup();
    if (!withheld_.empty()) {
        for (Held& h : held_) {
            Timing t;
            t.note = "withheld: " + withheld_;
            results_[h.id] = t;
            free_case(h);
        }
        held_.clear();
        held_bytes_ = 0;
        return;
    }
    /* The correctness work since the last pass left the card idle long enough to drop its clock. */
    if (!first && o_.warm_ms > 0 &&
        std::chrono::duration<double, std::milli>(Clock::now() - last_busy_).count() > 200.0)
        warm_up();
    for (Held& h : held_) time_case(h);
    held_.clear();
    held_bytes_ = 0;
    last_busy_ = Clock::now();
}

/* ================================================================== the host half */

Timing time_host(const TimedCase& c, const TimerOpts& o, int64_t footprint, double checked_s) {
    Timing out;
    out.threads = omp_threads();
    const size_t nk = c.opd.size();
    std::vector<RadTensor> t(nk);
    /* The timing draws, and private copies of everything the launch writes: the caller's
     * buffers hold the checked launch's output, which a comparison may still read. */
    std::vector<std::vector<uint8_t>> variants(nk), written(nk);
    struct Store { void* dst; const void* src; int64_t bytes; };
    std::vector<Store> restores;
    for (size_t k = 0; k < nk; ++k) {
        const TimedOperand& d = c.opd[k];
        t[k] = d.t;
        if (!d.variant.empty()) {
            variants[k] = d.variant;
            t[k].data = variants[k].data();
        }
        if ((d.role == RAD_OPD_OUT || d.role == RAD_OPD_INOUT) && d.t.data && d.bytes > 0) {
            written[k].assign((const uint8_t*)d.t.data, (const uint8_t*)d.t.data + d.bytes);
            t[k].data = written[k].data();
        }
        if (d.role == RAD_OPD_INOUT && d.pristine && d.bytes > 0)
            restores.push_back({ t[k].data, d.pristine, d.bytes });
    }

    /* Rotation, as on the device: copies in page-aligned host memory, weight tables through
     * pointer tables. */
    struct Owned {
        std::vector<void*> p;
        ~Owned() { for (void* x : p) std::free(x); }
    } owned;
    struct Rot { size_t k; uint8_t* arena; int64_t stride; };
    struct Tab { size_t k; std::vector<uint64_t> ptr; int64_t n; int64_t count; };
    std::vector<Rot> rot;
    std::vector<Tab> tab;
    TimedCase sized = c;     /* the variant sizes, for the plan */
    for (size_t k = 0; k < nk; ++k)
        if (!variants[k].empty()) {
            sized.opd[k].t.data = variants[k].data();
            sized.opd[k].bytes = (int64_t)variants[k].size();
        }
    const RotPlan plan = plan_rotation(sized, footprint);
    for (size_t k : plan.ops) {
        const TimedOperand& d = sized.opd[k];
        const int64_t stride = rotation_stride(d.bytes);
        uint8_t* base = (uint8_t*)d.t.data;
        uint8_t* arena = nullptr;
        if (plan.R > 1) {
            arena = (uint8_t*)std::aligned_alloc(
                4096, (size_t)((plan.R * stride + 4095) & ~(int64_t)4095));
            if (!arena) {
                out.note = "no memory for " + std::to_string(plan.R) + " rotated copies of " +
                           humanb(d.bytes);
                return out;
            }
            owned.p.push_back(arena);
            std::memcpy(arena, base, (size_t)d.bytes);
            for (int64_t have = 1; have < plan.R; have *= 2)
                std::memcpy(arena + have * stride, arena,
                            (size_t)(std::min(have, plan.R - have) * stride));
        }
        int64_t n = 0, entry = 0;
        if (table_form(d, &n, &entry)) {
            Tab b;
            b.k = k;
            b.n = n;
            b.count = table_count(plan.R);
            std::vector<uint64_t> bases;
            for (int64_t r = 0; r < plan.R; ++r)
                bases.push_back((uint64_t)(uintptr_t)(arena ? arena + r * stride : base));
            for (int64_t l = 0; l < b.count; ++l)
                table_pointers(bases, n, entry, c.seed, l, &b.ptr);
            t[k].stride[0] = 0;
            tab.push_back(std::move(b));
        } else if (arena) {
            rot.push_back({ k, arena, stride });
        }
    }

    int64_t next = 0;
    auto launch = [&]() -> int {
        const int64_t i = next++;
        for (const Rot& r : rot) t[r.k].data = r.arena + (i % plan.R) * r.stride;
        for (const Tab& b : tab) t[b.k].data = (void*)(b.ptr.data() + (i % b.count) * b.n);
        const RadArgs a = args_of(c, t);
        return c.info->launch(&a, nullptr);
    };
    auto restore = [&]() {
        for (const Store& s : restores) std::memcpy(s.dst, s.src, (size_t)s.bytes);
    };

    double est_us = checked_s * 1e6;
    if (checked_s < 1.0) {
        restore();
        const auto w0 = Clock::now();
        if (launch() < 0) {
            out.note = "timing failed: the launch refused";
            return out;
        }
        est_us = std::chrono::duration<double, std::micro>(Clock::now() - w0).count();
    }
    const bool batched = restores.empty();
    const int max_reps = std::max(1, o.max_reps);
    std::vector<double> us;
    const auto start = Clock::now();
    for (;;) {
        const int64_t b = batched ? batch_launches(est_us, kHostBatchUs, 0, o.fixed) : 1;
        if (!batched) restore();
        const auto t0 = Clock::now();
        for (int64_t j = 0; j < b; ++j)
            if (launch() < 0) {
                out.note = "timing failed: the launch refused";
                out.st = time_stats(us);
                return out;
            }
        const double el = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        us.push_back(el / (double)b);
        est_us = el / (double)b;
        if (timing_done(us, std::min(kMinReps, max_reps), max_reps, kCiPct)) break;
        if (std::chrono::duration<double>(Clock::now() - start).count() > kHostBudgetS) {
            if ((int)us.size() < std::min(kMinReps, max_reps))
                out.note = fmt("%zu timed launch(es) within the %.0f s host budget", us.size(),
                               kHostBudgetS);
            break;
        }
    }
    out.st = time_stats(us);
    out.rotation = rotation_of(plan.R, rot.size() + tab.size(), tab.size());
    if (o.verbose)
        std::printf("  timed  %-60s %9.2f us  n=%d ci=%.1f%%  %d thread(s)%s%s%s%s\n",
                    c.what.c_str(), out.st.median, out.st.n, out.st.ci_pct, out.threads,
                    out.rotation.empty() ? "" : ", ", out.rotation.c_str(),
                    out.note.empty() ? "" : "  ", out.note.c_str());
    return out;
}

int64_t host_llc_bytes() {
    /* Every cache index of every CPU at the highest level, each distinct instance (by the CPUs
     * sharing it) counted once. */
    std::set<std::string> seen;
    int64_t total = 0;
    int top = 0;
    std::vector<std::pair<int, std::pair<std::string, int64_t>>> found;
    DIR* d = opendir("/sys/devices/system/cpu");
    if (!d) return 0;
    while (dirent* e = readdir(d)) {
        if (std::strncmp(e->d_name, "cpu", 3) != 0 || !std::isdigit((unsigned char)e->d_name[3]))
            continue;
        for (int idx = 0; idx < 8; ++idx) {
            const std::string base = std::string("/sys/devices/system/cpu/") + e->d_name +
                                     "/cache/index" + std::to_string(idx) + "/";
            const std::string level = read_small(base + "level");
            if (level.empty()) break;
            const std::string size = read_small(base + "size");
            const std::string shared = read_small(base + "shared_cpu_list");
            char* end = nullptr;
            const long long v = std::strtoll(size.c_str(), &end, 10);
            if (v <= 0) continue;
            const int64_t bytes = (int64_t)v * (end && (*end == 'K' || *end == 'k') ? 1024
                                                : end && (*end == 'M' || *end == 'm') ? kMiB : 1);
            const int lv = std::atoi(level.c_str());
            top = std::max(top, lv);
            found.push_back({ lv, { shared, bytes } });
        }
    }
    closedir(d);
    for (const auto& [lv, f] : found) {
        if (lv != top || !seen.insert(f.first).second) continue;
        total += f.second;
    }
    return total;
}

}  /* namespace kb */
}  /* namespace rad */
