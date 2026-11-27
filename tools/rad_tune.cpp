/* rad-tune -- benchmark every point of each kernel's declared tuning space at the shapes the graph
 * actually issues, and write the cache the selector consults at declare.
 *
 * Constraint matching finds a correct kernel for a geometry; within it, the best tile configuration
 * depends on shape and machine (spec §15). This tool is the single tuning mechanism and tunecache.h
 * the single format, so plugins need no tuning tools of their own.
 *
 * The work list is Program::tune_requests: every (kernel, geometry) pair the declared graph can
 * reach, per band, rather than a swept grid.
 *
 * A candidate's time is the median over reps, not the best (a best-of hides clock-dependent
 * candidates), and the spread is reported so a measurement across a clock ramp is visible.
 */
#include "kfixture.h"
#include "opshapes.h"
#include "format/tunecache.h"

#include "rad_device.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

using namespace rad;

namespace {

void usage() {
    std::printf(
        "rad-tune -- benchmark each kernel's tunable space at this model's shapes, "
        "on this machine\n"
        "\n"
        "usage: rad-tune -m <model.rad> [options]\n"
        "\n"
        "  -m, --model FILE   the .rad whose declared graph supplies the shapes\n"
        "      --home DIR     $RADIANCE_HOME; the cache is written to <home>/tune/<machine>.tune\n"
        "      --kernels A:B  the kernel libraries to load, first wins; default every "
        "library found\n"
        "      --op NAME      tune only ops whose name starts with NAME\n"
        "      --machine KEY  override the machine key (for writing a cache for another box)\n"
        "      --reps N       timing reps, median reported (default 5)\n"
        "      --iters N      launches per rep (default 32). One launch is short enough that\n"
        "                     launch overhead and clock ramp would otherwise dominate.\n"
        "      --max-tok N    the max_tok declare is run at (default 8192)\n"
        "      --max-seqs N   the max_seqs declare is run at (default 256)\n"
        "      --tp N         tensor-parallel width to declare at (default 1). The shapes a\n"
        "                     sharded deployment runs are not the shapes tp=1 runs.\n"
        "      --kv-cache-dtype T  bf16 (default) or i8 -- attention bands are gated on it.\n"
        "                     Declare what the served engine runs, or the tune rows will not\n"
        "                     match the bands it selects.\n"
        "      --tp-wire MODE exact (default) or wht6 -- selects the all-reduce family.\n"
        "      --tp-wire-min-kb N  the message size from which the lossy wire is used.\n"
        "      --num-speculative-tokens N|auto\n"
        "                     draft depth to declare at (default auto: the container's "
        "own). It puts\n"
        "                     the drafter's kernels in the measured graph.\n"
        "      --dry-run      list what would be measured and stop\n"
        "  -v, --verbose      print every candidate, not just the winner\n"
        "  -h, --help\n");
}

/* A device allocation that frees itself, because the timing path has several ways out. Neither
 * copyable nor movable: a rank holds a vector of these, and a second handle to the same
 * allocation would free the buffer out from under a launch still reading it. */
struct DevBuf {
    void* p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&&) = delete;
    DevBuf& operator=(DevBuf&&) = delete;
    ~DevBuf() { if (p) rad_dev_free(p, RAD_MEM_DEVICE); }
};

/* Maximum relative difference from the defaults. A different split-K count reorders the reduction
 * and moves the last bits (at K=5120 in fp32); 1e-3 is two orders above that and far below a
 * transposed tile or a dropped k slice. */
constexpr double kVariantTol = 1e-3;

/* The candidates: every legal point of the kernel's declared space at this geometry, as the cross
 * product of its axes (a hand-written list of points can silently have holes). `tune_valid` prunes
 * points this geometry cannot run, e.g. a split that does not divide K; a pruned point is a skip,
 * not a failure.
 *
 * The all-defaults point, the baseline an untuned install runs, is moved to index 0 and
 * `*has_defaults` set. It is absent only when tune_valid pruned it.
 */
enum { kMaxCandidates = 4096 };

static std::vector<std::vector<std::pair<std::string, int64_t>>>
tune_candidates(const RadKernelInfo* k, const Geometry& g, int64_t* space_size,
                bool* has_defaults) {
    std::vector<std::vector<std::pair<std::string, int64_t>>> out;
    *has_defaults = false;
    if (!k || k->n_tunables <= 0) return out;

    int64_t total = 1;
    for (int i = 0; i < k->n_tunables; ++i) total *= (int64_t)k->tunables[i].n_values;
    if (space_size) *space_size = total;
    if (total <= 0 || total > (int64_t)kMaxCandidates) return out;

    std::vector<RadParam> cp((size_t)k->n_tunables);
    std::vector<int> idx((size_t)k->n_tunables, 0);
    std::vector<std::pair<std::string, int64_t>> point((size_t)k->n_tunables);

    for (int64_t n = 0; n < total; ++n) {
        int64_t rem = n;
        for (int i = k->n_tunables - 1; i >= 0; --i) {
            idx[(size_t)i] = (int)(rem % k->tunables[i].n_values);
            rem /= k->tunables[i].n_values;
        }
        for (int i = 0; i < k->n_tunables; ++i) {
            point[(size_t)i].first  = k->tunables[i].key;
            point[(size_t)i].second = k->tunables[i].values[idx[(size_t)i]];
            cp[(size_t)i].key  = k->tunables[i].key;
            cp[(size_t)i].kind = RAD_P_INT;
            cp[(size_t)i].ival = point[(size_t)i].second;
        }
        if (k->tune_valid &&
            k->tune_valid(g.params(), g.n_params(), cp.data(), (int)cp.size()) != 1) continue;
        out.push_back(point);
    }

    /* The loader requires each default to be one of its axis's values, so the all-defaults point
     * is in the product unless tune_valid pruned it. */
    for (size_t i = 0; i < out.size(); ++i) {
        bool all_default = true;
        for (int a = 0; a < k->n_tunables; ++a)
            if (out[i][(size_t)a].second != k->tunables[a].deflt) { all_default = false; break; }
        if (all_default) { std::swap(out[0], out[i]); *has_defaults = true; break; }
    }
    return out;
}

/* One rank's side of one candidate's launch: the instance its declare made, its operands on its
 * card, its scratch and its stream. A plain row has one; a collective has one per rank, fired
 * together, since a rank launching alone waits for a peer flag nobody raises. The instance is the
 * only route to the peer bundle (scratch, flags, sequence counters) init() builds, so a
 * collective cannot launch without it. */
struct Rank {
    int       rank = 0, world = 1;
    void*     instance = nullptr;
    RadStream stream = nullptr;              /* owned by the request's StreamSet */
    std::vector<RadTensor>            t;     /* the operands, as the kernel sees them */
    std::vector<DevBuf>               buf;   /* ...and the allocations behind them */
    std::vector<std::vector<uint8_t>> host;  /* the draw, only ever a copy source */
    DevBuf    scratch;
    int64_t   scratch_bytes = 0;
};

/* Streams die with the request, every way out of it, skips included. */
struct StreamSet {
    std::vector<RadStream> s;
    ~StreamSet() { for (RadStream x : s) if (x) rad_stream_destroy(x); }
};

/* The RadArgs a rank's launch takes. The candidate's axes ride in `g` exactly as at issue, so the
 * kernel reads `sk` with rad_args_geti() like `M`. */
void fill_args(const KernelRow& row, Rank& r, const Geometry& g, RadArgs* a) {
    std::memset(a, 0, sizeof(*a));
    a->t = r.t.data();  a->n_t = (int)r.t.size();
    a->p = g.params();  a->n_p = g.n_params();
    a->instance   = r.instance;
    a->rank       = r.rank;
    a->world_size = r.world;
    if (r.scratch.p) { a->scratch = r.scratch.p; a->scratch_bytes = r.scratch_bytes; }
}

/* Every rank's stream drained. A stream's device has to be current to sync it. */
int sync_all(std::vector<Rank>& rs) {
    int st = RAD_OK;
    for (Rank& r : rs) {
        if (!r.stream) continue;
        if (rad_dev_set(r.rank) < 0) return RAD_E_DEVICE;
        if (rad_stream_sync(r.stream) < 0) st = RAD_E_STATE;
    }
    return st;
}

/* Scratch zeroed before a candidate starts, outside the timed region: a candidate must not be
 * timed -- or judged -- on the leftovers of the one before it. */
int zero_scratch(std::vector<Rank>& rs) {
    for (Rank& r : rs) {
        if (!r.scratch.p) continue;
        if (rad_dev_set(r.rank) < 0) return RAD_E_DEVICE;
        if (rad_memset_async(r.scratch.p, 0, (uint64_t)r.scratch_bytes, r.stream) < 0)
            return RAD_E_STATE;
    }
    return RAD_OK;
}

/* `iters` launches on every rank. A collective's ranks launch from one thread each: the exchange
 * pipelines only once every card has work queued. A single rank launches inline, so the timed
 * region holds no thread start.
 *
 * A refusal is symmetric (every rank resolved the same kernel for the same geometry, and a launch
 * checks only the geometry and the instance), so no rank has queued work to drain; the caller
 * treats it as fatal for the candidate. */
int launch_all(const KernelRow& row, std::vector<Rank>& rs, const Geometry& g, int iters) {
    auto run = [&](Rank& r) {
        if (rad_dev_set(r.rank) < 0) return (int)RAD_E_DEVICE;
        RadArgs a;
        fill_args(row, r, g, &a);
        for (int k = 0; k < iters; ++k)
            if (const int s = row.info->launch(&a, r.stream); s < 0) return s;
        return (int)RAD_OK;
    };
    if (rs.size() == 1) return run(rs[0]);
    std::vector<int> st(rs.size(), (int)RAD_E_STATE);
    std::vector<std::thread> th;
    th.reserve(rs.size());
    for (size_t i = 0; i < rs.size(); ++i) th.emplace_back([&, i] { st[i] = run(rs[i]); });
    for (std::thread& t : th) t.join();
    for (int s : st) if (s < 0) return s;
    return RAD_OK;
}

/* Rank 0's OUT and INOUT bytes after one launch at `g`, which carries the candidate point as
 * ordinary parameters. Every candidate saw the same bytes on every rank, so a variant that
 * reorders a reduction, drops a peer's term, or writes the wrong slice differs from the defaults
 * here. */
int capture_outputs(const KernelRow& row, std::vector<Rank>& rs, const Geometry& g,
                    const RadOpSchema* schema, const std::vector<OpdShape>& sh,
                    std::vector<uint8_t>* out) {
    if (const int z = zero_scratch(rs)) return z;
    const int w = launch_all(row, rs, g, 1);
    if (w < 0) return w;
    if (const int s = sync_all(rs)) return s;

    Rank& r = rs[0];
    if (rad_dev_set(r.rank) < 0) return RAD_E_DEVICE;
    /* Size the destination before copying: growing it between async copies would reallocate it
     * under copies still in flight. */
    size_t total = 0;
    for (size_t k = 0; k < sh.size() && k < (size_t)schema->n_operands; ++k) {
        const int role = schema->operands[k].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        if (!r.t[k].data) continue;
        const int64_t bytes = rad_dtype_bytes(sh[k].dtype, rad_numel(sh[k].shape));
        if (bytes > 0) total += (size_t)bytes;
    }
    out->assign(total, 0);
    size_t at = 0;
    for (size_t k = 0; k < sh.size() && k < (size_t)schema->n_operands; ++k) {
        const int role = schema->operands[k].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        if (!r.t[k].data) continue;
        const int64_t bytes = rad_dtype_bytes(sh[k].dtype, rad_numel(sh[k].shape));
        if (bytes <= 0) continue;
        if (rad_memcpy_async(out->data() + at, r.t[k].data, bytes, r.stream) < 0)
            return RAD_E_STATE;
        at += (size_t)bytes;
    }
    if (rad_stream_sync(r.stream) < 0) return RAD_E_STATE;
    return RAD_OK;
}

/* The largest relative difference over the captured outputs, widened to f32 through each operand's
 * dtype. */
double max_rel_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                    const std::vector<OpdShape>& sh, const RadOpSchema* schema) {
    if (a.size() != b.size() || a.empty()) return a.size() == b.size() ? 0.0 : 1.0;
    double worst = 0.0;
    size_t at = 0;
    for (size_t k = 0; k < sh.size() && k < (size_t)schema->n_operands; ++k) {
        const int role = schema->operands[k].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        const int64_t n = rad_numel(sh[k].shape);
        const int64_t bytes = rad_dtype_bytes(sh[k].dtype, n);
        if (bytes <= 0 || at + (size_t)bytes > a.size()) break;
        std::vector<float> fa, fb;
        rad_widen(a.data() + at, sh[k].dtype, n, &fa);
        rad_widen(b.data() + at, sh[k].dtype, n, &fb);
        if ((int64_t)fa.size() < n || (int64_t)fb.size() < n) break;
        double num = 0, den = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double d = (double)fa[(size_t)i] - (double)fb[(size_t)i];
            num += d * d;
            den += (double)fa[(size_t)i] * (double)fa[(size_t)i];
        }
        const double rel = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
        /* Return NaN directly: `rel > worst` is false for it, so a max would drop it. The caller's
         * `!(err <= tol)` rejects it. */
        if (std::isnan(rel)) return rel;
        if (rel > worst) worst = rel;
        at += (size_t)bytes;
    }
    return worst;
}

/* Median microseconds per iteration at `g` (which carries the candidate point): the wall time of
 * `iters` launches on every rank, from the first launch to the last stream drained. Negative
 * means the kernel refused the shape at its entry point (spec 2.1), which is allowed: tune_valid
 * prunes only what the geometry alone can rule out.
 *
 * The clock stops only after the devices finish: launches are asynchronous, so without the sync
 * this would time host submission. For a collective that is doubly true -- the exchange's cost is
 * the wait for the peer, and the wait happens on the card, so a rank timed alone would report a
 * local copy. */
double time_point(const KernelRow& row, std::vector<Rank>& rs, const Geometry& g,
                  int reps, int iters, double* spread_pct) {
    if (const int z = zero_scratch(rs)) return (double)z;

    /* A warm-up launch; its status decides whether this candidate serves the shape at all. */
    const int w = launch_all(row, rs, g, 1);
    if (w < 0) return (double)w;
    if (const int s = sync_all(rs)) return (double)s;

    std::vector<double> us;
    us.reserve((size_t)reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (const int s = launch_all(row, rs, g, iters)) return (double)s;
        if (const int s = sync_all(rs)) return (double)s;
        const auto t1 = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
    }
    std::sort(us.begin(), us.end());
    if (spread_pct)
        *spread_pct = us[us.size() / 2] > 0
                    ? 100.0 * (us.back() - us.front()) / us[us.size() / 2] : 0.0;
    return us[us.size() / 2];
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string model, home = rad_home(), op_filter, machine;
    std::vector<std::string> hierarchy = rad_hierarchy_from_env();
    int      reps = 5, iters = 32;
    int64_t  max_tok = 8192;
    int64_t  max_seqs = 256;
    int      tp = 1, spec = -1;          /* -1 = the container's own drafter depth */
    int      kv_dtype = RAD_DT_INVALID;  /* RAD_I8 = the engine's --kv-cache-dtype i8 */
    int      wire_lossy = 0;             /* the engine's --tp-wire wht6 */
    int64_t  wire_min_kb = 128;          /* the engine's --tp-wire-min-kb */
    bool     dry = false, verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* w) { return rad_flag_value("rad-tune", argc, argv, &i, w); };
        auto num = [&](const char* w, int64_t lo, int64_t hi) {
            return rad_flag_int("rad-tune", w, next(w), lo, hi);
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-m" || a == "--model") model = next("--model");
        else if (a == "--home")    home = next("--home");
        else if (a == "--kernels") hierarchy = rad_split_list(next("--kernels"));
        else if (a == "--op")      op_filter = next("--op");
        else if (a == "--machine") machine = next("--machine");
        else if (a == "--reps")    reps = (int)num("--reps", 1, INT32_MAX);
        else if (a == "--iters")   iters = (int)num("--iters", 1, INT32_MAX);
        else if (a == "--max-tok") max_tok = num("--max-tok", 1, INT64_MAX);
        else if (a == "--max-seqs") max_seqs = num("--max-seqs", 1, INT64_MAX);
        else if (a == "--tp")      tp   = (int)num("--tp", 1, INT32_MAX);
        else if (a == "--kv-cache-dtype") {
            const char* v = next("--kv-cache-dtype");
            if (!std::strcmp(v, "i8"))       kv_dtype = RAD_I8;
            else if (!std::strcmp(v, "bf16")) kv_dtype = RAD_DT_INVALID;
            else { std::fprintf(stderr, "rad-tune: --kv-cache-dtype: '%s' is not one of "
                                "bf16, i8 (the tuner tunes the bands it declares; other "
                                "encodings are the engine's)\n", v); return 2; }
        }
        else if (a == "--tp-wire") {
            const char* v = next("--tp-wire");
            if (!std::strcmp(v, "wht6"))        wire_lossy = 1;
            else if (!std::strcmp(v, "exact"))  wire_lossy = 0;
            else { std::fprintf(stderr, "rad-tune: --tp-wire: '%s' is not one of exact, wht6\n",
                                v); return 2; }
        }
        else if (a == "--tp-wire-min-kb") wire_min_kb = num("--tp-wire-min-kb", 0, INT64_MAX);
        else if (a == "--num-speculative-tokens") {
            /* `auto` is the container's own depth, as the engine's flag of this name reads it. */
            const char* v = next("--num-speculative-tokens");
            spec = std::string(v) == "auto"
                       ? -1 : (int)rad_flag_int("rad-tune", "--num-speculative-tokens", v, 0,
                                                INT32_MAX);
        }
        else if (a == "--dry-run") dry = true;
        else if (a == "-v" || a == "--verbose") verbose = true;
        else { std::fprintf(stderr, "rad-tune: unrecognised argument '%s'. Try --help.\n",
                            a.c_str()); return 2; }
    }
    if (model.empty()) { usage(); return 2; }

    RadFile f;
    if (f.open(model.c_str()) < 0) return 1;

    LoadedPlugins plugins;
    if (rad_tools_load_plugins(home, hierarchy, f.str(f.header().arch_id),
                               f.str(f.header().quant), &plugins) < 0) return 1;

    const ContainerMeta cm(f);
    const RadModelMeta& meta = cm.meta;

    /* Declare the graph the deployment runs: the tensor-parallel width sets every sharded N, and
     * the speculative depth decides whether the drafter (and its shapes) is in the graph. */
    RadBuildCtx ctx{};
    ctx.rank = 0; ctx.world_size = tp; ctx.max_tok = max_tok;
    ctx.max_seqs = max_seqs;
    ctx.tp_wire_lossy = wire_lossy;
    ctx.tp_wire_min_bytes = wire_min_kb * 1024;
    ctx.kv_dtype = kv_dtype;
    RadArchProbe probe{};
    (void)rad_tools_probe(plugins, meta, &probe);
    ctx.max_spec = spec >= 0 ? spec : probe.draft_depth;
    /* Trees where the engine would ask for them (Engine::declare: the plugin's probe says they
     * pay and a drafter decodes speculatively): the verify's tree rows and the wider block are
     * then the graph that is tuned. */
    ctx.draft_tree = ctx.max_spec > 0 && probe.tree_pays != 0;
    ctx.max_seqs = max_seqs; ctx.max_ctx = meta.n_ctx_train; ctx.scope = "";

    /* Peer access, measured before selection, as Engine does: collectives constrain device.p2p
     * (abi/rad_types.h), and with the fact unset neither the peer-to-peer row nor its host-memory
     * twin can match, so every collective resolves to nothing -- silently, with no tune request
     * for it ever reaching the loop below. Every ordered pair is asked, since access is granted
     * one way at a time. */
    long long p2p = 1;
    for (int i = 0; i < tp; ++i)
        for (int j = 0; j < tp; ++j) {
            if (i == j) continue;
            const int a = rad_dev_peer_access(i, j);
            if (a < 0) {
                std::fprintf(stderr, "rad-tune: could not ask whether card %d can reach card %d's "
                                     "memory: %s\n", i, j, rad_dev_last_error());
                return 1;
            }
            if (a == 0) p2p = 0;
        }
    rad_tools_registry().set_device_fact(RAD_FACT_P2P, p2p);
    const char* p2p_note = p2p ? "peer to peer" : "the collectives run through pinned host memory";

    /* Every rank declares concurrently on its own thread and device, as Engine does: kernel init()
     * runs from declare, and a collective's init is a rendezvous (libr4d's all-reduce waits for its
     * peers), so declaring rank 0 alone would hang.
     *
     * All the programs stay alive, not only rank 0's. A collective's launch needs the instance each
     * rank's own declare made, and an instance is reachable only through the program that declared
     * it, so rank 0's program is the one measured while the others supply the peers. */
    std::vector<Program> progs((size_t)tp);
    {
        std::vector<int>     st((size_t)tp, 0);
        std::vector<std::thread> th;
        for (int ri = 0; ri < tp; ++ri) {
            th.emplace_back([&, ri] {
                if (tp > 1 && rad_dev_set(ri) < 0) {
                    std::fprintf(stderr, "rad-tune: --tp %d needs %d devices; there is no device "
                                         "%d to declare rank %d on\n", tp, tp, ri, ri);
                    st[(size_t)ri] = RAD_E_DEVICE;
                    return;
                }
                RadBuildCtx bc = ctx;
                bc.rank = ri;
                st[(size_t)ri] = rad_tools_declare(plugins, meta, bc, &progs[(size_t)ri], {},
                                                   nullptr, rad_container_sources(f));
            });
        }
        for (auto& t : th) t.join();
        if (tp > 1) rad_dev_set(0);
        for (int ri = 0; ri < tp; ++ri) if (st[(size_t)ri] < 0) return 1;
    }
    Program& prog = progs[0];

    /* Name what resolved to nothing, so a hole is a printed line instead of a tune request that
     * never appears: the engine's declare counts the misses, and this is the count's detail. */
    if (!prog.misses.empty()) {
        RadDeviceProps dp{};
        const bool host_backend = rad_dev_props(0, &dp) >= 0 && dp.is_host_backend;
        for (const Program::Miss* m : prog.device_holes(host_backend))
            std::printf("  unresolved  %s\n", m->line().c_str());
    }

    if (machine.empty()) machine = rad_machine_key();

    /* Load from the first home on the search path with a cache for this machine; save to the
     * first home on the path. */
    const std::string rel = TuneCache::rel_path(machine);
    const std::string src = rad_home_file(home, rel);
    const std::string save_home = rad_home_dirs(home).front();
    TuneCache cache;
    cache.load(src.substr(0, src.size() - rel.size() - 1), machine);   /* missing is normal */
    const size_t before = cache.rows().size();

    std::printf("rad-tune %s\n", model.c_str());
    std::printf("  machine      %s\n", machine.c_str());
    if (tp > 1) std::printf("  p2p          %s\n", p2p_note);
    std::printf("  cache        %s (%zu existing row(s))\n", src.c_str(), before);
    std::printf("  requests     %zu (kernel, shape) pair(s) the declared graph reaches\n\n",
                prog.tune_requests.size());

    Registry& reg = rad_tools_registry();
    auto row_for = [&](const std::string& plugin, const std::string& kernel) -> const KernelRow* {
        for (const KernelRow& r : reg.rows())
            if (r.plugin == plugin && r.info && kernel == r.info->name) return &r;
        return nullptr;
    };

    /* tune_requests lacks the operand list, so schemas are looked up by the kernel's op name. */
    int measured = 0, skipped = 0, single = 0;
    std::map<std::string, int> skip_why;

    for (const Program::TuneRequest& tr : prog.tune_requests) {
        const KernelRow* row = row_for(tr.plugin, tr.kernel);
        if (!row || !row->info || !row->info->launch) {
            skip_why[tr.plugin + "/" + tr.kernel + "  (no such kernel row)"]++;
            ++skipped;
            continue;
        }
        const std::string op = row->info->op;
        if (!op_filter.empty() && !op.starts_with(op_filter)) continue;

        /* A kernel with at most one candidate has nothing to choose; no row is written. */
        int64_t space = 0;
        bool has_defaults = false;
        const auto cands = tune_candidates(row->info, tr.geom, &space, &has_defaults);
        const int n_var = (int)cands.size();
        if (n_var <= 1) { ++single; continue; }
        if (!has_defaults) {
            skip_why[tr.plugin + "/" + tr.kernel + "  (tune_valid rejects the defaults: no "
                     "reference to check candidates against)"]++;
            ++skipped;
            continue;
        }

        const RadOpSchema* schema = reg.schema(op);
        if (!schema) { skip_why[op + "  (no schema)"]++; ++skipped; continue; }

        int64_t M = 1;
        long long mv = 0;
        for (const char* k : { "M", "q_len" })
            if (tr.geom.get_i(k, &mv)) { M = (int64_t)mv; break; }
        if (M <= 0) M = 1;

        /* Operand shapes as rad-kbench takes them: the reference library's description (which
         * defines the op), then the kernel's own, then the fallback table in opshapes.h. */
        std::vector<OpdShape> sh;
        int shst = RAD_E_UNSUPPORTED;
        for (const Plugin& p : reg.plugins()) {
            if (p.kind != RAD_PLUGIN_KERNEL || !p.reference) continue;
            if (const KernelRow* rr = reg.select_in(p.name, op, tr.geom, RAD_DOMAIN_HOST))
                shst = rad_operand_shapes_hook(rr->info, tr.geom, schema->n_operands, &sh);
            break;
        }
        if (shst < 0) shst = rad_operand_shapes_hook(row->info, tr.geom, schema->n_operands, &sh);
        if (shst < 0) shst = rad_operand_shapes(op, tr.geom, schema->n_operands, M, &sh);
        if (shst < 0) {
            skip_why[op + "  (neither the reference nor the kernel describes its operands, and "
                          "tools/opshapes.h has no recipe)"]++;
            ++skipped;
            continue;
        }

        if (dry) {
            std::printf("  %-22s %-46s %s/%s  %d of %lld point(s)\n", op.c_str(),
                        tr.geom.str().c_str(), tr.plugin.c_str(), tr.kernel.c_str(), n_var,
                        (long long)space);
            continue;
        }

        /* Inputs come from kf_draw, the same draw rad-kbench checks with, so each operand stays in
         * the domain its recipe states (constant `seqused`, cumulative lengths, distinct scatter
         * destinations, in-range block tables) and candidates are timed on realistic work. */
        std::vector<KfOperand> kop(sh.size());
        for (size_t k = 0; k < sh.size(); ++k) {
            KfOperand& d = kop[k];
            d.role       = k < (size_t)schema->n_operands ? schema->operands[k].role : RAD_OPD_IN;
            d.dtype      = sh[k].dtype;
            d.shape      = sh[k].shape;
            d.absent     = sh[k].absent;
            d.fill       = sh[k].fill;
            d.is_index   = sh[k].is_index;
            d.idx_unique = sh[k].idx_unique;
            d.idx_cu     = sh[k].idx_cu;
            d.idx_prefix = sh[k].idx_prefix;
            d.idx_history = sh[k].idx_history;
            d.idx_max    = sh[k].idx_max;
            d.idx_const  = sh[k].idx_const;
            d.fill_chunk = sh[k].fill_chunk;
        }
        const int64_t vocab_rows = kf_vocab_rows(kop);

        /* One set of operands per rank: a collective's ranks contribute their own bytes on their
         * own card, with the instance their own declare made. Each rank draws different bytes, so
         * a kernel that read its own contribution twice cannot pass. `rs` is sized once and never
         * grown: a moved DevBuf would leave its old handle to free the buffer too. */
        const int world = row->collective && tp > 1 ? tp : 1;
        std::string scale;
        if (world > 1) scale = ", " + std::to_string(world) + " ranks";
        std::vector<Rank> rs((size_t)world);
        StreamSet streams;
        bool ok = true;
        for (int ri = 0; ri < world && ok; ++ri) {
            Rank& r = rs[(size_t)ri];
            r.rank = ri;  r.world = world;
            /* The instance is found by the key the cache uses, not by position: a rank that
             * resolved a different kernel for this band has no peer to exchange with. A
             * collective cannot launch without it; a plain kernel has none, and null is exactly
             * what its launch expects. */
            const Program::TuneRequest* q = &tr;
            if (ri > 0) {
                q = nullptr;
                for (const auto& p : progs[(size_t)ri].tune_requests)
                    if (p.kernel == tr.kernel && p.plugin == tr.plugin &&
                        p.geom_key == tr.geom_key) { q = &p; break; }
            }
            if (world > 1 && (!q || !q->instance)) {
                skip_why[op + "  (rank " + std::to_string(ri) + " declared no instance for this "
                         "instantiation: no exchange to launch)"]++;
                ++skipped; ok = false; break;
            }
            r.instance = q ? q->instance : nullptr;
            if (rad_dev_set(ri) < 0) {
                skip_why[op + "  (no device to put rank " + std::to_string(ri) + " on)"]++;
                ++skipped; ok = false; break;
            }
            /* An explicit stream per rank; timing on the null stream would measure host
             * submission. Without ROCm the device layer is a synchronous host stub, and the
             * same path still works. */
            RadStream s = nullptr;
            if (rad_stream_create(&s, 0) < 0) {
                std::fprintf(stderr, "rad-tune: could not create a stream for rank %d: %s\n", ri,
                             rad_dev_last_error());
                return 1;
            }
            streams.s.push_back(s);
            r.stream = s;
            /* Inputs come from kf_draw, the same draw rad-kbench checks with, so each operand
             * stays in the domain its recipe states (constant `seqused`, cumulative lengths,
             * distinct scatter destinations, in-range block tables). */
            r.host.resize(sh.size());
            r.buf = std::vector<DevBuf>(sh.size());
            r.t   = std::vector<RadTensor>(sh.size());
            for (size_t k = 0; k < sh.size(); ++k) {
                /* An absent operand (attn's descales and cu_seqlens) is a zeroed tensor: no
                 * draw, no allocation, data null -- what kbench passes and the kernel checks. */
                if (sh[k].absent) continue;
                const int64_t bytes = rad_dtype_bytes(sh[k].dtype, rad_numel(sh[k].shape));
                if (bytes <= 0) {
                    skip_why[op + "  (an operand has a plugin-private dtype the core cannot "
                             "size)"]++;
                    ++skipped; ok = false; break;
                }
                kf_draw(kop, k, M, vocab_rows,
                        rad_case_seed(1, op, ri, row->info->domain, (int)k), &r.host[k]);
                r.buf[k].p = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
                if (!r.buf[k].p) {
                    skip_why[op + "  (no device memory for an operand on rank " +
                             std::to_string(ri) + ")"]++;
                    ++skipped; ok = false; break;
                }
                if (rad_memcpy_async(r.buf[k].p, r.host[k].data(), bytes, r.stream) < 0) {
                    skip_why[op + "  (the operands would not copy onto rank " +
                             std::to_string(ri) + ")"]++;
                    ++skipped; ok = false; break;
                }
                /* RadTensor::data must be the device allocation; a host address fails silently. */
                RadTensor& t = r.t[k];
                t = RadTensor{};
                t.data  = r.buf[k].p;
                t.dtype = sh[k].dtype;
                t.rank  = (uint32_t)sh[k].shape.size();
                for (size_t d = 0; d < sh[k].shape.size() && d < RAD_MAX_RANK; ++d)
                    t.shape[d] = sh[k].shape[d];
                rad_tensor_pack(&t);
            }
        }
        /* Scratch sized from each rank's own arguments: a split kernel writes its partials to
         * device memory, and every card needs its own. */
        if (ok && row->info->scratch) {
            for (Rank& r : rs) {
                if (rad_dev_set(r.rank) < 0) {
                    skip_why[op + "  (no device to size scratch on for rank " +
                             std::to_string(r.rank) + ")"]++;
                    ++skipped; ok = false; break;
                }
                RadArgs a;
                fill_args(*row, r, tr.geom, &a);
                const int64_t need = row->info->scratch(&a);
                if (need <= 0) continue;
                r.scratch.p = rad_dev_alloc(need, RAD_MEM_DEVICE);
                if (!r.scratch.p) {
                    skip_why[op + "  (no device memory for the scratch it asks for)"]++;
                    ++skipped; ok = false; break;
                }
                r.scratch_bytes = need;
            }
        }
        if (!ok) continue;                    /* the reason is already counted */
        if (world > 1) rad_dev_set(0);

        /* One geometry per candidate, built up front. As at issue, the chosen values are appended
         * to the resolved geometry, so a launch reads `sk` via rad_args_geti() like `M`. */
        std::vector<Geometry> gv((size_t)n_var);
        for (int v = 0; v < n_var; ++v) {
            gv[(size_t)v] = tr.geom;
            for (const auto& kv : cands[(size_t)v])
                gv[(size_t)v].set_i(kv.first.c_str(), kv.second);
        }

        /* Median over `reps`, with the spread. */
        std::vector<double> best_us((size_t)n_var, -1.0), spread((size_t)n_var, 0.0);
        for (int v = 0; v < n_var; ++v) {
            double sp = 0;
            best_us[(size_t)v] = time_point(*row, rs, gv[(size_t)v], reps, iters, &sp);
            spread[(size_t)v] = sp;
        }

        /* ---- each candidate must agree with the defaults before it can win on speed.
         *
         * No other check exercises non-default tile configurations, and a subtly wrong winner
         * would be served from the cache, showing up only as degraded output quality. Candidate 0,
         * the all-defaults point (what an untuned install runs, spec §15), is the reference.
         * kVariantTol allows the last-bit changes of a reordered reduction. */
        std::vector<uint8_t> ref;
        std::vector<double>  verr((size_t)n_var, 0.0);
        if (best_us[0] <= 0 || capture_outputs(*row, rs, gv[0], schema, sh, &ref) < 0) {
            skip_why[tr.plugin + "/" + tr.kernel + "  (the defaults refused the shape: no "
                     "reference to check candidates against)"]++;
            ++skipped;
            continue;
        }
        for (int v = 1; v < n_var; ++v) {
            if (best_us[(size_t)v] <= 0) continue;
            std::vector<uint8_t> got;
            if (capture_outputs(*row, rs, gv[(size_t)v], schema, sh, &got) < 0) {
                best_us[(size_t)v] = (double)RAD_E_STATE;
                continue;
            }
            verr[(size_t)v] = max_rel_diff(ref, got, sh, schema);
            if (!(verr[(size_t)v] <= kVariantTol)) best_us[(size_t)v] = (double)RAD_E_SHAPE;
        }

        int win = -1;
        for (int v = 0; v < n_var; ++v)
            if (best_us[(size_t)v] > 0 && (win < 0 || best_us[(size_t)v] < best_us[(size_t)win]))
                win = v;

        if (win < 0) {
            std::printf("  %-22s %-46s no candidate served it (%s/%s, %d tried of %lld)\n",
                        op.c_str(), tr.geom.str().c_str(), tr.plugin.c_str(), tr.kernel.c_str(),
                        n_var, (long long)space);
            ++skipped;
            continue;
        }

        const std::string wname = TuneCache::canon_choice(cands[(size_t)win]);

        if (verbose)
            for (int v = 0; v < n_var; ++v) {
                const std::string nm = TuneCache::canon_choice(cands[(size_t)v]);
                if (best_us[(size_t)v] > 0)
                    std::printf("      %-32s %9.3f us  spread %4.0f%%  vs default %8.2g%s\n",
                                nm.c_str(), best_us[(size_t)v], spread[(size_t)v], verr[(size_t)v],
                                v == win ? "  <-" : "");
                else if (!(verr[(size_t)v] <= kVariantTol))
                    std::printf("      %-32s rejected: disagrees with the defaults by %.3g\n",
                                nm.c_str(), verr[(size_t)v]);
                else
                    std::printf("      %-32s refused: %s\n", nm.c_str(),
                                rad_strerror((int)best_us[(size_t)v]));
            }

        TuneRow tr_out;
        tr_out.kernel   = tr.kernel;
        tr_out.plugin   = tr.plugin;
        tr_out.choice   = wname;
        /* Key on the geometry without the point (the chosen axes are in `choice`): the selector
         * looks the instantiation up before it knows what to choose. */
        tr_out.geom     = tr.geom_key;
        tr_out.us       = best_us[(size_t)win];
        tr_out.measured = TuneCache::now_iso8601();
        cache.put(tr_out);
        ++measured;

        std::printf("  %-22s %-46s %-28s %9.3f us  (%d of %lld%s)\n", op.c_str(),
                    tr.geom.str().c_str(), wname.c_str(), tr_out.us, n_var, (long long)space,
                    scale.c_str());
    }

    if (dry) {
        std::printf("\n--dry-run: nothing measured, nothing written.\n");
        return 0;
    }

    if (!skip_why.empty()) {
        std::printf("\nskipped\n");
        for (const auto& [what, n] : skip_why) std::printf("  %-70s x%d\n", what.c_str(), n);
        std::printf("  %d request(s) skipped in total.\n", skipped);
    }
    if (single)
        std::printf("\n%d request(s) had one point or none: nothing to choose, and the kernel's\n"
                    "own axis defaults are already what it runs.\n", single);

    cache.set_machine(machine);
    if (cache.save(save_home) < 0) return 1;

    std::printf("\n%d shape(s) measured, %zu row(s) in %s\n", measured, cache.rows().size(),
                TuneCache::path_for(save_home, machine).c_str());
    return 0;
}
