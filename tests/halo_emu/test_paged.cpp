/* tests/halo_emu: libhalo's paged attention rows (attn_paged: the decode and prefill bodies,
 * halo_attn_decode_h256_gqa6.hip and halo_attn_prefill_h256_gqa6.hip, through every unit that owns a
 * geometry) on the host SIMT emulator, against test_attn.cpp's port of libref's ref_attn_paged. The
 * operand construction and the reference are test_attn's (a shuffled block table with a spare
 * column, NaN in every slot past a context and every unnamed block, f64 accumulation); the rows come
 * from the row table (paged_rows.inc: every block-size-16 decode and prefill row, its launch, its
 * scratch hook and the q_len / window / causal its constraints allow). Decode at one token and at
 * verify heights, short and split-KV contexts, ragged batches, windows where a row takes one,
 * fp8 caches with and without per-head descales; prefill over ragged causal batches. */

#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(1234);
extern int emu_grids;
extern int emu_last_block;   /* threads a block of the last grid: 512 is the WMMA kernel */
static float nrm() { static std::normal_distribution<float> d(0.f, 1.f); return d(rng); }

static RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    return t;
}

using Fn = int (*)(const RadArgs*, RadStream);
struct Row {
    const char* name;
    Fn fn;
    int64_t (*scratch)(const RadArgs*);
    int hd, gqa;
    bool fp8;
    int qmax;        /* the row's q_len bound */
    int win;         /* 0: window EQ 0, 1: window GE 0, 2: window GE 1 */
    int causal1;     /* the row requires causal */
};
#include "paged_rows.inc"

struct Case {
    std::string name;
    int hd, gqa, kvh;
    bool fp8, causal, ragged, descale;
    int window;
    std::vector<int> qlen, ctx;          /* per sequence */
    int missing_block = -1;              /* sequence 0's block index to mark -1, or -1 */
    long long max_ctx = 0;               /* 0: not passed */
    int64_t scratch = 0;                 /* 0: no scratch (rows without a hook) */
    int qpitch_extra = 0;                /* q as a column slice of a wider buffer */
};

static int run(const Row& r, const Case& c) {
    const int n_seq = (int)c.qlen.size(), HD = c.hd, kvh = c.kvh, qh = kvh * c.gqa;
    int rows = 0; for (int q : c.qlen) rows += q;
    /* blocks: each sequence gets ceil(ctx/16) blocks at shuffled ids */
    int maxb = 0; for (int x : c.ctx) maxb = std::max(maxb, (x + 15) / 16);
    maxb += 1;  /* a spare column: the table is wider than any sequence needs */
    const int nblocks = n_seq * maxb + 3;
    std::vector<int> ids(nblocks); for (int i = 0; i < nblocks; ++i) ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int> bt(n_seq * maxb, -1);
    int nx = 0;
    for (int s = 0; s < n_seq; ++s) for (int b = 0; b < (c.ctx[s] + 15) / 16; ++b) bt[s * maxb + b] = ids[nx++];
    if (c.missing_block >= 0) bt[c.missing_block] = -1;
    /* the cache, filled everywhere -- including slots past each context, with NaN in some, so a
     * kernel that reads a masked key's V without guarding is caught */
    const int64_t slot = 2 * HD, nel = (int64_t)nblocks * kvh * 16 * slot;
    std::vector<float> kvf(nel);
    std::vector<uint16_t> kvb(nel); std::vector<uint8_t> kv8(nel);
    for (int64_t i = 0; i < nel; ++i) {
        float v = nrm();
        if (c.fp8) { kv8[i] = rad_f32_to_fp8e4m3(v * 4.f); kvf[i] = rad_fp8e4m3_to_f32(kv8[i]); }
        else { kvb[i] = rad_f32_to_bf16(v); kvf[i] = rad_bf16_to_f32(kvb[i]); }
    }
    /* poison: every slot at or past a sequence's context in its last block, and every block no
     * table names */
    std::vector<char> named(nblocks, 0);
    for (int s = 0; s < n_seq; ++s) for (int b = 0; b < maxb; ++b) if (bt[s * maxb + b] >= 0) named[bt[s * maxb + b]] = 1;
    auto poison = [&](int64_t i) { if (c.fp8) { kv8[i] = 0x7f; } else { kvb[i] = 0x7fc0; } kvf[i] = NAN; };
    for (int blk = 0; blk < nblocks; ++blk) if (!named[blk]) for (int64_t i = 0; i < kvh * 16 * slot; ++i) poison((int64_t)blk * kvh * 16 * slot + i);
    for (int s = 0; s < n_seq; ++s) {
        const int last = (c.ctx[s] - 1) / 16, blk = bt[s * maxb + last];
        if (blk < 0) continue;
        for (int o = c.ctx[s] % 16 ? c.ctx[s] % 16 : 16; o < 16; ++o)
            for (int h = 0; h < kvh; ++h) for (int d = 0; d < slot; ++d) poison((((int64_t)blk * kvh + h) * 16 + o) * slot + d);
    }
    const int qcols = qh * HD + c.qpitch_extra;
    std::vector<uint16_t> qb((size_t)rows * qcols); std::vector<float> qf(qb.size());
    for (size_t i = 0; i < qb.size(); ++i) { qb[i] = rad_f32_to_bf16(nrm()); qf[i] = rad_bf16_to_f32(qb[i]); }
    std::vector<int> su(c.ctx), cu(n_seq + 1, 0);
    for (int s = 0; s < n_seq; ++s) cu[s + 1] = cu[s] + c.qlen[s];
    std::vector<float> kd(n_seq * kvh, 1.f), vd(n_seq * kvh, 1.f);
    if (c.descale) for (auto* v : {&kd, &vd}) for (auto& x : *v) x = 0.5f + (float)(rng() % 100) / 100.f;
    std::vector<uint16_t> out((size_t)rows * qh * HD, 0xffff);
    std::vector<char> scratch(c.scratch + 64);
    char* sp = scratch.data() + ((16 - ((uintptr_t)scratch.data() & 15)) & 15);

    RadTensor t[8];
    t[0] = tens(qb.data(), RAD_BF16, {rows, qh * HD}); t[0].stride[0] = qcols;
    t[1] = tens(c.fp8 ? (void*)kv8.data() : (void*)kvb.data(), c.fp8 ? RAD_F8E4M3 : RAD_BF16, {nblocks, kvh, 16, slot});
    t[2] = tens(bt.data(), RAD_I32, {n_seq, maxb});
    t[3] = tens(su.data(), RAD_I32, {n_seq});
    t[4] = c.descale ? tens(kd.data(), RAD_F32, {n_seq, kvh}) : RadTensor{};
    t[5] = c.descale ? tens(vd.data(), RAD_F32, {n_seq, kvh}) : RadTensor{};
    t[6] = c.ragged ? tens(cu.data(), RAD_I32, {n_seq + 1}) : RadTensor{};
    t[7] = tens(out.data(), RAD_BF16, {rows, qh, HD});
    int qmax = 0; for (int q : c.qlen) qmax = std::max(qmax, q);
    std::vector<RadParam> p;
    auto pi = [&](const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); };
    auto ps = [&](const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); };
    auto pf = [&](const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); };
    pi("q_len", std::min(qmax + 3, r.qmax));  /* the band bound, above the step's max where the row allows */
    pi("head_dim", HD); pi("gqa", c.gqa); pi("block_size", 16); pi("causal", c.causal); pi("window", c.window);
    ps("q_dtype", "bf16"); ps("kv_dtype", c.fp8 ? "fp8_e4m3" : "bf16");
    const double scale = 1.0 / std::sqrt((double)HD) * 1.3;   /* not the default: the param must be read */
    pf("scale", scale); pi("n_head", qh); pi("max_seqs", n_seq);
    /* the host-visible context bound the split law needs (the engine always passes it) */
    pi("max_ctx", c.max_ctx ? c.max_ctx : (long long)maxb * 16);
    RadArgs a{}; a.t = t; a.n_t = 8; a.p = p.data(); a.n_p = (int)p.size();
    /* the split-KV scratch from the row's own hook, as the engine sizes it */
    std::vector<char> hook_scratch;
    if (r.scratch) {
        const int64_t need = r.scratch(&a);
        if (need < 0) { std::printf("%-44s scratch hook refused: %lld\n", c.name.c_str(), (long long)need); return 1; }
        hook_scratch.assign((size_t)need + 64, 0);
        a.scratch = hook_scratch.data() + ((16 - ((uintptr_t)hook_scratch.data() & 15)) & 15);
        a.scratch_bytes = need;
    } else {
        a.scratch = c.scratch ? sp : nullptr; a.scratch_bytes = c.scratch;
    }
    Fn fn = r.fn;
    const int g0 = emu_grids;
    const int rc = fn(&a, nullptr);
    const int launches = emu_grids - g0;
    if (rc != RAD_OK) { std::printf("%-44s launch returned %d\n", c.name.c_str(), rc); return 1; }

    /* the reference */
    double max_err = 0, max_ref = 0, se = 0, sr = 0; long bad = 0, nan_out = 0;
    for (int s = 0; s < n_seq; ++s) for (int i = 0; i < c.qlen[s]; ++i) for (int h = 0; h < qh; ++h) {
        const int tok = cu[s] + i, kh = h / c.gqa, pos = c.ctx[s] - c.qlen[s] + i;
        auto visible = [&](int j) {
            if (c.causal ? j > pos : j >= c.ctx[s]) return false;
            if (c.window > 0) { const int d = pos - j; if (c.causal ? d >= c.window : (d >= c.window || d <= -c.window)) return false; }
            return bt[s * maxb + j / 16] >= 0;
        };
        std::vector<double> sc(c.ctx[s], -INFINITY); double mx = -INFINITY;
        for (int j = 0; j < c.ctx[s]; ++j) {
            if (!visible(j)) continue;
            const int64_t kb = (((int64_t)bt[s * maxb + j / 16] * kvh + kh) * 16 + j % 16) * slot;
            double dot = 0; for (int d = 0; d < HD; ++d) dot += (double)qf[(size_t)tok * qcols + h * HD + d] * kvf[kb + d] * kd[s * kvh + kh];
            sc[j] = dot * scale; mx = std::max(mx, sc[j]);
        }
        std::vector<double> acc(HD, 0.0); double sum = 0;
        if (mx > -INFINITY)
            for (int j = 0; j < c.ctx[s]; ++j) {
                if (sc[j] == -INFINITY) continue;
                const double w = std::exp(sc[j] - mx); sum += w;
                const int64_t kb = (((int64_t)bt[s * maxb + j / 16] * kvh + kh) * 16 + j % 16) * slot;
                for (int d = 0; d < HD; ++d) acc[d] += w * kvf[kb + HD + d] * vd[s * kvh + kh];
            }
        for (int d = 0; d < HD; ++d) {
            const double ref = sum > 0 ? acc[d] / sum : 0.0;
            const float got = rad_bf16_to_f32(out[((size_t)tok * qh + h) * HD + d]);
            if (std::isnan(got)) { ++nan_out; continue; }
            const double e = std::fabs(got - ref);
            max_err = std::max(max_err, e); max_ref = std::max(max_ref, std::fabs(ref));
            se += e * e; sr += ref * ref;
            /* The absolute floor scales with V: an fp8 cache is drawn at 4x a bf16 one's magnitude. */
            const double vscale = c.fp8 ? 4.0 : 1.0;
            if (e > 1e-2 * std::max(vscale, std::fabs(ref)) + 4e-3 * std::fabs(ref)) ++bad;
        }
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const bool ok = bad == 0 && nan_out == 0 && rel < 5e-3;
    std::printf("%-44s rows %4d %s rel_l2 %.2e  max_abs %.2e  bad %ld  nan %ld  %s\n", c.name.c_str(), rows,
                launches > 1 ? "split" : emu_last_block == 512 ? "wmma " : "     ", rel, max_err, bad, nan_out,
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* only = argc > 1 ? argv[1] : nullptr;
    int fails = 0, cases = 0;
    for (const Row& r : kRows) {
        if (only && !std::strstr(r.name, only)) continue;
        const bool decode = std::strncmp(r.name, "attn_decode", 11) == 0;
        std::vector<Case> cs;
        auto add = [&](const char* tag, std::vector<int> ql, std::vector<int> ctx, bool causal, int window, bool descale) {
            Case c;
            c.name = std::string(r.name) + " " + tag;
            c.hd = r.hd; c.gqa = r.gqa; c.kvh = 2; c.fp8 = r.fp8;
            c.causal = r.causal1 ? true : causal;
            c.ragged = ql.size() > 1 || ql[0] > 1;
            c.descale = descale && r.fp8;
            c.window = window;
            c.qlen = ql; c.ctx = ctx;
            cs.push_back(c);
        };
        const int win = r.win == 2 ? 40 : 0;
        if (decode) {
            add("1 tok ctx 1", { 1 }, { 1 }, true, win, false);
            add("1 tok ctx 17", { 1 }, { 17 }, true, win, true);
            add("1 tok ctx 300", { 1 }, { 300 }, true, win, false);
            add("1 tok ctx 2049 (split)", { 1 }, { 2049 }, true, win, true);
            add("3 seqs ragged ctx", { 1, 1, 1 }, { 5, 64, 129 }, true, win, false);
            const int v = std::min(r.qmax, 4);
            add("verify", { v, 1 }, { 90, 33 }, true, win, true);
            if (r.win == 1) add("window 24", { 1, 2 }, { 200, 61 }, true, 24, false);
            if (!r.causal1) add("non-causal", { 1 }, { 77 }, false, 0, false);
        } else {
            add("prefill 37", { 37 }, { 37 }, true, win, false);
            add("prefill ragged", { 20, 1, 45 }, { 20, 9, 100 }, true, win, true);
            /* a short chunk over a long context: split-KV where the row has a scratch hook */
            add("prefill 11 ctx 2049", { 11 }, { 2049 }, true, win, true);
            add("prefill ragged ctx 1500", { 20, 1, 45 }, { 700, 9, 1500 }, true, win, false);
            if (r.win == 1) add("prefill window 16", { 30 }, { 50 }, true, 16, false);
        }
        for (const Case& c : cs) {
            if (std::getenv("HALO_EMU_TRACE")) { std::printf("-> %s\n", c.name.c_str()); std::fflush(stdout); }
            ++cases; fails += run(r, c);
        }
    }
    std::printf("paged attention: %d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
