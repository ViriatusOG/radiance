/* Inkling 2's attention (r4d_attn_rel.hip, attn_paged_rel) through the SIMT emulator, against a
 * host port of libref's ref_attn_paged_rel in double. The cases: decode (one row a sequence) at
 * gqa 3 with a full window over 1500 keys, so distances past the extent take no bias; a sliding
 * window of 1024 over the same; a verify block at gqa 4; a ragged prefill (cu_seqlens, mixed
 * lengths) at gqa 2 with the window; the int8 cache at gqa 3; r as a column slice of a wider row;
 * and a sink far above the scores, which must take the row. Output is bf16, so each case is held to
 * the bf16 rounding of its element (rel_l2 under 4e-3). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(91);
static float nrm() { static std::normal_distribution<float> d(0.f, 1.f); return d(rng); }

static RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    return t;
}
/* A tensor with explicit strides: a view of a wider buffer. */
static RadTensor view(void* p, uint32_t dt, std::vector<int64_t> shape, std::vector<int64_t> strides) {
    RadTensor t = tens(p, dt, shape);
    for (size_t i = 0; i < shape.size(); ++i) t.stride[i] = strides[i];
    return t;
}

struct Case {
    const char* name;
    std::vector<int> qlen, ctx;   /* per sequence: query rows, and total keys incl. its queries */
    int kvh, gqa, window, causal, extent;
    bool i8 = false;              /* the int8 cache (codes with an f16 scale of K and of V) */
    bool cu = true;               /* cu_seqlens present (false: uniform, qlen equal) */
    int rpitch = 0;               /* r as a column slice: this many extra columns per row */
    float sink_shift = 0.f;       /* added to every sink logit */
    int sink_every = 0;           /* a sink this far above the scores in every such row */
    bool use_scratch = true;      /* bind scratch buffer when requested by scratch hook */
};

static bool close(const char* what, const std::vector<double>& ref, const std::vector<float>& got, double tol) {
    double se = 0, sr = 0, worst = 0, scale = 1e-30;
    long nan = 0;
    for (double r : ref) scale = std::max(scale, std::fabs(r));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::isnan(got[i])) { ++nan; continue; }
        const double e = std::fabs(got[i] - ref[i]);
        se += e * e; sr += ref[i] * ref[i];
        worst = std::max(worst, e / scale);
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const bool ok = nan == 0 && rel < tol && worst < 8 * tol;
    std::printf("  %-44s rel_l2 %.2e  worst %.2e  nan %ld  %s\n", what, rel, worst, nan, ok ? "PASS" : "FAIL");
    return ok;
}

static constexpr int HD = 128, BS = 16, DR = 16;

static int run(const Case& c) {
    const int n_seq = (int)c.qlen.size(), nh = c.kvh * c.gqa;
    int rows = 0; std::vector<int> cu(n_seq + 1, 0);
    for (int s = 0; s < n_seq; ++s) { cu[s + 1] = cu[s] + c.qlen[s]; rows += c.qlen[s]; }
    int maxb = 0; std::vector<int> nb(n_seq);
    for (int s = 0; s < n_seq; ++s) { nb[s] = (c.ctx[s] + BS - 1) / BS; maxb = std::max(maxb, nb[s]); }

    /* Blocks: sequence s owns its own run of ids, in an order a table would not give
     * sequentially. Cache values are drawn, then quantised or rounded to bf16 as the cache holds
     * them; the reference reads the same values back. */
    int total_blocks = 0;
    std::vector<int> bt((size_t)n_seq * maxb, -1);
    for (int s = 0; s < n_seq; ++s)
        for (int b = 0; b < nb[s]; ++b) bt[(size_t)s * maxb + b] = total_blocks + (nb[s] - 1 - b);
    for (int s = 0; s < n_seq; ++s) total_blocks += nb[s];
    /* kv values in f32 as the cache's dequantised form, [block, kvh, slot, K then V] */
    const size_t slot_w = c.i8 ? (2 * HD + 4) : (2 * HD);
    std::vector<float> kvf((size_t)total_blocks * c.kvh * BS * 2 * HD, 0.f);
    std::vector<uint16_t> kvb;     /* bf16 cache */
    std::vector<uint8_t> kvi;      /* int8 cache */
    if (c.i8) kvi.assign((size_t)total_blocks * c.kvh * BS * slot_w, 0);
    else kvb.assign((size_t)total_blocks * c.kvh * BS * slot_w, 0);
    for (int blk = 0; blk < total_blocks; ++blk)
        for (int kh = 0; kh < c.kvh; ++kh)
            for (int sl = 0; sl < BS; ++sl) {
                const size_t row = (((size_t)blk * c.kvh + kh) * BS + sl);
                for (int half = 0; half < 2; ++half) {
                    float x[HD];
                    for (int d = 0; d < HD; ++d) x[d] = nrm() * (half ? 0.8f : 1.0f);
                    if (c.i8) {
                        float am = 0.f;
                        for (int d = 0; d < HD; ++d) am = std::max(am, std::fabs(x[d]));
                        const uint16_t sh = rad_f32_to_f16(std::min(am / 127.f, 515.f));
                        const float sc = rad_f16_to_f32(sh);
                        uint8_t* p = kvi.data() + row * slot_w;
                        for (int d = 0; d < HD; ++d) {
                            const float q = sc > 0 ? std::nearbyint(x[d] / sc) : 0.f;
                            p[half * HD + d] = (uint8_t)(int8_t)std::max(-127.f, std::min(127.f, q));
                            kvf[row * 2 * HD + half * HD + d] = (float)(int8_t)p[half * HD + d] * sc;
                        }
                        std::memcpy(p + 2 * HD + 2 * half, &sh, 2);
                    } else {
                        for (int d = 0; d < HD; ++d) {
                            const uint16_t b = rad_f32_to_bf16(x[d]);
                            kvb[row * slot_w + half * HD + d] = b;
                            kvf[row * 2 * HD + half * HD + d] = rad_bf16_to_f32(b);
                        }
                    }
                }
            }

    /* q, r, the rel_proj weight and the sinks. */
    std::vector<uint16_t> qb((size_t)rows * nh * HD);
    std::vector<float> qf(qb.size());
    for (size_t i = 0; i < qb.size(); ++i) { qb[i] = rad_f32_to_bf16(nrm()); qf[i] = rad_bf16_to_f32(qb[i]); }
    const int rstride = nh * DR + c.rpitch;
    std::vector<uint16_t> rb((size_t)rows * rstride, rad_f32_to_bf16(9.0f));   /* the pitch's extra columns */
    std::vector<float> rf((size_t)rows * nh * DR);
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < nh; ++h)
            for (int cc = 0; cc < DR; ++cc) {
                const uint16_t b = rad_f32_to_bf16(nrm());
                rb[(size_t)t * rstride + h * DR + cc] = b;
                rf[((size_t)t * nh + h) * DR + cc] = rad_bf16_to_f32(b);
            }
    const int E = c.extent;
    std::vector<uint16_t> pb((size_t)DR * E);
    std::vector<float> pf(pb.size());
    for (size_t i = 0; i < pb.size(); ++i) { pb[i] = rad_f32_to_bf16(0.3f * nrm()); pf[i] = rad_bf16_to_f32(pb[i]); }
    std::vector<float> sk((size_t)rows * nh);
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < nh; ++h) {
            float v = 1.5f * nrm() + c.sink_shift;
            if (c.sink_every && t % c.sink_every == 0) v = 40.f;
            sk[(size_t)t * nh + h] = v;
        }
    std::vector<uint16_t> ob((size_t)rows * nh * HD, 0);
    std::vector<int32_t> seqused(n_seq), cu32(cu.begin(), cu.end()), bt32(bt.begin(), bt.end());
    for (int s = 0; s < n_seq; ++s) seqused[s] = c.ctx[s];

    /* The reference, in double, one (row, head) at a time (ref_attn_paged_rel). */
    const double scale = 1.0 / std::sqrt((double)HD);
    std::vector<double> ref((size_t)rows * nh * HD, 0.0);
    for (int s = 0; s < n_seq; ++s)
        for (int i = 0; i < c.qlen[s]; ++i) {
            const int t = cu[s] + i;
            const int pos = c.ctx[s] - c.qlen[s] + i;
            for (int h = 0; h < nh; ++h) {
                const int kh = h / c.gqa;
                std::vector<double> lg; std::vector<int> js;
                for (int j = 0; j < c.ctx[s]; ++j) {
                    if (c.causal && j > pos) continue;
                    if (c.window > 0) {
                        const int d = pos - j;
                        if (c.causal ? d >= c.window : (d >= c.window || d <= -c.window)) continue;
                    }
                    const int blk = bt[(size_t)s * maxb + j / BS];
                    const size_t row = (((size_t)blk * c.kvh + kh) * BS + j % BS);
                    double dot = 0;
                    for (int d = 0; d < HD; ++d) dot += (double)qf[((size_t)t * nh + h) * HD + d] * kvf[row * 2 * HD + d];
                    double bias = 0;
                    const int d = pos - j;
                    if (d >= 0 && d < E)
                        for (int cc = 0; cc < DR; ++cc) bias += (double)rf[((size_t)t * nh + h) * DR + cc] * pf[(size_t)cc * E + d];
                    lg.push_back(dot * scale + bias); js.push_back(j);
                }
                if (lg.empty()) continue;   /* a zero row */
                double mx = sk[(size_t)t * nh + h];
                for (double x : lg) mx = std::max(mx, x);
                double den = std::exp(sk[(size_t)t * nh + h] - mx);
                std::vector<double> acc(HD, 0.0);
                for (size_t k = 0; k < lg.size(); ++k) {
                    const double w = std::exp(lg[k] - mx);
                    den += w;
                    const int j = js[k];
                    const int blk = bt[(size_t)s * maxb + j / BS];
                    const size_t row = (((size_t)blk * c.kvh + kh) * BS + j % BS);
                    for (int d = 0; d < HD; ++d) acc[d] += w * kvf[row * 2 * HD + HD + d];
                }
                for (int d = 0; d < HD; ++d) ref[((size_t)t * nh + h) * HD + d] = acc[d] / den;
            }
        }

    /* The launch. */
    RadTensor tq = tens(qb.data(), RAD_BF16, {rows, nh, HD});
    RadTensor tkv = c.i8 ? tens(kvi.data(), RAD_I8, {total_blocks, c.kvh, BS, 2 * HD + 4})
                         : tens(kvb.data(), RAD_BF16, {total_blocks, c.kvh, BS, 2 * HD});
    RadTensor tbt = tens(bt32.data(), RAD_I32, {n_seq, maxb});
    RadTensor tsu = tens(seqused.data(), RAD_I32, {n_seq});
    RadTensor tcu = tens(cu32.data(), RAD_I32, {n_seq + 1});
    RadTensor tr = c.rpitch ? view(rb.data(), RAD_BF16, {rows, nh, DR}, {rstride, DR, 1})
                            : tens(rb.data(), RAD_BF16, {rows, nh, DR});
    RadTensor tp = tens(pb.data(), RAD_BF16, {DR, E});
    RadTensor tsk = tens(sk.data(), RAD_F32, {rows, nh});
    RadTensor to = tens(ob.data(), RAD_BF16, {rows, nh, HD});

    RadTensor t[11] = {};
    t[0] = tq; t[1] = tkv; t[2] = tbt; t[3] = tsu;
    t[6] = c.cu ? tcu : RadTensor{};
    t[7] = tr; t[8] = tp; t[9] = tsk; t[10] = to;

    std::vector<RadParam> p;
    auto pi = [&](const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); };
    auto pf_ = [&](const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); };
    auto ps = [&](const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); };
    const int qmax = *std::max_element(c.qlen.begin(), c.qlen.end());
    pi("q_len", c.cu ? qmax : c.qlen[0]); pi("head_dim", HD); pi("gqa", c.gqa); pi("block_size", BS);
    pi("causal", c.causal); pi("window", c.window); ps("q_dtype", "bf16"); ps("kv_dtype", c.i8 ? "i8" : "bf16");
    pi("n_head", nh); pi("max_seqs", n_seq); pi("d_rel", DR); pi("extent", E);
    pf_("scale", scale);

    RadArgs a{}; a.t = t; a.n_t = 11; a.p = p.data(); a.n_p = (int)p.size();
    std::vector<uint8_t> scratch;
    if (c.use_scratch) {
        const int64_t sb = r4d_rad_attn_rel_scratch(&a);
        if (sb > 0) {
            scratch.assign((size_t)sb + 32, 0);
            uintptr_t addr = (uintptr_t)scratch.data();
            addr = (addr + 15u) & ~(uintptr_t)15u;
            a.scratch = (void*)addr;
            a.scratch_bytes = sb;
        }
    }
    const int rc = c.i8 ? r4d_rad_attn_rel_h128_i8kv(&a, nullptr) : r4d_rad_attn_rel_h128_bf16kv(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %-44s launch returned %d\n", c.name, rc); return 1; }

    std::vector<float> got((size_t)rows * nh * HD);
    for (size_t i = 0; i < got.size(); ++i) got[i] = rad_bf16_to_f32(ob[i]);
    return close(c.name, ref, got, 4e-3) ? 0 : 1;
}

int main() {
    /* One sequence of 1500 keys at decode: distances to 1499, so the bias's extent of 1024 cuts
     * the far keys off. The full window; the window of 1024 over the same keys. */
    std::vector<Case> cases = {
        { "decode 1500 keys, gqa 3, full, extent 1024", { 1 }, { 1500 }, 1, 3, 0, 1, 1024 },
        { "decode 1500 keys, gqa 3, window 1024", { 1 }, { 1500 }, 1, 3, 1024, 1, 1024 },
        { "decode x2 (1500, 90), gqa 1, 2 kv heads", { 1, 1 }, { 1500, 90 }, 2, 1, 1024, 1, 1024 },
        { "verify 4 and 8 rows, gqa 4, window 0", { 4, 8 }, { 300, 1500 }, 1, 4, 0, 1, 1024, false, true, 1 },
        { "ragged prefill 37, 5, 40 (1100 keys), gqa 2", { 37, 5, 40 }, { 1100, 5, 60 }, 1, 2, 1024, 1, 1024 },
        { "uniform prefill 3 x 8 rows, gqa 8, no cu", { 8, 8, 8 }, { 200, 60, 130 }, 1, 8, 0, 1, 40, false, false },
        { "int8 cache, gqa 3, window 0", { 2, 1 }, { 900, 500 }, 1, 3, 0, 1, 1024, true },
        { "r as a column slice of a wider row, gqa 3", { 3 }, { 700 }, 1, 3, 0, 1, 1024, false, true, 5 },
        { "sink far above the scores", { 1, 1 }, { 300, 40 }, 1, 2, 0, 1, 1024, false, true, 0, 0.f, 1 },
        { "decode 1500 keys unsplit fallback", { 1 }, { 1500 }, 1, 3, 0, 1, 1024, false, true, 0, 0.f, 0, false },
        { "wmma prefill 70 rows, gqa 3, bf16", { 70 }, { 200 }, 1, 3, 0, 1, 1024, false, false },
        { "wmma prefill 70 rows, gqa 3, i8, w1024", { 70 }, { 200 }, 1, 3, 1024, 1, 1024, true, true },
        { "wmma prefill 70 rows, sink far above", { 70 }, { 120 }, 1, 3, 0, 1, 64, false, false, 0, 0.f, 3 },
    };
    int fails = 0;
    for (const Case& c : cases) fails += run(c);
    std::printf("attn_paged_rel: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
