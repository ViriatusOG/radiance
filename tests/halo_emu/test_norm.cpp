/* tests/halo_emu (from tests/r4d_emu/test_norm.cpp): the sandwich-norm joins (halo_norm_join_bf16.hip) on the host SIMT emulator.
 *
 * Two references:
 *   Chain: the unfused sequence the arch plugin would otherwise issue (rmsnorm, add, rmsnorm; or
 *     rmsnorm x2, add, rmsnorm, add, mul_weight) in libhalo's per-element arithmetic, with sums of
 *     squares in this kernel's block order. The fused kernel must match it bit for bit: fusing
 *     changes only the launch count.
 *   Host arm: sums in index order (as a different libr4d block width would) and may differ by a
 *     flipped rounding, measured against the row's scale (see cmp).
 * Padding columns and rows past M must be left alone. The route form's chain adds the router's
 * rmsnorm (with its mult) and libhalo's rmsnorm_quant_fp8 of the experts' norm: scale amax / 448
 * per group, the clamped value encoded by the emulator's v_cvt_pk_fp8_f32, the saturating
 * encoder every fp8 cast here uses. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(11);

/* the kernel's sum of squares: 256 threads strided over the row, a butterfly a wave, waves in order */
static float block_factor(const std::vector<float>& v, int n, float eps) {
    float part[256];
    for (int t = 0; t < 256; ++t) {
        float s = 0.0f;
        for (int k = 0; k < 24; ++k) { const int i = t + 256 * k; const float x = i < n ? v[(size_t)i] : 0.0f; s += x * x; }
        part[t] = s;
    }
    float total = 0.0f;
    for (int w = 0; w < 8; ++w) {
        float l[32];
        for (int i = 0; i < 32; ++i) l[i] = part[w * 32 + i];
        for (int d = 16; d >= 1; d >>= 1) {
            float nl[32];
            for (int i = 0; i < 32; ++i) nl[i] = l[i] + l[i ^ d];
            std::memcpy(l, nl, sizeof l);
        }
        total += l[0];
    }
    return 1.0f / std::sqrt(total / (float)n + eps);
}
static std::vector<float> chain_norm(const std::vector<float>& v, const float* g, int n, float eps) {
    const float f = block_factor(v, n, eps);
    std::vector<float> y((size_t)n);
    for (int i = 0; i < n; ++i) y[(size_t)i] = rb((v[(size_t)i] * f) * g[i]);
    return y;
}

static RadTensor rows(void* p, uint32_t dt, int64_t M, int64_t n, int64_t ld) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = M; t.shape[1] = n; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
static RadTensor vec(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}
static std::vector<uint16_t> randb(size_t k, float sd) {
    std::normal_distribution<float> d(0.f, sd);
    std::vector<uint16_t> v(k);
    for (auto& x : v) x = f2b(d(rng));
    return v;
}
static std::vector<float> gains(int n) {
    std::normal_distribution<float> d(1.f, 0.4f);
    std::vector<float> g((size_t)n);
    for (auto& x : g) x = d(rng);
    return g;
}
/* Difference from the host arm in units of the row's RMS. A different summation order moves the
 * norm factor by an f32 ulp, which flips the odd bf16 rounding; where the residual add cancels,
 * one ulp of the addend is many ulps of the small sum, so output ulps are the wrong yardstick. */
struct Tally { long chain_bad = 0, host_diff = 0, host_bad = 0, pad = 0; double worst = 0.0; };
static void cmp(const uint16_t* got, const uint16_t* chain, const uint16_t* host, int n, Tally& t) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)b2f(host[i]) * b2f(host[i]);
    const double rms = std::sqrt(ss / n) + 1e-30;
    for (int i = 0; i < n; ++i) {
        if (got[i] != chain[i]) ++t.chain_bad;
        if (got[i] == host[i]) continue;
        ++t.host_diff;
        const double e = std::fabs((double)b2f(got[i]) - b2f(host[i])) / rms;
        if (e > t.worst) t.worst = e;
        /* one bf16 ulp of an element at 4x the row's rms is up to 3% of it: 1/16 passes a flipped
         * rounding of a large element and fails anything a real bug produces */
        if (e > 1.0 / 16) ++t.host_bad;
    }
}

static int case_nan(int M, int n, int pad, int extra_rows) {
    const int ld = n + pad, R = M + extra_rows;
    const float eps = 1e-6f;
    auto a = randb((size_t)R * ld, 3.f), x = randb((size_t)R * ld, 1.f), y0 = randb((size_t)R * ld, 1.f);
    auto wa = gains(n), wy = gains(n);
    auto xd = x, yd = y0, xh = x, yh = y0;
    RadTensor t[5] = { rows(a.data(), RAD_BF16, M, n, ld), vec(wa.data(), RAD_F32, n),
                       rows(xd.data(), RAD_BF16, M, n, ld), vec(wy.data(), RAD_F32, n),
                       rows(yd.data(), RAD_BF16, M, n, ld) };
    RadParam p[4]{}; p[0] = {"M", RAD_P_INT, M}; p[1] = {"n", RAD_P_INT, n};
    p[2].key = "eps"; p[2].kind = RAD_P_F64; p[2].dval = eps; p[3].key = "dtype"; p[3].kind = RAD_P_STR; p[3].sval = "bf16";
    RadArgs ar{}; ar.t = t; ar.n_t = 5; ar.p = p; ar.n_p = 4;
    if (halo_rad_norm_add_norm_bf16(&ar, nullptr) != RAD_OK) { std::puts("norm_add_norm: device launch failed"); return 1; }
    t[2].data = xh.data(); t[4].data = yh.data();
    if (host_norm_add_norm(&ar) != RAD_OK) { std::puts("norm_add_norm: host launch failed"); return 1; }
    Tally tx, ty;
    for (int m = 0; m < R; ++m) {
        const size_t o = (size_t)m * ld;
        if (m >= M) { for (int i = 0; i < ld; ++i) tx.pad += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]); continue; }
        for (int i = n; i < ld; ++i) tx.pad += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]);
        std::vector<float> av((size_t)n), xv((size_t)n);
        for (int i = 0; i < n; ++i) av[(size_t)i] = b2f(a[o + i]);
        const auto t1 = chain_norm(av, wa.data(), n, eps);                 /* rmsnorm */
        std::vector<uint16_t> cx((size_t)n), cy((size_t)n);
        for (int i = 0; i < n; ++i) { xv[(size_t)i] = rb(b2f(x[o + i]) + t1[(size_t)i]); cx[(size_t)i] = f2b(xv[(size_t)i]); }   /* add */
        const auto t2 = chain_norm(xv, wy.data(), n, eps);                 /* rmsnorm */
        for (int i = 0; i < n; ++i) cy[(size_t)i] = f2b(t2[(size_t)i]);
        cmp(&xd[o], cx.data(), &xh[o], n, tx);
        cmp(&yd[o], cy.data(), &yh[o], n, ty);
    }
    const bool ok = tx.chain_bad == 0 && ty.chain_bad == 0 && tx.host_bad == 0 && ty.host_bad == 0 && tx.pad == 0;
    std::printf("norm_add_norm  M %3d n %4d pitch %4d: vs chain x %ld y %ld differ; vs host arm %ld/%ld differ, worst "
                "%.1e of the row rms; padding touched %ld -> %s\n", M, n, ld, tx.chain_bad, ty.chain_bad, tx.host_diff,
                ty.host_diff, tx.worst > ty.worst ? tx.worst : ty.worst, tx.pad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* norm_add_norm_route: norm_add_norm, then the router's row r = norm(x) * w_r * mult and the experts'
 * E4M3 codes q and scales qs of norm(x) * w_e. `zero_group`: w_e zero over the second 128 columns,
 * so that group's amax is 0 and its scale the quantiser's 1. */
static int case_route(int M, int n, int pad, int extra_rows, bool zero_group = false) {
    const int ld = n + pad, R = M + extra_rows, G = n / 128, ldq = n + pad, lds = G + 3;
    const float eps = 1e-6f, mult = 1.0f / std::sqrt((float)n);
    auto a = randb((size_t)R * ld, 3.f), x = randb((size_t)R * ld, 1.f), y0 = randb((size_t)R * ld, 1.f);
    auto r0 = randb((size_t)R * ld, 1.f);
    auto wa = gains(n), wy = gains(n), wr = gains(n), we = gains(n);
    if (zero_group && n >= 256) for (int i = 128; i < 256; ++i) we[(size_t)i] = 0.0f;
    std::vector<uint8_t> q0((size_t)R * ldq);
    std::vector<float> s0((size_t)R * lds);
    for (auto& v : q0) v = (uint8_t)(rng() & 0xff);
    for (auto& v : s0) v = -7.0f;
    auto xd = x, yd = y0, rd = r0, xh = x, yh = y0, rh = r0;
    auto qd = q0, qh = q0;
    auto sd = s0, sh = s0;
    RadTensor t[10] = { rows(a.data(), RAD_BF16, M, n, ld), vec(wa.data(), RAD_F32, n),
                        rows(xd.data(), RAD_BF16, M, n, ld), vec(wy.data(), RAD_F32, n),
                        rows(yd.data(), RAD_BF16, M, n, ld), vec(wr.data(), RAD_F32, n),
                        rows(rd.data(), RAD_BF16, M, n, ld), vec(we.data(), RAD_F32, n),
                        rows(qd.data(), RAD_F8E4M3, M, n, ldq), rows(sd.data(), RAD_F32, M, G, lds) };
    RadParam p[6]{}; p[0] = {"M", RAD_P_INT, M}; p[1] = {"n", RAD_P_INT, n};
    p[2].key = "eps"; p[2].kind = RAD_P_F64; p[2].dval = eps; p[3].key = "dtype"; p[3].kind = RAD_P_STR; p[3].sval = "bf16";
    p[4].key = "mult"; p[4].kind = RAD_P_F64; p[4].dval = mult; p[5] = {"group", RAD_P_INT, 128};
    RadArgs ar{}; ar.t = t; ar.n_t = 10; ar.p = p; ar.n_p = 6;
    if (halo_rad_norm_add_norm_route_bf16(&ar, nullptr) != RAD_OK) { std::puts("norm_add_norm_route: device launch failed"); return 1; }
    t[2].data = xh.data(); t[4].data = yh.data(); t[6].data = rh.data(); t[8].data = qh.data(); t[9].data = sh.data();
    if (host_norm_add_norm_route(&ar) != RAD_OK) { std::puts("norm_add_norm_route: host launch failed"); return 1; }
    Tally tx, ty, tr;
    long q_chain = 0, s_chain = 0, q_host_bad = 0, touched = 0;
    for (int m = 0; m < R; ++m) {
        const size_t o = (size_t)m * ld, oq = (size_t)m * ldq, os = (size_t)m * lds;
        if (m >= M) {
            for (int i = 0; i < ld; ++i) touched += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]) + (rd[o + i] != r0[o + i]);
            for (int i = 0; i < ldq; ++i) touched += qd[oq + i] != q0[oq + i];
            for (int i = 0; i < lds; ++i) touched += sd[os + i] != s0[os + i];
            continue;
        }
        for (int i = n; i < ld; ++i) touched += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]) + (rd[o + i] != r0[o + i]);
        for (int i = n; i < ldq; ++i) touched += qd[oq + i] != q0[oq + i];
        for (int i = G; i < lds; ++i) touched += sd[os + i] != s0[os + i];
        std::vector<float> av((size_t)n), xv((size_t)n);
        for (int i = 0; i < n; ++i) av[(size_t)i] = b2f(a[o + i]);
        const auto t1 = chain_norm(av, wa.data(), n, eps);                 /* rmsnorm */
        std::vector<uint16_t> cx((size_t)n), cy((size_t)n), cr((size_t)n);
        for (int i = 0; i < n; ++i) { xv[(size_t)i] = rb(b2f(x[o + i]) + t1[(size_t)i]); cx[(size_t)i] = f2b(xv[(size_t)i]); }   /* add */
        const float f = block_factor(xv, n, eps);
        for (int i = 0; i < n; ++i) {
            cy[(size_t)i] = f2b((xv[(size_t)i] * f) * wy[(size_t)i]);                   /* rmsnorm */
            cr[(size_t)i] = f2b(((xv[(size_t)i] * f) * wr[(size_t)i]) * mult);         /* the router's */
        }
        cmp(&xd[o], cx.data(), &xh[o], n, tx);
        cmp(&yd[o], cy.data(), &yh[o], n, ty);
        cmp(&rd[o], cr.data(), &rh[o], n, tr);
        for (int g = 0; g < G; ++g) {                                       /* rmsnorm_quant_fp8 */
            float h[128], am = 0.0f;
            for (int j = 0; j < 128; ++j) { h[j] = rb((xv[(size_t)(g * 128 + j)] * f) * we[(size_t)(g * 128 + j)]); am = std::fmax(am, std::fabs(h[j])); }
            const float sc = am > 0.0f ? am * (1.0f / 448.0f) : 1.0f, inv = 1.0f / sc;
            s_chain += sd[os + g] != sc;
            if (zero_group && g == 1 && sd[os + g] != 1.0f) ++s_chain;
            for (int j = 0; j < 128; ++j) {
                const int i = g * 128 + j;
                q_chain += qd[oq + i] != rad_f32_to_fp8e4m3(std::fmin(std::fmax(h[j] * inv, -448.0f), 448.0f));
                /* vs the host arm: one code step is at most an eighth of its value, so at most an
                 * eighth of the group's amax; a scale an ulp off moves less */
                const double dv = std::fabs((double)rad_fp8e4m3_to_f32(qd[oq + i]) * sd[os + g] -
                                            (double)rad_fp8e4m3_to_f32(qh[oq + i]) * sh[os + g]);
                if (dv > (double)am / 8 + 1e-30) ++q_host_bad;
            }
        }
    }
    const bool ok = tx.chain_bad == 0 && ty.chain_bad == 0 && tr.chain_bad == 0 && q_chain == 0 && s_chain == 0 &&
                    tx.host_bad == 0 && ty.host_bad == 0 && tr.host_bad == 0 && q_host_bad == 0 && touched == 0;
    std::printf("norm_add_norm_route M %3d n %4d pitch %4d%s: vs chain x %ld y %ld r %ld q %ld scale %ld differ; vs host "
                "arm %ld/%ld/%ld differ, %ld codes past a step; padding touched %ld -> %s\n", M, n, ld,
                zero_group ? " (a zero group)" : "", tx.chain_bad, ty.chain_bad, tr.chain_bad, q_chain, s_chain,
                tx.host_diff, ty.host_diff, tr.host_diff, q_host_bad, touched, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* `next`: also the next layer's input norm, y = norm(x) * w_n over the x the join writes */
static int case_jas(int M, int n, int pad, int extra_rows, bool next = false) {
    const int ld = n + pad, R = M + extra_rows;
    const float eps = 1e-6f;
    /* the two arms as one [2M, n] buffer, as the plugin holds them: dense rows, then routed */
    auto ff = randb((size_t)2 * R * ld, 2.f), x = randb((size_t)R * ld, 4.f), y0 = randb((size_t)R * ld, 1.f);
    auto wd = gains(n), wr = gains(n), wj = gains(n), wn = gains(n);
    uint16_t sc = f2b(0.4375f + (float)(rng() % 64) / 128.f);
    auto xd = x, xh = x, yd = y0, yh = y0;
    RadTensor t[9] = { rows(ff.data(), RAD_BF16, M, n, ld), vec(wd.data(), RAD_F32, n),
                       rows(ff.data() + (size_t)M * ld, RAD_BF16, M, n, ld), vec(wr.data(), RAD_F32, n),
                       vec(wj.data(), RAD_F32, n), rows(xd.data(), RAD_BF16, M, n, ld), vec(&sc, RAD_BF16, 1),
                       vec(wn.data(), RAD_F32, n), rows(yd.data(), RAD_BF16, M, n, ld) };
    RadParam p[4]{}; p[0] = {"M", RAD_P_INT, M}; p[1] = {"n", RAD_P_INT, n};
    p[2].key = "eps"; p[2].kind = RAD_P_F64; p[2].dval = eps; p[3].key = "dtype"; p[3].kind = RAD_P_STR; p[3].sval = "bf16";
    RadArgs ar{}; ar.t = t; ar.n_t = next ? 9 : 7; ar.p = p; ar.n_p = 4;
    if (halo_rad_join_add_scale_bf16(&ar, nullptr) != RAD_OK) { std::puts("join_add_scale: device launch failed"); return 1; }
    t[5].data = xh.data(); t[8].data = yh.data();
    if (host_join_add_scale(&ar) != RAD_OK) { std::puts("join_add_scale: host launch failed"); return 1; }
    Tally tx, ty;
    for (int m = 0; m < R; ++m) {
        const size_t o = (size_t)m * ld;
        if (m >= M) { for (int i = 0; i < ld; ++i) tx.pad += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]); continue; }
        for (int i = n; i < ld; ++i) tx.pad += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]);
        std::vector<float> dv((size_t)n), rv((size_t)n), sv((size_t)n);
        for (int i = 0; i < n; ++i) { dv[(size_t)i] = b2f(ff[o + i]); rv[(size_t)i] = b2f(ff[(size_t)M * ld + o + i]); }
        const auto d1 = chain_norm(dv, wd.data(), n, eps);                  /* rmsnorm (arm 1) */
        const auto r1 = chain_norm(rv, wr.data(), n, eps);                  /* rmsnorm (arm 2) */
        for (int i = 0; i < n; ++i) sv[(size_t)i] = rb(d1[(size_t)i] + r1[(size_t)i]);   /* add */
        const auto j = chain_norm(sv, wj.data(), n, eps);                   /* rmsnorm */
        std::vector<uint16_t> cx((size_t)n);
        for (int i = 0; i < n; ++i) cx[(size_t)i] = f2b(rb(b2f(x[o + i]) + j[(size_t)i]) * b2f(sc));   /* add, mul_weight */
        cmp(&xd[o], cx.data(), &xh[o], n, tx);
        if (next) {                                                         /* the next rmsnorm */
            std::vector<float> xv((size_t)n);
            for (int i = 0; i < n; ++i) xv[(size_t)i] = b2f(cx[(size_t)i]);
            const auto yn = chain_norm(xv, wn.data(), n, eps);
            std::vector<uint16_t> cy((size_t)n);
            for (int i = 0; i < n; ++i) cy[(size_t)i] = f2b(yn[(size_t)i]);
            cmp(&yd[o], cy.data(), &yh[o], n, ty);
        }
    }
    const bool ok = tx.chain_bad == 0 && ty.chain_bad == 0 && tx.host_bad == 0 && ty.host_bad == 0 && tx.pad == 0;
    std::printf("join_add_scale%s M %3d n %4d pitch %4d: vs chain %ld/%ld differ; vs host arm %ld/%ld differ, worst %.1e "
                "of the row rms; padding touched %ld -> %s\n", next ? "+next" : "     ", M, n, ld, tx.chain_bad,
                ty.chain_bad, tx.host_diff, ty.host_diff, tx.worst > ty.worst ? tx.worst : ty.worst, tx.pad,
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    /* Gemma 4 26B's 2816 at decode and verify heights, a prefill height, odd widths, the limit */
    const int Ms[] = {1, 4, 16, 37};
    for (int M : Ms) fails += case_nan(M, 2816, M % 2 ? 64 : 0, 2);
    for (int M : Ms) fails += case_jas(M, 2816, M % 2 ? 0 : 128, 2);
    fails += case_nan(3, 1000, 24, 1) + case_jas(3, 1000, 24, 1);
    fails += case_nan(2, 6144, 0, 0) + case_jas(2, 6144, 16, 0);
    fails += case_nan(5, 256, 8, 1) + case_jas(5, 5376, 0, 1);
    /* the edge between the 12-a-thread kernels and the 24: 3072 the last of the first */
    fails += case_nan(2, 3072, 0, 1) + case_nan(2, 3073, 8, 1);
    fails += case_jas(2, 3072, 0, 1, true) + case_jas(2, 3073, 8, 1, true) + case_jas(3, 3073, 0, 1);
    /* and with the next layer's input norm folded in */
    for (int M : Ms) fails += case_jas(M, 2816, M % 2 ? 64 : 0, 2, true);
    fails += case_jas(3, 1000, 24, 1, true) + case_jas(2, 6144, 16, 0, true);
    /* the route form: the 26B-A4B's width at decode, verify and prefill heights, the edge between the
     * 12- and 24-a-thread kernels, one group, the limit, and a group whose norm is all zero */
    for (int M : Ms) fails += case_route(M, 2816, M % 2 ? 64 : 0, 2);
    fails += case_route(2, 3072, 0, 1) + case_route(2, 3200, 16, 1) + case_route(3, 128, 8, 1);
    fails += case_route(2, 6144, 0, 0) + case_route(4, 2816, 0, 1, true);
    std::printf("norm joins: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
