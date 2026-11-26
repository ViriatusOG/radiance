/* Inkling 2 device-only fusions (libr4d/r4d_inkling.hip) through the SIMT emulator:
 *   - loop_ctl (VEC8 and scalar fallback)
 *   - kv_clip_norm_store (bf16 and i8, head_dim 128 and 256, negative slot skip, zero-head edge)
 *   - qr_l2_clip (with and without position tau, grp <= 256 and > 256 fallback)
 *   - attn_res_mix3 (fast and general depth, no gain, f32 gain, bf16 gain)
 *   - attn_res_mix4 (fast and general depth)
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

std::mt19937 rng(4242);
float nrm() {
    static std::normal_distribution<float> d(0.f, 1.f);
    return d(rng);
}

struct Params {
    std::vector<RadParam> p;
    void i(const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); }
    void f(const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); }
    void str(const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); }
};

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape,
               std::vector<int64_t> stride = {}) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    if (stride.empty()) {
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            t.shape[i] = shape[i];
            t.stride[i] = st;
            st *= shape[i];
        }
    } else {
        for (size_t i = 0; i < shape.size(); ++i) {
            t.shape[i] = shape[i];
            t.stride[i] = stride[i];
        }
    }
    return t;
}

std::vector<uint16_t> bfv(size_t n, float sc, std::vector<float>* f) {
    std::vector<uint16_t> b(n);
    f->resize(n);
    for (size_t i = 0; i < n; ++i) {
        b[i] = rad_f32_to_bf16(nrm() * sc);
        (*f)[i] = rad_bf16_to_f32(b[i]);
    }
    return b;
}

bool close(const char* what, const std::vector<double>& ref, const std::vector<float>& got,
           double tol) {
    double se = 0, sr = 0, worst = 0, scale = 1e-30;
    long nan = 0;
    for (double r : ref) scale = std::max(scale, std::fabs(r));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::isnan(got[i])) { ++nan; continue; }
        const double e = std::fabs((double)got[i] - ref[i]);
        se += e * e;
        sr += ref[i] * ref[i];
        worst = std::max(worst, e / scale);
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const bool ok = nan == 0 && rel < tol && worst < 8 * tol;
    std::printf("  %-52s rel_l2 %.2e  worst %.2e  nan %ld  %s\n",
                what, rel, worst, nan, ok ? "PASS" : "FAIL");
    return ok;
}

int test_loop_ctl(int64_t M, int64_t n, int64_t rank, const char* what) {
    const double eps = 1e-6;
    std::vector<float> xf, mf, wdxf, wgf, wff;
    auto x   = bfv((size_t)(M * n), 1.2f, &xf);
    auto msg = bfv((size_t)(M * n), 0.9f, &mf);
    auto wdx = bfv((size_t)(rank * 2 * n), 0.08f, &wdxf);
    auto wg  = bfv((size_t)(n * rank), 0.08f, &wgf);
    auto wf  = bfv((size_t)(n * rank), 0.08f, &wff);
    std::vector<uint16_t> y((size_t)(M * n), 0);

    std::vector<double> ref((size_t)(M * n));
    for (int64_t m = 0; m < M; ++m) {
        double sx = 0, sm = 0;
        for (int64_t c = 0; c < n; ++c) {
            sx += xf[(size_t)(m * n + c)] * xf[(size_t)(m * n + c)];
            sm += mf[(size_t)(m * n + c)] * mf[(size_t)(m * n + c)];
        }
        const double rx = 1.0 / std::sqrt(sx / n + eps);
        const double rm = 1.0 / std::sqrt(sm / n + eps);
        std::vector<double> nx(n), nm(n), g(rank);
        for (int64_t c = 0; c < n; ++c) {
            nx[c] = rad_bf16_to_f32(rad_f32_to_bf16((float)(xf[(size_t)(m * n + c)] * rx)));
            nm[c] = rad_bf16_to_f32(rad_f32_to_bf16((float)(mf[(size_t)(m * n + c)] * rm)));
        }
        for (int64_t r = 0; r < rank; ++r) {
            double d = 0;
            for (int64_t c = 0; c < n; ++c) {
                d += wdxf[(size_t)(r * 2 * n + c)] * nx[c] +
                     wdxf[(size_t)(r * 2 * n + n + c)] * nm[c];
            }
            g[r] = rad_bf16_to_f32(rad_f32_to_bf16((float)d));
        }
        for (int64_t c = 0; c < n; ++c) {
            double gp = 0, fp = 0;
            for (int64_t r = 0; r < rank; ++r) {
                gp += wgf[(size_t)(c * rank + r)] * g[r];
                fp += wff[(size_t)(c * rank + r)] * g[r];
            }
            gp = rad_bf16_to_f32(rad_f32_to_bf16((float)gp));
            fp = rad_bf16_to_f32(rad_f32_to_bf16((float)fp));
            const double gt = rad_bf16_to_f32(rad_f32_to_bf16((float)std::tanh(gp)));
            const double ft = rad_bf16_to_f32(rad_f32_to_bf16((float)std::tanh(fp)));
            ref[(size_t)(m * n + c)] = (1.0 - ft) * xf[(size_t)(m * n + c)] + gt * mf[(size_t)(m * n + c)];
        }
    }

    Params P;
    P.i("M", M); P.i("n", n); P.i("rank", rank); P.f("eps", eps); P.str("dtype", "bf16");
    RadTensor ts[6] = {
        tens(x.data(), RAD_BF16, { M, n }),
        tens(msg.data(), RAD_BF16, { M, n }),
        tens(wdx.data(), RAD_BF16, { rank, 2 * n }),
        tens(wg.data(), RAD_BF16, { n, rank }),
        tens(wf.data(), RAD_BF16, { n, rank }),
        tens(y.data(), RAD_BF16, { M, n }),
    };
    RadArgs sa{}; sa.p = P.p.data(); sa.n_p = (int)P.p.size();
    const int64_t sb = r4d_rad_loop_ctl_scratch(&sa);
    std::vector<uint8_t> scratch((size_t)std::max<int64_t>(sb, 64), 0);
    RadArgs a{}; a.t = ts; a.n_t = 6; a.p = P.p.data(); a.n_p = (int)P.p.size();
    a.scratch = scratch.data(); a.scratch_bytes = sb;
    const int rc = r4d_rad_loop_ctl(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %-52s launch %d\n", what, rc); return 1; }
    std::vector<float> got((size_t)(M * n));
    for (size_t i = 0; i < got.size(); ++i) got[i] = rad_bf16_to_f32(y[i]);
    return close(what, ref, got, 5e-3) ? 0 : 1;
}

int test_kv_cns(int64_t T, int64_t kvh, int64_t hd, bool is_i8, bool zero_head,
                const char* what) {
    const int64_t bs = 8, nblk = 4;
    const double thr = 1.5, eps = 1e-6;
    const int64_t cols = kvh * hd;
    std::vector<float> kf, vf;
    auto kb = bfv((size_t)(T * cols), 1.8f, &kf);
    auto vb = bfv((size_t)(T * cols), 1.4f, &vf);
    if (zero_head) {
        for (int64_t d = 0; d < hd; ++d) {
            kb[(size_t)d] = 0; kf[(size_t)d] = 0.0f;
            vb[(size_t)d] = 0; vf[(size_t)d] = 0.0f;
        }
    }
    std::vector<int32_t> slot((size_t)T);
    for (int64_t t = 0; t < T; ++t) slot[t] = (t == 1 && T > 2) ? -1 : (int32_t)(t * 3 + 1);

    const int64_t row_elems = is_i8 ? (2 * hd + 4) : (2 * hd);
    const size_t kv_sz = (size_t)(nblk * kvh * bs * row_elems);
    std::vector<uint16_t> kv16(is_i8 ? 1 : kv_sz, 0xA5A5u);
    std::vector<uint8_t>  kv8(is_i8 ? kv_sz : 1, 0x5Au);

    RadTensor ts[4] = {
        tens(kb.data(), RAD_BF16, { T, cols }),
        tens(vb.data(), RAD_BF16, { T, cols }),
        tens(slot.data(), RAD_I32, { T }),
        tens(is_i8 ? (void*)kv8.data() : (void*)kv16.data(),
             is_i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_BF16,
             { nblk, kvh, bs, row_elems }),
    };
    Params P;
    P.i("M", T); P.i("head_dim", hd); P.i("n_head_kv", kvh); P.i("block_size", bs);
    P.f("threshold", thr); P.f("eps", eps); P.str("kv_dtype", is_i8 ? "i8" : "bf16");
    RadArgs a{}; a.t = ts; a.n_t = 4; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_kv_clip_norm_store(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %-52s launch %d\n", what, rc); return 1; }

    long bad = 0;
    double worst = 0;
    for (int64_t t = 0; t < T; ++t) {
        if (slot[t] < 0) continue;
        const int64_t blk = slot[t] / bs, off = slot[t] % bs;
        for (int64_t h = 0; h < kvh; ++h) {
            double sk = 0, sv = 0;
            for (int64_t d = 0; d < hd; ++d) {
                const double kv = kf[(size_t)(t * cols + h * hd + d)];
                const double vv = vf[(size_t)(t * cols + h * hd + d)];
                sk += kv * kv;
                sv += vv * vv;
            }
            const double fk = std::min(1.0, thr / (std::sqrt(sk) + 1e-6));
            const double rv = 1.0 / std::sqrt(sv / hd + eps);
            const size_t base = (size_t)(((blk * kvh + h) * bs + off) * row_elems);
            if (!is_i8) {
                for (int64_t d = 0; d < hd; ++d) {
                    const double wk = kf[(size_t)(t * cols + h * hd + d)] * fk;
                    const double wv = vf[(size_t)(t * cols + h * hd + d)] * rv;
                    const double gk = rad_bf16_to_f32(kv16[base + d]);
                    const double gv = rad_bf16_to_f32(kv16[base + hd + d]);
                    const double ek = std::fabs(gk - wk);
                    const double ev = std::fabs(gv - wv);
                    worst = std::max(worst, std::max(ek / (std::fabs(wk) + 1e-2),
                                                     ev / (std::fabs(wv) + 1e-2)));
                    if (ek > 0.01 * std::fabs(wk) + 2e-3 || ev > 0.01 * std::fabs(wv) + 2e-3)
                        ++bad;
                }
            } else {
                float ak = 0, av = 0;
                std::vector<float> kbv(hd), vbv(hd);
                for (int64_t d = 0; d < hd; ++d) {
                    kbv[d] = rad_bf16_to_f32(rad_f32_to_bf16((float)(kf[(size_t)(t * cols + h * hd + d)] * fk)));
                    vbv[d] = rad_bf16_to_f32(rad_f32_to_bf16((float)(vf[(size_t)(t * cols + h * hd + d)] * rv)));
                    ak = std::fmax(ak, std::fabs(kbv[d]));
                    av = std::fmax(av, std::fabs(vbv[d]));
                }
                const _Float16 k16 = (_Float16)std::fmin(ak / 127.0f, 515.0f);
                const _Float16 v16 = (_Float16)std::fmin(av / 127.0f, 515.0f);
                uint16_t k16_u = 0, v16_u = 0, got_ks = 0, got_vs = 0;
                std::memcpy(&k16_u, &k16, 2);
                std::memcpy(&v16_u, &v16, 2);
                std::memcpy(&got_ks, kv8.data() + base + 2 * hd, 2);
                std::memcpy(&got_vs, kv8.data() + base + 2 * hd + 2, 2);
                if (got_ks != k16_u || got_vs != v16_u) ++bad;
                const float ksc = (float)k16, vsc = (float)v16;
                for (int64_t d = 0; d < hd; ++d) {
                    const int want_kc = (int)std::fmin(std::fmax(ksc > 0.f ? std::nearbyint(kbv[d] / ksc) : 0.f, -127.f), 127.f);
                    const int want_vc = (int)std::fmin(std::fmax(vsc > 0.f ? std::nearbyint(vbv[d] / vsc) : 0.f, -127.f), 127.f);
                    const int got_kc = (int8_t)kv8[base + d];
                    const int got_vc = (int8_t)kv8[base + hd + d];
                    if (std::abs(got_kc - want_kc) > 1 || std::abs(got_vc - want_vc) > 1) ++bad;
                    worst = std::max(worst, (double)std::max(std::abs(got_kc - want_kc),
                                                             std::abs(got_vc - want_vc)));
                }
            }
        }
    }
    /* Negative slot row must keep its sentinel pattern. */
    if (T > 2) {
        const size_t skip_off = (size_t)((0 * kvh * bs + 0) * row_elems);
        if (!is_i8 && kv16[skip_off] != 0xA5A5u) ++bad;
        if (is_i8 && kv8[skip_off] != 0x5Au) ++bad;
    }
    const bool ok = bad == 0 && (is_i8 || worst < 1e-2);
    std::printf("  %-52s bad %ld  worst %.2e  %s\n", what, bad, worst, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int test_qr_l2(int64_t M, int64_t nq, int64_t gq, int64_t nr, int64_t gr, bool has_pos,
               const char* what) {
    const double thrq = 1.5, thrr = 1.2, floorv = 2.0, alpha = 0.3;
    std::vector<float> xqf, wqf, xrf, wrf;
    auto xq = bfv((size_t)(M * nq), 2.0f, &xqf);
    auto wq = bfv((size_t)nq, 0.9f, &wqf);
    auto xr = bfv((size_t)(M * nr), 1.8f, &xrf);
    auto wr = bfv((size_t)nr, 0.9f, &wrf);
    std::vector<int32_t> pos((size_t)M);
    for (int64_t r = 0; r < M; ++r) pos[r] = (int32_t)(r * 37);
    std::vector<uint16_t> yq((size_t)(M * nq), 0), yr((size_t)(M * nr), 0);

    std::vector<double> ref_q((size_t)(M * nq)), ref_r((size_t)(M * nr));
    for (int64_t t = 0; t < M; ++t) {
        const double tau = has_pos ? 1.0 + alpha * std::log(std::max(1.0, (pos[t] + 1.0) / floorv))
                                   : 1.0;
        for (int64_t g0 = 0; g0 < nq; g0 += gq) {
            double ss = 0;
            for (int64_t i = g0; i < g0 + gq; ++i) {
                const double v = (double)xqf[(size_t)(t * nq + i)] * (double)wqf[(size_t)i];
                ss += v * v;
            }
            const double f = std::min(1.0, thrq / (std::sqrt(ss) + 1e-6)) * tau;
            for (int64_t i = g0; i < g0 + gq; ++i)
                ref_q[(size_t)(t * nq + i)] = (double)xqf[(size_t)(t * nq + i)] * (double)wqf[(size_t)i] * f;
        }
        for (int64_t g0 = 0; g0 < nr; g0 += gr) {
            double ss = 0;
            for (int64_t i = g0; i < g0 + gr; ++i) {
                const double v = (double)xrf[(size_t)(t * nr + i)] * (double)wrf[(size_t)i];
                ss += v * v;
            }
            const double f = std::min(1.0, thrr / (std::sqrt(ss) + 1e-6)) * tau;
            for (int64_t i = g0; i < g0 + gr; ++i)
                ref_r[(size_t)(t * nr + i)] = (double)xrf[(size_t)(t * nr + i)] * (double)wrf[(size_t)i] * f;
        }
    }

    Params P;
    P.i("M", M); P.i("nq", nq); P.i("gq", gq); P.f("thrq", thrq);
    P.i("nr", nr); P.i("gr", gr); P.f("thrr", thrr); P.str("dtype", "bf16");
    if (has_pos) { P.f("floor", floorv); P.f("alpha", alpha); }
    RadTensor ts[7] = {
        tens(xq.data(), RAD_BF16, { M, nq }), tens(wq.data(), RAD_BF16, { nq }),
        tens(xr.data(), RAD_BF16, { M, nr }), tens(wr.data(), RAD_BF16, { nr }),
        tens(yq.data(), RAD_BF16, { M, nq }), tens(yr.data(), RAD_BF16, { M, nr }),
        has_pos ? tens(pos.data(), RAD_I32, { M }) : RadTensor{},
    };
    RadArgs a{}; a.t = ts; a.n_t = has_pos ? 7 : 6; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_qr_l2_clip(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %-52s launch %d\n", what, rc); return 1; }
    std::vector<float> got_q((size_t)(M * nq)), got_r((size_t)(M * nr));
    for (size_t i = 0; i < got_q.size(); ++i) got_q[i] = rad_bf16_to_f32(yq[i]);
    for (size_t i = 0; i < got_r.size(); ++i) got_r[i] = rad_bf16_to_f32(yr[i]);
    return (!close(what, ref_q, got_q, 5e-3) || !close(what, ref_r, got_r, 5e-3)) ? 1 : 0;
}

template <int NQ>
int test_mix_multi(int64_t M, int64_t n, int64_t nS, int w_mode, const char* what) {
    /* w_mode: 0 = none, 1 = f32, 2 = bf16 (only NQ == 3) */
    const int64_t MB = nS + 2, MT = M + 2;
    const double eps = 1e-6;
    std::vector<float> stf, curf;
    auto st  = bfv((size_t)(MB * MT * n), 1.4f, &stf);
    auto cur = bfv((size_t)(M * n), 1.1f, &curf);
    std::vector<std::vector<float>> qv(NQ), wf(NQ);
    std::vector<std::vector<uint16_t>> wb(NQ), yb(NQ);
    for (int k = 0; k < NQ; ++k) {
        qv[k].resize((size_t)n);
        for (auto& x : qv[k]) x = nrm() * 0.25f;
        yb[k].assign((size_t)(M * n), 0);
        if (w_mode == 1) {
            wf[k].resize((size_t)n);
            for (auto& x : wf[k]) x = 1.0f + 0.1f * nrm();
        } else if (w_mode == 2) {
            wb[k] = bfv((size_t)n, 0.3f, &wf[k]);
            for (int64_t c = 0; c < n; ++c) {
                wb[k][(size_t)c] = rad_f32_to_bf16(1.0f + 0.1f * nrm());
                wf[k][(size_t)c] = rad_bf16_to_f32(wb[k][(size_t)c]);
            }
        }
    }

    std::vector<std::vector<double>> ref(NQ, std::vector<double>((size_t)(M * n)));
    for (int64_t t = 0; t < M; ++t) {
        std::vector<std::vector<double>> v;
        for (int64_t i = 0; i <= nS; ++i) {
            std::vector<double> x((size_t)n);
            for (int64_t c = 0; c < n; ++c)
                x[(size_t)c] = i < nS ? stf[(size_t)(i * MT * n + t * n + c)]
                                      : curf[(size_t)(t * n + c)];
            double ms = 0;
            for (double e : x) ms += e * e;
            const double r = 1.0 / std::sqrt(ms / n + eps);
            for (double& e : x) e *= r;
            v.push_back(x);
        }
        for (int q = 0; q < NQ; ++q) {
            std::vector<double> sc;
            double mx = -1e300;
            for (auto& x : v) {
                double d = 0;
                for (int64_t c = 0; c < n; ++c) d += x[(size_t)c] * qv[q][(size_t)c];
                sc.push_back(d / std::sqrt((double)n));
                mx = std::max(mx, sc.back());
            }
            double den = 0;
            for (double& s : sc) { s = std::exp(s - mx); den += s; }
            std::vector<double> m((size_t)n, 0.0);
            for (size_t i = 0; i < v.size(); ++i)
                for (int64_t c = 0; c < n; ++c) m[(size_t)c] += sc[i] / den * v[i][(size_t)c];
            if (w_mode != 0) {
                double ms = 0;
                for (double e : m) ms += e * e;
                const double r = 1.0 / std::sqrt(ms / n + eps);
                for (int64_t c = 0; c < n; ++c) m[(size_t)c] *= r * wf[q][(size_t)c];
            }
            for (int64_t c = 0; c < n; ++c) ref[q][(size_t)(t * n + c)] = m[(size_t)c];
        }
    }

    RadTensor ts[11] = {};
    ts[0] = nS > 0 ? tens(st.data(), RAD_BF16, { MB, MT, n }) : RadTensor{};
    ts[1] = tens(cur.data(), RAD_BF16, { M, n });
    for (int k = 0; k < NQ; ++k) {
        ts[2 + k] = tens(qv[k].data(), RAD_F32, { 1, n });
        ts[2 + NQ + k] = tens(yb[k].data(), RAD_BF16, { M, n });
        if (NQ == 3 && w_mode == 1) ts[8 + k] = tens(wf[k].data(), RAD_F32, { n });
        if (NQ == 3 && w_mode == 2) ts[8 + k] = tens(wb[k].data(), RAD_BF16, { n });
    }
    Params P;
    P.i("M", M + 16); P.i("n", n); P.i("n_state", nS); P.f("eps", eps); P.str("dtype", "bf16");
    RadArgs a{};
    a.t = ts;
    a.n_t = (NQ == 3 && w_mode != 0) ? 11 : (2 + 2 * NQ);
    a.p = P.p.data();
    a.n_p = (int)P.p.size();
    const int rc = NQ == 3 ? r4d_rad_attn_res_mix3(&a, nullptr)
                           : r4d_rad_attn_res_mix4(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %-52s launch %d\n", what, rc); return 1; }
    for (int q = 0; q < NQ; ++q) {
        std::vector<float> got((size_t)(M * n));
        for (size_t i = 0; i < got.size(); ++i) got[i] = rad_bf16_to_f32(yb[q][i]);
        if (!close(what, ref[q], got, 5e-3)) return 1;
    }
    return 0;
}

}  // namespace

int main() {
    std::printf("inkling2 fusions (r4d emulator):\n");
    int fails = 0;
    fails += test_loop_ctl(4, 128, 8, "loop_ctl VEC8 M=4 n=128 rank=8");
    fails += test_loop_ctl(1, 2304, 64, "loop_ctl VEC8 M=1 n=2304 rank=64 (the model's)");
    fails += test_loop_ctl(2, 1096, 12, "loop_ctl VEC8 M=2 n=1096 rank=12 (past the prefetch, ragged)");
    fails += test_loop_ctl(2, 100, 5, "loop_ctl scalar M=2 n=100 rank=5");
    fails += test_kv_cns(4, 4, 128, false, false, "kv_clip_norm_store bf16 hd=128");
    fails += test_kv_cns(5, 4, 128, true,  true,  "kv_clip_norm_store i8 hd=128 + skip + zero");
    fails += test_kv_cns(3, 2, 256, true,  false, "kv_clip_norm_store i8 hd=256 (MAX_E=8)");
    fails += test_qr_l2(4, 512, 128, 128, 16, true,  "qr_l2_clip with positions (tau)");
    fails += test_qr_l2(3, 320, 320, 64,  16, false, "qr_l2_clip grp=320 fallback, no positions");
    fails += test_mix_multi<3>(3, 2304, 0, 0, "attn_res_mix3 nS=0 n=2304 no w");
    fails += test_mix_multi<3>(4, 2304, 4, 1, "attn_res_mix3 nS=4 n=2304 f32 w");
    fails += test_mix_multi<3>(3, 2304, 2, 2, "attn_res_mix3 nS=2 n=2304 bf16 w");
    fails += test_mix_multi<3>(2, 96,   6, 2, "attn_res_mix3 general nS=6 bf16 w");
    fails += test_mix_multi<4>(4, 2304, 1, 0, "attn_res_mix4 nS=1 n=2304");
    fails += test_mix_multi<4>(2, 96,   6, 0, "attn_res_mix4 general nS=6");
    std::printf("inkling: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
