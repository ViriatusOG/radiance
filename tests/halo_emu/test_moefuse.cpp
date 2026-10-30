/* tests/halo_emu (from tests/r4d_emu/test_moefuse.cpp): GLM's decode fusions of the MoE block (halo_glm.hip) on the host SIMT emulator,
 * each byte for byte against the separate launches it replaces.
 *
 * router_sigmoid_topk_scatter against router_sigmoid_topk (the emulated kernel), then moe_scatter's
 * stable counting sort over its ids written out here, then the `fill` of the shared gate with 1:
 * random, tied and NaN ranks (a row with fewer finite ranks than top_k selects expert 0 for its
 * empty slots, and the sort takes those as written), slices and protected experts, rows past 64
 * wrapping over the waves, the sort's bound and its refusal past it. Pitch padding, guards past
 * every output and the gate's untouched rows show an overrun.
 *
 * swiglu_clamped_quant_fp8 against swiglu_clamped then quant_act_fp8 (both emulated): codes,
 * scales and the optional bf16 product, at GLM's widths and ragged ones, with values far past the
 * clamp, exactly on it and one bf16 step beyond, all-zero groups (scale 1), a gate so negative the
 * sigmoid underflows, and NaN and infinite inputs.
 *
 * moe_scatter_inv and moe_gather_inv: the small sort's kernel with the inverse against itself
 * without, and the inverse against the table the gather builds from `sorted`; then moe_gather's
 * kernel reading that inverse against its self-scan (T <= 512) or the prepass's table (above), at
 * GLM's f32 routing weights and bf16 ones, with the ungated shared arm, a sigmoid-gated one and
 * none, and slices that drop slots. moe_scatter's three-pass form (T > 1024) is not here: its
 * kernels take dynamic LDS through `extern __shared__`, which the emulator cannot give; the card
 * checks it (halo_selftest --case moe_scatter_inv_i32). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"
#include "ref_host.h"
#include "halo_moe_sort.h"
#include "halo_moe_gather.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int32_t kPoison = -77;
constexpr uint16_t kGatePoison = 0x7e57;

/* moe_scatter's contract: slot i = token * top_k + slot goes to its expert's run in ascending i;
 * an expert outside the slice drops the slot; protected experts are placed last in list order;
 * sorted rows past the live count are -1. */
struct SortModel {
    std::vector<int32_t> sorted, off, cnt;
};
int32_t place_of(int32_t e, int nl, const std::vector<int>& prot) {
    int32_t below = 0;
    for (size_t j = 0; j < prot.size(); ++j) {
        if (prot[j] == e) return nl - (int32_t)prot.size() + (int32_t)j;
        below += prot[j] < e;
    }
    return e - below;
}
SortModel sort_model(const std::vector<int32_t>& ids, int M, int k, int ip, int nl, int base,
                     const std::vector<int>& prot) {
    const int T = M * k;
    std::vector<int32_t> place((size_t)T);
    SortModel s;
    s.cnt.assign((size_t)nl, 0);
    s.off.assign((size_t)nl + 1, 0);
    s.sorted.assign((size_t)T, -1);
    for (int i = 0; i < T; ++i) {
        const int32_t e = ids[(size_t)(i / k) * ip + i % k] - base;
        place[(size_t)i] = e >= 0 && e < nl ? place_of(e, nl, prot) : -1;
        if (place[(size_t)i] >= 0) ++s.cnt[(size_t)place[(size_t)i]];
    }
    for (int e = 0; e < nl; ++e) s.off[(size_t)e + 1] = s.off[(size_t)e] + s.cnt[(size_t)e];
    std::vector<int32_t> seen((size_t)nl, 0);
    for (int i = 0; i < T; ++i)
        if (place[(size_t)i] >= 0) {
            const int32_t e = place[(size_t)i];
            s.sorted[(size_t)(s.off[(size_t)e] + seen[(size_t)e]++)] = i;
        }
    return s;
}

/* The table the gather builds from `sorted` (gather_kernel's self-scan, or its prepass): slot
 * sorted[p] at row p, every other slot -1. */
std::vector<int32_t> inverse_of(const std::vector<int32_t>& sorted) {
    std::vector<int32_t> inv(sorted.size(), -1);
    for (size_t p = 0; p < sorted.size(); ++p)
        if (sorted[p] >= 0) inv[(size_t)sorted[p]] = (int32_t)p;
    return inv;
}

/* moe_scatter's small-form kernel (halo_moe_sort.h) over contiguous ids, with or without the
 * inverse. */
struct Scatter {
    int rc = RAD_OK;
    std::vector<int32_t> sorted, off, cnt, inv;
};
Scatter scatter_kernel(const std::vector<int32_t>& ids, int nl, int base, const char* protect,
                       bool with_inv) {
    const int T = (int)ids.size();
    Scatter s;
    s.sorted.assign((size_t)T, kPoison);
    s.off.assign((size_t)nl + 1, kPoison);
    s.cnt.assign((size_t)nl, kPoison);
    s.inv.assign((size_t)T, kPoison);
    RadParam pp{};
    pp.key = "protect";
    pp.kind = RAD_P_STR;
    pp.sval = protect;
    RadArgs pa{};
    pa.p = &pp;
    pa.n_p = protect ? 1 : 0;
    halo_moesort::Protect prot;
    s.rc = halo_moesort::protect_of(&pa, nl, &prot);
    if (s.rc != RAD_OK) return s;
    if (with_inv)
        HALO_EMIT(nullptr, (halo_moesort::scatter_small_kernel<true>), dim3(1), dim3(256), 0,
                 ids.data(), s.sorted.data(), s.off.data(), s.cnt.data(), (uint32_t)T, (uint32_t)nl,
                 (int32_t)base, prot, s.inv.data());
    else
        HALO_EMIT(nullptr, (halo_moesort::scatter_small_kernel<false>), dim3(1), dim3(256), 0,
                 ids.data(), s.sorted.data(), s.off.data(), s.cnt.data(), (uint32_t)T, (uint32_t)nl,
                 (int32_t)base, prot, (int32_t*)nullptr);
    return s;
}

struct RouterShape {
    int M, ne, k, norm;
    int mode;              /* 0 random, 1 tied ranks, 2 NaN rows */
    int nl, base;          /* the slice: nl 0 is the whole router */
    const char* protect;
    bool gate;
};

void router_scatter_cases(std::mt19937& rng, int& cases, int& fails) {
    const RouterShape shapes[] = {
        { 1, 288, 8, 1, 0, 0, 0, nullptr, true },      { 2, 288, 8, 1, 0, 0, 0, nullptr, true },
        { 6, 288, 8, 1, 0, 0, 0, nullptr, true },      { 8, 288, 8, 1, 1, 0, 0, nullptr, true },
        { 9, 288, 8, 1, 0, 0, 0, nullptr, false },     { 16, 288, 8, 1, 2, 0, 0, nullptr, true },
        { 17, 288, 8, 1, 0, 0, 0, nullptr, true },     { 24, 288, 8, 1, 1, 0, 0, nullptr, true },
        { 33, 288, 8, 1, 0, 0, 0, nullptr, true },     { 48, 288, 8, 1, 0, 0, 0, nullptr, true },
        { 48, 288, 8, 1, 2, 0, 0, nullptr, true },     { 65, 288, 8, 1, 1, 0, 0, nullptr, true },
        { 100, 288, 8, 0, 0, 0, 0, nullptr, true },    { 128, 288, 8, 1, 0, 0, 0, nullptr, true },
        { 24, 288, 8, 1, 0, 144, 144, nullptr, true }, { 24, 288, 8, 1, 1, 96, 32, "3,7,90", true },
        { 48, 512, 8, 1, 0, 0, 0, nullptr, true },     { 32, 32, 32, 0, 1, 0, 0, nullptr, true },
        { 5, 300, 10, 1, 2, 0, 0, "0,299", true },
    };
    std::uniform_int_distribution<int> ud(-1000, 1000);
    for (const RouterShape& sh : shapes) {
        const int M = sh.M, ne = sh.ne, k = sh.k, T = M * k;
        const int nl = sh.nl ? sh.nl : ne, base = sh.base;
        const int lp = ne + 5, bp = 2, ip = k + 3, wp = k + 1, gp = 3;
        const float factor = 2.5f;
        std::vector<float> lg((size_t)M * lp), bias((size_t)ne * bp);
        for (auto& v : lg) v = sh.mode == 1 ? (float)(ud(rng) % 8) * 0.5f : (float)ud(rng) / 200;
        for (auto& v : bias)
            v = sh.mode == 1 ? (float)(ud(rng) % 5) * 0.125f : (float)ud(rng) / 2000;
        if (sh.mode == 2)
            for (int r = 0; r < M; r += 2) {
                /* every other row keeps (r % 5) finite ranks, fewer than top_k */
                for (int e = 0; e < ne; ++e) lg[(size_t)r * lp + e] = NAN;
                for (int j = 0; j < r % 5; ++j)
                    lg[(size_t)r * lp + (size_t)(ud(rng) + 1000) % ne] = 0.25f * (float)j;
            }
        std::vector<int> prot;
        if (sh.protect)
            for (const char* p = sh.protect; *p;) {
                char* end = nullptr;
                prot.push_back((int)std::strtol(p, &end, 10));
                p = *end == ',' ? end + 1 : end;
            }

        RadTensor t[8]{};
        t[0].data = lg.data();
        t[0].dtype = RAD_F32;
        t[0].rank = 2;
        t[0].shape[0] = M;
        t[0].shape[1] = ne;
        t[0].stride[0] = lp;
        t[0].stride[1] = 1;
        t[1].data = bias.data();
        t[1].dtype = RAD_F32;
        t[1].rank = 1;
        t[1].shape[0] = ne;
        t[1].stride[0] = bp;
        t[2].dtype = RAD_I32;
        t[2].rank = 2;
        t[2].shape[0] = M;
        t[2].shape[1] = k;
        t[2].stride[0] = ip;
        t[2].stride[1] = 1;
        t[3].dtype = RAD_F32;
        t[3].rank = 2;
        t[3].shape[0] = M;
        t[3].shape[1] = k;
        t[3].stride[0] = wp;
        t[3].stride[1] = 1;
        t[4].dtype = RAD_I32;
        t[4].rank = 1;
        t[4].shape[0] = T;
        t[4].stride[0] = 1;
        t[5].dtype = RAD_I32;
        t[5].rank = 1;
        t[5].shape[0] = nl + 1;
        t[5].stride[0] = 1;
        t[6].dtype = RAD_I32;
        t[6].rank = 1;
        t[6].shape[0] = nl;
        t[6].stride[0] = 1;
        t[7].dtype = RAD_BF16;
        t[7].rank = 2;
        t[7].shape[0] = M;
        t[7].shape[1] = 1;
        t[7].stride[0] = gp;
        t[7].stride[1] = 1;
        std::vector<RadParam> p(9);
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "n_expert", RAD_P_INT, ne };
        p[2] = { "top_k", RAD_P_INT, k };
        p[3] = { "norm", RAD_P_INT, sh.norm };
        p[4].key = "route_scale";
        p[4].kind = RAD_P_F64;
        p[4].dval = factor;
        p[5].key = "dtype";
        p[5].kind = RAD_P_STR;
        p[5].sval = "f32";
        p[6] = { "n_local", RAD_P_INT, nl };
        p[7] = { "expert_base", RAD_P_INT, base };
        p[8].key = "protect";
        p[8].kind = RAD_P_STR;
        p[8].sval = sh.protect;
        const int n_p = sh.protect ? 9 : 8;

        /* the separate launches: the selection, then the sort and fill modelled */
        std::vector<int32_t> ids_ref((size_t)M * ip, kPoison);
        std::vector<float> w_ref((size_t)M * wp, -7.25f);
        t[2].data = ids_ref.data();
        t[3].data = w_ref.data();
        RadArgs sel{};
        sel.t = t;
        sel.n_t = 4;
        sel.p = p.data();
        sel.n_p = 6;
        ++cases;
        if (halo_rad_router_sigmoid_topk(&sel, nullptr) != RAD_OK) {
            std::printf("FAIL router_sigmoid_topk launch M=%d ne=%d\n", M, ne);
            ++fails;
            continue;
        }
        const SortModel ref = sort_model(ids_ref, M, k, ip, nl, base, prot);
        /* moe_scatter's own kernel over the ids, contiguous as it reads them, held to the model;
         * with the inverse, which must be the table the gather builds from `sorted`. */
        std::vector<int32_t> flat((size_t)T);
        for (int i = 0; i < T; ++i) flat[(size_t)i] = ids_ref[(size_t)(i / k) * ip + i % k];
        Scatter ks = scatter_kernel(flat, nl, base, sh.protect, true);
        ++cases;
        if (ks.rc != RAD_OK || ks.sorted != ref.sorted || ks.off != ref.off || ks.cnt != ref.cnt ||
            ks.inv != inverse_of(ref.sorted)) {
            std::printf("FAIL moe_scatter kernel against its model at M=%d ne=%d slice %d+%d\n", M,
                        ne, base, nl);
            ++fails;
        }

        for (int with_inv = 0; with_inv < 2; ++with_inv) {
            std::vector<int32_t> ids((size_t)M * ip, kPoison), sorted((size_t)T + 4, kPoison),
                off((size_t)nl + 5, kPoison), cnt((size_t)nl + 4, kPoison),
                inv((size_t)T + 4, kPoison);
            std::vector<float> w((size_t)M * wp, -7.25f);
            std::vector<uint16_t> gate((size_t)(M + 2) * gp, kGatePoison);
            RadTensor tf[9];
            std::memcpy(tf, t, sizeof t);
            tf[2].data = ids.data();
            tf[3].data = w.data();
            tf[4].data = sorted.data();
            tf[5].data = off.data();
            tf[6].data = cnt.data();
            tf[7].data = sh.gate ? gate.data() : nullptr;
            tf[8] = tf[4];
            tf[8].data = inv.data();
            RadArgs a{};
            a.t = tf;
            a.n_t = with_inv ? 9 : 8;
            a.p = p.data();
            a.n_p = n_p;
            ++cases;
            const int rc = halo_rad_router_sigmoid_topk_scatter(&a, nullptr);
            int bad = 0;
            bad += ids != ids_ref;
            bad += std::memcmp(w.data(), w_ref.data(), w.size() * 4) != 0;
            bad += std::memcmp(sorted.data(), ks.sorted.data(), (size_t)T * 4) != 0;
            bad += std::memcmp(off.data(), ks.off.data(), (size_t)(nl + 1) * 4) != 0;
            bad += std::memcmp(cnt.data(), ks.cnt.data(), (size_t)nl * 4) != 0;
            if (with_inv) bad += std::memcmp(inv.data(), ks.inv.data(), (size_t)T * 4) != 0;
            for (int i = 0; i < 4; ++i)
                bad += sorted[(size_t)T + i] != kPoison || off[(size_t)nl + 1 + i] != kPoison ||
                       cnt[(size_t)nl + i] != kPoison || inv[(size_t)T + i] != kPoison;
            if (!with_inv)
                for (int i = 0; i < T; ++i) bad += inv[(size_t)i] != kPoison;
            for (int r = 0; r < M + 2; ++r)
                for (int c = 0; c < gp; ++c) {
                    const uint16_t want = sh.gate && r < M && c == 0 ? 0x3f80 : kGatePoison;
                    bad += gate[(size_t)r * gp + c] != want;
                }
            if (rc != RAD_OK || bad) {
                std::printf(
                    "FAIL router_sigmoid_topk_scatter at M=%d ne=%d k=%d mode %d slice %d+%d%s%s%s "
                    "(rc %d, %d mismatching outputs)\n",
                    M, ne, k, sh.mode, base, nl, sh.protect ? " protect " : "",
                    sh.protect ? sh.protect : "", with_inv ? " with inverse" : "", rc, bad);
                ++fails;
            }
        }
    }
    /* Past the sort's one workgroup: refused, nothing written. */
    {
        const int M = 129, ne = 288, k = 8;
        std::vector<float> lg((size_t)M * ne, 0.5f), bias((size_t)ne, 0.f),
            w((size_t)M * k, -7.25f);
        std::vector<int32_t> ids((size_t)M * k, kPoison), sorted((size_t)M * k, kPoison),
            off((size_t)ne + 1, kPoison), cnt((size_t)ne, kPoison);
        RadTensor t[7]{};
        auto mat = [](RadTensor& x, void* d, uint32_t dt, int r, int c) {
            x.data = d;
            x.dtype = dt;
            x.rank = 2;
            x.shape[0] = r;
            x.shape[1] = c;
            x.stride[0] = c;
            x.stride[1] = 1;
        };
        auto vec = [](RadTensor& x, void* d, int n) {
            x.data = d;
            x.dtype = RAD_I32;
            x.rank = 1;
            x.shape[0] = n;
            x.stride[0] = 1;
        };
        mat(t[0], lg.data(), RAD_F32, M, ne);
        t[1].data = bias.data();
        t[1].dtype = RAD_F32;
        t[1].rank = 1;
        t[1].shape[0] = ne;
        t[1].stride[0] = 1;
        mat(t[2], ids.data(), RAD_I32, M, k);
        mat(t[3], w.data(), RAD_F32, M, k);
        vec(t[4], sorted.data(), M * k);
        vec(t[5], off.data(), ne + 1);
        vec(t[6], cnt.data(), ne);
        RadParam p[6]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "n_expert", RAD_P_INT, ne };
        p[2] = { "top_k", RAD_P_INT, k };
        p[3] = { "norm", RAD_P_INT, 1 };
        p[4].key = "route_scale";
        p[4].kind = RAD_P_F64;
        p[4].dval = 2.5;
        p[5].key = "dtype";
        p[5].kind = RAD_P_STR;
        p[5].sval = "f32";
        RadArgs a{};
        a.t = t;
        a.n_t = 7;
        a.p = p;
        a.n_p = 6;
        ++cases;
        const int rc = halo_rad_router_sigmoid_topk_scatter(&a, nullptr);
        if (rc == RAD_OK || ids[0] != kPoison || sorted[0] != kPoison) {
            std::printf("FAIL router_sigmoid_topk_scatter served M*top_k=%d past the sort's bound "
                        "(rc %d)\n",
                        M * k, rc);
            ++fails;
        }
    }
}

struct QuantShape {
    int M, n;
    float limit;
    int mode; /* 0 random past the clamp, 1 clamp edges and zero groups, 2 NaN and inf */
    bool y; /* the fused op also writes the bf16 product */
};

void swiglu_quant_cases(std::mt19937& rng, int& cases, int& fails) {
    const QuantShape shapes[] = {
        { 1, 256, 10.f, 0, false },   { 8, 256, 10.f, 0, true },  { 48, 256, 10.f, 1, false },
        { 384, 256, 10.f, 0, false }, { 6, 1536, 10.f, 0, true }, { 3, 2048, 10.f, 1, true },
        { 5, 132, 7.f, 1, true },     { 7, 4, 10.f, 0, true },    { 9, 256, 10.f, 2, true },
        { 2, 260, 0.5f, 2, false },
    };
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ud(0, 15);
    for (const QuantShape& sh : shapes) {
        const int M = sh.M, n = sh.n, nkb = (n + 127) / 128;
        const int gp = 2 * n + 6, yp = n + 5, qp = n + 8, sp = nkb + 3;
        std::vector<uint16_t> gu((size_t)M * gp, f2b(-3.f));
        const uint16_t lim = f2b(sh.limit), past = (uint16_t)(lim + 1);
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < 2 * n; ++c) {
                float v = nd(rng) * 6.f;
                uint16_t h = f2b(v);
                if (sh.mode == 1) {
                    const int pick = ud(rng);
                    if (pick == 0)
                        h = lim;
                    else if (pick == 1)
                        h = past;
                    else if (pick == 2)
                        h = (uint16_t)(lim | 0x8000u);
                    else if (pick == 3)
                        h = (uint16_t)(past | 0x8000u);
                    else if (pick == 4)
                        h = f2b(-200.f);
                    /* rows divisible by 3 hold an all-zero first group */
                    if (r % 3 == 0 && (c % n) < 128) h = (c < n && (c & 1)) ? f2b(-0.f) : 0;
                } else if (sh.mode == 2) {
                    const int pick = ud(rng);
                    if (pick == 0)
                        h = 0x7fc0; /* NaN */
                    else if (pick == 1)
                        h = 0x7f80; /* +inf */
                    else if (pick == 2)
                        h = 0xff80; /* -inf */
                }
                gu[(size_t)r * gp + c] = h;
            }
        RadTensor t[4]{};
        t[0].data = gu.data();
        t[0].dtype = RAD_BF16;
        t[0].rank = 2;
        t[0].shape[0] = M;
        t[0].shape[1] = 2 * n;
        t[0].stride[0] = gp;
        t[0].stride[1] = 1;
        auto mat = [](RadTensor& x, uint32_t dt, int r, int c, int pitch) {
            x.dtype = dt;
            x.rank = 2;
            x.shape[0] = r;
            x.shape[1] = c;
            x.stride[0] = pitch;
            x.stride[1] = 1;
        };
        RadParam pc[4]{};
        pc[0] = { "M", RAD_P_INT, M };
        pc[1] = { "n", RAD_P_INT, n };
        pc[2].key = "limit";
        pc[2].kind = RAD_P_F64;
        pc[2].dval = sh.limit;
        pc[3].key = "dtype";
        pc[3].kind = RAD_P_STR;
        pc[3].sval = "bf16";
        RadParam pq[4]{};
        pq[0] = { "M", RAD_P_INT, M };
        pq[1] = { "n", RAD_P_INT, n };
        pq[2] = { "group", RAD_P_INT, 128 };
        pq[3].key = "dtype";
        pq[3].kind = RAD_P_STR;
        pq[3].sval = "bf16";
        RadParam pf[5]{};
        pf[0] = pc[0];
        pf[1] = pc[1];
        pf[2] = pc[2];
        pf[3] = pq[2];
        pf[4] = pc[3];

        std::vector<uint16_t> y_ref((size_t)M * yp, 0x7e57), y((size_t)M * yp, 0x7e57);
        std::vector<uint8_t> q_ref((size_t)M * qp, 0xa5), q((size_t)M * qp, 0xa5);
        std::vector<float> s_ref((size_t)M * sp, -7.25f), s((size_t)M * sp, -7.25f);
        RadTensor tc[2] = { t[0], {} };
        mat(tc[1], RAD_BF16, M, n, yp);
        tc[1].data = y_ref.data();
        RadArgs ac{};
        ac.t = tc;
        ac.n_t = 2;
        ac.p = pc;
        ac.n_p = 4;
        RadTensor tq[3] = { tc[1], {}, {} };
        mat(tq[1], RAD_F8E4M3, M, n, qp);
        tq[1].data = q_ref.data();
        mat(tq[2], RAD_F32, M, nkb, sp);
        tq[2].data = s_ref.data();
        RadArgs aq{};
        aq.t = tq;
        aq.n_t = 3;
        aq.p = pq;
        aq.n_p = 4;
        ++cases;
        if (halo_rad_swiglu_clamped(&ac, nullptr) != RAD_OK ||
            halo_rad_quant_act_fp8(&aq, nullptr) != RAD_OK) {
            std::printf("FAIL swiglu_clamped + quant_act_fp8 launch M=%d n=%d\n", M, n);
            ++fails;
            continue;
        }
        RadTensor tf[4] = { t[0], {}, {}, {} };
        mat(tf[1], RAD_F8E4M3, M, n, qp);
        tf[1].data = q.data();
        mat(tf[2], RAD_F32, M, nkb, sp);
        tf[2].data = s.data();
        mat(tf[3], RAD_BF16, M, n, yp);
        tf[3].data = sh.y ? y.data() : nullptr;
        RadArgs af{};
        af.t = tf;
        af.n_t = 4;
        af.p = pf;
        af.n_p = 5;
        const int rc = halo_rad_swiglu_clamped_quant_fp8(&af, nullptr);
        int bad = 0;
        bad += q != q_ref;
        bad += std::memcmp(s.data(), s_ref.data(), s.size() * 4) != 0;
        if (sh.y)
            bad += y != y_ref;
        else
            for (uint16_t v : y) bad += v != 0x7e57;
        if (rc != RAD_OK || bad) {
            std::printf("FAIL swiglu_clamped_quant_fp8 at M=%d n=%d limit %g mode %d%s (rc %d, %d "
                        "mismatching outputs)\n",
                        M, n, (double)sh.limit, sh.mode, sh.y ? " with bf16" : "", rc, bad);
            ++fails;
        }
    }
}

struct GatherShape {
    int M, n, nl, base;
    bool f32w; /* GLM's f32 routing weights, else bf16 */
    int shared; /* 0 none, 1 the ungated arm (gate 1, act none), 2 a sigmoid gate */
};

/* moe_gather's kernel (halo_moe_gather.h) as its launcher runs it, reading the table moe_scatter_inv
 * wrote, against the same kernel building the table itself from `sorted` (T <= 512) or reading the
 * prepass's table (above). Ids are distinct per row from 288 experts, a slice drops slots. */
void gather_inverse_cases(std::mt19937& rng, int& cases, int& fails) {
    const GatherShape shapes[] = {
        { 1, 4096, 288, 0, true, 1 },  { 6, 4096, 288, 0, true, 1 }, { 8, 512, 288, 0, true, 1 },
        { 24, 512, 144, 72, true, 1 }, { 48, 256, 288, 0, true, 1 }, { 64, 256, 288, 0, true, 0 },
        { 100, 256, 96, 0, true, 1 },  { 6, 300, 288, 0, false, 2 }, { 33, 256, 288, 0, false, 0 },
    };
    const int k = 8, ne = 288;
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const GatherShape& sh : shapes) {
        const int M = sh.M, n = sh.n, T = M * k, nl = sh.nl;
        std::vector<int32_t> ids((size_t)T);
        for (int r = 0; r < M; ++r) {
            std::vector<int32_t> pool(ne);
            for (int e = 0; e < ne; ++e) pool[(size_t)e] = e;
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int j = 0; j < k; ++j) ids[(size_t)(r * k + j)] = pool[(size_t)j];
        }
        Scatter plain = scatter_kernel(ids, nl, sh.base, nullptr, false);
        Scatter inv = scatter_kernel(ids, nl, sh.base, nullptr, true);
        ++cases;
        if (plain.rc != RAD_OK || inv.rc != RAD_OK || inv.sorted != plain.sorted ||
            inv.off != plain.off || inv.cnt != plain.cnt || inv.inv != inverse_of(plain.sorted)) {
            std::printf("FAIL moe_scatter with the inverse at M=%d slice %d+%d\n", M, sh.base, nl);
            ++fails;
            continue;
        }
        const int ye_ld = n + 3, ew_ld = k + 2, y_ld = n + 5, sh_ld = n + 1, sg_ld = 2;
        std::vector<uint16_t> ye((size_t)T * ye_ld), shp((size_t)M * sh_ld), sg((size_t)M * sg_ld);
        for (auto& v : ye) v = f2b(nd(rng));
        for (auto& v : shp) v = f2b(nd(rng));
        for (size_t i = 0; i < sg.size(); ++i) sg[i] = sh.shared == 1 ? f2b(1.f) : f2b(nd(rng));
        std::vector<float> ewf((size_t)M * ew_ld);
        std::vector<uint16_t> ewb((size_t)M * ew_ld);
        for (size_t i = 0; i < ewf.size(); ++i) {
            ewf[i] = std::fabs(nd(rng)) * 0.3f;
            ewb[i] = f2b(ewf[i]);
        }
        const void* ew = sh.f32w ? (const void*)ewf.data() : (const void*)ewb.data();
        const uint16_t* shv = sh.shared ? shp.data() : nullptr;
        const uint16_t* sgv = sh.shared ? sg.data() : nullptr;
        const int sig = sh.shared == 2;
        const bool small = T <= (int)halo_moegather::kGatherSelfT;
        /* the launcher's rule: the self-scan up to 512 slots, the prepass's table above */
        const std::vector<int32_t> table = inverse_of(plain.sorted);
        const unsigned gy = (unsigned)((n + 255) / 256);
        auto gather = [&](const int32_t* tab, const int32_t* sorted, std::vector<uint16_t>& y) {
            y.assign((size_t)M * y_ld, 0x7e57);
            if (sh.f32w)
                HALO_EMIT(nullptr, (halo_moegather::gather_kernel<256, true>), dim3((unsigned)M, gy),
                         dim3(256), 0, ye.data(), ew, tab, y.data(), (uint32_t)M, (uint32_t)n,
                         (uint32_t)k, (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld, sorted,
                         (uint32_t)T, shv, sgv, (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
            else
                HALO_EMIT(nullptr, (halo_moegather::gather_kernel<256, false>), dim3((unsigned)M, gy),
                         dim3(256), 0, ye.data(), ew, tab, y.data(), (uint32_t)M, (uint32_t)n,
                         (uint32_t)k, (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld, sorted,
                         (uint32_t)T, shv, sgv, (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
        };
        std::vector<uint16_t> y_old, y_new;
        if (small)
            gather(nullptr, plain.sorted.data(), y_old);
        else
            gather(table.data(), nullptr, y_old);
        gather(inv.inv.data(), nullptr, y_new);
        ++cases;
        if (y_new != y_old) {
            size_t first = 0;
            while (first < y_new.size() && y_new[first] == y_old[first]) ++first;
            std::printf("FAIL moe_gather from the inverse at M=%d n=%d slice %d+%d %s weights, "
                        "shared %d: first difference at %zu\n",
                        M, n, sh.base, nl, sh.f32w ? "f32" : "bf16", sh.shared, first);
            ++fails;
        }
    }
}

}   // namespace

int main() {
    std::mt19937 rng(23);
    int fails = 0, cases = 0;
    router_scatter_cases(rng, cases, fails);
    swiglu_quant_cases(rng, cases, fails);
    gather_inverse_cases(rng, cases, fails);
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
