/* tests/r4d_emu: Muse Glimmer's vision-tower coordinate ops (r4d_vit_muse.hip) on the host SIMT
 * emulator.
 *
 * posemb_resample: held to hand-computed constants on a 4 x 4 table (identity, a 2 x 2 blend,
 * clamped corners of an 8 x 8 grid, bf16 output).
 *
 * grid_embed_pad: held to hand-computed constants -- a side-4 table resampled to an 8 x 8 grid,
 * whose last row and column reach taps off the table (zero-weight, grid_sample's zeros padding),
 * and an in-range 4 x 4 grid that reads the table whole. window_order: held to an independent host
 * reference that sorts the patches by (segment, window row, window column, raster), which shares
 * no arithmetic with the kernel's base-plus-rank slot formula; the inverse permutation, the rope
 * positions and every cuw boundary follow from that order.
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static RadTensor rows(void* p, uint32_t dt, int64_t r0, int64_t r1, int64_t ld) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = r0; t.shape[1] = r1;
    t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
static RadTensor vec(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}
static RadParam pi_(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam ps_(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

// ------------------------------------------------------------------ grid_embed_pad

static int test_pad() {
    const int64_t n = 3, side = 4;
    /* table[h * 4 + w] = (h * 4 + w + 1) / 4, three channels offset by 0.1 */
    std::vector<float> tabf((size_t)(side * side * n));
    for (int h = 0; h < side; ++h)
        for (int w = 0; w < side; ++w)
            for (int c = 0; c < n; ++c) tabf[(size_t)((h * side + w) * n + c)] = (h * side + w + 1) * 0.25f + 0.1f * c;
    std::vector<uint16_t> tab((size_t)(side * side * n));
    for (size_t i = 0; i < tab.size(); ++i) tab[i] = f2b(tabf[i]);

    /* An 8 x 8 grid: src(i) = i / 2 - 0.25 on each axis; per axis the taps are floor(src) and
     * floor(src) + 1 with hat weights, the off-table tap's weight zero. Row 0: taps -1 (masked)
     * and 0 at 3/4; row 7: tap 3 at 3/4 and 4 (masked); interior rows blend two adjacent taps. */
    const int64_t M = 6;
    /* Rows (0,0), (7,7), (1,3), (4,2), (7,0), (3,7) of the 8 x 8 grid. */
    const int rr[6] = { 0, 7, 1, 4, 7, 3 }, cc[6] = { 0, 7, 3, 2, 0, 7 };
    std::vector<int32_t> crd(4 * M);
    for (int m = 0; m < 6; ++m) {
        crd[m] = rr[m]; crd[M + m] = cc[m]; crd[2 * M + m] = 8; crd[3 * M + m] = 8;
    }
    std::vector<uint16_t> x((size_t)(M * n));
    for (size_t i = 0; i < x.size(); ++i) x[i] = f2b(7.0f);

    RadTensor t[3] = { rows(x.data(), RAD_BF16, M, n, n), rows(tab.data(), RAD_BF16, side * side, n, n),
                       rows(crd.data(), RAD_I32, 4, M, M) };
    RadParam p[4] = { pi_("M", M), pi_("n", n), pi_("side", side), ps_("dtype", "bf16") };
    RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = 4;
    if (r4d_rad_grid_embed_pad(&a, nullptr) != RAD_OK) { std::puts("grid_embed_pad: launch failed"); return 1; }

    /* Hand-computed: each axis's in-range weight (the off-table tap contributes nothing), the
     * joint weight the product h*w, the taps summed in libref's order (top-left, top-right,
     * bottom-left, bottom-right). Channel 0; the others sit 0.1 apart, and x starts at 7. */
    /* (0,0): both axes tap 0 only, at 3/4 each -> 9/16 * 0.25 = 0.140625. */
    /* (7,7): both axes tap 3 only at 3/4 -> 9/16 * 4 = 2.25. */
    /* (1,3): rows 0,1 at 3/4, 1/4; cols 1,2 at 3/4, 1/4 -> 0.8125. */
    /* (4,2): rows 1,2 at 1/4, 3/4; cols 1,2 at 3/4, 1/4 -> 2.3125. */
    /* (7,0): row tap 3 at 3/4 (tap 4 masked), col tap 0 at 3/4 (tap -1 masked) -> 1.828125. */
    /* (3,7): rows 1,2 at 3/4, 1/4; col tap 3 at 3/4 (tap 4 masked) -> 1.6875. */
    /* Hand-derived per row: the literal taps and weights (channel 0 verified against the printed
     * fractions, every channel against this table). Taps summed in libref's order: top-left,
     * top-right, bottom-left, bottom-right, each joint weight the product h*w. */
    struct Taps { int th0, th1, tw0, tw1; float wh0, wh1, ww0, ww1; };
    const Taps taps[6] = {
        { 0, 0, 0, 0, 0.0f, 0.75f, 0.0f, 0.75f },    /* (0,0): taps -1/4 masked, 0 at 3/4 */
        { 3, 3, 3, 3, 0.75f, 0.0f, 0.75f, 0.0f },    /* (7,7): taps 3 at 3/4, 4 masked */
        { 0, 1, 1, 2, 0.75f, 0.25f, 0.75f, 0.25f },  /* (1,3) */
        { 1, 2, 0, 1, 0.25f, 0.75f, 0.25f, 0.75f },  /* (4,2) */
        { 3, 3, 0, 0, 0.75f, 0.0f, 0.0f, 0.75f },    /* (7,0): row 4 and col -1 masked */
        { 1, 2, 3, 3, 0.75f, 0.25f, 0.75f, 0.0f },   /* (3,7): col 4 masked */
    };
    int bad = 0;
    for (int m = 0; m < 6; ++m) {
        const Taps& t6 = taps[m];
        for (int c = 0; c < n; ++c) {
            const float acc =
                b2f(tab[(size_t)((t6.th0 * side + t6.tw0) * n + c)]) * (t6.wh0 * t6.ww0) +
                b2f(tab[(size_t)((t6.th0 * side + t6.tw1) * n + c)]) * (t6.wh0 * t6.ww1) +
                b2f(tab[(size_t)((t6.th1 * side + t6.tw0) * n + c)]) * (t6.wh1 * t6.ww0) +
                b2f(tab[(size_t)((t6.th1 * side + t6.tw1) * n + c)]) * (t6.wh1 * t6.ww1);
            const float w = rb(7.0f + rb(acc));
            const float got = b2f(x[(size_t)(m * n + c)]);
            if (got != w) {
                ++bad;
                std::printf("grid_embed_pad: row %d ch %d: got %g want %g\n", m, c, got, w);
            }
        }
    }

    /* In range: a 4 x 4 grid on the side-4 table reads each row whole (src = i exactly). */
    const int64_t M2 = 2;
    std::vector<int32_t> crd2(4 * M2);
    crd2[0] = 2; crd2[1] = 3; crd2[4] = 4; crd2[5] = 4; crd2[6] = 4; crd2[7] = 4;
    crd2[2] = 1; crd2[3] = 2;
    std::vector<uint16_t> x2((size_t)(M2 * n));
    for (size_t i = 0; i < x2.size(); ++i) x2[i] = f2b(0.0f);
    RadTensor t2[3] = { rows(x2.data(), RAD_BF16, M2, n, n),
                        rows(tab.data(), RAD_BF16, side * side, n, n),
                        rows(crd2.data(), RAD_I32, 4, M2, M2) };
    RadParam p2[4] = { pi_("M", M2), pi_("n", n), pi_("side", side), ps_("dtype", "bf16") };
    RadArgs a2{}; a2.t = t2; a2.n_t = 3; a2.p = p2; a2.n_p = 4;
    if (r4d_rad_grid_embed_pad(&a2, nullptr) != RAD_OK) { std::puts("grid_embed_pad in-range: launch failed"); return 1; }
    for (int m = 0; m < 2; ++m)
        for (int c = 0; c < n; ++c) {
            const int h = m == 0 ? 2 : 3, w = m == 0 ? 1 : 2;
            const float want = rb(rb(tabf[(size_t)((h * side + w) * n + c)]) + 0.0f);
            if (b2f(x2[(size_t)(m * n + c)]) != want) {
                ++bad;
                std::printf("grid_embed_pad in-range: row %d ch %d: got %g want %g\n", m, c,
                            b2f(x2[(size_t)(m * n + c)]), want);
            }
        }
    return bad ? 1 : 0;
}

// ------------------------------------------------------------------ window_order

/* The independent reference: sort patches by (segment, window row, window column, row, col). */
struct Ord { int seg, wr, wc, r, c, p; };
static int test_window(int window, const std::vector<std::pair<int, int>>& grids) {
    /* Pack the grids back to back: segment s covers base .. base + gh*gw. */
    int P = 0;
    std::vector<int> cu{ 0 };
    std::vector<int> gh_(grids.size()), gw_(grids.size());
    for (size_t s = 0; s < grids.size(); ++s) {
        gh_[s] = grids[s].first; gw_[s] = grids[s].second;
        P += gh_[s] * gw_[s];
        cu.push_back(P);
    }
    std::vector<int32_t> crd(4 * (size_t)P, 0);
    std::vector<Ord> ord;
    for (size_t s = 0; s < grids.size(); ++s) {
        const int base = cu[s];
        for (int r = 0; r < gh_[s]; ++r)
            for (int c = 0; c < gw_[s]; ++c) {
                const int p = base + r * gw_[s] + c;
                crd[p] = r; crd[(size_t)P + p] = c;
                crd[2 * (size_t)P + p] = gh_[s]; crd[3 * (size_t)P + p] = gw_[s];
                ord.push_back({ (int)s, window > 0 ? r / window : 0, window > 0 ? c / window : 0,
                                r, c, p });
            }
    }
    std::stable_sort(ord.begin(), ord.end(), [](const Ord& l, const Ord& r) {
        if (l.seg != r.seg) return l.seg < r.seg;
        if (l.wr != r.wr) return l.wr < r.wr;
        if (l.wc != r.wc) return l.wc < r.wc;
        if (l.r != r.r) return l.r < r.r;
        return l.c < r.c;
    });
    std::vector<int32_t> widx((size_t)P), winv((size_t)P), wpos(2 * (size_t)P), wcuw((size_t)P + 1, P);
    for (int slot = 0; slot < P; ++slot) {
        widx[(size_t)slot] = ord[(size_t)slot].p;
        winv[(size_t)ord[(size_t)slot].p] = slot;
        wpos[(size_t)slot] = ord[(size_t)slot].c + 1;          /* pos[0][slot] = col + 1 */
        wpos[(size_t)P + (size_t)slot] = ord[(size_t)slot].r + 1;
    }
    if (window <= 0) {
        for (int i = 0; i <= (int)grids.size() && i <= P; ++i) wcuw[(size_t)i] = cu[(size_t)i];
    } else {
        /* cuw[k] is the base slot of window k: the patches before it in the permuted order. */
        int k = 0, base = 0;
        for (size_t s = 0; s < grids.size(); ++s) {
            const int nh = (gh_[s] + window - 1) / window, nw = (gw_[s] + window - 1) / window;
            for (int wr = 0; wr < nh; ++wr)
                for (int wc = 0; wc < nw; ++wc) {
                    wcuw[(size_t)k] = base;
                    int cnt = 0;
                    for (const Ord& o : ord)
                        if (o.seg == (int)s && o.wr == wr && o.wc == wc) ++cnt;
                    base += cnt;
                    ++k;
                }
        }
        wcuw[(size_t)k] = P;
    }

    std::vector<int32_t> idx((size_t)P, -1), inv((size_t)P, -1), pos(2 * (size_t)P, -1),
        cuw((size_t)P + 1, -1);
    RadTensor t[6] = { rows(crd.data(), RAD_I32, 4, P, P),   rows(cu.data(), RAD_I32, 1, (int64_t)cu.size(), (int64_t)cu.size()),
                       rows(idx.data(), RAD_I32, 1, P, P),   rows(inv.data(), RAD_I32, 1, P, P),
                       rows(pos.data(), RAD_I32, 2, P, P),   rows(cuw.data(), RAD_I32, 1, P + 1, P + 1) };
    RadParam p[2] = { pi_("M", P), pi_("window", window) };
    RadArgs a{}; a.t = t; a.n_t = 6; a.p = p; a.n_p = 2;
    if (r4d_rad_window_order(&a, nullptr) != RAD_OK) {
        std::printf("window_order w=%d: launch failed\n", window);
        return 1;
    }
    int bad = 0;
    for (int i = 0; i < P; ++i) {
        if (idx[(size_t)i] != widx[(size_t)i]) { ++bad; std::printf("window_order w=%d: idx[%d] %d want %d\n", window, i, idx[(size_t)i], widx[(size_t)i]); }
        if (inv[(size_t)i] != winv[(size_t)i]) { ++bad; std::printf("window_order w=%d: inv[%d] %d want %d\n", window, i, inv[(size_t)i], winv[(size_t)i]); }
        if (pos[(size_t)i] != wpos[(size_t)i]) { ++bad; std::printf("window_order w=%d: pos c[%d] %d want %d\n", window, i, pos[(size_t)i], wpos[(size_t)i]); }
        if (pos[(size_t)P + (size_t)i] != wpos[(size_t)P + (size_t)i]) { ++bad; std::printf("window_order w=%d: pos r[%d] %d want %d\n", window, i, pos[(size_t)P + (size_t)i], wpos[(size_t)P + (size_t)i]); }
        if (idx[(size_t)inv[(size_t)i]] != i) ++bad;           /* the inverse property */
    }
    for (size_t i = 0; i <= (size_t)P; ++i)
        if (cuw[i] != wcuw[i]) { ++bad; std::printf("window_order w=%d: cuw[%zu] %d want %d\n", window, i, cuw[i], wcuw[i]); }
    return bad ? 1 : 0;
}

// ------------------------------------------------------------------ posemb_resample

/* posemb_resample on the emulator, against hand-computed values. The table is a 4 x 4 grid of
 * two channels, t(h, w, c) = (4h + w) + 0.5c, in f32 so its values are exact. The same grid is
 * the table; a 2 x 2 grid of 4 x 4 blends each quadrant's four rows (coordinate 0.5 and 2.5); a
 * 8 x 8 grid's first and last cells clamp to the table's corners; and a bf16 output rounds once. */
static int test_posemb() {
    const int64_t side = 4, n = 2;
    std::vector<float> tab((size_t)(side * side * n));
    for (int h = 0; h < side; ++h)
        for (int w = 0; w < side; ++w)
            for (int c = 0; c < n; ++c) tab[(size_t)((h * side + w) * n + c)] = (float)(4 * h + w) + 0.5f * c;
    int bad = 0;
    auto run_pr = [&](const std::vector<float>& src, int64_t Th, int64_t Tw, uint32_t ydt,
                      std::vector<float>* outf, std::vector<uint16_t>* outb) {
        RadTensor t[2];
        t[0] = rows((void*)src.data(), RAD_F32, side * side, n, n);
        if (ydt == RAD_F32) {
            outf->assign((size_t)(Th * Tw * n), -99.0f);
            t[1] = rows(outf->data(), RAD_F32, Th * Tw, n, n);
        } else {
            outb->assign((size_t)(Th * Tw * n), 0);
            t[1] = rows(outb->data(), RAD_BF16, Th * Tw, n, n);
        }
        RadParam p[6] = { pi_("S_h", side), pi_("S_w", side), pi_("T_h", Th), pi_("T_w", Tw),
                          pi_("n", n), ps_("dtype", ydt == RAD_F32 ? "f32" : "bf16") };
        RadArgs a{}; a.t = t; a.n_t = 2; a.p = p; a.n_p = 6;
        return r4d_rad_posemb_resample(&a, nullptr);
    };

    /* The same grid: the table, bit for bit. */
    std::vector<float> same;
    if (run_pr(tab, side, side, RAD_F32, &same, nullptr) != RAD_OK) { std::puts("posemb_resample: launch failed"); return 1; }
    for (size_t i = 0; i < tab.size(); ++i)
        if (same[i] != tab[i]) { ++bad; std::printf("posemb_resample identity: i %zu got %g want %g\n", i, same[i], tab[i]); }

    /* 2 x 2: cell (0, 0) blends rows and columns {0, 1}, cell (1, 1) {2, 3}; mean of the four
     * products at weight 1/4, so y = 2.5 + 0.5c and 12.5 + 0.5c. */
    std::vector<float> two;
    if (run_pr(tab, 2, 2, RAD_F32, &two, nullptr) != RAD_OK) { std::puts("posemb_resample 2x2: launch failed"); return 1; }
    for (int c = 0; c < n; ++c) {
        const float want00 = 2.5f + 0.5f * c, want11 = 12.5f + 0.5f * c;
        if (two[(size_t)(0 * n + c)] != want00) { ++bad; std::printf("posemb_resample 2x2 (0,0) ch %d: got %g want %g\n", c, two[(size_t)(0 * n + c)], want00); }
        if (two[(size_t)((1 * 2 + 1) * n + c)] != want11) { ++bad; std::printf("posemb_resample 2x2 (1,1) ch %d: got %g want %g\n", c, two[(size_t)((1 * 2 + 1) * n + c)], want11); }
    }

    /* 8 x 8: the first cell's coordinate is -0.25 and clamps to the table's corner; the last is
     * 3.0 and reads the corner too. Both are one tap at weight 1. */
    std::vector<float> eight;
    if (run_pr(tab, 8, 8, RAD_F32, &eight, nullptr) != RAD_OK) { std::puts("posemb_resample 8x8: launch failed"); return 1; }
    for (int c = 0; c < n; ++c) {
        const float first = eight[(size_t)c], last = eight[(size_t)((8 * 8 - 1) * n + c)];
        if (first != 0.0f + 0.5f * c) { ++bad; std::printf("posemb_resample 8x8 first ch %d: got %g\n", c, first); }
        if (last != 15.0f + 0.5f * c) { ++bad; std::printf("posemb_resample 8x8 last ch %d: got %g\n", c, last); }
    }

    /* A bf16 output: 3 x 5 onto the table's f32 values, each cell rounded once to bf16. The
     * same arithmetic as the f32 output, so each stored value is that value rounded. */
    std::vector<float> f32out;
    std::vector<uint16_t> bf16out;
    if (run_pr(tab, 3, 5, RAD_F32, &f32out, nullptr) != RAD_OK) { std::puts("posemb_resample 3x5 f32: launch failed"); return 1; }
    if (run_pr(tab, 3, 5, RAD_BF16, nullptr, &bf16out) != RAD_OK) { std::puts("posemb_resample 3x5 bf16: launch failed"); return 1; }
    for (size_t i = 0; i < bf16out.size(); ++i) {
        const float want = rb(f32out[i]);
        if (b2f(bf16out[i]) != want) { ++bad; std::printf("posemb_resample 3x5 bf16 i %zu: got %g want %g\n", i, b2f(bf16out[i]), want); }
    }
    return bad ? 1 : 0;
}

int main() {
    int rc = 0;
    rc |= test_pad();
    rc |= test_posemb();
    /* One 48 x 40 segment, window 32: ragged right and bottom, windows 32x32, 32x8, 16x32, 16x8. */
    rc |= test_window(32, { { 48, 40 } });
    /* Video-style: two segments back to back, window 16. */
    rc |= test_window(16, { { 20, 24 }, { 10, 12 } });
    /* Window-divisible. */
    rc |= test_window(8, { { 16, 24 } });
    /* A window larger than the grid: one window holding the whole segment. */
    rc |= test_window(64, { { 12, 10 } });
    /* window <= 0: the identity, cuw = cu. */
    rc |= test_window(0, { { 9, 7 }, { 5, 5 } });
    if (!rc) std::puts("r4d_emu_vit: ok");
    return rc;
}
