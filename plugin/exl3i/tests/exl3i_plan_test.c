/* exl3i_plan_test.c -- exl3i_plan.h against the oracle (common/exl3i_decode.c), on the host: for every fractional
 * rate (m even, 2..126, m % 16 != 0), every lane and both runs, the lane's extraction exactly as the device does it
 * (alignbit of words woff, a third word and `mid` past KA 4, ubfe at off, the salt XOR) must give the oracle's state
 * for each of the lane's 8 positions, over random tiles. Exit 1 on any difference. */
#include "exl3i_plan.h"
#include "exl3i_decode.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t alignbit(uint32_t hi, uint32_t lo, unsigned s) { return (uint32_t)((((uint64_t)hi << 32) | lo) >> (s & 31)); }
static uint32_t ubfe(uint32_t x, unsigned o, unsigned w) { return (x >> o) & ((1u << w) - 1u); }

int main(void) {
    long bad = 0, n = 0;
    uint32_t seed = 12345;
    for (int m = 2; m <= 126; m += 2) {
        if (m % 16 == 0) continue;
        const int ka = m / 16, W = m / 2, nw = ka >= 5 ? 3 : 2;
        const unsigned mask = exl3i_mask_n(m % 16);
        if ((int)mask != exl3i_mask(m % 16)) { printf("mask %d differs\n", m); return 1; }
        for (int t = 0; t < 64; ++t) {
            uint32_t w[64];
            uint16_t val[256], st[256];
            for (int i = 0; i < W; ++i) { seed = seed * 1664525u + 1013904223u; w[i] = seed ^ (seed >> 13); }
            exl3i_decode_tile(w, m, EXL3I_CB_MAD, val, st);   /* the states do not depend on the codebook */
            for (int lane = 0; lane < 32; ++lane) {
                Exl3iPlan P;
                exl3i_plan(&P, lane, ka, mask);
                const int nn = lane & 15, h = lane >> 4, base = 32 * (nn & 7) + 4 * (nn >> 3) + 16 * h;
                for (int r = 0; r < 2; ++r) {
                    const uint32_t a0 = w[P.woff[r][0]], a1 = w[P.woff[r][1]], a2 = w[P.woff[r][2]];
                    const uint32_t lo = alignbit(a1, a2, P.shift[r]);
                    const uint32_t mid = nw == 3 ? alignbit(alignbit(a0, a1, P.shift[r]), lo, 16) : 0;
                    for (int i = 0; i < 4; ++i) {
                        const unsigned off = P.off[r][i];
                        if (off > (nw == 3 ? 32u : 16u)) { printf("m %d lane %d run %d: off %u out of reach\n", m, lane, r, off); return 1; }
                        uint32_t s = off <= 16 ? ubfe(lo, off, 16) : ubfe(mid, off - 16, 16);
                        const int q = base + 8 * r + i;
                        /* the oracle's states are the raw windows; the codebook reads state ^ salt at a 0-bit step */
                        bad += s != st[q];
                        bad += (s ^ P.xr[r][i]) != (exl3i_d(q, ka, mask) ? st[q] : (st[q] ^ exl3i_salt16(q)));
                        ++n;
                    }
                }
            }
        }
    }
    printf("exl3i plan: %ld lane-positions over 56 fractional rates, %ld differences\nPLAN %s\n", n, bad, bad ? "FAIL" : "PASS");
    return bad != 0;
}
