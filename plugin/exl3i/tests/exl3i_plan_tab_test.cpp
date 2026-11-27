/* exl3i_plan_tab_test.cpp -- exl3i_plan_tab.h (the compile-time lane plans RateFrac reads) against exl3i_plan() (the
 * run-time plan, itself checked against the oracle by exl3i_plan_test.c): every fractional / sub-1 rate (m even 2..126,
 * m % 16 != 0), every lane, both runs, every field the kernels read (woff, shift, off, xr), and zero rows at integer
 * m. The table is a constexpr object here, so this also proves it fits the compiler's constant-evaluation budget. */
#include "exl3i_plan.h"
#include "exl3i_plan_tab.h"
#include <stdio.h>

static constexpr exl3i_pt::Tab kTab = exl3i_pt::build();

int main() {
    long bad = 0, n = 0;
    for (int m = 2; m <= 126; m += 2) {
        for (int lane = 0; lane < 32; ++lane) {
            const uint32_t* e = kTab.e[m / 2][lane];
            if (m % 16 == 0) { bad += (e[0] | e[1] | e[2] | e[3]) != 0; continue; }
            Exl3iPlan P;
            exl3i_plan(&P, lane, m / 16, exl3i_mask_n(m % 16));
            for (int r = 0; r < 2; ++r) {
                const uint32_t w = e[2 * r], o = e[2 * r + 1];
                bad += (w >> 24) != P.shift[r];
                for (int x = 0; x < 3; ++x) bad += (int)((w >> (8 * x)) & 0xFF) != P.woff[r][x];
                for (int i = 0; i < 4; ++i) {
                    bad += ((o >> (8 * i)) & 0xFF) != P.off[r][i];
                    const uint32_t xr = m < 16 ? (kTab.xr[m / 2][lane][2 * r + (i >> 1)] >> (16 * (i & 1))) & 0xFFFF : 0;
                    bad += xr != P.xr[r][i];
                    ++n;
                }
            }
        }
    }
    printf("exl3i plan table: %ld lane-positions over 56 rates vs exl3i_plan(), %ld differences\nPLAN TAB %s\n", n, bad,
           bad ? "FAIL" : "PASS");
    return bad != 0;
}
