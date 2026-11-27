/* exl3i_plan_tab.h -- the fractional and sub-1 rates' lane plans (exl3i_plan.h) as a table built at compile time, so
 * a kernel reads its lane's plan (four dwords) instead of computing it: RateFrac's run-time plan was ~1000 VALU a lane
 * (S() loops, % WORDS), paid once a wave, and wdec8 / T2's wdec run a wave per 1-4 tiles, so at K2.5 the plan
 * cost more than the decode (wdec8 at K2.5 takes 1.35x K4's time per layer, up to 2.85x on gdn_z).
 *
 * Entry [m / 2][lane], m even 2..126 (integer m's rows are zero: RateInt keeps Plan<BITS>): per run r,
 * d[2r] = woff[r][0] | woff[r][1] << 8 | woff[r][2] << 16 | shift[r] << 24 and d[2r + 1] = off[r][0..3] as bytes;
 * xr (sub-1 only, m < 16) in its own table [m / 2][lane][r] = xr[r][0..1] | ..., two dwords a run. Bit for bit
 * exl3i_plan()'s fields (tests/exl3i_plan_tab_test.cpp checks every rate and lane on the host). C++17 constexpr: the
 * same functions as exl3i_plan.h, restated so they can run in a constant expression. */
#ifndef EXL3I_PLAN_TAB_H
#define EXL3I_PLAN_TAB_H
#include <stdint.h>

#ifdef __HIPCC__
#define EXL3I_CX constexpr __host__ __device__
#else
#define EXL3I_CX constexpr
#endif

namespace exl3i_pt {
EXL3I_CX unsigned mask_n(int n) {
    unsigned m = 0;
    for (int i = 0; i < 16; i++) if (((i + 1) * n) / 16 - (i * n) / 16) m |= 1u << i;
    return m;
}
EXL3I_CX int d(int q, int ka, unsigned mask) { return ka + (int)((mask >> (q & 15)) & 1u); }
EXL3I_CX int bpb(int ka, unsigned mask) {
    int b = 16 * ka;
    for (int i = 0; i < 16; i++) b += (int)((mask >> i) & 1u);
    return b;
}
EXL3I_CX unsigned salt16(int q) { return (0x9E37u * (unsigned)((q & 15) + 1)) & 0xFFFFu; }

struct Tab {
    uint32_t e[64][32][4];    /* [m / 2][lane]: per run (woff | shift), off bytes */
    uint32_t xr[8][32][4];    /* [m / 2][lane]: per run xr[0] | xr[1] << 16, xr[2] | xr[3] << 16 (m < 16) */
};

EXL3I_CX Tab build() {
    Tab t{};
    for (int m = 2; m <= 126; m += 2) {
        if (m % 16 == 0) continue;
        const int ka = m / 16;
        const unsigned mask = mask_n(m % 16);
        const int B = bpb(ka, mask), WORDS = B / 2, R = 32 * WORDS;
        int P[17] = {};   /* P[j] = d(0) + .. + d(j - 1): S(q) = (q >> 4) B + P[(q & 15) + 1], within the constexpr step budget */
        for (int j = 0; j < 16; j++) P[j + 1] = P[j] + d(j, ka, mask);
        auto Sq = [&](int q) { return (q >> 4) * B + P[(q & 15) + 1]; };
        for (int lane = 0; lane < 32; lane++) {
            const int n = lane & 15, h = lane >> 4;
            const int base = 32 * (n & 7) + 4 * (n >> 3) + 16 * h;
            for (int r = 0; r < 2; r++) {
                const int q0 = base + 8 * r;
                int e3 = Sq(q0 + 3) % R;
                if (e3 == 0) e3 = R;
                const int de = (e3 - 1) >> 5;
                const unsigned shift = (unsigned)(32 * (de + 1) - e3);
                uint32_t w = shift << 24;
                for (int x = 0; x < 3; x++) w |= (uint32_t)(((de - 2 + x) % WORDS + WORDS) % WORDS) << (8 * x);
                const int s3 = Sq(q0 + 3);
                uint32_t o = 0;
                uint32_t xr[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4; i++) {
                    o |= (uint32_t)(s3 - Sq(q0 + i)) << (8 * i);
                    xr[i] = d(q0 + i, ka, mask) == 0 ? salt16(q0 + i) : 0u;
                }
                t.e[m / 2][lane][2 * r] = w;
                t.e[m / 2][lane][2 * r + 1] = o;
                if (m < 16) {
                    t.xr[m / 2][lane][2 * r] = xr[0] | xr[1] << 16;
                    t.xr[m / 2][lane][2 * r + 1] = xr[2] | xr[3] << 16;
                }
            }
        }
    }
    return t;
}
}  // namespace exl3i_pt
#endif
