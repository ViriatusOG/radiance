/* Derived from libr4d/r4d_gemm_mxfp4a8_gemv.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_mxfp4a8_gemv.hip - gemm_mxfp4a8_gemv (M <= 8) and gemm_mxfp4a8_gemv_wide
// (9..64 rows): the MXFP4 GEMM at decode heights.
//
// A row's output bits must not depend on how many rows ride with it (the selftest compares
// row 0 bit for bit from M = 1 through the wide row's 64). Both rows therefore run the
// skinny kernel's own accumulation -- per-32-block folds in ascending K, split partials
// summed in split order -- whose order is a function of (N, K) alone: the split law reads
// neither M nor the tile counts. libr4d reaches the same invariance with a register-streamed
// matvec at M <= 8 and a second WMMA form past it. Here the skinny kernel serves one row tile
// and narrow weights, and past 16 rows of a weight 3072 or more wide the LDS-staged tile
// (halo_gemm_mxfp4a8_tiled.hip) runs the same order at the same split: the skinny kernel
// re-reads and re-unpacks the weight once a row tile, the tile once (34816 x 5120 at 64 rows:
// 2.68 ms -> ~1.1 ms on gfx1151). The rows are told apart only by the M each accepts.
#include "halo.h"
#include "halo_gemm_args.h"
#include <cstdint>

static int rad_mxfp4_gemv(const RadArgs* a, RadStream s, long long m_lo, long long m_hi) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return HALO_NO(RAD_E_SHAPE);
    if (g.M < m_lo || g.M > m_hi) return HALO_NO(RAD_E_SHAPE);
    if (!g.ascale) return HALO_NO(RAD_E_INVAL);
    if (!g.bref) return HALO_NO(RAD_E_INVAL);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    Cfg c;
    /* NPW 2 untuned: two tiles a wave hide the unpack behind the second tile's loads
     * (measured 1367 -> ~970 us at N=5120 K=25600 M=1 on gfx1151; NPW 4 starves). */
    rc = cfg_for(a, g.K, g.M, HALO_MXFP4_GROUP, BR_FLAT, &c, 0, 2, 16);
    if (rc != RAD_OK) return rc;
    /* Past one row tile, on a weight wide enough to fill the card with 128-column blocks, the
     * LDS-staged tile at this split: the same bytes, since a row's order is the split law's
     * (K, SK) alone, but the weight is read and unpacked once for all the rows instead of once a
     * row tile, and the activation is shared through LDS. One row tile (M <= 16) and narrow N
     * stay here, where the skinny kernel's split-K waves are the parallelism. */
    if (g.M > 16 && g.N >= 3072) {
        rc = halo_mxfp4a8_tiled_split(a, s, g, mp, as_group, as_rs, c.sk);
        if (rc != RAD_E_UNSUPPORTED) return rc;
    }
    const long long kb = (g.K + 127) / 128;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        HALO_TRY("gemm_mxfp4a8_gemv",
            halo_gemm_mxfp4a8_nt_m64_plane(halo_addr(g.a) + p * g.K,
                                     halo_addr(g.ascale) + (long)(p * kb * 4),
                                     halo_addr(g.b) + n0 * g.K / 2, halo_addr(g.bscale) + n0,
                                     halo_addr(g.bref) + n0, halo_addr(g.y) + n0 * 2,
                                     (int)g.M, (int)g.K, (int)np, c.wv, c.sk, c.mb, c.npw,
                                     as_group, as_rs, (long)(uintptr_t)s,
                                     (int)(mp.lda ? mp.lda : g.K), (int)g.N, (int)g.N,
                                     (long)(uintptr_t)a->scratch, a->scratch_bytes));
    }
    return RAD_OK;
}

/* One entry point serves both rows (the wide row's table entry names this same function, as
 * in libr4d): the band is M itself, and the accumulation order does not change across it. */
extern "C" int halo_rad_gemm_mxfp4a8_gemv(const RadArgs* a, RadStream s) {
    return rad_mxfp4_gemv(a, s, 1, 64);
}
