/* tests/halo_emu: the tuned-configuration readers the emulated GEMM units call (halo_rows.cpp
 * defines them over its tunable tables; the emulator links the units without the row table).
 * These answer an untuned install's defaults -- tGemmSkinny's -- or the launch's own parameter. */
#include "rad_abi.h"

struct HaloGemmCfg { int wv, sk, mb, npw, nt; };
extern "C" void halo_cfg_gemm(const RadArgs* a, HaloGemmCfg* out) {
    auto g = [&](const char* k, int d) { long long v; return rad_args_geti(a, k, &v) ? (int)v : d; };
    out->wv = g("wv", 1); out->sk = g("sk", 0); out->mb = g("mb", 0); out->npw = g("npw", 1);
    out->nt = g("nt", 1);
}

struct HaloSplitCfg { int splits; };
extern "C" void halo_cfg_split(const RadArgs* a, HaloSplitCfg* out) {   /* tAttnSplit: 0, the split law */
    long long v;
    out->splits = rad_args_geti(a, "splits", &v) ? (int)v : 0;
}

/* halo_gemm_fp8a8.hip's gated row hands its tiled pair's product to halo_fused_quant_fp8.hip's
 * quantiser, a unit the emulator does not build (it carries the all-reduce headers). The emulated
 * gated cases run the fold (no scratch, M <= 16), so this is never reached there; it refuses rather
 * than answer. The pair itself is checked on the card (halo_selftest's fp8_numeric, byte for byte
 * against gemm_nt_q + gated_quant_fp8). */
extern "C" int halo_gated_quant_fp8(long, long, long, long, int, int, int, int, int, int, int, long) {
    return -1;
}
