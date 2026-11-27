#include <string.h>
/* prefill.c -- the launch and scratch hooks of exl3r4d's M > 16 rows (rows.c, the prefill section).
 * Plain C against the ABI: nothing allocated, nothing synchronised, no environment read.
 *
 * THE FORM IS CHOSEN BY SLOTS PER EXPERT, NOT BY M. A band of M > 16 tokens holds two different workloads: a decode
 * step at C >= 5 with MTP depth 3 (M = 20..64; on the real model T = 200..640 rows over 512 experts, runs of a row
 * or two) and a prefill chunk (runs of tens of rows). The first is the decode band's problem and takes its kernels
 * (exl3r4d_launch: one workgroup an expert's run, MTX 1); the second takes the run-aligned MTX GEMM
 * (exl3r4d_prefill.hip). The crossover is libr4d's, a run length: decode-like while 2 T <= 3 n_expert -- an average
 * of 1.5 routed rows an expert (r4d_moe.hip r4d_moe_decode_band). Re-measured for these kernels: the measurements at
 * the end of exl3r4d_prefill.hip.
 *
 * WHAT A LAUNCH REFUSES is the decode adapter's list (launch.c, via exl3r4d_fill), plus RAD_E_SCRATCH when the
 * prefill form was not given the scratch its map and rows need, and RAD_E_SHAPE when the prep's rows would not be
 * addressable by the 32-bit lane offsets (T rows, or a chunk's under the scratch cap, x mats x K x 2 bytes >= 4 GiB). */
#include <stddef.h>

#include "exl3r4d_prefill.h"

/* X3D-F: fn4 switch "exl3_x3d" 0 / 1 -- 1: K = 2's decode reads a lane's eight states from one word pair
 * (exl3r4d_common.hip.h decode_frag_x3d; the same f16 weights by construction). 0, the start value, keeps the regular
 * kernels. The engine reaches it through these hooks (core/runtime/fn4_switch.cpp: every loaded object exporting
 * rad_fn4_switch); every other name answers 0 (not mine).
 * K25U: 2 / 3 / 4 = X3D-F (K = 2, as 1) + K = 2.5's decode with unpacked states (exl3r4d.hip k25_windows /
 * decode_frags_k25; the same f16 weights): gate_up G3 (2), G2D (3) or UB96 (4), down U4 at all three. 0 and 1 change
 * no K = 2.5 launch. */
static volatile int g_fn4_x3d = 0;
int exl3r4d_fn4_x3d(void) { return g_fn4_x3d; }
/* SPB-G: fn4 switch "exl3_dng" 0 / 2 / 4 / 8 -- the K = 2 (X3D-F) and K = 2.5 (K25U) down kernels take that many
 * 128-column blocks a workgroup (exl3r4d.hip spb_cb; launch.c fill() -> Exl3r4dGemm::dng). Every output byte is the
 * one-block grid's: the same code on the same data a block. 0, the start value, keeps the one-block grid.
 * SPB-P: 101 / 102 / 104 / 108 = the same kernels as a persistent grid over the launch's live units only, G = value - 100
 * column blocks a unit (exl3r4d.hip exl3r4d_gemm_pdx_kernel); the same bytes. */
static volatile int g_fn4_dng = 0;
int exl3r4d_fn4_dng(void) { return g_fn4_dng; }
/* fn4 switch "exl3_xr" 0 .. 31 -- gate_up's map form with host-read codes (exl3r4d_prefill.hip, "host-read codes:
 * order and copy ring"): bits 0-1 the order of a chunk's groups (0 longest first, 1 VRAM-read first, 2 host-read
 * first), bit 2 the ring copy (with order 1; only where the op was declared with `xring` MiB of ring). 0, the start
 * value, keeps the regular launches.
 * Order 3 (bits 0-1 = 3, bit 2 clear) = INTERLEAVE: each set longest first, the host-read entries spread evenly over
 * the first (4 - s) / 4 of the chunk's positions, s = bits 3-4 (3 = even over the whole chunk, 11 = the first 3/4, 19 =
 * the first half, 27 = the first quarter). The same entries in another order: the same bytes. Values with bits 3-4
 * and another order act as their bits 0-2. */
static volatile int g_fn4_xr = 0;
unsigned long long* exl3r4d_xr_rank_init(int rank);   /* exl3r4d_prefill.hip: the rank's device counters */
void exl3r4d_xr_note(int rank, void* stream);        /* exl3r4d_prefill.hip: the counters' periodic print */
/* The engine's host pools, per rank (radiance core: Mover::pool_ranges, called once a rank at start with the rank's card
 * current): [lo, hi) as the cards address them, kept here and copied into every two-class prefill launch of that rank
 * (Exl3r4dPrefill.xr_lo / xr_hi). Returns the ranges kept (at most EXL3R4D_XR_ARGPOOLS), or a negative RAD_E_*. */
#define XR_RANKS 64
static unsigned long long g_xr_plo[XR_RANKS][EXL3R4D_XR_ARGPOOLS], g_xr_phi[XR_RANKS][EXL3R4D_XR_ARGPOOLS];
static int g_xr_pn[XR_RANKS];
static unsigned long long* g_xr_cnt[XR_RANKS];
__attribute__((visibility("default"))) int rad_host_pool_ranges(int rank, int n, const unsigned long long* lo,
                                                                const unsigned long long* hi) {
    if (rank < 0 || rank >= XR_RANKS || n < 0 || (n && (!lo || !hi))) return RAD_E_INVAL;
    int m = 0;
    for (int i = 0; i < n && m < EXL3R4D_XR_ARGPOOLS; ++i)
        if (hi[i] > lo[i]) g_xr_plo[rank][m] = lo[i], g_xr_phi[rank][m] = hi[i], ++m;
    g_xr_cnt[rank] = exl3r4d_xr_rank_init(rank);
    g_xr_pn[rank] = m;
    return m;
}
/* The fn4 hooks (0 = not mine, -1 = refused value, 1 = taken; rad_fn4_value reports it): each switch name is one test
 * in each of the three functions. */
__attribute__((visibility("default"))) int rad_fn4_accepts(const char* name, int value) {
    if (!name) return 0;
    if (!strcmp(name, "exl3_x3d")) return value >= 0 && value <= 4 ? 1 : -1;   /* K25U: 2 .. 4 */
    if (!strcmp(name, "exl3_dng"))   /* SPB-G 2 / 4 / 8; SPB-P 101 / 102 / 104 / 108 (the persistent live-units form) */
        return value == 0 || value == 2 || value == 4 || value == 8 || value == 101 || value == 102 || value == 104 ||
               value == 108 ? 1 : -1;
    if (!strcmp(name, "exl3_xr")) return value >= 0 && value <= 31 ? 1 : -1;   /* order / ring; 8 .. 31: interleave */
    return 0;
}
__attribute__((visibility("default"))) int rad_fn4_switch(const char* name, int value) {
    const int ok = rad_fn4_accepts(name, value);
    if (ok <= 0) return ok;
    if (!strcmp(name, "exl3_x3d")) g_fn4_x3d = value;
    if (!strcmp(name, "exl3_dng")) g_fn4_dng = value;
    if (!strcmp(name, "exl3_xr")) g_fn4_xr = value;
    return 1;
}
__attribute__((visibility("default"))) int rad_fn4_value(const char* name, int* applied) {
    if (!name || !applied) return 0;
    if (!strcmp(name, "exl3_x3d")) { *applied = g_fn4_x3d; return 1; }
    if (!strcmp(name, "exl3_dng")) { *applied = g_fn4_dng; return 1; }
    if (!strcmp(name, "exl3_xr")) { *applied = g_fn4_xr; return 1; }
    return 0;
}

/* The configuration (exl3r4d_prefill.h) as this file uses it. Compile-time, not tunables: rad-tune cannot see a WTAB
 * op. */
/* AM 1 and 3 rotate in the GEMM: no prep, no rows in scratch. */
static int rot_am(int am) { return am == 1 || am == 3 || am == 4 || am == 5; }
/* AM 4 / 5 (gate_up's map form, fused: a workgroup an entry across every column) are taken only where they hold:
 * widths of at most 3 column blocks (tp2's 384 / 256: AM 4's registers, AM 5's 24 waves and LDS); elsewhere AM 0. */
static int am_of(const Exl3Geom* g, int run) {
    const int am = run ? EXL3R4D_PF_AM_RUN : EXL3R4D_PF_AM_MAP, a = g->gate_up ? am % 8 : am / 8;
    int wide = 0;
    for (int c = 0; c < g->n_class; ++c) wide |= g->rows[c] > 768;
    if ((a == 4 || a == 5) && (!g->gate_up || run || wide)) return 0;
    return a;
}

/* RUN FORM OR MAP FORM, by slots per expert: the run form (no map: one workgroup an expert's whole run) while
 * RD * T <= RN * n_expert, libr4d's decode band rule (2 T <= 3 n_expert) as the default. */
#ifndef EXL3R4D_PF_RN_GU
#define EXL3R4D_PF_RN_GU 3
#endif
#ifndef EXL3R4D_PF_RN_DN
#define EXL3R4D_PF_RN_DN 0   /* down: the map form at every T -- the run form was measured +8..+14% */
#endif
#ifndef EXL3R4D_PF_RD
#define EXL3R4D_PF_RD 2
#endif
/* gate_up's run form at K = 4 only. At K = 3 its MT 1 K loop compiles dependency-bound (256 instructions, 22
 * s_delay_alu, against the map form's 236 / 1) and runs +33..+37% at M = 17..64, where the map form is at K = 4's time;
 * K 2, 5, 6 take the map form likewise (RN_GU_KX). */
#ifndef EXL3R4D_PF_RN_GU_KX
#define EXL3R4D_PF_RN_GU_KX 0
#endif
static int64_t run_rn(int gate_up, int bits) {
    return gate_up ? (bits == 4 ? EXL3R4D_PF_RN_GU : EXL3R4D_PF_RN_GU_KX) : EXL3R4D_PF_RN_DN;
}
static int run_form(const Exl3Geom* g, int64_t T) {
    return EXL3R4D_PF_RD * T <= run_rn(g->gate_up, g->bits) * g->n_expert;
}

/* THE DECODE BAND'S KERNELS AT M > 16 (exl3r4d_launch), an arm: while XD * T <= XN * n_expert. MEASURED SLOWER than
 * the MTX GEMM at every M > 16 (down M=32 -15%, M=64 -20% on the MTX form; gate_up 0..-2%), so XN is 0. */
#ifndef EXL3R4D_PF_XN_GU
#define EXL3R4D_PF_XN_GU 0
#endif
#ifndef EXL3R4D_PF_XN_DN
#define EXL3R4D_PF_XN_DN 0
#endif
#ifndef EXL3R4D_PF_XD
#define EXL3R4D_PF_XD 2
#endif
static int decode_like(int gate_up, int64_t T, int64_t n_expert) {
    return EXL3R4D_PF_XD * T <= (gate_up ? EXL3R4D_PF_XN_GU : EXL3R4D_PF_XN_DN) * n_expert;
}

/* THE STREAMED y: down's map form writes y with whole-row nontemporal stores (exl3r4d_prefill.hip, the YS kernels)
 * when y (T rows x y_cols bf16) is at least YNT_MB MiB; 0: never. Measured, tp2 / tp1, us a layer, the _pf chain
 * gate_up -> down plus the next op: M=256 -6.5 / -14.2, 512 -34.7 / -46.3, 1024 -34.9 / -35.1, 2048 -26.1 / -15.9;
 * below that it does not pay (M=32 / 64 / 128 at tp2: -2.3 / +1.2 / +0.4, single op down M=32 +4.9) and y would leave
 * the L2 its reader (the gather) finds it in. 8: gfx1201's L2 is 8 MiB -- a
 * y larger than that cannot stay there for its reader anyway (M=128: 6.6 MB, M=256: 13 MB at 2560 columns). */
#ifndef EXL3R4D_PF_YNT_MB
#define EXL3R4D_PF_YNT_MB 8
#endif
static int ystream(const Exl3Geom* g, int64_t T, int64_t y_cols, int run, int am) {
    return EXL3R4D_PF_YNT_MB > 0 && !g->gate_up && !run && am == 0 &&
           T * y_cols * 2 >= ((int64_t)EXL3R4D_PF_YNT_MB << 20);
}

static int64_t round256(int64_t b) { return (b + 255) / 256 * 256; }

/* Rows a map entry at most: 16 MTX, twice that with PAIR in gate_up's map form (AM 0). */
static int64_t fm_of(const Exl3Geom* g, int run) {
    return 16 * EXL3R4D_PF_MTX * (EXL3R4D_PF_PAIR && g->gate_up && !run && am_of(g, run) == 0 ? 2 : 1);
}

/* The prep's rows (f16 [T][mats * x_cols]) then the map (a 16-byte header and 16 bytes an entry), then -- chunked --
 * the chunk table (16 bytes a chunk). */
/* v3 EPI 1: with h_rot = 1 down's input arrives rotated (gate_up's epilogue did down's step 1), so down has no prep:
 * its GEMM reads h itself as the prep's rows. */
static int dn_prerot(const Exl3Geom* g) { return !g->gate_up && g->h_rot; }
static int64_t row_bytes(const Exl3Geom* g) { return (g->gate_up ? 2 : 1) * g->x_cols * 2; }
static int64_t map_entries(const Exl3Geom* g, int64_t T) {
    return exl3r4d_pf_map_cap(T, g->n_expert, fm_of(g, run_form(g, T)));
}
/* THE SCRATCH CAP (EXL3R4D_PF_SCRATCH_MB): gate_up's map form chunked by row when T rows would not fit. A chunk is the
 * entries that START in R consecutive rows; its rows fit R + fm (an entry is at most fm rows), so the scratch holds
 * RMAX + fm rows whatever T is, and a launch takes the fewest chunks of at most RMAX rows, balanced (equal R). */
static int64_t chunk_rmax(const Exl3Geom* g, int64_t T, int run) {   /* 0: not chunked */
    const int64_t fm = fm_of(g, run), cap = (int64_t)EXL3R4D_PF_SCRATCH_MB << 20, rb = row_bytes(g);
    if (EXL3R4D_PF_SCRATCH_MB <= 0 || run || !g->gate_up || rot_am(am_of(g, run)) || T * rb <= cap) return 0;
    const int64_t R = cap / rb - fm;
    return R < fm ? fm : R;
}
static int64_t n_chunks(const Exl3Geom* g, int64_t T, int run) {
    const int64_t R = chunk_rmax(g, T, run);
    return R ? (T + R - 1) / R : 1;
}
static int64_t chunk_rows(const Exl3Geom* g, int64_t T, int run) {
    const int64_t n = n_chunks(g, T, run);
    return chunk_rmax(g, T, run) ? (T + n - 1) / n : 0;
}
static int64_t rows_bytes(const Exl3Geom* g, int64_t T, int run) {
    if (dn_prerot(g) || rot_am(am_of(g, run))) return 0;
    const int64_t R = chunk_rmax(g, T, run);
    return round256((R ? R + fm_of(g, run) : T) * row_bytes(g));
}
static int64_t map_only_bytes(const Exl3Geom* g, int64_t T) { return round256(16 + 16 * map_entries(g, T)); }
static int64_t ctab_bytes(const Exl3Geom* g, int64_t T, int run) {
    return chunk_rows(g, T, run) ? round256(16 * n_chunks(g, T, run)) : 0;
}
/* EXL3R4D_PF_LPT: gate_up's map form keeps an order array (4 bytes an entry) after the chunk table. */
static int lpt(const Exl3Geom* g, int run) { return EXL3R4D_PF_LPT && g->gate_up && !run; }
static int64_t map_bytes(const Exl3Geom* g, int64_t T, int run) {
    if (run) return 0;
    return map_only_bytes(g, T) + ctab_bytes(g, T, run) + (lpt(g, run) ? round256(4 * map_entries(g, T)) : 0);
}
/* The copy ring: the op's `xring` MiB (v10), gate_up of two classes only; 0 = none. */
static int64_t xring_mb(const RadArgs* a, const Exl3Geom* g) {
    if (!g->gate_up || g->n_class != 2) return 0;
    const int64_t v = rad_param_geti(a->p, a->n_p, "xring", 0);
    return v > 0 && v <= 1024 ? v : 0;
}
/* The copy ring's tables (xoff, xhead, xflag, xlist: an entry each; xcnt: 64 chunks) then the ring, after the order
 * array -- the layout exl3r4d_prefill.hip's xr_tab / xr_ring read. Only where the map form keeps an order array. */
static int64_t xr_bytes(const Exl3Geom* g, int64_t T, int run, int64_t mb) {
    if (run || !lpt(g, run) || mb <= 0) return 0;
    return round256(4 * (4 * map_entries(g, T) + 64)) + (mb << 20);
}
static int64_t pf_bytes(const Exl3Geom* g, int64_t T, int64_t mb) {
    const int run = run_form(g, T);
    return rows_bytes(g, T, run) + map_bytes(g, T, run) + xr_bytes(g, T, run, mb);
}

/* `op`: EXL3_GATE_UP or EXL3_DOWN, or its _mod4 form (v6). */
int exl3r4d_pf_launch(const RadArgs* a, RadStream s, int op) {
    Exl3Geom g;
    Exl3r4dPrefill d;
    Exl3r4dCls hi[2];
    const int gate_up = exl3_op_base(op) != EXL3_DOWN;
    memset(&d, 0, sizeof d);
    int rc = exl3r4d_fill(a, op, &g, &d.g, hi);
    if (rc != RAD_OK) return rc;
    if (d.g.T == 0) return RAD_OK;   /* no rows: nothing to write */
#if EXL3R4D_OPSTAMP   /* reading only: the op bracketed by device-clock stamps (exl3r4d_prefill.hip) */
    if (decode_like(gate_up, d.g.T, g.n_expert)) {
        (void)exl3r4d_opstamp((gate_up ? 2 : 3) * 2, d.g.T, s);
        const int lr = exl3r4d_launch(a, s, op);
        (void)exl3r4d_opstamp((gate_up ? 2 : 3) * 2 + 1, d.g.T, s);
        return lr;
    }
#else
    if (decode_like(gate_up, d.g.T, g.n_expert)) return exl3r4d_launch(a, s, op);
#endif
    const int64_t T = d.g.T, xr_pitch = dn_prerot(&g) ? d.g.x_pitch : (gate_up ? 2 : 1) * g.x_cols;
    const int run = run_form(&g, T), am = am_of(&g, run);
    /* A rotation in the GEMM would rotate an h_rot input a second time: those arms refuse it by name. */
    if (dn_prerot(&g) && rot_am(am)) return RAD_E_UNSUPPORTED;
    {   /* the A operand's 32-bit lane offsets: T rows, or a chunk's at most (the scratch cap) */
        const int64_t R = chunk_rmax(&g, T, run), xrows = R ? R + fm_of(&g, run) : T;
        if (!rot_am(am) && xrows * xr_pitch * 2 >= ((int64_t)1 << 32)) return RAD_E_SHAPE;
    }
    const int64_t xmb = xring_mb(a, &g), xrb = xr_bytes(&g, T, run, xmb);
    const int64_t need = rows_bytes(&g, T, run) + map_bytes(&g, T, run) + xrb;
    if (need > 0 && (!a->scratch || a->scratch_bytes < need)) return RAD_E_SCRATCH;
    if (need > 0 && !rad_ptr_aligned(a->scratch, 16)) return RAD_E_ALIGN;   /* the map's 16-byte entries */
    d.g.xr = dn_prerot(&g) ? (void*)d.g.x : rot_am(am) ? 0 : a->scratch;
    d.g.xr_pitch = xr_pitch;
    d.g.chunk = 16;
    d.g.sk = 1;
    d.map = run ? 0 : (uint32_t*)((char*)a->scratch + rows_bytes(&g, T, run));
    d.fm = (int32_t)fm_of(&g, run);
    d.cap = run ? 0 : (int32_t)exl3r4d_pf_map_cap(T, g.n_expert, d.fm);
    d.chunk_n = 1;
    d.ystream = ystream(&g, T, d.g.y_cols, run, am);
    d.order = lpt(&g, run) && !rot_am(am) && am == 0
                  ? (uint32_t*)((char*)d.map + map_only_bytes(&g, T) + ctab_bytes(&g, T, run)) : 0;
    /* fn4 exl3_xr: the order (bits 0-1) wherever the order array is; the ring (bit 2, order 1) where the op carries
     * one. 0: the regular launch. */
    {
        const int v = g_fn4_xr;
        int mode = 0;
        if (v && d.order && g.n_class == 2)
            mode = (v & 4) && xrb > 0 ? 5 : (v & 7) == 3 ? 3 | (v & 0x18) : (v & 3) == 3 ? 0 : (v & 3);
        const int ring = mode & 4 ? (int)xmb : 0;
        d.xr = (int32_t)((uint32_t)mode | ((uint32_t)exl3_k2(g.bits) << 8) | ((uint32_t)ring << 16));
        if (mode && a->rank >= 0 && a->rank < XR_RANKS) {
            const int r = a->rank;
            for (int i = 0; i < g_xr_pn[r]; ++i) d.xr_lo[i] = g_xr_plo[r][i], d.xr_hi[i] = g_xr_phi[r][i];
            d.xr_n = g_xr_pn[r];
            d.xr_cnt = g_xr_cnt[r];
            exl3r4d_xr_note(r, s);
        }
    }
    {
        const int64_t R = chunk_rows(&g, T, run);
        if (R) {   /* the scratch cap: chunks by row; a chunk's entries <= R / fm + 1 whole-or-partial entries of the
                    * runs it cuts, + one partial entry a run (expert or zero range) ending in it -- and <= cap */
            const int64_t fm = d.fm, runs = g.n_expert + 2 < R ? g.n_expert + 2 : R;
            const int64_t gy = R / fm + 1 + runs;
            d.chunk_n = (int32_t)n_chunks(&g, T, run);
            d.chunk_rows = (int32_t)R;
            d.chunk_gy = (int32_t)(gy < d.cap ? gy : d.cap);
            d.ctab = (uint32_t*)((char*)d.map + map_only_bytes(&g, T));
        }
    }
    /* prep 3: none (h_rot's down reads h as it is) */
    const int prep = dn_prerot(&g) ? 3 : gate_up ? EXL3R4D_PF_PREP % 4 : EXL3R4D_PF_PREP / 4;
#if EXL3R4D_OPSTAMP
    if (g.n_class != 4) {
        (void)exl3r4d_opstamp((gate_up ? 0 : 1) * 2, d.g.T, s);
        const int lr = exl3r4d_launch_prefill(&d, gate_up, g.bits, am, prep, run, s);
        (void)exl3r4d_opstamp((gate_up ? 0 : 1) * 2 + 1, d.g.T, s);
        return lr;
    }
#else
    if (g.n_class != 4) return exl3r4d_launch_prefill(&d, gate_up, g.bits, am, prep, run, s);
#endif
    /* v6: the four-class block -- the same fields after the decode-band block (exl3r4d_prefill_fields.h) */
    Exl3r4dPrefill4 d4;
    if ((rc = exl3r4d_gemm4(&d.g, hi, &d4.g)) != RAD_OK) return rc;
    memcpy(&d4.map, &d.map, offsetof(Exl3r4dPrefill, xr_lo) - offsetof(Exl3r4dPrefill, map));
    return exl3r4d_launch_prefill4(&d4, gate_up, g.bits, am, prep, run, s);
}
_Static_assert(offsetof(Exl3r4dPrefill, xr_lo) - offsetof(Exl3r4dPrefill, map) ==
                   sizeof(Exl3r4dPrefill4) - offsetof(Exl3r4dPrefill4, map),
               "the two prefill blocks share every field after their decode-band blocks (XR's pools: two-class only)");

/* Sized at declare for the band's top (M = its upper bound, T = M * top_k), from the parameters alone: the larger of
 * the two forms' needs, because a launch anywhere in the band may take either -- the decode form below the
 * crossover (the decode band's hook answers for it), the prefill form above it. Both grow with T, so the top bounds
 * the band. */
int64_t exl3r4d_pf_scratch(const RadArgs* a, int op) {
    Exl3Geom g;
    const int gate_up = exl3_op_base(op) != EXL3_DOWN;
    if (exl3_geom_parse(a->p, a->n_p, op, exl3_geom_two_classes(a->p, a->n_p), &g) != RAD_OK) return 0;
    /* The decode band's kernels only where the XN arm can send a launch to them (XN 0: never). */
    const int64_t T = g.M * g.top_k;
    const int64_t dec = (gate_up ? EXL3R4D_PF_XN_GU : EXL3R4D_PF_XN_DN) ? exl3r4d_scratch(a, op) : 0;
    if (decode_like(gate_up, T, g.n_expert)) return dec;
    /* Every form a launch in the band can take, at its largest T: the run form up to the crossover, the map form
     * above it; both grow with T. */
    int64_t need = pf_bytes(&g, T, xring_mb(a, &g));
    const int64_t Tr = run_rn(gate_up, g.bits) * g.n_expert / EXL3R4D_PF_RD;   /* the run form's largest T */
    if (Tr < T) {
        const int64_t r = rows_bytes(&g, Tr, 1);
        need = r > need ? r : need;
    }
    return need > dec ? need : dec;
}
