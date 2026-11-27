/* exl3r4d_prefill.h -- the prefill band (M > 16) of exl3r4d: what its host adapter (prefill.c) hands its device code
 * (exl3r4d_prefill.hip), and the row hooks rows.c registers. The decode band's interface is exl3r4d_dev.h; this one
 * only adds to it.
 *
 * TWO FORMS IN ONE BAND, chosen at launch by SLOTS PER EXPERT, not by M (prefill.c exl3r4d_pf_decode_like):
 *   decode-like  runs of a row or two an expert (decode at C >= 5 with MTP depth 3 lands at M = 20..64): the decode
 *                band's kernels (exl3r4d_launch, MTX 1, one workgroup an expert's run);
 *   prefill      runs of tens of rows: the run-aligned map in scratch, one workgroup a map entry of up to 16 * MTX
 *                rows of one expert, one decoded weight fragment feeding MTX WMMAs (MTX 4 or 8). */
#ifndef EXL3R4D_PREFILL_H
#define EXL3R4D_PREFILL_H

#include <stdint.h>

#include "exl3r4d_dev.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The prefill form's argument block: the decode band's (operands, routing, the prep's rows in g.xr) plus the
 * run map. Passed by value to every kernel of the form; ~260 bytes, under the 768-byte kernel-argument limit
 * (docs/PLUGIN.md §5). */
/* The copy ring's host-pool ranges in the two-class block (the four-class block has no room: 504 of 512 explicit
 * bytes). */
#ifndef EXL3R4D_XR_ARGPOOLS
#define EXL3R4D_XR_ARGPOOLS 6
#endif
typedef struct Exl3r4dPrefill {
    Exl3r4dGemm g;      /* g.xr: the prep kernel's rotated rows (scratch); g.sk is 1 */
#include "exl3r4d_prefill_fields.h"
    /* The copy ring (two-class block only, after the shared fields; prefill.c copies the shared part into the
     * four-class block): this rank's host pools [lo, hi) as the cards address them (rad_host_pool_ranges), and the
     * rank's device counters (a reading; null: none). The kernels run on the engine's own AQL queues, which load the
     * code objects themselves, so the pools travel in the arguments rather than in device globals HIP would write. */
    unsigned long long  xr_lo[EXL3R4D_XR_ARGPOOLS], xr_hi[EXL3R4D_XR_ARGPOOLS];
    unsigned long long* xr_cnt;
    int32_t             xr_n, xr_pad2;
} Exl3r4dPrefill;
/* The four-class form's (interface v6; exl3r4d_dev.h, Exl3r4dGemm4): ~470 bytes. */
typedef struct Exl3r4dPrefill4 {
    Exl3r4dGemm4 g;
#include "exl3r4d_prefill_fields.h"
} Exl3r4dPrefill4;

/* The copy ring's word (exl3r4d_prefill_fields.h `xr`). */
#define xr_mode_of(a) ((int)((a).xr & 0xff))
#define xr_k2_of(a) ((int)(((a).xr >> 8) & 0xff))
#define xr_ring_of(a) ((int)((unsigned)(a).xr >> 16))

/* A map entry that owns no expert: rows [r0, r1) no run covers ([0, off[0]) and [off[n_expert], T)), zeroed. */
#define EXL3R4D_PF_ZERO 0xffffffffu

/* THE CONFIGURATION, read by both sides (prefill.c and exl3r4d_prefill.hip instantiate the same kernels):
 *   MTX, NPW   16-row blocks a workgroup owns at most (MTX 4 or 8), column tiles a wave
 *   AM_RUN,    the GEMM's A operand in the run form / the map form, a digit per op (gate_up + 8 * down):
 *   AM_MAP       0 fragments from the prep's rows in global memory each K step; 1 the rows rotated into LDS in the
 *                K loop, no prep; 2 the prep's rows staged through LDS once a workgroup; 3 AM 1 pipelined;
 *                4 gate_up's map form fused: a workgroup an entry across every column, each row rotated once
 *   PREP       the map form's prep, a digit per op (gate_up + 4 * down): 1 by map entry, 2 the map and the prep by row
 *              in one launch, 0 the decode band's prep kernel (one workgroup a live expert); the launch passes 3
 *              (none) for down with h_rot = 1 (interface v3: gate_up's epilogue already did down's step 1)
 * Every value here is a measured choice; the measurements at the end of exl3r4d_prefill.hip have the numbers. */
#ifndef EXL3R4D_PF_MTX
#define EXL3R4D_PF_MTX 4
#endif
#ifndef EXL3R4D_PF_NPW
#define EXL3R4D_PF_NPW 2
#endif
#ifndef EXL3R4D_PF_AM_RUN
#define EXL3R4D_PF_AM_RUN 0
#endif
#ifndef EXL3R4D_PF_AM_MAP
#define EXL3R4D_PF_AM_MAP 0
#endif
#ifndef EXL3R4D_PF_PREP
#define EXL3R4D_PF_PREP (1 + 4 * 2)
#endif
/* PAIR (gate_up's map form, an arm): map entries of up to 32 MTX rows; those of more than MTX 16-row blocks go to a
 * 16-wave workgroup of two 8-wave sets that share each tile's decode through LDS (exl3r4d_prefill.hip). */
#ifndef EXL3R4D_PF_PAIR
#define EXL3R4D_PF_PAIR 0
#endif
/* SCRATCH_MB (gate_up's map form): the most scratch the prep's rotated rows may take, in MiB; 0: no cap. Above it the
 * map form runs in chunks of map entries by row (a chunk = the entries starting in R consecutive sorted rows, so its
 * rows fit R + fm rows), prep then GEMM a chunk, each chunk's rows from the start of the same scratch. The weights are
 * still read once (a chunk is a range of experts). WHY A CAP: the scratch is charged to the card's VRAM budget at the
 * band top (M = the batched-token budget: 200 MiB at 2048 tokens, x 2 arenas), and that VRAM is expert slots the KV
 * cache would otherwise lend the slab: 46 keeps this op at or under the baseline's 47 MiB, and below ~46 the chunking
 * costs more (exl3r4d_prefill.hip's measurements). */
#ifndef EXL3R4D_PF_SCRATCH_MB
#define EXL3R4D_PF_SCRATCH_MB 46
#endif
/* LPT (gate_up's map form): each GEMM launch takes its entries longest first, through an order array the map kernel
 * writes after the map (scratch: 4 bytes an entry); 0: the map's order (expert by expert). 1 orders single entries,
 * 2 (the default) whole experts, so an expert's entries stay together and share its weight tile in cache: a launch's
 * tail of long workgroups shrinks (tp2 -2.0..-4.35% at M = 512..2048). */
#ifndef EXL3R4D_PF_LPT
#define EXL3R4D_PF_LPT 2
#endif
/* PGY: grid.y of a chunk's prep (its workgroups loop over the chunk's entries). */
#ifndef EXL3R4D_PF_PGY
#define EXL3R4D_PF_PGY 64
#endif

/* The device side, on `stream`. run 1 (the run form): one workgroup a (column block, live expert) found by a scan of
 * expert_offset (the decode band's find_run), passes of 16 * MTX rows over its run, no map; the decode band's prep
 * kernel unless am is 1. run 0 (the map form): the map kernel, the prep (prep 1 by map entry, 0 the decode band's),
 * one workgroup a (column block, map entry). am as above (g.xr is not read at am 1). bits: the codes' K (2..6; the A
 * arms are K = 4's). RAD_OK or a negative RAD_E_*; must not allocate, must not synchronise. */
int exl3r4d_launch_prefill(const Exl3r4dPrefill* a, int gate_up, int bits, int am, int prep, int run, void* stream);
#if EXL3R4D_OPSTAMP   /* a timing reading only (exl3r4d_prefill.hip) */
int exl3r4d_opstamp(int site, long long T, void* stream);   /* site: op * 2 + phase */
#endif
int exl3r4d_launch_prefill4(const Exl3r4dPrefill4* a, int gate_up, int bits, int am, int prep, int run,
                            void* stream);

/* Entries a map can need at most for T rows over n_expert experts at fm rows an entry: one partial entry an
 * expert, T / fm whole ones, and the two zero ranges' partials. The scratch hook and the launch use this one
 * function, so they cannot disagree about the map's size. */
static inline int64_t exl3r4d_pf_map_cap(int64_t T, int64_t n_expert, int64_t fm) {
    return n_expert + T / fm + 3;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* The host side (C, against the ABI): the prefill rows' hooks (prefill.c), registered in rows.c's M > 16 section,
 * and the decode band's operand checks they reuse (launch.c). Not seen by the device code. */
#ifndef __HIP__
#include "exl3r4d.h"
int     exl3r4d_pf_launch(const RadArgs* a, RadStream s, int op);
int64_t exl3r4d_pf_scratch(const RadArgs* a, int op);
/* launch.c's operand checks and argument block (the decode adapter's `fill`), exported for the prefill adapter; a
 * _mod4 op's classes 2 and 3 go to `hi`, and exl3r4d_gemm4 packs the four-class block from both (v6). */
int     exl3r4d_fill(const RadArgs* a, int op, Exl3Geom* g, Exl3r4dGemm* d, Exl3r4dCls hi[2]);
int     exl3r4d_gemm4(const Exl3r4dGemm* d, const Exl3r4dCls hi[2], Exl3r4dGemm4* o);
#endif

#endif /* EXL3R4D_PREFILL_H */
