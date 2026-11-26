/* exl3r4d_dev.h -- what the C host side (launch.c) hands the device side (exl3r4d.hip): one plain
 * struct of kernel arguments and the extern "C" entry that launches it. No radiance types cross
 * this line except RadStream (an opaque pointer), so the .hip file needs nothing of rad_plugin.h.
 *
 * The struct is the kernel's argument block, passed by value: it has to stay under the engine's
 * 768-byte kernel-argument limit (docs/PLUGIN.md §5). It is about 240 bytes. */
#ifndef EXL3R4D_DEV_H
#define EXL3R4D_DEV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* WGQ: gate_up's decode GEMM drains per-WGP unit lists (exl3r4d.hip); its counter blocks come from the same
 * instance pool as the single-launch gate_up's (EXL3R4D_FUSE1). 0 = the grid. Set it for the C and the HIP side
 * alike (the argument block depends on it). */
#ifndef EXL3R4D_WGQ
#define EXL3R4D_WGQ 0
#endif
/* PDN: down's decode GEMM (the h_rot path) as a persistent grid: each workgroup lists the live experts once
 * in LDS and claims (expert, column block) units from one ticket in `ctr` (the counter block of the single-launch
 * form, EXL3R4D_FUSE1, from the down row's instance pool).
 * 0 = the grid. */
#ifndef EXL3R4D_PDN
#define EXL3R4D_PDN 0
#endif

/* One parity class of experts: the four plane tables (suh1 only for gate_up) and the class's
 * weight geometry. A table is the engine's pointer array when its *_s0 is 0, else a stacked base
 * with *_s0 bytes an entry (the form a tool holding plain buffers passes). */
typedef struct Exl3r4dCls {
    const void* codes;      /* the stored trellis record (layout.c), one entry an expert */
    const void* suh;        /* f16 [cols] */
    const void* suh1;       /* f16 [cols], gate_up only (the up half's input scales) */
    const void* svh;        /* f16 [rows] */
    const void* dn_suh;     /* f16 [rows / 2], gate_up with h_rot only: DOWN's suh of the same experts (v3 EPI 1) */
    int64_t     codes_s0, suh_s0, suh1_s0, svh_s0, dn_suh_s0;
    int32_t     rows, cols; /* this class's [N, K]; gate_up rows = 2 * width (gate then up) */
} Exl3r4dCls;

/* EXL3R4D_SUHC: the decode prep (prep_gu's PREP_ROW kernel, gate_up_down's GD_PROW prologue) reads each
 * expert's suh / suh.1 from a resident copy its row instance owns, never across the link for a pooled expert. Give it
 * to BOTH the C and the HIP flags: it changes this argument block. */
#ifndef EXL3R4D_SUHC   /* OFF: its 5.2 MB an instance (~504 MB a rank, allocated at init, before load) costs the expert
                        * pool ~490 resident units, and the pool's higher miss rate (0.347 -> 0.402 %) costs more
                        * than the copy saves: one stream's step 10.89 -> 11.00 ms with it on */
#define EXL3R4D_SUHC 0
#endif
typedef struct Exl3r4dGemm {
    Exl3r4dCls     cls[2];
#include "exl3r4d_gemm_fields.h"
} Exl3r4dGemm;

/* THE FOUR-CLASS BLOCK (interface v6, the _mod4 ops): four class records and the same fields after them. A record is
 * compact -- 32-bit entry strides, 72 bytes against 88 -- because four full records put the prefill form's block
 * (Exl3r4dPrefill4) past the 512 explicit bytes the engine's kernarg slot leaves (exl3r4d.hip, kRadKernargSlot); a
 * stacked table whose entry stride does not fit is refused at launch (launch.c). The device code is exl3r4d.hip and
 * exl3r4d_prefill.hip compiled a second time with EXL3R4D_NCLS 4 (exl3r4d_mod4.hip, exl3r4d_prefill_mod4.hip), where
 * these are the types its code names (exl3r4d_common.hip.h); the two-class build is the same source at 2, unchanged. */
typedef struct Exl3r4dCls4 {
    const void* codes, *suh, *suh1, *svh, *dn_suh;
    int32_t     codes_s0, suh_s0, suh1_s0, svh_s0, dn_suh_s0;
    int32_t     rows, cols;
} Exl3r4dCls4;
typedef struct Exl3r4dGemm4 {
    Exl3r4dCls4    cls[4];
#include "exl3r4d_gemm_fields.h"
} Exl3r4dGemm4;

/* The two ops' device launches. `gate_up` 1: moe_exl3_gate_up (two matrices, SiLU epilogue, f16 h);
 * 0: moe_exl3_down (one matrix, bf16 y). `npw` column tiles a wave (1, 2, 4). With a->xr set the
 * prep kernel runs first (into a->xr) and the GEMM reads it; else the GEMM rotates its rows into
 * LDS. RAD_OK, or a negative RAD_E_* naming what the device side refuses; launch errors come back
 * as RAD_E_DEVICE. Must not allocate and must not synchronise. */
int exl3r4d_launch_gemm(const Exl3r4dGemm* a, int gate_up, int bits, int npw, void* stream);
/* ...and the four-class build's (EXL3R4D_NCLS 4): the same launches over an Exl3r4dGemm4. */
int exl3r4d_launch_gemm4(const Exl3r4dGemm4* a, int gate_up, int bits, int npw, void* stream);

/* The prep kernel alone: every row of every live run of a->sorted / a->off rotated into a->xr (f16 [T][xr_pitch],
 * matrix m at m * cols), dropped rows zeros -- what the GEMM's A fragments read when a->xr is set. One workgroup a
 * live expert (grid a->nslot). exl3r4d_launch_gemm calls it; a second kernel of this library (the prefill band)
 * may too. RAD_OK, RAD_E_INVAL without a->xr, RAD_E_DEVICE on a launch error. */
int exl3r4d_launch_prep(const Exl3r4dGemm* a, int gate_up, void* stream);
int exl3r4d_fn4_dng(void);   /* the multi-block down: the fn4 switch exl3_dng 0 / 2 / 4 / 8 (prefill.c), 0 at start */
int exl3r4d_fn4_x3d(void);   /* the K = 2 / 2.5 decode forms: the fn4 switch exl3_x3d (prefill.c) 0 .. 4, 0 at start */
int exl3r4d_launch_prep4(const Exl3r4dGemm4* a, int gate_up, void* stream);
int exl3r4d_launch_prep_form(const Exl3r4dGemm* a, int gate_up, void* stream, int grid2d);
int exl3r4d_launch_prep_form4(const Exl3r4dGemm4* a, int gate_up, void* stream, int grid2d);

/* WGQ: every WGP's HW_ID1 key, found by a probe dispatch (instance init only: it synchronises), written to map_dev
 * [EXL3R4D_WGQ_KEYS] as dense ids 0.. in key order (-1: no such WGP). Returns the WGP count, or <= 0. */
#define EXL3R4D_WGQ_KEYS 256
int exl3r4d_wgp_probe(int* map_dev);

/* The single-launch forms' counter blocks: device memory allocated and zeroed (hipMalloc + hipMemset), and freed --
 * for a row instance's init / fini only, never the launch path. Null when the device refuses. */
void* exl3r4d_dev_zalloc(size_t bytes);
void  exl3r4d_dev_free(void* p);

/* The bytes of LDS one GEMM launch of this shape asks for, or a negative RAD_E_* when it does not
 * fit (the launch refuses the same shape the same way). */
int64_t exl3r4d_gemm_lds(int gate_up, int npw, int sk, int fused_rows, int64_t cols_max);

/* Workgroups of the decode gate_up GEMM (K = bits, npw, split-K sk, h_rot's staged block) resident at once on the
 * current device, or 0 when unknown (EXL3R4D_SK_FIT, launch.c). Cached a device after the first call. */
int exl3r4d_gu_resident(int bits, int npw, int sk, int h_rot);
/* The current device's WGPs, cached; 1 when unknown (it sizes the four-class gate_up's K slices). */
int exl3r4d_wgps(void);

/* Rows a fused-prep pass can hold in LDS for this op at K = cols_max (0: none), at most 16. */
int exl3r4d_fused_rows(int gate_up, int64_t cols_max);

/* Dynamic LDS a launch may ask for: 64 KiB less room for the kernels' static __shared__ (a launch
 * asking for all 64 KiB aborts the queue). */
#define EXL3R4D_LDS_MAX (64 * 1024 - 256)

/* In-GEMM rotation: no prep kernel for an op -- each GEMM workgroup rotates its rows into its own copy of the
 * scratch rows (one copy a 128-column block of the output) -- when that op's flag is set and the sorted rows are
 * at most EXL3R4D_GSCR_MAXT; above it the prep kernel runs. Shared by the launcher (scratch size) and the kernel.
 * gate_up's (off): bit-identical but slower -- the split chain with a_rot 0 against the prep_gu chain is tp2 M=1/2/3
 * 45.0/69.2/99.5 vs 38.8/60.1/80.7 us, tp1 +9.7..+19.7 us -- with and without 4 KB kernel alignment alike, so the
 * in-GEMM rotation's cost is not code placement (the measurements at the end of exl3r4d.hip). */
#ifndef EXL3R4D_GSCR_GU
#define EXL3R4D_GSCR_GU 0
#endif
#ifndef EXL3R4D_GSCR_DN   /* on for down: tp2 M=1 -21%, M=4 -9%, M=8 -4%; M=16 +1.7%, hence the cap */
#define EXL3R4D_GSCR_DN 1
#endif
#ifndef EXL3R4D_GSCR_MAXT
#define EXL3R4D_GSCR_MAXT 80
#endif
/* Single launch: gate_up's prep and GEMM in ONE launch, ordered by a counter block a (stream, output)
 * taken from a pool the row instance's init allocates and zeroes (never the launch path). 0 = two dispatches. */
#ifndef EXL3R4D_FUSE1
#define EXL3R4D_FUSE1 0
#endif
#define EXL3R4D_CTR_ROUND 64    /* a counter block's words rounded up to this (256 B): no two blocks share a line */
#define EXL3R4D_CTR_BLOCKS 64   /* counter blocks an instance: distinct (stream, output) pairs it serves fused */
/* A down workgroup owns EXL3R4D_DNG consecutive 128-column blocks of one run -- the expert search, the plane pointers
 * and (with the in-GEMM rotation) the rotation paid once for all of them, one scratch copy a workgroup.
 * 1 = a block a workgroup. */
#ifndef EXL3R4D_DNG
#define EXL3R4D_DNG 1
#endif


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3R4D_DEV_H */
