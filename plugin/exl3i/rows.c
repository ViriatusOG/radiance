/* rows.c -- exl3i, the gfx1201 kernel library for EXL3-V3's dense linear (exl3i_dense): its identity, its rows, the
 * shape hook.
 *
 * Two rows of exl3i_dense, one launch (nt.c picks the path), one stored weight form (the same layout hooks), and
 * exl3i_actq (version 3: the producer-side A8 quantiser, AQ):
 *   exl3i_gemm_nt     act f16, codebook mad: the f16 WMMA kernel (nt_pass; T2 from 512 rows)
 *   exl3i_gemm_nt_a8  act a8t / a8g128 .. a8g1024, codebook mad: the A8 path (exl3i_a8.h) at launches
 *                     of 512 real rows or more, nt_pass below; mad at a8g128 and 1 .. 8 rows: the byte-sum GEMV
 *                     (exl3i_mad.hip.h, the rows quantised as the A8 path quantises them)
 * Refused at selection, by constraint: a codebook other than mad, an act no row lists, K or N not a multiple of
 * 128; by the shape hook and at launch: a rate
 * exl3i_rate_parse refuses (1/8 steps, 0.125 .. 8); by the scratch hook (declare) and at launch: an A8 group that does
 * not divide K; at declare, by the layout hook: an encoding whose rate, codebook or shape is not the op's; at launch
 * (nt.c): the same again plus N % 256 with suh.1, the alignments, and an A8 launch past the rows its scratch was
 * sized for.
 * And exl3i_rerank (include/exl3i_rerank.h): the drafter's candidates rescored against an exl3i head in the form
 * exl3i_dense stores it (its layout hooks: one copy of the head).
 * And the two token-embedding gathers (embed_exl3i.h: the token table as an exl3i trellis,
 * embed_tq.h: the rtn fwht128 table): a device row and a host row each, with the same ops, priorities and domains as
 * the dense op's rows. The host rows are the references (rad_kernel_reference); neither gather is concurrent. */
#include "exl3i.h"
#include "exl3i_rerank.h"
#include "embed_exl3i.h"
#include "embed_tq.h"
#include "rad_abi_layout.h"

#include <string.h>

static void desc(RadOpdDesc* d, uint32_t dt, uint32_t rank, const int64_t* shape, uint32_t fill) {
    memset(d, 0, sizeof *d);
    d->dtype = dt;
    d->rank = rank;
    for (uint32_t r = 0; r < rank; ++r) d->shape[r] = shape[r];
    d->fill = fill;
    d->idx_const = -1;
}

/* The operands as a tool synthesises them: a [M, K], the record as the container's plane (u32 [N/16, (K/16) m/2]: any
 * words are a valid trellis), suh / suh.1 [K], svh [N], y [M, N] f16. */
static int nt_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    Exl3iGeom g;
    if (exl3i_geom_parse(p, n_p, &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    if (operand < 0 || operand >= EXL3I_N_OPD || !out) return RAD_E_SHAPE;
    RadOpdDesc v[EXL3I_N_OPD];
    desc(&v[EXL3I_OPD_A], RAD_F16, 2, (int64_t[]){ g.M, g.K }, RAD_FILL_NORMAL);
    desc(&v[EXL3I_OPD_RECORD], RAD_U32, 2, (int64_t[]){ g.N / 16, exl3i_row_words(g.K, g.m) }, RAD_FILL_INDEX);
    v[EXL3I_OPD_RECORD].idx_max = (int64_t)1 << 32;
    desc(&v[EXL3I_OPD_SUH], RAD_F16, 1, (int64_t[]){ g.K }, RAD_FILL_NORMAL);
    desc(&v[EXL3I_OPD_SVH], RAD_F16, 1, (int64_t[]){ g.N }, RAD_FILL_NORMAL);
    desc(&v[EXL3I_OPD_SUH1], RAD_F16, 1, (int64_t[]){ g.K }, RAD_FILL_NORMAL);
    desc(&v[EXL3I_OPD_Y], RAD_F16, 2, (int64_t[]){ g.M, g.N }, RAD_FILL_NORMAL);
    desc(&v[EXL3I_OPD_AQ], RAD_U8, 2, (int64_t[]){ g.M, exl3i_aq_row_bytes(g.K, 1) }, RAD_FILL_INDEX);   /* read only with aq_planes */
    v[EXL3I_OPD_AQ].idx_max = 256;
    *out = v[operand];
    return RAD_OK;
}

static const RadConstraint c_nt[] = {
    RAD_CGE("M", 1),
    RAD_CIN("codebook", "mad"),
    RAD_CIN("act", "f16"),      /* the f16 WMMA row; the int8 / int4 activation modes are other rows' */
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
};
#define N_NT (int)(sizeof c_nt / sizeof c_nt[0])

/* The A8 modes the row lists (exl3_act.h spells any multiple of 128; these are the options the allocation search prices). */
static const RadConstraint c_a8[] = {
    RAD_CGE("M", 1),
    RAD_CIN("codebook", "mad"),
    RAD_CIN("act", "a8t a8g128 a8g256 a8g512 a8g1024"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
};
#define N_A8 (int)(sizeof c_a8 / sizeof c_a8[0])

/* exl3i_actq: K in 128s, 1 .. 4 planes */
static const RadConstraint c_aq[] = {
    RAD_CGE("M", 1),
    RAD_CDIV("K", 128),
    RAD_CGE("planes", 1),
    RAD_CLE("planes", EXL3I_AQ_MAX_PLANES),
};
#define N_AQ (int)(sizeof c_aq / sizeof c_aq[0])

/* exl3i_rerank: the device row's candidate ceiling, K and N in 128s */
static const RadConstraint c_rr[] = {
    RAD_CGE("M", 1),
    RAD_CGE("R", 1),
    RAD_CLE("R", EXL3I_RERANK_MAX_R),
    RAD_CIN("codebook", "mad"),
    RAD_CDIV("K", 128),
    RAD_CDIV("N", 128),
};
#define N_RR (int)(sizeof c_rr / sizeof c_rr[0])

/* embed_tq / embed_exl3i (embed_tq.h, embed_exl3i.h): the token-embedding gathers, one row a token; the same constraints
 * as the dense op's rows. */
static const RadConstraint c_embed_tq[] = {
    RAD_CDIV("n_embd", 128),
};
#define N_EMBED_TQ (int)(sizeof c_embed_tq / sizeof c_embed_tq[0])

static const RadConstraint c_embed_exl3i[] = {
    RAD_CDIV("n_embd", 128),
    RAD_CDIV("n_vocab", 128),
    RAD_CIN("codebook", "mad"),
};
#define N_EMBED_EXL3I (int)(sizeof c_embed_exl3i / sizeof c_embed_exl3i[0])

/* rows 0 .. 3 are the dense, quant and rerank rows above; 4 .. 7 the embedding gathers (embed_tq and embed_exl3i, device
 * then host) */
#define EXL3I_N_DENSE_ROWS 4

static const RadKernelInfo k_rows[] = {
    {
        .name = "exl3i_gemm_nt", .op = EXL3I_OP_NT, .family = "gemm",
        .computes = "EXL3-V3 dense linear: the dense GEMM's arithmetic (a x suh / suh.1 rotated, trellis decode in "
                    "registers, f16 WMMA, output rotation, x svh, bf16 / f16 / f32 y; T2 two-pass prefill) with the "
                    "decode at any rate m / 16 (integer K Plan<BITS>, fractional and sub-1 the plan table) in "
                    "IQR's computed code mad (on the int8 grid)",
        .shape = "any M; K % 128, N % 128 (N % 256 with suh.1); rates 0.125 .. 8 in 1/8 steps",
        .dtypes = "exl3i (rate=, codebook= mad); a f16 or bf16; y bf16, f16 or f32",
        .domain = RAD_DOMAIN_DEVICE, .priority = 100,
        .constraints = c_nt, .n_constraints = N_NT,
        .launch = exl3i_nt_launch, .scratch = exl3i_nt_scratch,
        .layout = exl3i_nt_layout, .relayout = exl3i_nt_relayout, .opd_shape = nt_shape,
        .unrelayout = exl3i_nt_unrelayout,
    },
    {
        .name = "exl3i_gemm_nt_a8", .op = EXL3I_OP_NT, .family = "gemm",
        .computes = "EXL3-V3 dense linear, int8 activations (A8): from 512 real rows a two-pass prefill -- the rows x suh "
                    "/ suh.1 rotated and quantised to int8 with one f32 scale a (row, G inputs), the mad trellis "
                    "decoded once into radiance's i8a8 fragment plane (int8 grid index q), radiance's i8a8 prefill tile "
                    "(activation scale folded a 128-K block), the output rotation x the grid step (4 K_INV) / sqrt(128) "
                    "x svh at the store; below 512 rows "
                    "exl3i_gemm_nt's f16 kernel on the same weight; mad at a8g128 from 1 to 8 rows the byte-sum GEMV "
                    "(each row quantised as the A8 path does, sudot4(hash, x 0x01010101) a weight, the -126 bias from the "
                    "rows' sums)",
        .shape = "any M (A8 from 512 rows); K % 128 (and % G), N % 128 (N % 256 with suh.1); rates 0.125 .. 8 in 1/8 "
                 "steps",
        .dtypes = "exl3i (rate=, codebook= mad); act a8t | a8g128 | a8g256 | a8g512 | a8g1024; a f16 or bf16; "
                  "y bf16, f16 or f32",
        .domain = RAD_DOMAIN_DEVICE, .priority = 100,
        .constraints = c_a8, .n_constraints = N_A8,
        .launch = exl3i_nt_launch, .scratch = exl3i_nt_scratch,
        .layout = exl3i_nt_layout, .relayout = exl3i_nt_relayout, .opd_shape = nt_shape,
        .unrelayout = exl3i_nt_unrelayout,
    },
    {
        .name = "exl3i_actq", .op = EXL3I_OP_ACTQ, .family = "quant",
        .computes = "EXL3-V3's A8 activation quantiser for up to four consumers of one x (exl3i_abi.h version 3): per row "
                    "and consumer i, x x s_i rotated (128 points), one f32 scale a 128 inputs, int8 -- the dense op's own "
                    "a8g128 planes, byte for byte, packed per row as [planes][K] int8 then [planes][K / 128] f32; with pre "
                    "rmsnorm / silu_mul / mul of the producer's bf16 output (libr4d's arithmetic), optionally stored to h; "
                    "with res rmsnorm's input is x + res, stored to xo (libr4d's add_rmsnorm)",
        .shape = "any M; K % 128; planes 1 .. 4",
        .dtypes = "x f16 or bf16 (bf16 with a pre stage); s_i f16; out u8; g f32 or bf16; x2, h, res, xo bf16",
        .domain = RAD_DOMAIN_DEVICE, .priority = 100,
        .constraints = c_aq, .n_constraints = N_AQ,
        .launch = exl3i_aq_launch, .opd_shape = exl3i_aq_shape,
        .layout = exl3i_aq_layout, .relayout = exl3i_aq_relayout, .unrelayout = exl3i_aq_unrelayout,
    },
    {
        .name = "exl3i_rerank", .op = EXL3I_OP_RERANK, .family = "moe",
        .computes = "logit_rerank against an exl3i head (include/exl3i_rerank.h): per row, x x suh rotated to f16 "
                    "(the dense op's rotation), each candidate's 128-row output-Hadamard block of the stored trellis decoded "
                    "(grid indices x the step for mad, as the decode form's DQ) and dotted in f32, its element of the "
                    "output rotation x 1/sqrt(128) x svh -- exl3i_dense's decode-form logit at that row to f32 rounding; "
                    "then the R re-sorted descending, ties to the lower id, pads last",
        .shape = "any M; R 1 .. 256; K % 128, N % 128; rates 0.125 .. 8 in 1/8 steps; R16 rows by the record's dtype",
        .dtypes = "exl3i (rate=, codebook= mad); x bf16 or f16; idx i32; val, pairs f32",
        .domain = RAD_DOMAIN_DEVICE, .priority = 100,
        .constraints = c_rr, .n_constraints = N_RR,
        .launch = exl3i_rerank_launch, .opd_shape = exl3i_rerank_shape,
        .layout = exl3i_nt_layout, .relayout = exl3i_nt_relayout, .unrelayout = exl3i_nt_unrelayout,
    },
    {
        .name = "embed_tq_gather", .op = EMBED_TQ_OP, .family = "vocab",
        .computes = "x[m] = bf16(H128(table[code] * scale)) of row tokens[m], one workgroup a token, the core's decode order",
        .shape = "n_embd % 128 == 0, bits 1..8; an id outside [0, n_vocab) writes a zero row",
        .dtypes = "i32 tokens, uB codes, bf16 scale per 128, f32 table, bf16 x",
        .domain = RAD_DOMAIN_DEVICE, .priority = 50,
        .constraints = c_embed_tq, .n_constraints = N_EMBED_TQ,
        .launch = embed_tq_device_launch, .layout = embed_tq_layout, .opd_shape = embed_tq_shape,
    },
    {
        .name = "ref_embed_tq", .op = EMBED_TQ_OP, .family = "vocab",
        .computes = "the host row: the same decode, products and butterfly in the same order",
        .shape = "n_embd % 128 == 0, bits 1..8",
        .dtypes = "i32 tokens, uB codes, bf16 scale per 128, f32 table, bf16 x",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_embed_tq, .n_constraints = N_EMBED_TQ,
        .launch = embed_tq_host_launch, .layout = embed_tq_layout, .opd_shape = embed_tq_shape,
    },
    {
        .name = "embed_exl3i_gather", .op = EMBED_EXL3I_OP, .family = "vocab",
        .computes = "x[m] = the exl3i table's row tokens[m]: a workgroup a (token, 128 columns) decodes its block's 64 tiles "
                    "from the stored record, folds the 128 rows with the token's Hadamard signs, rotates, scales; the host "
                    "row's order",
        .shape = "n_embd, n_vocab % 128 == 0, m even 2..128, mad, natural or R16 rows; an id outside "
                 "[0, n_vocab) writes a zero row",
        .dtypes = "i32 tokens, u32 record, f16 suh, f16 svh, bf16 x",
        .domain = RAD_DOMAIN_DEVICE, .priority = 50,
        .constraints = c_embed_exl3i, .n_constraints = N_EMBED_EXL3I,
        .launch = embed_exl3i_device_launch, .layout = embed_exl3i_layout, .opd_shape = embed_exl3i_shape,
    },
    {
        .name = "ref_embed_exl3i", .op = EMBED_EXL3I_OP, .family = "vocab",
        .computes = "the host row: the oracle's tile decode, then the same fold, butterfly and products in the same order",
        .shape = "n_embd, n_vocab % 128 == 0, m even 2..128",
        .dtypes = "i32 tokens, u32 record, f16 suh, f16 svh, bf16 x",
        .domain = RAD_DOMAIN_HOST, .priority = 0,
        .constraints = c_embed_exl3i, .n_constraints = N_EMBED_EXL3I,
        .launch = embed_exl3i_host_launch, .layout = embed_exl3i_layout, .opd_shape = embed_exl3i_shape,
    },
};

#ifndef EXL3I_BUILD_TARGET
#define EXL3I_BUILD_TARGET "gfx1201"
#endif
#ifndef EXL3I_BUILD_ID
#define EXL3I_BUILD_ID "unknown"
#endif

static const RadPluginInfo k_info = {
    RAD_PLUGIN_KERNEL, "exl3i", "0.3.0",
    "EXL3-V3's dense linear (exl3i_dense) on gfx1201: any rate in 1/8 steps, mad, f16 WMMA, A8 "
    "two-pass prefill on mad (build "
    EXL3I_BUILD_ID ")",
    EXL3I_BUILD_TARGET,
};

uint32_t             rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
const RadPluginInfo* rad_plugin_info(void) { return &k_info; }

int                rad_kernel_schema_count(void) { return exl3i_schema_count() + 3; }
const RadOpSchema* rad_kernel_schema_at(int i) {
    const int n = exl3i_schema_count();
    if (i < n) return exl3i_schema_at(i);
    if (i == n) return exl3i_rerank_schema();
    if (i == n + 1) return embed_tq_schema();
    return i == n + 2 ? embed_exl3i_schema() : 0;
}
int                rad_kernel_count(void) { return (int)(sizeof k_rows / sizeof k_rows[0]); }
const RadKernelInfo* rad_kernel_at(int i) { return i >= 0 && i < rad_kernel_count() ? &k_rows[i] : 0; }

/* as the dense GEMM's row: reads its operands and its scratch slice, keeps nothing between launches */
int rad_kernel_concurrent(int i) { return i >= 0 && i < EXL3I_N_DENSE_ROWS; }

/* the host rows of the two embedding gathers (rows 5 and 7) are their references; every other row
 * serves */
int rad_kernel_reference(int i) { return i == EXL3I_N_DENSE_ROWS + 1 || i == EXL3I_N_DENSE_ROWS + 3; }
