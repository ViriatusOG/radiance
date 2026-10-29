/* r3d_rows.cpp -- what libr3d can compute, and for which geometry.
 *
 * One row per (kernel, band). A row is a `RadKernelInfo`: names, prose, the geometry it serves as
 * constraints, the tuning axes, and the entry points that launch it. The engine reads this table at
 * load, matches a requested op and geometry against it, and ranks the candidates by `priority`; it
 * never calls a function in this file to do that, and it never looks at a kernel's name. Nothing in
 * the engine, the graph, the containers or a test names libr3d's kernels: a card runs this library
 * because the rows say they cover its ops, and it stops running them when they do not.
 *
 * ============================================================ the schema table
 *
 * The first plugin in hierarchy order to declare an op fixes its schema, and a plugin whose schema
 * disagrees is refused at load, both plugins named (rad_abi.h, spec §2.3): arguments are positional,
 * so two schemas for one op would make one call mean two things depending on which row won. libref
 * declares the vocabulary in `docs/OPS.md`, so every schema below is libref's -- the same keys, the
 * same types, the same requiredness, the same operand roles and optionality, in the same order. The
 * prose is ours, because it states what *this* row computes; the loader's comparison
 * covers the positional slots and not the documents.
 *
 * An op libr3d invents needs a schema here too, and libref's side of it: an op whose schema is
 * declared only by a kernel plugin cannot be checked against a reference, and an op that exists in
 * two kernels with two schemas breaks the rule above. That is why libr3d declares no new op: every
 * row below implements an op libref declares.
 *
 * ============================================================ the constraint table
 *
 * A constraint is a necessary condition on the geometry (spec §2.1). "bf16 only" is not one: the
 * dtype of an operand is a property of the call, and the shim reads the tensor. What goes here is
 * what the *kernel* cannot do -- the divisibility its lane map assumes, the band its grid is sized
 * for, the scheme its decoders implement -- and what the selector needs to choose between rows.
 * A constraint the shim also checks costs nothing; a constraint the kernel does not actually
 * enforce is a promise the tests will break.
 *
 * ============================================================ rows are positional
 *
 * `RadKernelInfo` is initialised positionally, as `rad_abi.h` requires: a field added anywhere but
 * the end shifts every later initialiser, and `nullptr` converts to any function pointer, so part
 * of the shift compiles. The order below is the struct's; the trailing slots a row does not use are
 * left off, which aggregate initialisation zeroes. A row's entry points are, in order: init, fini,
 * launch, scratch, layout, relayout, opd_shape, unrelayout, replaces.
 *
 * The only `layout` and `relayout` hooks in this table present the container's own planes as the
 * joined operands they store -- the w2a8 row's `b_scale`, the fp8 weight's codes and scale, the
 * logits head -- and repack nothing, so a container converted on this card keeps row-major codes
 * against row-major scales. A dot2 GEMM reads a weight in the order it is stored; libr4d's fragment
 * permutations are a WMMA fact, and RDNA3's wave-matrix instructions use a 16x16 fragment layout
 * this table's dot2 kernels do not (r3d_common.h has the measurements).
 *
 * Pure host C++: no r3d.h, no HIP, so this table compiles and can be read and tested without ROCm.
 */

#include "r3d_plugin.h"

#ifndef R3D_BUILD_TARGET
#define R3D_BUILD_TARGET "gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151"
#endif

#include <cstddef>
#include <cstdlib>

#define ROW(a)  (a), (int)(sizeof(a) / sizeof((a)[0]))
#define NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* Parameter and constraint spellings, shared with libref's table. */
#define P_INT(k)   { (k), RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { (k), RAD_P_INT, RAD_OPTIONAL }
#define P_INTD(k)  { (k), RAD_P_INT, RAD_DERIVED }
#define P_STR(k)   { (k), RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { (k), RAD_P_STR, RAD_OPTIONAL }
#define P_F64(k)   { (k), RAD_P_F64, RAD_REQUIRED }
#define P_F64O(k)  { (k), RAD_P_F64, RAD_OPTIONAL }

#define C_EQ(k, v)  { (k), RAD_C_EQ,  (long long)(v), nullptr }
#define C_LE(k, v)  { (k), RAD_C_LE,  (long long)(v), nullptr }
#define C_GE(k, v)  { (k), RAD_C_GE,  (long long)(v), nullptr }
#define C_DIV(k, v) { (k), RAD_C_DIV, (long long)(v), nullptr }
#define C_IN(k, s)  { (k), RAD_C_IN,  0, (s) }

/* A row with one implementation and nothing to measure. */
#define NOTUNE nullptr, 0, nullptr

#define SCHEMA(op, p, o, doc) { (op), ROW(p), ROW(o), (doc) }

/* ================================================================== schemas
 *
 * Operand names are diagnostics only; roles and optionality are positional and must match. Where
 * libref's array carries a name libr3d's reads differently (`b` for a gate, say) the names differ
 * and the schema does not.
 */

/* --- norms and elementwise --- */
static const RadParamSpec pRmsNorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                         P_F64O("wadd"), P_F64O("mult") };
static const RadOperandSpec oRmsNorm[] = { { "x", RAD_OPD_IN, 0 },
                                           { "w", RAD_OPD_WEIGHT, 1 },
                                           { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pLayernorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype") };
static const RadOperandSpec oLayernorm[] = { { "x", RAD_OPD_IN, 0 },
                                             { "w", RAD_OPD_WEIGHT, 0 },
                                             { "b", RAD_OPD_WEIGHT, 1 },
                                             { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pAdd[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("div") };
static const RadParamSpec pMnDtype[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oBinary[] = { { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_IN, 0 },
                                          { "y", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oUnary[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pActMul[] = { P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype") };

static const RadParamSpec pCast[] = { P_INT("M"), P_INT("n"), P_STR("from"), P_STR("to") };

static const RadParamSpec pScale[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
/* `s` is a WEIGHT, which is what the reference declares and what the core acts on: a weight operand
 * has its issue resolve a handle and wait on the mover, and a scale plane comes from the container
 * or from a quantiser, not from a buffer the planner allocated for this step. Declaring it IN is
 * not a cosmetic difference -- the loader compares roles positionally and refuses the library that
 * disagrees, every row with it. */
static const RadOperandSpec oScale[] = { { "x", RAD_OPD_IN, 0 }, { "s", RAD_OPD_WEIGHT, 0 },
                                         { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pFill[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("value") };
static const RadOperandSpec oFill[] = { { "y", RAD_OPD_OUT, 0 } };

/* --- the model's odds and ends --- */
static const RadParamSpec pRope[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                      P_INT("n_head_kv"), P_F64("theta"), P_F64("scale"),
                                      P_STR("mode"), P_INTO("rotary_dim"), P_STRO("sections"),
                                      P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                      P_F64O("yarn_high"), P_F64O("yarn_magnitude"),
                                      P_STRO("ladder"), P_F64O("llama3_factor"),
                                      P_F64O("llama3_low"), P_F64O("llama3_high"),
                                      P_F64O("llama3_orig") };
static const RadOperandSpec oRope[] = { { "qkv", RAD_OPD_INOUT, 0 },
                                        { "positions", RAD_OPD_IN, 0 } };

static const RadParamSpec pKvStore[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head_kv"),
                                         P_INT("block_size"), P_STR("kv_dtype"),
                                         P_F64O("k_scale"), P_F64O("v_scale") };
static const RadOperandSpec oKvStore[] = { { "k", RAD_OPD_IN, 0 }, { "v", RAD_OPD_IN, 0 },
                                           { "slot_mapping", RAD_OPD_IN, 0 },
                                           { "kv_cache", RAD_OPD_INOUT, 0 } };

static const RadParamSpec pEmbed[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                       P_STR("dtype"), P_INTO("vocab_offset"), P_F64O("wscale"),
                                       P_F64O("mult") };
static const RadOperandSpec oEmbed[] = { { "tokens", RAD_OPD_IN, 0 },
                                         { "wte", RAD_OPD_WEIGHT, 0 },
                                         { "x", RAD_OPD_OUT, 0 } };

static const RadOperandSpec oGather[] = { { "x", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 },
                                          { "y", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oScatter[] = { { "v", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 },
                                           { "x", RAD_OPD_INOUT, 0 } };

/* --- quantisation --- */
static const RadParamSpec pQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype") };
static const RadOperandSpec oQuantFp8[] = { { "x", RAD_OPD_IN, 0 }, { "q", RAD_OPD_OUT, 0 },
                                            { "scale", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oQuant[] = { { "x", RAD_OPD_IN, 0 }, { "q", RAD_OPD_OUT, 0 },
                                         { "scale", RAD_OPD_OUT, 0 },
                                         { "asum", RAD_OPD_OUT, 1 } };

static const RadParamSpec pGatedQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                            P_STRO("act") };
static const RadOperandSpec oGatedQuant[] = { { "gate_up", RAD_OPD_IN, 0 },
                                              { "q", RAD_OPD_OUT, 0 },
                                              { "scale", RAD_OPD_OUT, 0 },
                                              { "out_bf16", RAD_OPD_OUT, 1 } };

static const RadOperandSpec oDequant[] = { { "q", RAD_OPD_IN, 0 }, { "scale", RAD_OPD_IN, 0 },
                                           { "y", RAD_OPD_OUT, 0 } };

/* --- gemm --- */
static const RadParamSpec pGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype") };
static const RadOperandSpec oGemm[] = { { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
                                        { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                       P_STR("dtype"), P_INTO("pb1"), P_INTO("pb2"),
                                       P_INTO("split_n") };
static const RadOperandSpec oGemmQ[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 1 },
                                         { "b", RAD_OPD_WEIGHT, 0 },
                                         { "b_scale", RAD_OPD_WEIGHT, 0 },
                                         { "y", RAD_OPD_OUT, 0 }, { "a_sum", RAD_OPD_IN, 1 },
                                         { "b_ref", RAD_OPD_WEIGHT, 1 } };


/* ================================================================== the vocab and the sampler
 *
 * The vocab and sampler bands, transcribed from libref. A schema is inert on its own -- the loader
 * matches a row to a schema, and a row naming an entry point no unit defines is an undefined symbol
 * that kills `dlopen` and takes every row in the library with it -- so `r3d_selftest`'s schema phase
 * compares every declared schema against libref's, slot for slot, the moment the library loads, and
 * a misspelled key is refused at load, not when a row first meets it.
 *
 * The sampler is a law, not a shape: the candidate set, the tie-break in the top-k, the order of the
 * renormalisations and the seeded draw are all observable, and libref is the oracle for all of them.
 */

/* The gated MLP's activation, as the llama-family graphs emit it: one fused [M, 2n] projection, the
 * gate's half activated and multiplied by the other. `act_mul` is the same arithmetic over two
 * operands; these are different ops, and an architecture emits one or the other. */
static const RadOperandSpec oGateUp[] = { { "gate_up", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };

/* Paged attention. `block_size` is RAD_DERIVED here (the container's cache geometry, admitted by a
 * constraint that only refuses its absence), and `n_head`, `max_seqs`, `max_ctx` and `prefill_fp8`
 * are on the list because a plugin's scratch hook sizes a split-KV buffer from them and both
 * libraries must agree on the parameter list -- libref ignores them. */
/* DFlash2's candidate walk: a draft block of `steps` proposals, `top_k` candidates at each step,
 * rescored against the target's rank-d codebooks. `anchor_stride` is optional and defaults to 1;
 * `hp` is the optional per-edge scaling of the predecessor's codebook row, and absent it is a plane
 * of ones, which is what the reference computes. */
/* The gated MLP band: one weight of N = 2n rows (gate first, then up), the GEMM, the gate and the
 * output quantiser in one launch, so the [M, N] intermediate the unfused chain writes and re-reads
 * never exists. `act` is a required string: an implementation compiles one gate, and an optional
 * would let a row resolve for the gate it does not have. */
static const RadParamSpec pGemmQGated[] = { P_INT("M"), P_INT("N"), P_INT("K"),
                                            P_INT("group"), P_STR("dtype"), P_STR("act") };
/* The vocabulary projection's schema, slot for slot with the reference's: the geometry is the head's
 * (`n_vocab` x `n_embd`), the dtype is the activation's, and `logits` is f32 -- the op exists
 * separately from gemm_nt for that reason, and the row that serves it narrows nothing. */
static const RadParamSpec pLogits[] = { P_INT("M"), P_INT("n_vocab"), P_INT("n_embd"),
                                        P_STR("dtype") };
static const RadOperandSpec oLogits[] = {
    { "x", RAD_OPD_IN, 0 }, { "lm_head", RAD_OPD_WEIGHT, 0 }, { "logits", RAD_OPD_OUT, 0 } };

static const RadOperandSpec oGemmQGated[] = {
    { "a", RAD_OPD_IN, 0 },        { "a_scale", RAD_OPD_IN, 0 },
    { "b", RAD_OPD_WEIGHT, 0 },    { "b_scale", RAD_OPD_WEIGHT, 0 },
    { "q", RAD_OPD_OUT, 0 },       { "scale", RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};

static const RadParamSpec pDflashSelect[] = { P_INT("M"), P_INT("steps"), P_INT("top_k"),
                                              P_INT("rank"), P_INT("n_vocab"),
                                              P_INTO("anchor_stride"), P_STR("dtype") };
static const RadOperandSpec oDflashSelect[] = {
    { "cand", RAD_OPD_IN, 0 },    { "unary", RAD_OPD_IN, 0 },
    { "hp", RAD_OPD_IN, 1 },      { "anchor", RAD_OPD_IN, 0 },
    { "pred", RAD_OPD_WEIGHT, 0 },{ "succ", RAD_OPD_WEIGHT, 0 },
    { "tokens", RAD_OPD_OUT, 0 },
};

static const RadParamSpec pAttnPaged[] = {
    P_INT("q_len"), P_INT("head_dim"), P_INT("gqa"), P_INTD("block_size"), P_INT("causal"),
    P_INT("window"), P_STR("q_dtype"), P_STR("kv_dtype"), P_F64O("scale"), P_INTO("max_ctx"),
    P_INTO("n_head"), P_INTO("max_seqs"), P_INTO("prefill_fp8")
};
static const RadOperandSpec oAttnPaged[] = {
    { "q", RAD_OPD_IN, 0 }, { "kv_cache", RAD_OPD_IN, 0 }, { "block_table", RAD_OPD_IN, 0 },
    { "seqused", RAD_OPD_IN, 0 }, { "k_descale", RAD_OPD_IN, 1 }, { "v_descale", RAD_OPD_IN, 1 },
    { "cu_seqlens", RAD_OPD_IN, 1 }, { "out", RAD_OPD_OUT, 0 }
};

/* The row's top-R over a vocabulary slice, and the sampler's stages. The history stages write this
 * rank's columns and read global ids, so they carry `vocab_off`; the narrowing stages work on the
 * candidate set the top-k produced and need only its width. */
static const RadParamSpec pRowTopk[]  = { P_INT("M"), P_INT("N"), P_INT("R"), P_INTO("vocab_off"),
                                          P_STR("dtype") };
static const RadOperandSpec oRowTopk[] = { { "x", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_OUT, 0 },
                                           { "val", RAD_OPD_OUT, 0 },
                                           { "pairs", RAD_OPD_OUT, 1 } };
static const RadParamSpec pMVocab[]  = { P_INT("M"), P_INT("n_vocab") };
static const RadParamSpec pMCand[]   = { P_INT("M"), P_INT("n_cand") };
static const RadParamSpec pPenal[]   = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadParamSpec pTopk[]    = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadParamSpec pLogprobs[] = { P_INT("M"), P_INT("n_vocab"), P_INT("k"),
                                          P_INTO("vocab_off") };
static const RadOperandSpec oTemp[]      = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 } };
static const RadOperandSpec oMask[]      = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "bitmask", RAD_OPD_IN, 0 } };
static const RadOperandSpec oPenalties[] = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 } };
static const RadOperandSpec oDry[]       = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 },
                                             { "breakers", RAD_OPD_IN, 1 } };
static const RadOperandSpec oNgramBan[]  = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 },
                                             { "ngram", RAD_OPD_IN, 0 },
                                             { "whitelist", RAD_OPD_IN, 1 } };
static const RadOperandSpec oTopk[]      = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "cand_idx", RAD_OPD_OUT, 0 },
                                             { "cand_val", RAD_OPD_OUT, 0 },
                                             { "pairs", RAD_OPD_OUT, 1 } };
static const RadOperandSpec oArgmax[]    = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "token", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oLogprobs[]  = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "stats", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oNarrow[]    = { { "cand_idx", RAD_OPD_INOUT, 0 },
                                             { "cand_val", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 } };
static const RadOperandSpec oPick[]      = { { "cand_idx", RAD_OPD_IN, 0 },
                                             { "cand_val", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "token", RAD_OPD_OUT, 0 } };


/* The fused MLP activation and the fused norm-quantiser, both from the model unit. The first takes
 * one [M, 2n] operand and produces [M, n]; the second is libref's, slot for slot, with `residual`
 * INOUT because the sum it stores is the one the variance reads. */
static const RadParamSpec pRmsQuantFp8[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INT("group"),
                                             P_STR("dtype"), P_F64O("wadd") };
/* `residual` is optional because the op has a no-residual form, which is what the launcher's `res`
 * pointer being 0 means; libref spells it INOUT_O and the loader compares the flag. */
static const RadOperandSpec oRmsQuantFp8[] = {
    { "x", RAD_OPD_IN, 0 },        { "residual", RAD_OPD_INOUT, 1 }, { "w", RAD_OPD_WEIGHT, 0 },
    { "q", RAD_OPD_OUT, 0 },       { "scale", RAD_OPD_OUT, 0 },      { "out_bf16", RAD_OPD_OUT, 1 }
};

static const RadOpSchema kSchemas[] = {
    SCHEMA("rmsnorm", pRmsNorm, oRmsNorm,
           "y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult. `wadd` is GemmaRMSNorm's `1 +`, "
           "added to the gain in f32; `mult` scales the normed row before its one rounding. No "
           "gain operand at all means a gain of 1 and a `wadd` of 0."),
    SCHEMA("layernorm", pLayernorm, oLayernorm,
           "y = (x - mean) / sqrt(var + eps) * w + b, the mean and the variance taken in two "
           "passes over the row."),
    SCHEMA("add", pAdd, oBinary,
           "y = a + b. `b` carrying exactly n elements against an output of M rows broadcasts "
           "along rows, and nothing else broadcasts at all. With `div`, y = rd(rd(a + b) / div)."),
    SCHEMA("mul", pMnDtype, oBinary, "y = a * b, with the same one broadcast rule as `add`."),
    SCHEMA("act_mul", pActMul, oBinary,
           "y = act(a) * b over two [M, n] operands, `act` naming silu, gelu (the exact erf form), "
           "gelu_tanh, sigmoid or quick_gelu. The feed-forward's gated middle."),
    SCHEMA("silu", pMnDtype, oUnary, "y = x * sigmoid(x)."),
    SCHEMA("gelu", pMnDtype, oUnary,
           "y = 0.5 * x * (1 + erf(x / sqrt(2))), the exact form; the tanh approximation is a "
           "different op name."),
    SCHEMA("sigmoid", pMnDtype, oUnary, "y = 1 / (1 + exp(-x))."),
    SCHEMA("softmax", pMnDtype, oUnary, "Row-wise, f32, with the maximum subtracted."),
    SCHEMA("cast", pCast, oUnary,
           "y = x, from the dtype `from` to the dtype `to`, one rounding on the way. An integer "
           "pair goes through without touching f32, so ids above 2^24 survive."),
    SCHEMA("scale", pScale, oScale,
           "y = x * s, `s` one element broadcast over the whole output. A row of scales is "
           "`mul_weight`."),
    SCHEMA("fill", pFill, oFill, "y = value over [M, n]. The output is the only operand."),
    SCHEMA("rope", pRope, oRope,
           "Rotary positions applied in place to a fused qkv row: q heads, then the key heads, the "
           "value heads untouched. `mode` is neox, gptj, mrope, imrope or axial; the yarn, llama3 "
           "and ladder parameters adjust the frequency ladder."),
    SCHEMA("kv_store", pKvStore, oKvStore,
           "K and V into a paged cache at slot_mapping's slots, converting to the cache's dtype. A "
           "negative slot is skipped; the cache is [blocks, kv_heads, block_size, 2*head_dim]."),
    SCHEMA("embed_lookup", pEmbed, oEmbed,
           "x[r] = wte[tokens[r] - vocab_offset] * wscale * mult, the id global and the table "
           "sharded by `vocab_offset`. An id outside the shard's range yields a zero row."),
    SCHEMA("gather_rows", pMnDtype, oGather, "y[r] = x[idx[r]], rows of n elements."),
    SCHEMA("scatter_rows", pMnDtype, oScatter, "x[idx[r]] = v[r], rows of n elements."),
    SCHEMA("quant_act_fp8", pQuant, oQuantFp8,
           "Symmetric per-(row, group) E4M3: scale = amax * (1/448) with the reciprocal "
           "multiplied, not divided; an all-zero group gets scale 1 and the codes 0."),
    SCHEMA("quant_act_i8", pQuant, oQuant,
           "Symmetric int8 with ONE SCALE PER ROW, max(amax, 1e-8) / 127, codes round-to-nearest-"
           "ties-to-even and saturating, stored in the STEP ORDER the int8 WMMA reads. `asum` takes "
           "the group-major row sums the zero-pointed GEMMs reduce."),
    SCHEMA("quant_act_i8g", pQuant, oQuantFp8,
           "The same int8 code at `group` granularity: one f32 scale per (row, group), amax / 127, "
           "1 for an all-zero group, codes row-major."),
    SCHEMA("gated_quant_fp8", pGatedQuant, oGatedQuant,
           "act(gate) * up over a fused [M, 2n] projection, quantised in the same pass: the codes "
           "are taken over the rounded product, which `out_bf16` receives when present."),
    SCHEMA("dequant", pQuant, oDequant,
           "y = code * scale, the read of a container's own quantised weight: `group`-wide f32 "
           "scales against E4M3, E5M2 or int8 codes."),
    SCHEMA("gemm_nt", pGemm, oGemm, "y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate."),
    SCHEMA("gemm_nt_q", pGemmQ, oGemmQ,
           "The same product against a quantised weight. `dtype` names the scheme: fp8a16 a bf16 "
           "activation against E4M3 codes, fp8a8 quant_act_fp8's codes and scales, i8a8 int8 on "
           "both axes. `group` is the activation's scale width; `pb1` and `pb2` the weight's block, "
           "which the container states. `a_sum` and `b_ref` are the row sums a zero-pointed form "
           "reduces."),

    SCHEMA("rmsnorm_quant_fp8", pRmsQuantFp8, oRmsQuantFp8,
           "RMS norm with an optional residual add, then the E4M3 quantiser, in one pass. The "
           "residual add is STORED BEFORE THE VARIANCE READS IT, so the sum the norm is taken over "
           "is the bf16 one and not the f32 one -- that is what makes this the `add` + `rmsnorm` "
           "pair and not `rmsnorm_add`, which normalises the unrounded sum. `wadd` is added to every "
           "gain in f32. `out_bf16` holds exactly the value the codes were taken over."),

    /* ---- the vocabulary and the sampler: schemas first, rows with their kernels ---- */
    SCHEMA("silu_mul", pMnDtype, oGateUp,
           "y[i] = silu(gate_up[i]) * gate_up[n + i] over one fused [M, 2n] plane, the gate the "
           "first half -- rad-convert's fusion order -- and the product of an activation that was "
           "never narrowed. act_mul is the same arithmetic over two operands, narrowing first, "
           "because the pair it replaces stored the activation."),
    SCHEMA("attn_paged", pAttnPaged, oAttnPaged,
           "Scaled dot-product attention over a paged KV cache: q [tokens, n_head, head_dim], the "
           "cache [blocks, kv_heads, block_size, 2*head_dim], `block_table` a row's block list, "
           "`seqused` its length. `gqa` is the head ratio; `causal` and `window` are the mask; "
           "`scale` is 1/sqrt(head_dim) unless the graph says otherwise. k_descale and v_descale "
           "come with an fp8 cache, cu_seqlens with a packed prefill."),
    SCHEMA("row_topk", pRowTopk, oRowTopk,
           "Exact per-row top-R, descending, ties to the lower column. `vocab_off` shifts every "
           "column reported, so a rank's slice reports global ids; `pairs` is the [M, R, 2] plane an "
           "all_gather moves, with id -1 for a slot the row could not fill."),
    SCHEMA("sample_argmax", pMVocab, oArgmax,
           "The most likely token of each row, the lowest id on a tie."),
    SCHEMA("sample_temp", pMVocab, oTemp,
           "logits / temperature, in place. The temperature is a row parameter, so a batch mixes "
           "greedy and sampled rows."),
    SCHEMA("sample_topk", pTopk, oTopk,
           "The k most likely candidates a row: ids and values, descending, ties to the lower id. "
           "`pairs` is the vocab-parallel form the merge reads after the all-gather."),
    SCHEMA("sample_topp", pMCand, oNarrow,
           "Narrow the candidate set to the smallest prefix whose probabilities reach p, in the "
           "order the candidates arrive -- not in sorted order, which is a different set at a tie."),
    SCHEMA("sample_minp", pMCand, oNarrow,
           "Drop a candidate whose probability is under min_p times the best candidate's."),
    SCHEMA("sample_typical", pMCand, oNarrow,
           "Drop the candidates furthest from the set's entropy: keep those within `sigma` of the "
           "mean surprise, the token set a reader would not have predicted least and most alike."),
    SCHEMA("sample_xtc", pMCand, oNarrow,
           "The exclusion-the-model-would-not-choose filter: drop every candidate above p, keeping "
           "the last one standing when all of them are."),
    SCHEMA("sample_pick", pMCand, oPick,
           "The seeded inverse-CDF draw over the candidate set: the law the request's seed selects, "
           "in the order the candidates arrive. A kernel that draws from the same distribution "
           "differs from the reference on every token, and kbench reads that as a failure, because "
           "for a sampler it is one."),
    SCHEMA("sample_penalties", pPenal, oPenalties,
           "Presence, frequency and repetition penalties in place, over the row's history, with the "
           "DRY and penalty terms libref spells out."),
    SCHEMA("sample_dry", pPenal, oDry,
           "Distributed Repetition Penalty: a sequence of `last_n` that repeats a sequence of "
           "`baseline` is penalised, growing with the length of the repeat. `breakers` marks the "
           "positions that end a sequence and are never penalised."),
    SCHEMA("sample_ngram_ban", pPenal, oNgramBan,
           "Ban the n-grams of length `n` that the history shows inside `window`, to -inf. "
           "`whitelist` exempts a set of ids from the ban."),
    SCHEMA("sample_mask", pMVocab, oMask,
           "The grammar's and the tool-caller's doing: set to -inf every id whose bit is clear in "
           "the row's bitmask, addressed by the row's mask_row, which is -1 for a row with no "
           "constraint and so shares a short plane."),
    SCHEMA("sample_logprobs", pLogprobs, oLogprobs,
           "The row's log-softmax statistics for the response: max, sum of exp, and the k best "
           "(id, logit) pairs, in [M, 2 + 2k]."),
    SCHEMA("logits_gemm", pLogits, oLogits,
           "logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T. Separate from gemm_nt "
           "because lm_head is its own access class and the sampler wants to fuse against it: the "
           "head is the largest read in a decode step and the output is f32, which every sampler "
           "stage downstream reads."),
    SCHEMA("gemm_nt_q_gated", pGemmQGated, oGemmQGated,
           "The gated MLP band in one launch: `b` is N = 2n rows, gate over up, the dot products are "
           "narrowed to bf16 as the unfused pair would leave them, the gate is `gt * sigmoid(gt)`, the "
           "product is narrowed, and the row is coded per group of `group` columns with the amax that "
           "group's scale comes from. `out_bf16`, when asked for, is the narrowed gated tensor before "
           "the coding."),
    SCHEMA("dflash_select", pDflashSelect, oDflashSelect,
           "DFlash2's greedy walk over a draft block: at each of `steps` steps the candidate with "
           "the best `unary + pred[pid] . (hp * succ[cand])` is picked, and becomes the predecessor "
           "of the next step. `cand`/`unary` have one row per (sequence, step); `anchor` is the "
           "block's first predecessor, at `m * anchor_stride`."),
};

extern "C" int rad_kernel_schema_count(void) {
    return (int)(sizeof(kSchemas) / sizeof(kSchemas[0]));
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0 || i >= rad_kernel_schema_count()) return nullptr;
    return &kSchemas[i];
}

/* ================================================================== tuning
 *
 * The axes a kernel is instantiated along, measured on this card (`rad-tune`, spec §15). Every row
 * without a matrix instruction to size a tile against has few axes: what is left to measure is how
 * the work is split across the card, and how the weight stream is read.
 */

/* Split-K. 0 is not a point in the space, it is the kernel's own law: "the largest split legal at
 * this K", because no fixed default is legal at every shape and an illegal default would make an
 * untuned install fail rather than run slowly. */
static const int64_t kSk[] = { 0, 1, 2, 4, 8, 16 };
/* Waves per block. 8 is 256 threads, which is the largest block that keeps the register budget of
 * a wave-per-N-tile kernel inside gfx11's 1024 VGPRs a SIMD at four waves. */
static const int64_t kWv[] = { 1, 2, 4, 8 };
/* Non-temporal weight streaming: a decode GEMM reads a weight once per token, so the L2 is a
 * latency cache and not a reuse cache. RDNA3 has no transposed-load instruction, so this axis
 * selects the plain 128-bit stream with the cache hints off -- a knob worth measuring, not an
 * instruction. */
static const int64_t kNt[] = { 0, 1 };
/* No `kb` axis. Only a GEMM that stages a K slab through LDS has a slab width to measure, and
 * the skinny rows stream the weight through registers; the axis joins the table with the tiled
 * row. */

static const RadTunable tGemmM16[] = {
    { "sk", kSk, NELEM(kSk), 0, "split-K slices; 0 is the widest legal for this K" },
    { "wv", kWv, NELEM(kWv), 8, "waves per block" },
    { "nt", kNt, NELEM(kNt), 1, "non-temporal weight streaming" },
};
/* No `wv` here. The quantised kernel's launcher takes a fixed block of eight waves -- its weight
 * decode, sixteen codes a lane into eight dot2, is the whole of a wave's issue budget at any block
 * size -- so an axis the tuner would sweep is an axis the kernel ignores, and a tunable that changes
 * nothing is worse than none: it spends the tuner on a knob and teaches the reader that the table
 * lies. The axis joins the row with a launcher that takes it. */
static const RadTunable tGemmQ[] = {
    { "sk", kSk, NELEM(kSk), 0, "split-K slices; 0 is the widest legal for this K" },
    { "nt", kNt, NELEM(kNt), 1, "non-temporal weight streaming" },
};

/* Cross-axis legality, in one place so the tuner, the scratch hook and the launch agree. A split's
 * slice must end on a k multiple the kernel's inner loop consumes: 8 elements for a bf16 dot2 pair
 * group, and one whole scale group for a quantised weight, since a slice that ends mid-block cannot
 * apply its block scale. */
static int gemm_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    const long long K = rad_param_getdim(p, n_p, "K", 0);
    const long long sk = rad_param_geti(c, n_c, "sk", 0);
    const long long wv = rad_param_geti(c, n_c, "wv", 8);
    if (K <= 0 || wv < 1 || wv > 8) return 0;
    const long long unit = rad_param_getdim(p, n_p, "group", 0) > 0
                               ? rad_param_getdim(p, n_p, "group", 0)
                               : 8;
    return sk >= 0 && (sk == 0 || K % (sk * unit) == 0);
}

/* The chosen point, read back. `TUNE` falls to the axis's default when the core appended nothing,
 * so a checker with no tuned key runs what an untuned install runs. */
static long long tune_i(const RadArgs* a, const RadTunable* t, int n, const char* key) {
    if (a && a->p)
        for (int i = 0; i < a->n_p; ++i)
            if (a->p[i].key && !std::strcmp(a->p[i].key, key) && a->p[i].kind == RAD_P_INT)
                return a->p[i].ival;
    for (int i = 0; i < n; ++i)
        if (!std::strcmp(t[i].key, key)) return t[i].deflt;
    return 0;
}
#define TUNE(a, tab, key) ((int)tune_i((a), (tab), NELEM(tab), (key)))

extern "C" void r3d_cfg_gemm(const RadArgs* a, R3dGemmCfg* out) {
    out->wv = TUNE(a, tGemmM16, "wv");
    out->sk = TUNE(a, tGemmM16, "sk");
    out->kb = 0;
    out->nt = TUNE(a, tGemmM16, "nt");
}
extern "C" void r3d_cfg_split(const RadArgs* a, R3dSplitCfg* out) {
    out->splits = TUNE(a, tGemmM16, "sk");
}
extern "C" void r3d_cfg_topk(const RadArgs* a, R3dTopkCfg* out) {
    out->rc = 0;
    out->nch = 0;
}

/* A split's partials are f32 over the output, plus one arrival counter per split. Sized at the
 * band's bounds and the axis's widest split, because a scratch hook runs at declare, where the
 * range may still be open: the buffer must hold any point the tuner can pick. */
/* Split-KV's partials: one record of 128 accumulator floats plus the running max and sum -- 132
 * floats -- per (query row, head, slice). The shim takes the slice count from the bytes the engine
 * offers and halves it until they fit, so this is the want, stated at the slice count the walk
 * benefits from and capped at what these bands can carry: 15 MB at the decode band (1792 rows over
 * 16 heads) and 17 MB at the prefill band. Attaching scratch makes the row non-concurrent, which is
 * the ABI's rule (rad_kernel_concurrent is `scratch == nullptr`) and the price of the split. */
static int64_t attn_scratch(const RadArgs* a) {
    /* The size comes from the SHAPE, and a scratch hook runs at declare, where the engine has bound
     * the operands' shapes and has no bytes to hand out yet -- `data` is null at declare for every
     * tensor in the plan. A hook that requires bytes answers 0 for the whole run, the engine
     * allocates an arena of 0 bytes, and the shim's split pass finds no partials to write. The
     * sibling `gemm_scratch` sizes off `RadParam` dims for exactly this reason. */
    if (!a || a->n_t < 1 || a->t[0].rank == 0) return 0;
    long long n_head = 0, head_dim = 0;
    const int64_t nh = rad_args_geti(a, "n_head", &n_head) ? n_head : 0;
    if (!rad_args_geti(a, "head_dim", &head_dim) || !head_dim || !nh) return 0;
    const long long rows = rad_tensor_numel(&a->t[0]) / (nh * head_dim);
    if (rows <= 0) return 0;
    const long long want = rows * nh * 8 * 132 * (long long)sizeof(float);
    const long long cap = 16LL << 20;
    return want < cap ? want : cap;
}

static int64_t gemm_scratch(const RadParam* p, int n_p) {
    const long long M = rad_param_getdim(p, n_p, "M", 0), N = rad_param_getdim(p, n_p, "N", 0);
    const long long sk = kSk[NELEM(kSk) - 1];
    if (M <= 0 || N <= 0) return 0;
    return sk * M * N * 4 + sk * 64;
}

/* ================================================================== constraints
 *
 * What the kernel cannot do, and what the selector needs. The divisibility in each is the lane map
 * in the unit: a wave covering a bf16 row reads four elements a lane (two pairs), so a row of 4m
 * elements needs no tail; where the kernel has a tail path the constraint says 2, not 4.
 */

/* bf16 elementwise, norms and the GEMMs: rows of an even width, so a lane owns a pair. */
static const RadConstraint cBf16Row[]  = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cF32Row[]   = { C_IN("dtype", "f32"), C_DIV("n", 2) };
static const RadConstraint cNormBf16[] = { C_IN("dtype", "bf16"), C_DIV("n", 2),
                                           C_GE("n", 2) };
static const RadConstraint cNormF32[]  = { C_IN("dtype", "f32"), C_DIV("n", 2), C_GE("n", 2) };
static const RadConstraint cSoftmax[]  = { C_IN("dtype", "f32"), C_GE("n", 2) };

/* A cast row is its pair of dtype names. */
static const RadConstraint cCastBf16F32[] = { C_IN("from", "bf16"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastF32Bf16[] = { C_IN("from", "f32"), C_IN("to", "bf16"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastBf16[]    = { C_IN("from", "bf16"), C_IN("to", "bf16"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastF32[]     = { C_IN("from", "f32"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI64F32[]  = { C_IN("from", "i64"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI32I64[]  = { C_IN("from", "i32"), C_IN("to", "i64"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI64I32[]  = { C_IN("from", "i64"), C_IN("to", "i32"),
                                              C_DIV("n", 2) };

/* act_mul names its activation, because a row for gelu and a row for silu are different template
 * instantiations and a model asks for one of them. */
static const RadConstraint cActSilu[]    = { C_IN("act", "silu"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };
static const RadConstraint cActGelu[]    = { C_IN("act", "gelu"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };
static const RadConstraint cActSigmoid[] = { C_IN("act", "sigmoid"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };

/* A fill row is its dtype; the value is a parameter, and every dtype takes every value. */
static const RadConstraint cFillBf16[] = { C_IN("dtype", "bf16") };
static const RadConstraint cFillF32[]  = { C_IN("dtype", "f32") };
static const RadConstraint cFillI32[]  = { C_IN("dtype", "i32") };
static const RadConstraint cFillI64[]  = { C_IN("dtype", "i64") };

/* rope: the two modes this kernel implements, and a rotary width that is a whole even head. The
 * yarn, llama3 and ladder parameters are absent from these rows' bands, so a geometry carrying them
 * is not routed here. */
/* `rotary_dim` is optional on the op, and a constraint on a parameter a band does not carry does
 * not hold -- the engine's rule, quoted in r3d_selftest.cpp's band-reachability phase -- so it
 * cannot appear here: an architecture that rotates the whole head, which is every llama-family
 * graph, emits a band without the parameter and would find no row. Its evenness is the shim's to
 * refuse (RAD_E_SHAPE), and `rotary_dim` absent is head_dim, which is even by the constraint above. */
static const RadConstraint cRope[] = { C_IN("mode", "neox gptj"), C_DIV("head_dim", 2) };

/* The cache's dtype is the row: a bf16 cache stores the value, an E4M3 cache encodes it with the
 * store's scale. head_dim even because a lane writes a pair; block_size a multiple of 16 so a
 * block's slice of a row is a whole number of lanes. */
static const RadConstraint cKvBf16[] = { C_IN("kv_dtype", "bf16"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };
static const RadConstraint cKvFp8[]  = { C_IN("kv_dtype", "fp8_e4m3"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };
static const RadConstraint cKvF16[]  = { C_IN("kv_dtype", "f16"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };

static const RadConstraint cEmbed[] = { C_IN("dtype", "bf16"), C_DIV("n_embd", 2) };

/* The gather's and scatter's dtype, and a row wide enough for a lane to own a pair. */
static const RadConstraint cGatherBf16[]   = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cGatherF32[]    = { C_IN("dtype", "f32"), C_DIV("n", 2) };
static const RadConstraint cScatterBf16[]  = { C_IN("dtype", "bf16"), C_DIV("n", 2) };

/* The quantisers: the scale group is the checkpoint's block, 128, and a row wide enough that a
 * wave's four-elements-a-lane store is whole. */
static const RadConstraint cQuantFp8[]   = { C_IN("dtype", "bf16"), C_EQ("group", R3D_FP8_BLOCK),
                                             C_DIV("n", 4) };
/* `quant_act_i8` is the per-row int8 whose codes the int8 WMMA reads, so its width is a whole
 * 16-byte step, not a lane's four. `quant_act_i8g` is the per-group one whose codes are row-major,
 * which is what a dot2 GEMM reads; it is the activation of this library's i8a8 row. */
static const RadConstraint cQuantI8[]    = { C_IN("dtype", "bf16"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 16) };
static const RadConstraint cQuantI8g[]   = { C_IN("dtype", "bf16"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 4) };
/* Not `C_IN("act", "silu sigmoid")`: `act` is optional on the op (a graph may carry libref's integer
 * `mode` instead, and libref's own default is silu), and a band that states neither would resolve to
 * nothing. The shim reads `act`, falls back to `mode`, then to silu, and refuses a gelu -- the
 * activation whose kernel this unit does not carry -- with RAD_E_UNSUPPORTED. */
static const RadConstraint cGateQuant[]  = { C_IN("dtype", "bf16"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantFp8[] = { C_IN("dtype", "fp8_e4m3"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantE5[]  = { C_IN("dtype", "fp8_e5m2"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantI8[]  = { C_IN("dtype", "i8"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 4) };

/* The GEMMs. The skinny row's band is the decode and verify height: 16 rows, which is a speculative
 * verify's ceiling with room to spare, and below the height where staging the activations through
 * LDS starts to pay. K even, because the inner loop consumes pairs; N unconstrained, because a wave
 * covering N tiles the tail.
 *
 * No upper bound on N: the vocabulary and the feed-forward widths are 11008, 14336, 128256 and
 * whatever a given checkpoint's `intermediate_size` is, and a bound would refuse them. */
static const RadConstraint cGemmM16[] = { C_IN("dtype", "bf16"), C_GE("M", 1), C_LE("M", 16),
                                          C_DIV("K", 2), C_GE("N", 1) };
/* The vocabulary projection. **No `dtype` constraint, and the reason is the engine's, not the
 * kernel's**: the band this op is declared with carries the geometry (M, n_vocab, n_embd) and not a
 * dtype key -- the head's dtype is a property of the weight, which is why libref's own logits row
 * answers it with a `check_dtypes` hook and not with a constraint. A constraint on a key the
 * geometry does not carry serves nobody: with `dtype in {bf16}` declared, the engine resolves a
 * host kernel and no device one and refuses to start, so the dtype is refused where it is read, at
 * the shim, naming the widening the kernel does.
 * `n_embd` in 32-code chunks is the entry's own rule (K % 32), stated so a band outside it is
 * refused at declare rather than at launch. `n_vocab`'s floor is the store's granularity: a row
 * writes 16-byte runs, so a plane four ids wide is the narrowest it can read, and a band narrower
 * than that -- a shard's padding, a 1-id test vocabulary -- is refused by the constraint and not by
 * the launch. Above the floor the grid is vocab-wide at every M, so a 320-wide vocabulary and a
 * 130560-wide one are the same kernel. */
static const RadConstraint cLogitsGemm[] = { C_GE("M", 1), C_GE("n_vocab", 4),
                                             C_DIV("n_embd", 32) };
/* The tiled GEMM at prefill heights. M >= 17 is the complement of cGemmM16's M <= 16 by
 * construction, so the two rows never compete for a band: the skinny shape (a wave a run of
 * columns, the weight streamed past registers) is right up to 16 rows and wrong above it, and the
 * 128x128 tile is the reverse. K % 32 is the tile's k-chunk and N >= 8 the bf16 store run, for the
 * same reason cLogitsGemm states its floor: what the kernel cannot read is in the constraints. */
/* The quantised GEMM above the decode heights. Here `dtype` IS the operand scheme -- gemm_q_shape
 * switches on it -- so C_IN filters, which is the distinction from logits_gemm, where the band's
 * dtype names the activation and C_IN filters nothing. M >= 17 is the complement of
 * gemm_nt_q_fp8a8_m16's M <= 16 -- read from its constraint, not from its name: a higher floor
 * would leave 17..64 unserved, and a 209-row batched decode is the shape a serving card spends
 * most of its life in. The tile's k-chunk (32) divides the container's scale
 * block (128), which is why `group` is an equality and `K` a divisibility. */
static const RadConstraint cGemmQFP8Tiled[] = { C_IN("dtype", "fp8a8 fp8a8f32"),
                                                C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 17),
                                                C_DIV("K", 128) };
/* The OCP-MXFP4 rows: the weight is e2m1 with one e8m0 scale per 32 of K, the activation's
 * scale per 128 -- both fixed by the format, so the `group` key carries no kernel-side meaning
 * and the rows do not constrain it (the reference derives the planes from it). K dividing 128
 * carries the 32-block; the band split is the fp8a8 one, 16 at the ceiling and 17 at the floor,
 * and the dtype carries the whole scheme. */
static const RadConstraint cGemmQMxfp4a8M16[]  = { C_IN("dtype", "mxfp4a8"),
                                                   C_GE("M", 1),
                                                   C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQMxfp4a8Tiled[] = { C_IN("dtype", "mxfp4a8"),
                                                    C_GE("M", 17),
                                                    C_DIV("K", 128) };

/* The symmetric int4 / int8 weight grids at decode heights: the block is the weight's scale
 * group (128), and the band is the m16 one -- M > 16 is the tiled rows', and the 8-bit grid at
 * prefill heights is declared out. */
static const RadConstraint cGemmQW4A8M16[]  = { C_IN("dtype", "w4a8"),
                                                C_GE("M", 1), C_LE("M", 16),
                                                C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQW8A8M16[]  = { C_IN("dtype", "w8a8"),
                                                C_GE("M", 1), C_LE("M", 16),
                                                C_DIV("K", 128), C_GE("N", 1) };

static const RadConstraint cGemmNTTiled[] = { C_IN("dtype", "bf16"), C_GE("M", 17), C_DIV("K", 32),
                                              C_GE("N", 8) };
/* The quantised GEMM: the scheme names the row, the weight's block is the container's, and a split
 * must land on a block boundary (gemm_tune_valid). */
/* The block: `group` is required and every band names it, so the row states it. `pb1` and `pb2` are
 * OPTIONAL, and a constraint on a key the band omits does not hold -- a model whose container
 * declares the default block says nothing about pb1, and constraining it would make every one of
 * that model's GEMM bands fall through to the reference and its host kernel -- 259 ops on one
 * 2B model. The launcher defaults them to `group` (r3d_gemm_fp8a16_nt_m1.hip) and refuses a
 * pb2 that is not the code group, which is the container's own invariant: the K block IS the code
 * group, so a band naming pb2 unequal to group resolves to this row and is refused at launch. */
static const RadConstraint cGemmQFp8a16[] = { C_IN("dtype", "fp8a16"),
                                              C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQFp8a8[]  = { C_IN("dtype", "fp8a8"),
                                              C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
/* The int8 GEMM's constraints are written ahead of its row: a row naming an entry point that is
 * not in the library is an undefined symbol, and the plugin would fail to load, taking every other
 * row with it. */
static const RadConstraint cGemmQI8[]     = { C_IN("dtype", "i8a8"),
                                              C_EQ("group", R3D_W8_GROUP), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };

/* ================================================================== the rows
 *
 * Declaration order is the tie-break within one priority, so rows that share an op are declared
 * narrowest first.
 */
/* The constraints above are written ahead of the rows that use them, for the same reason the
 * schemas are: the table reads as one document. A constraint array with no row is not an error, so
 * the file builds; the comment above each says which row it serves. */

/* ================================================================== the fused-MLP rows
 *
 * The two rows the llama graphs ask for that the model unit serves. `silu_mul` constrains
 * nothing but the dtype and an even `n`: `act` is not in this op's parameter list (it is the fused
 * form of one fixed activation), and a constraint on a key the geometry lacks does not hold, so a
 * row built on the act_mul constraints would resolve to no band at all. `rmsnorm_quant_fp8` states
 * the group the kernel covers with a wave and requires it to divide `n`; the standalone quantiser's
 * `n % 4` would admit a band this kernel refuses, and a row that resolves at declare and fails at
 * launch is the shape of bug this file exists to prevent.
 */
static const RadConstraint cSiluMul[]     = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cRmsQuantFp8[] = { C_IN("dtype", "bf16"), C_EQ("group", R3D_FP8_BLOCK),
                                              C_DIV("n", R3D_FP8_BLOCK) };

/* The sampler's pre-filters constrain `n_vocab` and nothing else. The bands an architecture emits
 * for these ops carry `n_vocab` and `M` (plus `vocab_off`, and `k` for logprobs) and no `dtype`
 * key, so a row that constrained a dtype would constrain a key the geometry lacks and resolve to no
 * band at all. The dtype is f32; the shims check it on the tensor, which is where a dtype belongs
 * when the geometry does not name it. `k <= 64` is the depth of the logprobs kernel's pool, so the
 * row serves exactly the bands its launcher accepts.
 */
static const RadConstraint cSampleF32[]      = { C_GE("n_vocab", 1) };
static const RadConstraint cSampleLogprobs[] = { C_GE("n_vocab", 1), C_GE("k", 1), C_LE("k", 64) };

/* The attention rows bind the two dtype keys, the head width the kernel's tile is built for, and
 * the q_len bucket boundary the two kernels are split at. `gqa`, `window` and `causal` are left
 * free: the kernel walks the mask for any gqa the shim accepts and any window, and a constraint on
 * them would only exclude bands it serves. */
static const RadConstraint cAttnBf16Decode[]  = { C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
                                                  C_EQ("head_dim", 128), C_LE("q_len", 16) };
static const RadConstraint cAttnBf16Prefill[] = { C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
                                                  C_EQ("head_dim", 128), C_GE("q_len", 17) };
static const RadConstraint cAttnFp8Decode[]   = { C_IN("q_dtype", "bf16"),
                                                  C_IN("kv_dtype", "fp8_e4m3"),
                                                  C_EQ("head_dim", 128), C_LE("q_len", 16) };
static const RadConstraint cAttnFp8Prefill[]  = { C_IN("q_dtype", "bf16"),
                                                  C_IN("kv_dtype", "fp8_e4m3"),
                                                  C_EQ("head_dim", 128), C_GE("q_len", 17) };

/* `row_topk` is banded on dtype, so it gets two rows; the candidate stages are banded on `n_cand`,
 * whose ceiling is the kernel's: five f32 arrays of n_cand in LDS is 20 KiB of the 64 KiB, and a
 * wider band is refused by the shim (RAD_E_SHAPE), so the row states the ceiling and resolves for
 * exactly what it serves. The vocab stages carry `n_vocab` and no dtype key, and the logits are f32
 * by the shim's check on the tensor.
 */
static const RadConstraint cRowTopkBf16[] = { C_IN("dtype", "bf16"), C_GE("N", 1), C_GE("R", 1) };
static const RadConstraint cRowTopkF32[]  = { C_IN("dtype", "f32"),  C_GE("N", 1), C_GE("R", 1) };
static const RadConstraint cSampleCand[]  = { C_GE("n_cand", 1), C_LE("n_cand", 1024) };

/* Every key the walk reads is required and in the band, so all of them are bound -- `top_k <= 256`
 * is one lane a candidate. `anchor_stride` is deliberately NOT bound: it is optional, defaults to
 * 1, and the kernel serves any value at or above 1, so a band that omits it must still resolve. */
/* The block is one key for three things at once: the weight's K block, its row block and the
 * quantiser's group. `act` is C_IN because it is a required string here, not the optional `mode`
 * form. M <= 64 is the ceiling the launcher serves; 47 of MiniCPM5-2B's 48 bands are M=64. */
/* The container's 2-bit vocab weights. `group` is one key for the weight's K block and the scale's
 * block, as everywhere in this family; K must also be a multiple of the launcher's chunk
 * (2048 / M-tile), which the launcher refuses and the row does not pretend to cover. */
static const RadConstraint cGemmQW2A8[]  = { C_IN("dtype", "w2a8"), C_EQ("group", R3D_FP8_BLOCK),
                                             C_GE("M", 1), C_LE("M", 16), C_DIV("K", 128),
                                             C_GE("N", 1) };

/* The packed-code grids at the heights gemm_nt_q_w2a8_m16 declines. The grid is the container's --
 * one scale and one zero per `group` of codes on a {1, group} plane -- so `group` is the block, and
 * K divides by it because a staged 32-code chunk must sit inside one scale group. */
static const RadConstraint cGemmQW2A8Tiled[] = { C_IN("dtype", "w2a8 w4a8_asym"),
                                                 C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 17),
                                                 C_DIV("K", 128) };

static const RadConstraint cGemmGated[]  = { C_IN("dtype", "fp8a8"),
                                             C_IN("act", "silu sigmoid"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                             C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 2),
                                             C_GE("N", 2) };

static const RadConstraint cDflashSelect[] = { C_IN("dtype", "bf16"), C_GE("M", 1),
                                               C_GE("steps", 1), C_GE("top_k", 1),
                                               C_LE("top_k", 256), C_GE("rank", 1),
                                               C_GE("n_vocab", 1) };

static const RadKernelInfo kKernels[] = {
/* ------------------------------------------------------------------ r3d_model_bf16.hip
 *
 * One workgroup a row. The `shape` column states the geometry the grid is built for; the
 * `computes` column states the arithmetic, in the order the kernel performs it.
 */
{ "rmsnorm_bf16", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; a workgroup a row; the sum of squares accumulates in f32 and only the store narrows",
  "bf16 x and y, bf16 or f32 gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_rmsnorm, nullptr, nullptr, nullptr,
  r3d_shape_rmsnorm },

{ "rmsnorm_f32", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; a workgroup a row",
  "f32 x, y and gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormF32), NOTUNE,
  nullptr, nullptr, r3d_rad_rmsnorm, nullptr, nullptr, nullptr,
  r3d_shape_rmsnorm },

{ "layernorm_bf16", "layernorm", "norm",
  "layer norm with a gain and an optional bias, mean and variance in two passes",
  "any n; a workgroup a row; both passes accumulate in f32",
  "bf16 x, y, gain and bias, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_layernorm, nullptr, nullptr, nullptr,
  r3d_shape_layernorm },

{ "add_bf16", "add", "elem",
  "a + b, with b's row broadcast and an optional divisor",
  "any n; a workgroup a row; f32 arithmetic, one narrowing at the store",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_add, nullptr, nullptr, nullptr,
  r3d_shape_add },

{ "add_f32", "add", "elem",
  "a + b, with b's row broadcast and an optional divisor",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_add, nullptr, nullptr, nullptr,
  r3d_shape_add },

{ "mul_bf16", "mul", "elem",
  "a * b, with b's row broadcast",
  "any n; a workgroup a row",
  "bf16 in and out, f32 product, one narrowing at the store",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_mul, nullptr, nullptr, nullptr,
  r3d_shape_mul },

{ "mul_f32", "mul", "elem",
  "a * b, with b's row broadcast",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_mul, nullptr, nullptr, nullptr,
  r3d_shape_mul },

/* The casts, one row per pair the engine issues. bf16 to f32 and back is the residual stream's
 * width change; i64 to f32 is a position table read by a rotary; the integer pairs move token ids
 * between planes and must not pass through f32, which loses the low bits above 2^24. */
{ "cast_bf16_f32", "cast", "elem", "bf16 to f32, exact", "any n; a wave 128 elements",
  "bf16 in, f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastBf16F32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_f32_bf16", "cast", "elem", "f32 to bf16, one rounding", "any n; a wave 128 elements",
  "f32 in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastF32Bf16), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_bf16_bf16", "cast", "elem", "bf16 to bf16, a copy", "any n; a wave 128 elements",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_f32_f32", "cast", "elem", "f32 to f32, a copy", "any n; a wave 128 elements",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastF32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i64_f32", "cast", "elem", "i64 to f32", "any n; a wave 128 elements",
  "i64 in, f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI64F32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i32_i64", "cast", "elem", "i32 to i64, exact", "any n; a wave 128 elements",
  "i32 in, i64 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI32I64), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i64_i32", "cast", "elem", "i64 to i32, the ids' narrowing", "any n; a wave 128 elements",
  "i64 in, i32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI64I32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "silu_bf16", "silu", "elem", "x * sigmoid(x), the sigmoid in f32", "any n; a wave 128 elements",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_silu, nullptr, nullptr, nullptr, r3d_shape_silu },

{ "silu_f32", "silu", "elem", "x * sigmoid(x)", "any n; a wave 128 elements", "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_silu, nullptr, nullptr, nullptr, r3d_shape_silu },

{ "gelu_bf16", "gelu", "elem",
  "the exact erf GELU: 0.5 * x * (1 + erf(x / sqrt(2)))",
  "any n; a wave 128 elements; erff, the accurate form, not the tanh approximation",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_gelu, nullptr, nullptr, nullptr, r3d_shape_gelu },

{ "sigmoid_bf16", "sigmoid", "elem", "1 / (1 + exp(-x))", "any n; a wave 128 elements",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_sigmoid, nullptr, nullptr, nullptr, r3d_shape_sigmoid },

{ "act_mul_silu_bf16", "act_mul", "elem",
  "silu(gate) * up, the gated feed-forward's middle",
  "any n; a workgroup a row; the gate is operand 0",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActSilu), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "act_mul_gelu_bf16", "act_mul", "elem",
  "gelu(gate) * up, the exact erf form",
  "any n; a workgroup a row",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActGelu), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "act_mul_sigmoid_bf16", "act_mul", "elem", "sigmoid(gate) * up", "any n; a workgroup a row",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActSigmoid), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "softmax_f32", "softmax", "elem",
  "row softmax with the maximum subtracted, the denominator summed in ascending order",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSoftmax), NOTUNE,
  nullptr, nullptr, r3d_rad_softmax, nullptr, nullptr, nullptr, r3d_shape_softmax },

{ "scale_bf16", "scale", "elem", "x * s, one scalar over the whole output",
  "any n; a wave 128 elements", "bf16 in and out, fp32 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_scale, nullptr, nullptr, nullptr, r3d_shape_scale },

{ "scale_f32", "scale", "elem", "x * s, one scalar over the whole output",
  "any n; a wave 128 elements", "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_scale, nullptr, nullptr, nullptr, r3d_shape_scale },

{ "fill_bf16", "fill", "elem", "the value over [M, n]", "any n; a wave 128 elements",
  "bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_f32", "fill", "elem", "the value over [M, n]", "any n; a wave 128 elements", "f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillF32), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_i32", "fill", "elem", "the value over [M, n], the value as an integer",
  "any n; a wave 128 elements", "i32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillI32), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_i64", "fill", "elem", "the value over [M, n], the value as an integer",
  "any n; a wave 128 elements", "i64 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillI64), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "rope_bf16", "rope", "attn",
  "rotary positions applied in place to a fused qkv row, neox and gptj pair layouts",
  "head_dim even; rotary_dim optional and even, defaulting to head_dim; a wave a head; the "
  "angles from accurate powf and sincosf, the position divided by `scale`",
  "bf16 qkv in place, i32 positions, fp32 angles",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRope), NOTUNE,
  nullptr, nullptr, r3d_rad_rope, nullptr, nullptr, nullptr, r3d_shape_rope },

{ "kv_store_bf16", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, bf16 to bf16",
  "any head_dim and block size; a wave a (token, head) pair; a negative slot skipped",
  "bf16 k and v, i32 slot mapping, bf16 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "kv_store_f16", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, bf16 to f16",
  "any head_dim and block size; a wave a (token, head) pair",
  "bf16 k and v, i32 slot mapping, f16 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvF16), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "kv_store_fp8", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, encoding E4M3 with the store's scale",
  "any head_dim and block size; a wave a (token, head) pair; the scale divides before the encode",
  "bf16 k and v, i32 slot mapping, fp8_e4m3 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "embed_lookup_bf16", "embed_lookup", "gemm",
  "the codebook row an id names, scaled, with out-of-shard ids yielding a zero row",
  "any n_embd; a workgroup a row of the table",
  "bf16 table and output, i32 ids",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cEmbed), NOTUNE,
  nullptr, nullptr, r3d_rad_embed_lookup, nullptr, nullptr, nullptr, r3d_shape_embed_lookup },

{ "gather_rows_bf16", "gather_rows", "elem", "y[r] = x[idx[r]]",
  "any n; a workgroup a row", "bf16 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatherBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_gather_rows, nullptr, nullptr, nullptr, r3d_shape_gather_rows },

{ "gather_rows_f32", "gather_rows", "elem", "y[r] = x[idx[r]]",
  "any n; a workgroup a row", "f32 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatherF32), NOTUNE,
  nullptr, nullptr, r3d_rad_gather_rows, nullptr, nullptr, nullptr, r3d_shape_gather_rows },

{ "scatter_rows_bf16", "scatter_rows", "elem", "x[idx[r]] = v[r]",
  "any n; a workgroup a row; distinct indices, so no two rows write one destination",
  "bf16 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cScatterBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_scatter_rows, nullptr, nullptr, nullptr, r3d_shape_scatter_rows },

/* ------------------------------------------------------------------ r3d_quant_fp8.hip
 *
 * Plain row-major code planes against one f32 scale per (row, group). No fragment permutation for
 * the codes a dot2 GEMM reads: it reads them in the order they are stored, which is the whole
 * reason a container converted on this card is readable by libref and by `rad-convert` on any other
 * card. The one plane that keeps libref's permutation is `quant_act_i8`'s, whose order is part of
 * what the op means -- see the row.
 */
{ "quant_act_fp8_bf16", "quant_act_fp8", "quant",
  "symmetric per-(row, group) E4M3: scale = amax * (1/448), the codes the value times 1/scale",
  "group 128; n a multiple of 4; a wave covers four columns a lane, so a group is a quarter-wipe",
  "bf16 in, fp8_e4m3 codes, f32 scales",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_fp8, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_fp8 },

/* The per-row int8, in libref's step order. The order is the op's, not the kernel's convenience:
 * the codes are the WMMA A-fragment's, and the plane is what the oracle compares, so this row keeps
 * the permutation and the kernel's store address carries it. libr3d's own int8 GEMM reads the
 * row-major `quant_act_i8g` plane two rows below, which is the pairing libref's comment states; a
 * gfx11 card has no consumer for the step-ordered plane and serves the op so an architecture that
 * emits it resolves to a kernel here rather than to the reference. */
{ "quant_act_i8_bf16", "quant_act_i8", "quant",
  "symmetric int8 with one scale per row: scale = max(amax, 1e-8) / 127, ties to even, saturating",
  "n a multiple of 16, a whole WMMA step; a wave a row of the row sums; the codes stored in the "
  "step order, the middle two dwords of every 16 bytes exchanged",
  "bf16 in, i8 codes, f32 scale per row and the group-major row sums",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_i8, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_i8 },

{ "quant_act_i8g_bf16", "quant_act_i8g", "quant",
  "symmetric per-(row, group) int8: scale = amax / 127, 1 for an all-zero group, ties to even",
  "group 128; n a multiple of 4; a wave covers four columns a lane, so a lane's codes are one "
  "dword and a wave's are 128 contiguous bytes",
  "bf16 in, i8 codes row-major, f32 scales",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantI8g), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_i8g, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_i8g },

{ "gated_quant_fp8_bf16", "gated_quant_fp8", "quant",
  "silu(gate) * up over a fused [M, 2n] projection, quantised in the same pass",
  "group 128; n a multiple of 4; the product is rounded to bf16 before its amax, as every fused "
  "implementation rounds it",
  "bf16 gate_up in, fp8_e4m3 codes, f32 scales, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGateQuant), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_quant_fp8, nullptr, nullptr, nullptr,
  r3d_shape_gated_quant_fp8 },

{ "dequant_fp8", "dequant", "quant",
  "E4M3 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "fp8_e4m3 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

{ "dequant_e5m2", "dequant", "quant",
  "E5M2 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "fp8_e5m2 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantE5), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

{ "dequant_i8", "dequant", "quant",
  "int8 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "i8 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

/* ------------------------------------------------------------------ r3d_gemm_bf16_nt_m16.hip
 *
 * The decode GEMM: M is 1 to 16, so a wave owns a run of N columns and streams the weight's rows
 * past an activation kept in registers. There is no fragment layout to fill, which is why this
 * kernel reads the checkpoint's rows in order.
 */
{ "gemm_nt_bf16_m16", "gemm_nt", "gemm",
  "a[M,K] @ b[N,K]^T for the decode and verify heights, f32 accumulate, one narrowing at the store",
  "M <= 16; a wave a run of N columns, one output element a lane, k ascending in pairs; split-K "
  "over scratch with the partials reduced in split order; a ragged K runs the scalar tail at sk=1",
  "bf16 a, b and y, fp32 accumulation; the pair is folded into the accumulator by one rounding, so "
  "about 0.3% of a long-K output lands one bf16 store ulp from the reference's, and no output is "
  "further. bf16 reaches the dot as f16, exact over f16's normal range, which every checkpoint's "
  "weights and activations live inside",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmM16), tGemmM16, NELEM(tGemmM16), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_bf16_m16,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt },

/* ------------------------------------------------------------------ r3d_gemm_fp8a16_nt_m1.hip
 *
 * The same shape against a block-scaled weight. The codes are E4M3, widened to f16 exactly and
 * accumulated by dot2; the block scale is applied once per 128-element slab, after the slab's
 * products, which is where the reference applies it.
 */
{ "gemm_nt_q_fp8a16_m16", "gemm_nt_q", "gemm",
  "a[M,K] @ b[N,K]^T against E4M3 codes with a scale per 128x128 block",
  "M <= 16, K a multiple of the block, group = pb2 = 128 and pb1 the rows sharing a scale row; a "
  "wave a run of N columns; the codes widened to f16 exactly and the block scale applied after each "
  "slab; a split ends on a block boundary (gemm_tune_valid)",
  "bf16 a, fp8_e4m3 b, f32 b_scale, bf16 y, fp32 accumulation; an f32 a, a bf16 scale plane, a "
  "ragged last block and the zero-point planes a_sum and b_ref are refused, not approximated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFp8a16), tGemmQ, NELEM(tGemmQ), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_fp8a8_m16", "gemm_nt_q", "gemm",
  "quant_act_fp8's codes and scales against E4M3 codes with a scale per 128x128 block",
  "M <= 16, K a multiple of the block; both axes coded, so both scales apply",
  "fp8_e4m3 a and b, f32 scales on both, bf16 y, fp32 accumulation",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFp8a8), tGemmQ, NELEM(tGemmQ), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_mxfp4a8_m16", "gemm_nt_q", "gemm",
  "fp8 a against OCP-MXFP4 weights: e2m1 codes, two a byte, with an e8m0 scale per 32 of K, the "
  "block's partials scaled and summed in ascending k",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane, the lane's two 32-blocks "
  "scaled at the block boundary",
  "fp8_e4m3 a with an f32 scale per (row, 128 of K), fp4_e2m1 codes with an e8m0 scale per (row, "
  "32 of K) -- `b_ref`, the same exponent plane, is accepted and ignored, bf16 y, fp32 "
  "accumulation; a reserved e8 (0, 255) is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQMxfp4a8M16), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_mxfp4a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_mxfp4a8 },

{ "gemm_nt_q_w4a8_m16", "gemm_nt_q", "gemm",
  "the symmetric 4-bit weight grid at decode: signed int4 codes against int8 activations, the "
  "group's integer sub-sum scaled once by the activation's row scale and the weight's group scale",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane from two 16-byte loads, "
  "the pair fold at the group boundary, the group terms scaled in the epilogue",
  "I8 a in the step order with an f32 scale per row, I4 codes b (two a byte) with an f16 scale "
  "per (row, 128 of K), bf16 y, fp32 accumulation; the sub-sums are exact integers under 2^24, "
  "and the store is the only narrowing; no zero point, so no a_sum",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQW4A8M16), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w4a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_w4a8 },

{ "gemm_nt_q_w8a8_m16", "gemm_nt_q", "gemm",
  "the symmetric 8-bit weight grid at decode: signed int8 codes against int8 activations, the "
  "group's integer sub-sum scaled once by the activation's row scale and the weight's group scale",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane from four 16-byte loads, "
  "the pair fold at the group boundary, the group terms scaled in the epilogue",
  "I8 a in the step order with an f32 scale per row, I8 codes b with an f16 scale per (row, 128 "
  "of K), bf16 y, fp32 accumulation; the sub-sums are exact integers under 2^24, and the store is "
  "the only narrowing; no zero point, so no a_sum; M > 16 is declared out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQW8A8M16), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w8a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_w8a8 },

    { "silu_mul_bf16", "silu_mul", "elem",
      "silu(gate) * up over one fused gate_up plane, the gate the first half",
      "any n; a workgroup a row; the activation is NOT narrowed before the product, as ref's is not "
      "-- act_mul narrows it, over two operands, because the pair it replaces stored it",
      "bf16 in and out, fp32 arithmetic",
      RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
      ROW(cSiluMul), NOTUNE,
      nullptr, nullptr, r3d_rad_silu_mul, nullptr, nullptr, nullptr, r3d_shape_silu_mul },

    { "rmsnorm_quant_fp8_bf16", "rmsnorm_quant_fp8", "quant",
      "residual add, RMS norm and the E4M3 quantiser in one pass: a 5120-element bf16 plane a token "
      "is written once and read once inside one kernel, where the pair writes it and reads it back",
      "group 128 and a multiple of 128; a workgroup a row; the residual sum is stored before the "
      "variance reads it, so the norm is over the bf16 sum, and the codes are taken over the bf16 "
      "normalised value, which out_bf16 receives",
      "bf16 x, residual, gain and optional out_bf16, fp8_e4m3 codes, f32 scales, fp32 arithmetic",
      RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
      ROW(cRmsQuantFp8), NOTUNE,
      nullptr, nullptr, r3d_rad_rmsnorm_quant_fp8, nullptr, nullptr, nullptr,
      r3d_shape_rmsnorm_quant_fp8 },


/* ------------------------------------------------------------------ r3d_sample_pre_f32.hip
 *
 * The sampler's pre-filters: one workgroup a row over the logits plane. The history stages are
 * sparse -- a row's history is a few hundred tokens against a vocabulary of 130560, so the
 * reference's own shape, walk the window and not the row, is the fast shape too.
 */
{ "sample_penalties_f32", "sample_penalties", "sample",
  "presence, frequency and repetition penalties from the row's history, in place",
  "a workgroup a row; the count is taken from the token's FIRST occurrence forward, and the "
  "repetition penalty is sign-dependent -- v > 0 divides, v < 0 multiplies -- as the reference has "
  "it; a penalty of zero writes nothing",
  "f32 logits, f32 params, i32 history; writes only inside the row's own n_vocab columns",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_penalties, nullptr, nullptr, nullptr,
  r3d_shape_sample_penalties },

{ "sample_dry_f32", "sample_dry", "sample",
  "the DRY penalty: a suffix of the history that repeats after a token penalises it, by its length",
  "a workgroup a row; the longest repeat per token, capped at dry_allowed_length; the multiplier "
  "grows by repeated multiply and stops at FLT_MAX/base, which is where an exponent would overflow; "
  "one penalty per token, from its longest repeat, and the FIRST occurrence is the one written",
  "f32 logits, f32 params, i32 history, u8 breakers; a breaker position ends a sequence and is "
  "never penalised; a history no longer than dry_allowed_length penalises nothing",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_dry, nullptr, nullptr, nullptr,
  r3d_shape_sample_dry },

{ "sample_ngram_ban_f32", "sample_ngram_ban", "sample",
  "the n-grams the history shows inside the window are banned to -inf",
  "a workgroup a row; the prefix compare over n-1 tokens, the ban landing on s[idx+n-1]; the "
  "whitelist is removed from the banned set BEFORE the ban, so an exempt id survives; the window is "
  "dry_penalty_last_n, and n = 0 bans nothing",
  "f32 logits, f32 params, i32 history, i32 n-gram pairs, an optional i32 whitelist plane",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_ngram_ban, nullptr, nullptr, nullptr,
  r3d_shape_sample_ngram_ban },

{ "sample_mask_f32", "sample_mask", "sample",
  "the grammar's and the tool-caller's doing: every id whose bit is clear becomes -inf",
  "a workgroup a row over 4080 words at this vocabulary; a word of all ones is skipped and the last "
  "partial word honours n_vocab; a NEGATIVE mask_row skips the row, which is how an unconstrained "
  "request ends up with no mask applied -- the descriptor calls the same thing "
  "a shared row, and the body is the spec; the bound is the bitmask plane's own row count, not M, "
  "so a row addressing the plane's second half is masked as the reference masks it",
  "f32 logits, f32 params, i32 bitmask; a plane with fewer rows than the logits is refused "
  "(RAD_E_SHAPE), the case the geometry never produces",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_mask, nullptr, nullptr, nullptr,
  r3d_shape_sample_mask },

{ "sample_logprobs_f32", "sample_logprobs", "sample",
  "the row's summary for the response: max, the sum of exp, and the k best (id, logit) pairs",
  "one read of the logits plane where the reference reads it three (max, sum, k-list); the sum is "
  "SUM exp(v - max) in DOUBLE, the reference's law, and a row's k-list is one insertion list in "
  "registers, not an upper_bound per element; stats is [M][2 + 2k], and rows the flag leaves alone "
  "are not written",
  "f32 logits, f32 params, f32 stats; k <= 64 is the pool's depth and above it the shim refuses "
  "(RAD_E_UNSUPPORTED); on a row that contains a NaN the reference's list freezes at the first k it "
  "saw, because upper_bound over a NaN returns end and every later compare then fails -- this kernel "
  "reports the row's true k-best, a divergence a plane has to be fed NaNs to reach",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleLogprobs), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_logprobs, nullptr, nullptr, nullptr,
  r3d_shape_sample_logprobs },


/* ------------------------------------------------------------------ r3d_attn_paged_h128.hip
 *
 * Paged attention, head_dim 128, banded on q_len. The two kernels share the block walk and differ
 * in what a workgroup is: decode is one head group over 16 query rows with split-KV, prefill stages
 * a 16-token query tile. The walk's block bounds are the WORKGROUP's -- a tile is staged once into
 * LDS and every wave of the workgroup consumes it, and waves that narrowed the walk to their own
 * query positions would stage different blocks over the same LDS, the last writer per 16-byte chunk
 * winning.
 */
{ "attn_paged_bf16_decode", "attn_paged", "attn",
  "paged attention over a bf16 cache: one head group a workgroup, 16-token tiles, split-KV",
  "head_dim 128; q_len <= 16; two passes over the blocks -- scores are recomputed in the second so "
  "the softmax sum accumulates ascending like the reference, and a masked key is never exponentiated; "
  "the block bounds come from the workgroup's 16 rows, because the staged tile is shared",
  "bf16 q and out, bf16 cache, i32 block_table and seqused; a descale on a bf16 cache is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  ROW(cAttnBf16Decode), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_decode_h128, attn_scratch, nullptr, nullptr,
  r3d_shape_attn_paged },

{ "attn_paged_bf16_prefill", "attn_paged", "attn",
  "the same walk over a 16-token query tile, for a chunk that is wider than the decode bucket",
  "head_dim 128; q_len >= 17; a workgroup is 4 waves over 16 query rows and one KV head; the tile's "
  "idle lanes clamp to the last query row instead of reading past the q plane",
  "bf16 q and out, bf16 cache, i32 block_table and seqused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_PREFILL,
  ROW(cAttnBf16Prefill), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_prefill_h128, attn_scratch, nullptr, nullptr,
  r3d_shape_attn_paged },

{ "attn_paged_fp8_decode", "attn_paged", "attn",
  "paged attention over an E4M3 cache, the descale riding the widening into the tile",
  "head_dim 128; q_len <= 16; a 16-code chunk is the aligned 16-byte unit -- eight codes are 8 bytes "
  "and a 16-byte load at a slot's odd half returns garbage; the descale "
  "multiplies into the f16 tile, exact for a power-of-two scale inside f16's range",
  "bf16 q and out, fp8_e4m3 cache, i32 block_table and seqused, f32 k_descale and v_descale "
  "optional (absent is scale 1.0)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  ROW(cAttnFp8Decode), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_decode_h128, attn_scratch, nullptr, nullptr,
  r3d_shape_attn_paged },

{ "attn_paged_fp8_prefill", "attn_paged", "attn",
  "the E4M3 cache over a 16-token query tile",
  "head_dim 128; q_len >= 17; the same staged walk as the bf16 prefill, with the codes widened as "
  "they land in the tile",
  "bf16 q and out, fp8_e4m3 cache, i32 block_table and seqused, f32 descales optional (absent is "
  "scale 1.0)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_PREFILL,
  ROW(cAttnFp8Prefill), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_prefill_h128, attn_scratch, nullptr, nullptr,
  r3d_shape_attn_paged },


/* ------------------------------------------------------------------ r3d_sample_f32.hip
 *
 * The sampler's spine: the row's top-R, and the stages that turn the top-R's candidate plane into
 * one token. The candidate set is the contract between them -- `cand_val` carries LOGITS, so each
 * narrowing stage takes its own softmax and the stages compose in any order; a dropped candidate is
 * `cand_idx` -1 and `cand_val` -inf; the set is descending by value then ascending by id; and every
 * stage leaves a greedy row byte-identical, because a greedy row's token is `sample_argmax`'s.
 * `topk_next` is a sort by (-value, +column) wearing a loop -- the rule `v == *pv && i <= *pi` is
 * "the next one in a run of equal values" -- so one pass keeping the best R returns the reference's
 * R pairs, in its order, with its tie-breaks to the lower column.
 */
{ "row_topk_bf16", "row_topk", "moe",
  "the row's top R, with the reference's tie-breaks, in one scan of the row",
  "one pass keeps the best R in the reference's order, where the reference scans the row R times; "
  "R beyond a pass's pool rescans with the chain's state carried, so the answer is the reference's "
  "at any R and only the time grows; a slot a short row cannot fill is idx -1 and val -inf; a row "
  "that opens with a NaN reports (0, NaN) R times, because the reference's seed takes the first "
  "element it sees and nothing displaces a NaN",
  "bf16 x, i32 idx [M,R], f32 val [M,R], an optional f32 pairs [M,R,2] (the id's bits and the value)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRowTopkBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_row_topk, nullptr, nullptr, nullptr, r3d_shape_row_topk },

{ "row_topk_f32", "row_topk", "moe",
  "the row's top R over an f32 plane, in one scan",
  "the same one-pass insertion as the bf16 row; the band's router row costs the reference 7.0 GB of "
  "scans and this kernel 468 MB, and at the band it is the sifts that cost, not the bytes: the f32 "
  "plane moves half again the traffic for the same 23 ms",
  "f32 x, i32 idx, f32 val, optional f32 pairs",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRowTopkF32), NOTUNE,
  nullptr, nullptr, r3d_rad_row_topk, nullptr, nullptr, nullptr, r3d_shape_row_topk },

{ "sample_argmax_f32", "sample_argmax", "sample",
  "the greedy token: the row's argmax, ties to the lower id",
  "a workgroup a row; the max is a strict `>`, so a NaN is never adopted -- the reference's scan "
  "keeps the first element it sees, which is why a row opening with a NaN yields token 0, and a row "
  "of all -inf yields 0; the stage that reads no scalar at all, so a missing params operand is not "
  "a refusal: the reference describes the operand and does not read it",
  "f32 logits, i32 token [M]; the params plane is described, not read",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_argmax, nullptr, nullptr, nullptr,
  r3d_shape_sample_argmax },

{ "sample_temp_f32", "sample_temp", "sample",
  "temperature, in place: every logit multiplied by the reciprocal of temp",
  "a workgroup a row; a row is skipped when temp is not positive or is exactly 1; the reciprocal is "
  "formed ONCE, as the reference forms it, not a divide per element -- a divide per element is a "
  "different f32, and the argmax after it is only bit-exact under the transcription",
  "f32 logits INOUT, f32 params; the most expensive row in the slice at 11 ms a step, and the cost "
  "is the plane: 1 GiB in and 1 GiB out for one multiply an element",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_temp, nullptr, nullptr, nullptr,
  r3d_shape_sample_temp },

{ "sample_topk_f32", "sample_topk", "sample",
  "the candidate set the rest of the sampler draws from, written whole every step",
  "the set's width is the BUFFER's, not the row's: top_k 40 in a 1024-wide plane is 40 candidates "
  "and 984 holes, and a stale candidate left in a hole is a token the pick can draw; the row's "
  "top_k only narrows inside the width, and `<= 0` or a width above it takes the whole width, as "
  "the reference clamps; cand_val carries the LOGIT, which is what lets the narrowing stages "
  "compose in any order; a greedy row takes none",
  "f32 logits, f32 params, i32 cand_idx [M,n_cand] (local columns), f32 cand_val, an optional i32 "
  "pairs [M,2*n_cand] of (global id, logit bits) with 0xffffffff for a hole; n_cand <= 1024",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_topk, nullptr, nullptr, nullptr,
  r3d_shape_sample_topk },

{ "sample_pick_f32", "sample_pick", "sample",
  "the draw: one uniform over the candidate set's own softmax, in index order",
  "splitmix64 over the seed and the position, the top 24 bits over 2^24, keyed by the pick's own "
  "key; the exps are computed by the wave in parallel and the CDF accumulates on one lane in "
  "candidate index order, because the order is the law -- parallel values, serial sums; an empty "
  "set writes -1; a set whose mass falls below u falls back to the last candidate above -infinity, "
  "which is why a masked tail is never the fallback",
  "i32 cand_idx, f32 cand_val, f32 params (seed, pos), i32 token [M]; only token is written",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_pick, nullptr, nullptr, nullptr,
  r3d_shape_sample_pick },

{ "sample_topp_f32", "sample_topp", "sample",
  "nucleus: the shortest prefix whose cumulative probability reaches top_p",
  "the stage takes its own softmax over the set (max by strict `>` over live candidates, then exp, "
  "then the predicate), because the set carries logits; at least one candidate survives; it is a "
  "PREFIX rule in the set's order, which is why the set's order is contract and why no stage sorts "
  "it",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when top_p is out of range; a greedy row "
  "and an all-holes row are left alone",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_topp, nullptr, nullptr, nullptr,
  r3d_shape_sample_topp },

{ "sample_minp_f32", "sample_minp", "sample",
  "min-p: drop what is below p times the best candidate's probability",
  "the stage's own softmax again; the best always survives; a relative cut under the best, not a "
  "prefix, so it composes with top-p in either order",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when min_p is out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_minp, nullptr, nullptr, nullptr,
  r3d_shape_sample_minp },

{ "sample_typical_f32", "sample_typical", "sample",
  "locally typical: drop the candidates whose surprisal is furthest from the set's entropy",
  "the set's entropy from the same exps, the kept mass overshooting p by design and not depending "
  "on the order, ties kept -- a stage that sorted the set would genuinely disagree with the schema",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when typical_p is out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_typical, nullptr, nullptr, nullptr,
  r3d_shape_sample_typical },

{ "sample_xtc_f32", "sample_xtc", "sample",
  "the exclusion of top candidates: a coin, then drop all but the least likely of the clearers",
  "its own coin (the seed keyed by XTC) decides whether the stage fires at all; when it fires it "
  "drops every candidate at or above the threshold EXCEPT the least likely of them, and does nothing "
  "when fewer than two clear it -- a stage that dropped unconditionally would fire on 100% of steps",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when the probability or the threshold is "
  "out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_xtc, nullptr, nullptr, nullptr,
  r3d_shape_sample_xtc },

/* ------------------------------------------------------------------ r3d_dflash_f32.hip
 *
 * The speculative decoder's accept step. It is a gather, not arithmetic: two 63.8 MiB codebooks, a
 * walk gathering `M * steps * rank` elements, so the score's rank sum is one lane's serial chain --
 * there is nothing to hide the gather behind, and a tree would be a different sum for no gain.
 */
/* ------------------------------------------------------------------ r3d_gemm_nt_q_w2a8.hip
 *
 * `output.weight`: U2 codes, int8 activations, one f32 scale per activation ROW (ZP_I8_ROW), and a
 * joined `b_scale` the container writes as two planes -- scale f16 and zero u8 -- behind one operand
 * that only a layout hook can present. libref's shape hook declines the dtype for exactly that
 * reason, so kbench has no case for the band: the row's opd_shape is null, which is the tool's
 * named skip, and the joined plane is built with rad_planes_relayout, the same builder the driver
 * uses.
 *
 * Exactness, measured and not assumed: `v_fdot2_f32_f16` carries ~22 bits of mantissa on integer
 * operands, not 24, so an integer dot through it is inexact above 2^20 (71027 of 262144 sampled dots
 * at bias 1024, worst |err| 0.125; 70115 and 0.015625 at bias 128). The bias is 128 because the
 * error's ratio to the magnitude is ~2^-22 and shrinking the bias shrinks the tail. The result is
 * that 0.1-0.2% of outputs land one bf16 ulp off the reference: rel_l2 5.0e-3 at the band,
 * max_abs 9.8e-4, no non-finite anywhere. This row is NOT bit-exact by construction, and the dot
 * probe in the unit is the check that states it.
 *
 * M <= 16 is decode. 46 of these per token, 135 GFLOP a token of CPU work that the host row puts on
 * the card at 3.0-4.2 ms (66.8 MB of codes at the card's 22 GB/s is the floor; the warm-L2 run is
 * 2.43 ms at 30 GB/s).
 */
{ "gemm_nt_q_w2a8_m16", "gemm_nt_q", "gemm",
  "the vocab projection at decode: 2-bit codes, int8 activations, the zero removed with a_sum",
  "a wave takes one weight row; lane l owns group l&15 and half of it, 64 codes to a 16-byte load, "
  "the pair (l, l^16) folding the group in one shuffle; lanes 0..15 hold the per-group terms and "
  "lane 0 sums the groups in ascending order in one accumulator, which is the reference's order. "
  "The codes widen to f16 through a bias of 128, so the dot's integer answer comes back with "
  "B*sum(a) on it and a_sum -- group-major, the reference's own 16-row pad -- removes it",
  "U2 codes b, I8 activations a, f32 a_scale (one per row), the joined b_scale (f16 scale + u8 zero "
  "as rad_planes stores them), f32 a_sum, bf16 y. fp8a8, fp8a16 and w4a8 take their own rows; a zero "
  "plane stored sub-byte (U2/U4/U5) is refused, not approximated -- the bit-extract path is 10 lines "
  "of rad_load_bits and is not written; M > 16 is refused, prefill's vocab is logits_gemm",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cGemmQW2A8), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w2a8, nullptr,
  r3d_layout_gemm_nt_q_w2a8, r3d_relayout_gemm_nt_q_w2a8, nullptr },

{ "gemm_nt_q_w2a8_tiled", "gemm_nt_q", "gemm",
  "the packed-code grids at the heights the skinny row declines: 2-bit and 4-bit codes, int8 "
  "activations, the batched decode and the draft head of a 2-bit vocab projection",
  "the same 128x128 tile as the fp8a8 arm, with the codes unpacked on the way into LDS -- 8 codes "
  "from a 4-byte load at U2, 16 from a 16-byte load at U4 -- and the epilogue folding "
  "(code - zero) * scale per staged element, so the dot runs over values and no a_sum term is "
  "reduced: the zero comes off term by term, which is why Σ a(q-z) needs no Σ a here while the m16 "
  "row needs it to lift the integer bias off its f16-as-int dot. The row's scale and zero are one "
  "per ROW on a {1, group} plane, so the staging reads 128 scalars a chunk where the fp8a8 arm reads "
  "one, at the row the grid puts it; issued one chunk ahead of the fold that consumes them, because "
  "a scalar load with nothing to overlap it is the whole band (210.8 ms against 2.396 ms with the "
  "answer hoisted, the same output bits). r3d_tiling.h's swizzle is shared, proven byte-identical",
  "M <= 16 is gemm_nt_q_w2a8_m16's by its declared C_LE(\"M\", 16); fp8a8 and fp8a16 take their own "
  "rows, and the fp8a8 arm refuses a band carrying a_sum at its own line -- a row that silently "
  "ignores a band operand passes a checkpoint whose a_sum is garbage. A zero plane stored sub-byte "
  "(U2/U4) is refused and named, not approximated: rad_load_bits is the ten-line fix and it is "
  "written for the codes, not the zero. Measured, VRAM: M=64 N=130560 K=2048 w2a8 24.151 ms, 1417 "
  "GFLOP/s, 3.7 GB/s useful against a 0.449 ms read floor of the 73.1 MB operand -- at kKC=32 a "
  "staged chunk draws 8 bytes from a row of pitch 512, so the band moves 8x the bytes it uses and "
  "the lever is a wider k-chunk, not more loads in flight (the codes lead measured 0.6%, "
  "bit-identical)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cGemmQW2A8Tiled), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w2a8_tiled, nullptr,
  r3d_layout_gemm_q_fp8_weight, r3d_relayout_gemm_q_fp8_weight, r3d_shape_gemm_nt_q },

/* ------------------------------------------------------------------ r3d_gemm_fp8a8_gated.hip
 *
 * The gate couples column i to column n+i, so a lane owns both dots and the coupling costs no extra
 * weight traffic; the block is one group by 256/group rows so the group's amax closes in-block,
 * which is also why there is nothing to tune (the block shape is the quantiser's rule, not a knob)
 * and why the row keeps its M ceiling in the prose. At the band's own numbers this is ~341 GFLOP/s
 * at M=64 -- 0.7% of the card's fp8 peak, 46 of these per forward pass -- and it is still the row
 * the model wants, because the alternative it replaces is the reference's host GEMM: the same 3.2
 * GFLOP per layer on the CPU, which is 135 GFLOP a token for this checkpoint.
 */
{ "gemm_nt_q_gated_fp8a8", "gemm_nt_q_gated", "gemm",
  "the gated MLP band's GEMM, its gate and its output quantiser in one launch",
  "a block is one 128-wide group by 256/group rows, so the group's amax closes in-block: wave max "
  "then an LDS fold over the lanes that share a group. The gate couples column i to column n+i, so a "
  "lane owns both dots and the coupling costs no extra weight traffic. No [M,N] bf16 intermediate "
  "exists: the row is narrowed, gated and coded from registers, which is why there is no scratch and "
  "no second pass. The narrowing matches the unfused chain -- both halves bf16 before the gate, the "
  "activation NOT narrowed before the product, the product formed in f32 then narrowed -- so the "
  "codes are the codes the chain would have written, byte for byte at the band",
  "fp8_e4m3 a and b, f32 a_scale, bf16 b_scale (f32 accepted and equal bit for bit; fp8a8f32 is a "
  "different dtype and takes no row here), f32 scale, E4M3 q, bf16 out_bf16 optional; f32 "
  "accumulation with -ffp-contract=off; f16x4 dot pairing, the code widening exact. M > 64, N odd, K "
  "% group != 0, group outside {32,64,128,256} or wider than n, sk > 1 and an absent act are refused, "
  "not approximated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmGated), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_gated, nullptr, nullptr, nullptr,
  r3d_shape_gemm_nt_q_gated },

{ "dflash_select_bf16", "dflash_select", "draft",
  "DFlash2's greedy walk over the candidate lattice, one lane a (sequence, candidate) pair",
  "score[l][c] = unary[l][c] + SUM_r pred[pid(l)][r] * hp[l][r] * succ[cand[l][c]][r], the rank sum "
  "ascending in r in f32 with ax[r] = pred * hp rounded to f32 BEFORE it meets succ, because the "
  "reference writes it to a float array first; argmax strict `>` over c ascending, so the earliest "
  "candidate wins a tie and a NaN never takes the lead; pid(0) is the anchor at m * anchor_stride "
  "and pid(l) the walk's own previous pick, so the steps loop is serial with the two LDS round "
  "trips its dependence requires; hp absent is a plane of ones; a predecessor outside the vocabulary "
  "contributes zero and a candidate outside it is scored by unary alone, which is what makes a "
  "shard's padding rows legal input",
  "bf16 codebooks (pred, succ, optional hp) with contiguous rows, f32 unary, i32 cand, anchor and "
  "tokens; pitched rows served on cand, unary, tokens and hp, a row stride INSIDE a codebook row "
  "refused (RAD_E_STRIDE); top_k > 256 refused (RAD_E_UNSUPPORTED); a cand plane describing more "
  "sequences than the band's M is refused, not truncated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDflashSelect), NOTUNE,
  nullptr, nullptr, r3d_rad_dflash_select, nullptr, nullptr, nullptr,
  r3d_shape_dflash_select },

/* ------------------------------------------------------------------ r3d_vocab_bf16.hip
 *
 * The head is the largest read in a decode step -- 510 MiB at this checkpoint's 130560 x 2048 -- and
 * the sampler reads f32, which is why the store is typed rather than narrowed and why a prefill band
 * writes 1.07 GB. Measured on the card: 3.29 ms a token at M=1 (163 GB/s of the head), 0.507 ms a
 * token at M=16 (the head read once for sixteen rows), 269 ms for a 2048-token prefill at 4067
 * GFLOP/s. Against the reference's law the f32 plane is 1.77e-6 in rel_l2, which is ~30 f32 ulps at
 * K=2048 and is the pair-dot folding two products into one rounding -- measured identical against
 * the same law with the operands widened to f16 first, so the widening contributes nothing and the
 * head's own bf16 quantisation (4.9e-4 an element) is 300x coarser than the row.
 */
{ "logits_gemm_bf16", "logits_gemm", "vocab",
  "logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T, the head read once a token",
  "M <= 16: a wave a run of vocabulary columns, one output element a lane, the head's rows streamed "
  "past an activation kept in registers, no wave reduction. M > 16: 128x128 tiles through LDS, 8x8 "
  "f32 accumulators a lane, the head staged with an 8-element swizzle, k ascending in 32-code "
  "chunks. The grid is vocab-wide at every M, so there is no split-K and no scratch",
  "bf16 x, f32 logits: every sampler stage reads f32, so the narrowing the schema does not ask for "
  "is not performed, and a prefill band's output plane is M x n_vocab f32. The head arrives in the "
  "form the container stores and the row reads both, chosen by the operand's header, not by a "
  "parameter: bf16 rows in the order they are stored, or -- what this checkpoint's output.weight "
  "actually is -- E4M3 codes with a bf16 scale per 128x128 block as two container planes, which the "
  "row's layout hook presents as the one operand the plan says they are, moving no byte. E4M3 and "
  "int8 widen to f16 exactly, so the fold into f16 is the only arithmetic the codes pay: 2.77e-04 "
  "from the reference's exact law, one f16 rounding, 250x finer than the code's own quantum. A head "
  "with a zero plane is w2a8's grid and is refused at the shim's own line; so is a scale block that "
  "is not 128x128, which is gemm_nt_q's grid",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cLogitsGemm), NOTUNE,
  nullptr, nullptr, r3d_rad_logits_gemm_bf16, nullptr,
  r3d_layout_logits_head, r3d_relayout_logits_head, r3d_shape_logits_gemm_bf16 },

/* ------------------------------------------------------------------ r3d_gemm_fp8a8_tiled.hip
 *
 * The prefill projection: this checkpoint declares 176 of these a forward pass (44 of qkv, 44 of
 * the gated MLP's 12288 columns, 44 of o, 44 of the 2048-wide one, three of the drafter at
 * K=10240), and the reference does their 135 GFLOP a token in a host loop.
 * Measured at the graph's own bands, weights in VRAM: 28.171 ms at N=12288 K=2048 (3659 GFLOP/s),
 * 14.699 at N=2048 K=6144, 5.911 at N=2560, 4.381 at N=2048 K=2048, 29.798 at N=2048 K=10240 --
 * 27 to 36% of the card's dense f16 peak against a byte floor of 0.051 to 0.257 ms. These are the
 * first bands in this library where the dot, not the wire, is the limit, and it is the reason the
 * E4M3 decoder is a table: at the tile's densities the decode ALU is on the critical path.
 */
{ "gemm_nt_q_fp8a8_tiled", "gemm_nt_q", "gemm",
  "y[M,N] = a[M,K] @ dequant(b)^T at prefill heights, the group sub-sum from the raw codes and one "
  "scale per group, which is the reference's law and the reason the codes are never pre-scaled",
  "128x128 tiles through LDS, 64 f32 accumulators a lane, k in 32-code chunks, the family's 8-element "
  "XOR swizzle on the staged rows; both sides decode through the block's 512-byte E4M3->f16 table "
  "and leave staging as f16 with their scale folded in -- a_scale per staged row, b_scale one scalar "
  "per tile, because the tile's 128 columns are exactly the scale plane's row extent and 32 divides "
  "128. One accumulator, no per-group partials, 237 VGPRs and no spill",
  "E4M3 codes both sides, f32 a_scale per (row, group of 128), bf16 b_scale per 128x128 block -- the "
  "container's own block; a {1,group} scale block is w2a8's grid and is refused, as is a zero plane "
  "and an a_sum, fp8a8 being symmetric. y bf16 with one narrowing at the store, and at a bf16 output "
  "the store's ulp (3.9e-3) is the binding floor, not the fold's (9.8e-4 of the quadrature scale): "
  "83% of outputs are bit-exact, 99.93% land within the fold plus one store ulp, none beyond. The "
  "fold is v_mul_f16, never f16x2_pack, which rounds toward zero and flushes subnormals",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFP8Tiled), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8_tiled, nullptr,
  r3d_layout_gemm_q_fp8_weight, r3d_relayout_gemm_q_fp8_weight, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_mxfp4a8_tiled", "gemm_nt_q", "gemm",
  "y[M,N] = a[M,K] @ dequant(b)^T at prefill heights, the weight in OCP MXFP4: e2m1 codes, two a "
  "byte, one e8m0 scale per 32 of K; the 32-block partial scales at the block boundary and the "
  "blocks sum in ascending k",
  "128x128 tiles through LDS, 64 f32 accumulators a lane for the block partials and 64 more for the "
  "running sum, k in 32-code chunks -- one OCP 32-block a chunk -- the family's 8-element XOR "
  "swizzle on the staged rows; the codes unpack to e4m3 through eight fixed magnitude bytes and "
  "decode through the block's 512-byte table, a_scale folded per staged row, the e8 read at "
  "staging and held in registers for the flush",
  "E4M3 a with an f32 scale per (row, 128 of K), fp4_e2m1 codes with an e8m0 scale per (row, 32 "
  "of K) -- `b_ref`, the same exponent plane, is accepted and ignored, bf16 y, fp32 accumulation. "
  "The e2m1 magnitude is exact in e4m3, the code products are "
  "exact in f16, the 32-block partial scales by a power of two in f32, and the store is the only "
  "narrowing; a reserved e8 (0, 255) is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_TILED,
  ROW(cGemmQMxfp4a8Tiled), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_mxfp4a8_tiled, nullptr,
  nullptr, nullptr, r3d_shape_gemm_nt_q_mxfp4a8 },

/* The same tiling on gemm_nt's operands, the prefill projection the reference runs on the host: at
 * M=2048 this checkpoint's q/k/v/o and the MLP's up/down are 135 GFLOP a token of f32-accumulating
 * bf16, and the reference does them on the CPU. The exactness is the m16 row's, because it is the
 * same dot. */
{ "gemm_nt_bf16_tiled", "gemm_nt", "gemm",
  "a[M,K] @ b[N,K]^T at prefill heights, f32 accumulate, one narrowing at the store",
  "128x128 tiles through LDS, 8x8 f32 accumulators a lane, k ascending in 32-code chunks, the "
  "partials closed in the tile; M <= 16 stays with gemm_nt_bf16_m16, whose wave-per-run-of-columns "
  "shape is the right one there -- the two rows' M ranges are complements, so neither competes for "
  "the other's band",
  "bf16 a, b and y, fp32 accumulation; the pair is folded into the accumulator by one rounding, so "
  "about 0.3% of a long-K output lands one bf16 store ulp from the reference's, and no output is "
  "further. 99.93% bit-exact measured at the 512-column twin, 100% within two ulps",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_TILED,
  ROW(cGemmNTTiled), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_bf16_tiled, nullptr,
  nullptr, nullptr, r3d_shape_gemm_nt },

};

/* ================================================================== the exports
 *
 * The two every plugin must export, the four that make a kernel library, and `rad_kernel_concurrent`.
 *
 * libr3d exports no `rad_kernel_chain`: chaining replaces the queue's ordering bit between two
 * dependent launches with a spin in the kernel, and it pays where a wave is idle during that spin.
 * Nothing here is built as a chained form, so the answer would be 0 for every row; the symbol is
 * left out, which is the same answer with one less thing to keep in step (ABI,
 * `rad_kernel_chain`: "without the symbol, or with 0, the engine never asks").
 *
 * It exports no `rad_kernel_collective` either: there is no second rank in this plugin's world. The
 * cards that run these rows are single-GPU cards, and an all-reduce over the PCIe x1 slot most of
 * them sit in is slower than sending the bytes through the host.
 */
extern "C" int rad_kernel_count(void) {
    return (int)(sizeof(kKernels) / sizeof(kKernels[0]));
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    return &kKernels[i];
}

/* Which rows touch nothing but their operands and the scratch they were handed.
 *
 * The elementwise, norm, quant, cast, rope, cache and gather rows do: one launch, one pass, no
 * state between launches, so two of them may run at once and the engine can start a row before an
 * independent earlier one finishes.
 *
 * The GEMM rows do not, and the reason is the scratch, not the kernel. The scratch arena belongs to
 * the op instance, so two layers' decode GEMMs launched at the same time share it: a split-K row's
 * partials and arrival counters from the two launches would land on each other. With sk = 1 a GEMM
 * launch has no partials and is safe, but the promise is per row and the axis is chosen per
 * geometry, so the row states the worst case. When split-K is off on a given card (a tuned point of
 * sk = 1 for every shape), the loss is one overlap the engine would have taken.
 */
extern "C" int rad_kernel_concurrent(int i) {
    const RadKernelInfo* k = rad_kernel_at(i);
    if (!k) return 0;
    return k->scratch == nullptr;
}

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libr3d",
    "0.1.0",
    "the RDNA3 (gfx11) kernel library: the elementwise, norm, rope, cache and embedding rows, fp8 "
    "and int8 activation quantisation with its dequant and gated forms, and the skinny bf16 and "
    "fp8a16 GEMMs, every one of them through v_dot2acc_f32_f16 because RDNA3 has no matrix "
    "instruction a compiler will emit",
    /* The architectures this build's code objects run on, spelled as `RadDeviceProps::arch`. The
     * loader compares a device's arch against these words and warns once when the code object it
     * picked names a card this list does not (`target_covers`), so the string has to be the build's
     * real targets rather than an author's guess at them.
     * RAD_GPU_TARGETS reaches this source as a compile definition from CMakeLists.txt; the fallback
     * is the whole family, which is what a build that did not say is claiming. */
    R3D_BUILD_TARGET,
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
