/* r4d_ds41_ggml.inc -- the ggml ops (GEMMs over llama.cpp's native blocks) in r4d_rows.cpp, included three times:
 * R4D_DS41_SPECS (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (entries of
 * kSchemas) and R4D_DS41_ROWS (entries of kKernels). No include guard, on purpose.
 *
 * gemm_nt_ggml, moe_gemm_ggml, logits_gemm_ggml and embed_lookup_ggml over llama.cpp's own GGUF
 * blocks (r4d_ds41_ggml.hip). Schemas are libref's (ref_registry.cpp), field for field and doc for
 * doc. Rows come in two block families, because K must be whole blocks of the type: the 256-value
 * K-quants and I-quants (q5_K q6_K iq2_xxs iq1_s) and the 32-value types (q8_0 iq4_nl). Each GEMM
 * has a weight-streaming `_gemv` row for the decode band (M <= 16; for the grouped GEMM M counts
 * tokens) above a `_tiled` row that serves any M. Every row relayouts a GGUF import's planes into
 * the type's native blocks (r4d_ds41_ggml_layout.cpp, libref's bytes and tag) and also reads the
 * float logical form, which is what rad-kbench's random operands pass. Built only with libquant's
 * sources (R4D_HAVE_GGML), which the layout needs. */
#if defined(R4D_DS41_SPECS)

static const RadParamSpec pDs41GemmGgml[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("type"),
    P_STR("dtype") };
static const RadOperandSpec oDs41GemmGgml[] = {
    { "a", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pDs41MoeGemmGgml[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
    P_INT("top_k"), P_STR("a_order"), P_STR("type"), P_STR("dtype") };
static const RadOperandSpec oDs41MoeGemmGgml[] = {
    { "a", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WTAB, 0 }, { "sorted_tok", RAD_OPD_IN, 0 },
    { "expert_offset", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pDs41LogitsGgml[] = { P_INT("M"), P_INT("n_vocab"), P_INT("n_embd"),
    P_STR("type"), P_STR("dtype") };
static const RadOperandSpec oDs41LogitsGgml[] = {
    { "x", RAD_OPD_IN, 0 }, { "lm_head", RAD_OPD_WEIGHT, 0 }, { "logits", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pDs41EmbedGgml[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
    P_STR("type"), P_STR("dtype"), P_INTO("vocab_offset"), P_F64O("mult") };
static const RadOperandSpec oDs41EmbedGgml[] = {
    { "tokens", RAD_OPD_IN, 0 }, { "wte", RAD_OPD_WEIGHT, 0 }, { "x", RAD_OPD_OUT, 0 },
};

#define R4D_DS41_GG_T256 "q5_K q6_K iq2_xxs iq1_s iq3_xxs"
#define R4D_DS41_GG_T32  "q8_0 iq4_nl"
/* The decode band: up to 16 activation rows (two passes of eight over the streamed weight). */
static const RadConstraint cDs41GgGemv256[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("N", 1), C_GE("K", 256), C_DIV("K", 256),
    C_IN("type", R4D_DS41_GG_T256), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgGemv32[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_IN("type", R4D_DS41_GG_T32), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgTiled256[] = {
    C_GE("M", 1), C_GE("N", 1), C_GE("K", 256), C_DIV("K", 256),
    C_IN("type", R4D_DS41_GG_T256), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgTiled32[] = {
    C_GE("M", 1), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_IN("type", R4D_DS41_GG_T32), C_IN("dtype", "bf16"),
};
/* The grouped GEMM: up to 1024 experts (the live-expert list is in LDS), either row order. */
static const RadConstraint cDs41GgMoeGemv256[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("N", 1), C_GE("K", 256), C_DIV("K", 256),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", R4D_DS41_GG_T256), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgMoeGemv32[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", R4D_DS41_GG_T32), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgMoeTiled256[] = {
    C_GE("M", 1), C_GE("N", 1), C_GE("K", 256), C_DIV("K", 256),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", R4D_DS41_GG_T256), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgMoeTiled32[] = {
    C_GE("M", 1), C_GE("N", 1), C_GE("K", 32), C_DIV("K", 32),
    C_GE("n_expert", 1), C_LE("n_expert", 1024), C_GE("top_k", 1), C_IN("a_order", "token sorted"),
    C_IN("type", R4D_DS41_GG_T32), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgLogitsGemv256[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("n_vocab", 1), C_GE("n_embd", 256), C_DIV("n_embd", 256),
    C_IN("type", R4D_DS41_GG_T256), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgLogitsGemv32[] = {
    C_GE("M", 1), C_LE("M", 16), C_GE("n_vocab", 1), C_GE("n_embd", 32), C_DIV("n_embd", 32),
    C_IN("type", R4D_DS41_GG_T32), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgLogitsTiled256[] = {
    C_GE("M", 1), C_GE("n_vocab", 1), C_GE("n_embd", 256), C_DIV("n_embd", 256),
    C_IN("type", R4D_DS41_GG_T256), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgLogitsTiled32[] = {
    C_GE("M", 1), C_GE("n_vocab", 1), C_GE("n_embd", 32), C_DIV("n_embd", 32),
    C_IN("type", R4D_DS41_GG_T32), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgEmbed256[] = {
    C_GE("M", 1), C_GE("n_vocab", 1), C_GE("n_embd", 256), C_DIV("n_embd", 256),
    C_IN("type", R4D_DS41_GG_T256), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41GgEmbed32[] = {
    C_GE("M", 1), C_GE("n_vocab", 1), C_GE("n_embd", 32), C_DIV("n_embd", 32),
    C_IN("type", R4D_DS41_GG_T32), C_IN("dtype", "bf16"),
};
#undef R4D_DS41_GG_T256
#undef R4D_DS41_GG_T32

#elif defined(R4D_DS41_SCHEMAS)

SCHEMA("gemm_nt_ggml", pDs41GemmGgml, oDs41GemmGgml,
       "y = a . w^T with w llama.cpp's own blocks of GGUF type `type` (\"iq1_s\", \"q6_K\", ...), "
       "relayouted from the type's planes at load: each row dequantised as llama.cpp does, f32 dot in "
       "index order, rounded once to dtype. A float [N, K] w is the logical form."),
SCHEMA("moe_gemm_ggml", pDs41MoeGemmGgml, oDs41MoeGemmGgml,
       "moe_gemm with each expert's weight in llama.cpp's blocks of GGUF type `type`: y[i] = a[row(i)] . "
       "w[e]^T in sorted order, row(i) per `a_order` as moe_gemm, a dropped slot a row of zeros. `w` is a "
       "WEIGHT TABLE of n_expert entries, each read as gemm_nt_ggml reads its weight (native blocks, or "
       "the float [N, K] logical form), so each output row equals gemm_nt_ggml's for its row and "
       "expert."),
SCHEMA("logits_gemm_ggml", pDs41LogitsGgml, oDs41LogitsGgml,
       "logits_gemm with lm_head in llama.cpp's blocks of GGUF type `type`, read as gemm_nt_ggml reads its "
       "weight: logits[M, n_vocab] (f32) = x[M, n_embd] . lm_head^T."),
SCHEMA("embed_lookup_ggml", pDs41EmbedGgml, oDs41EmbedGgml,
       "embed_lookup from a table in llama.cpp's blocks of GGUF type `type`: the id's row dequantised as "
       "llama.cpp does, times `mult` when given, rounded once. Padding, bounds and vocab_offset behave as "
       "embed_lookup's. A float [n_vocab, n_embd] wte is the logical form."),

#elif defined(R4D_DS41_ROWS)
#if defined(R4D_HAVE_GGML)

#define R4D_DS41_GG_HOOKS(shape)                                                                   \
    r4d_ds41_ggml_layout, r4d_ds41_ggml_relayout, shape, r4d_ds41_ggml_unrelayout

{ "gemm_nt_ggml_qk256_gemv", "gemm_nt_ggml", "gemm",
  "y = a . w^T over llama.cpp's blocks, the weight streamed once: a wave per weight row (two from 3 "
  "rows on wide weights), lanes over its 32-value units, the activation chunk staged in LDS as f32, "
  "a wave butterfly per sum",
  "M 1..16 (eight rows a pass), any N, K a multiple of 256", "bf16 a and y; q5_K, q6_K, iq2_xxs or "
  "iq1_s blocks (or the f32 logical form), dequantised to f32 as llama.cpp does, f32 products and sum",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41GgGemv256), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_ggml_gemv, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_gemm_nt_ggml) },
{ "gemm_nt_ggml_qk32_gemv", "gemm_nt_ggml", "gemm",
  "gemm_nt_ggml_qk256_gemv over the 32-value block types", "M 1..16, any N, K a multiple of 32",
  "bf16 a and y; q8_0 or iq4_nl blocks (or the f32 logical form)",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41GgGemv32), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_ggml_gemv, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_gemm_nt_ggml) },
{ "gemm_nt_ggml_qk256_tiled", "gemm_nt_ggml", "gemm",
  "y = a . w^T over llama.cpp's blocks in 128 x 128 tiles: the weight tile dequantised into LDS once a "
  "32-value slab, the next slab's blocks prefetched, f32 8 x 8 register tiles",
  "any M, any N, K a multiple of 256", "bf16 a and y; q5_K, q6_K, iq2_xxs or iq1_s blocks (or the f32 "
  "logical form), f32 products and sum",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41GgTiled256), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_ggml_tiled, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_gemm_nt_ggml) },
{ "gemm_nt_ggml_qk32_tiled", "gemm_nt_ggml", "gemm",
  "gemm_nt_ggml_qk256_tiled over the 32-value block types", "any M, any N, K a multiple of 32",
  "bf16 a and y; q8_0 or iq4_nl blocks (or the f32 logical form)",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41GgTiled32), NOTUNE,
  nullptr, nullptr, r4d_rad_gemm_nt_ggml_tiled, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_gemm_nt_ggml) },

{ "moe_gemm_ggml_qk256_gemv", "moe_gemm_ggml", "moe",
  "the grouped GEMM over llama.cpp-block experts, each live expert's weight streamed once: a block "
  "scans the offsets for the experts with rows, then a wave per weight row (two on K-quant/q8_0 "
  "experts of 1024 rows or more; eight I-quant units a lane in flight) dots it with that expert's "
  "gathered rows (eight a pass)",
  "M 1..16 tokens, any N, K a multiple of 256, up to 1024 experts", "bf16 a and y; q5_K, q6_K, "
  "iq2_xxs or iq1_s experts (table or stacked, or the f32 logical form), i32 routing",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41GgMoeGemv256), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_ggml_gemv, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_moe_gemm_ggml) },
{ "moe_gemm_ggml_qk32_gemv", "moe_gemm_ggml", "moe",
  "moe_gemm_ggml_qk256_gemv over the 32-value block types",
  "M 1..16 tokens, any N, K a multiple of 32, up to 1024 experts",
  "bf16 a and y; q8_0 or iq4_nl experts (or the f32 logical form), i32 routing",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41GgMoeGemv32), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_ggml_gemv, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_moe_gemm_ggml) },
{ "moe_gemm_ggml_qk256_tiled", "moe_gemm_ggml", "moe",
  "the grouped GEMM over llama.cpp-block experts in 128-row x 128-column tiles of one expert's run "
  "(a block finds its expert by scanning the offsets), gemm_nt_ggml_qk256_tiled's slab loop",
  "any M, any N, K a multiple of 256, up to 1024 experts", "bf16 a and y; q5_K, q6_K, iq2_xxs or "
  "iq1_s experts (or the f32 logical form), i32 routing",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41GgMoeTiled256), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_ggml_tiled, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_moe_gemm_ggml) },
{ "moe_gemm_ggml_qk32_tiled", "moe_gemm_ggml", "moe",
  "moe_gemm_ggml_qk256_tiled over the 32-value block types",
  "any M, any N, K a multiple of 32, up to 1024 experts",
  "bf16 a and y; q8_0 or iq4_nl experts (or the f32 logical form), i32 routing",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41GgMoeTiled32), NOTUNE,
  nullptr, nullptr, r4d_rad_moe_gemm_ggml_tiled, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_moe_gemm_ggml) },

{ "logits_gemm_ggml_qk256_gemv", "logits_gemm_ggml", "vocab",
  "logits[M, n_vocab] f32 = x . lm_head^T over llama.cpp's blocks, gemm_nt_ggml_qk256_gemv's kernel",
  "M 1..16, any n_vocab, n_embd a multiple of 256", "bf16 x, F32 LOGITS; q5_K, q6_K, iq2_xxs or iq1_s "
  "head (or the f32 logical form)",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41GgLogitsGemv256), NOTUNE,
  nullptr, nullptr, r4d_rad_logits_gemm_ggml_gemv, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_logits_gemm_ggml) },
{ "logits_gemm_ggml_qk32_gemv", "logits_gemm_ggml", "vocab",
  "logits_gemm_ggml_qk256_gemv over the 32-value block types",
  "M 1..16, any n_vocab, n_embd a multiple of 32", "bf16 x, F32 LOGITS; q8_0 or iq4_nl head",
  RAD_DOMAIN_DEVICE, 58, ROW(cDs41GgLogitsGemv32), NOTUNE,
  nullptr, nullptr, r4d_rad_logits_gemm_ggml_gemv, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_logits_gemm_ggml) },
{ "logits_gemm_ggml_qk256_tiled", "logits_gemm_ggml", "vocab",
  "logits[M, n_vocab] f32 over llama.cpp's blocks, gemm_nt_ggml_qk256_tiled's kernel",
  "any M, any n_vocab, n_embd a multiple of 256", "bf16 x, F32 LOGITS; q5_K, q6_K, iq2_xxs or iq1_s "
  "head (or the f32 logical form)",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41GgLogitsTiled256), NOTUNE,
  nullptr, nullptr, r4d_rad_logits_gemm_ggml_tiled, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_logits_gemm_ggml) },
{ "logits_gemm_ggml_qk32_tiled", "logits_gemm_ggml", "vocab",
  "logits_gemm_ggml_qk256_tiled over the 32-value block types",
  "any M, any n_vocab, n_embd a multiple of 32", "bf16 x, F32 LOGITS; q8_0 or iq4_nl head",
  RAD_DOMAIN_DEVICE, 30, ROW(cDs41GgLogitsTiled32), NOTUNE,
  nullptr, nullptr, r4d_rad_logits_gemm_ggml_tiled, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_logits_gemm_ggml) },

{ "embed_lookup_ggml_qk256", "embed_lookup_ggml", "vocab",
  "embedding gather from llama.cpp's blocks: a block per token, the row's units dequantised, times "
  "`mult`, rounded once",
  "any M and n_vocab, n_embd a multiple of 256; a negative or out-of-range id writes a zero row",
  "i32 tokens, bf16 x; q5_K, q6_K, iq2_xxs or iq1_s table (or the f32 logical form)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41GgEmbed256), NOTUNE,
  nullptr, nullptr, r4d_rad_embed_lookup_ggml, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_embed_lookup_ggml) },
{ "embed_lookup_ggml_qk32", "embed_lookup_ggml", "vocab",
  "embed_lookup_ggml_qk256 over the 32-value block types",
  "any M and n_vocab, n_embd a multiple of 32; a negative or out-of-range id writes a zero row",
  "i32 tokens, bf16 x; q8_0 or iq4_nl table (or the f32 logical form)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41GgEmbed32), NOTUNE,
  nullptr, nullptr, r4d_rad_embed_lookup_ggml, nullptr, R4D_DS41_GG_HOOKS(r4d_shape_embed_lookup_ggml) },

#undef R4D_DS41_GG_HOOKS
#endif  /* R4D_HAVE_GGML */
#endif
