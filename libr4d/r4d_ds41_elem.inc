/* r4d_ds41_elem.inc -- DeepSeek-V4.1's elem ops in r4d_rows.cpp, included three times:
 * R4D_DS41_SPECS (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (entries of
 * kSchemas) and R4D_DS41_ROWS (entries of kKernels). No include guard, on purpose.
 *
 * The specs and docs are libref's (libref/ref_registry.cpp) field for field. The constraints are
 * what each launch in r4d_ds41_elem.hip accepts; a condition between parameters (n % block,
 * off + rot <= head_dim) is refused there and by libref alike. */
#if defined(R4D_DS41_SPECS)

static const RadParamSpec pDs41TokenMap[] = { P_INT("M"), P_INT("n_vocab") };
static const RadOperandSpec oDs41TokenMap[] = {
    { "ids", RAD_OPD_IN, 0 }, { "table", RAD_OPD_WEIGHT, 0 }, { "mapped", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41TokenMap[] = { C_GE("M", 1), C_GE("n_vocab", 1) };

static const RadParamSpec pDs41StreamGate[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"),
                                                P_F64("gate_eps"), P_STR("dtype") };
static const RadOperandSpec oDs41StreamGate[] = {
    { "h", RAD_OPD_IN, 0 }, { "k", RAD_OPD_IN, 0 },       { "v", RAD_OPD_IN, 0 },
    { "w", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },
    { "mask", RAD_OPD_IN, RAD_OPD_F_ABSENT },
};
static const RadConstraint cDs41StreamGate[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 1), C_LE("hc", 16), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41RopeRows[] = { P_INT("M"),   P_INT("n_head"), P_INT("head_dim"),
                                              P_INT("rot"), P_INT("off"),    P_STR("mode"),
                                              P_INT("inverse"), P_STR("dtype") };
static const RadOperandSpec oDs41RopeRows[] = {
    { "x", RAD_OPD_INOUT, 0 }, { "cos_sin", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 },
};
static const RadConstraint cDs41RopeRows[] = {
    C_GE("M", 1),       C_GE("n_head", 1),   C_GE("head_dim", 2), C_GE("rot", 2),
    C_DIV("rot", 2),    C_GE("off", 0),      C_GE("inverse", 0),  C_LE("inverse", 1),
    C_IN("mode", "gptj neox"), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41QuantRoundtrip[] = { P_INT("M"),      P_INT("n"),
                                                    P_INT("block"),  P_STR("format"),
                                                    P_STR("scale"),  P_STR("dtype") };
static const RadOperandSpec oDs41QuantRoundtrip[] = { { "x", RAD_OPD_INOUT, 0 } };
/* One row a format, so each states the scales that format takes. */
static const RadConstraint cDs41QuantFp8[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("block", 1), C_IN("format", "fp8_e4m3"),
    C_IN("scale", "ue8m0 f32"), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41QuantFp4[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("block", 1), C_IN("format", "fp4_e2m1"),
    C_IN("scale", "ue8m0 e4m3"), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41MhcMix[] = { P_INT("M"),          P_INT("n"),   P_INT("hc"),
                                            P_INT("iterations"), P_F64("eps"), P_F64("hc_eps"),
                                            P_STR("dtype") };
static const RadOperandSpec oDs41MhcMix[] = {
    { "h", RAD_OPD_IN, 0 },      { "fn", RAD_OPD_WEIGHT, 0 }, { "scale", RAD_OPD_WEIGHT, 0 },
    { "base", RAD_OPD_WEIGHT, 0 }, { "pre", RAD_OPD_OUT, 0 }, { "post", RAD_OPD_OUT, 0 },
    { "comb", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41MhcMix[] = {
    C_GE("M", 1),          C_GE("n", 1),           C_GE("hc", 2), C_LE("hc", 8),
    C_GE("iterations", 1), C_LE("iterations", 256), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41MhcCollapse[] = { P_INT("M"), P_INT("n"), P_INT("hc"),
                                                 P_STR("dtype") };
static const RadOperandSpec oDs41MhcCollapse[] = {
    { "h", RAD_OPD_IN, 0 }, { "pre", RAD_OPD_IN, 0 }, { "x", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41MhcCollapse[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41SoftmaxPool[] = { P_INT("M"), P_INT("n"), P_INT("ratio"),
                                                 P_STR("dtype") };
static const RadOperandSpec oDs41SoftmaxPool[] = {
    { "stage", RAD_OPD_IN, 0 }, { "out", RAD_OPD_OUT, 0 }, { "page", RAD_OPD_IN, 1 },
};
static const RadConstraint cDs41SoftmaxPool[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("ratio", 1), C_LE("ratio", 64), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41EntryStore[] = { P_INT("M"), P_INT("n"), P_INT("block_size"),
                                                P_STR("dtype"), P_STRO("format") };
static const RadOperandSpec oDs41EntryStore[] = {
    { "x", RAD_OPD_IN, 0 }, { "page", RAD_OPD_IN, 0 }, { "cache", RAD_OPD_INOUT, 0 },
};
static const RadConstraint cDs41EntryStore[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("block_size", 1), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41RouterScored[] = { P_INT("M"),           P_INT("n_expert"),
                                                  P_INT("top_k"),       P_INT("norm"),
                                                  P_F64("route_scale"), P_STR("score"),
                                                  P_STR("dtype") };
static const RadOperandSpec oDs41RouterScored[] = {
    { "logits", RAD_OPD_IN, 0 }, { "bias", RAD_OPD_WEIGHT, 0 },
    { "ids", RAD_OPD_OUT, 0 },   { "weights", RAD_OPD_OUT, 0 },
};
/* top_k <= 32: round j parks its pick on lane j of the row's wave. */
static const RadConstraint cDs41RouterScored[] = {
    C_GE("M", 1),     C_GE("n_expert", 1), C_LE("n_expert", 512),
    C_GE("top_k", 1), C_LE("top_k", 32),   C_IN("score", "sigmoid sqrtsoftplus softmax"),
    C_IN("dtype", "f32"),
};

/* router_scored_topk_vl: HF's gate swaps its correction bias for bias_vl on image-span rows (the
 * `mask`). Same limits; selection moves, weights don't. */
static const RadParamSpec pDs41RouterScoredVL[] = { P_INT("M"),           P_INT("n_expert"),
                                                    P_INT("top_k"),       P_INT("norm"),
                                                    P_F64("route_scale"), P_STR("score"),
                                                    P_STR("dtype") };
static const RadOperandSpec oDs41RouterScoredVL[] = {
    { "logits", RAD_OPD_IN, 0 },  { "bias", RAD_OPD_WEIGHT, 0 },
    { "bias_vl", RAD_OPD_WEIGHT, 0 }, { "mask", RAD_OPD_IN, 0 },
    { "ids", RAD_OPD_OUT, 0 },    { "weights", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41RouterScoredVL[] = {
    C_GE("M", 1),     C_GE("n_expert", 1), C_LE("n_expert", 512),
    C_GE("top_k", 1), C_LE("top_k", 32),   C_IN("score", "sigmoid sqrtsoftplus softmax"),
    C_IN("dtype", "f32"),
};

/* ds41_vit_rope2d / ds41_pack3x3: DeepSeek-V4.1's tower geometry -- both rotary axes on one ladder,
 * and the aligner's 3x3 zero-padded pack. */
static const RadParamSpec pDs41VitRope[] = { P_INT("M"),        P_INT("head_dim"),
                                             P_INT("n_head"),   P_F64("theta"),
                                             P_STR("dtype") };
static const RadOperandSpec oDs41VitRope[] = {
    { "qkv", RAD_OPD_INOUT, 0 }, { "coord", RAD_OPD_IN, 0 },
};
static const RadConstraint cDs41VitRope[] = {
    C_GE("M", 1), C_GE("head_dim", 4), C_GE("n_head", 1), C_IN("dtype", "bf16 f32"),
};
static const RadParamSpec pDs41Pack[] = { P_INT("M"), P_INT("n"), P_INT("merge"), P_STR("dtype") };
static const RadOperandSpec oDs41Pack[] = {
    { "x", RAD_OPD_IN, 0 }, { "coord", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41Pack[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("merge", 2), C_IN("dtype", "bf16 f32"),
};
/* ds41_span_rows: the learned rows a typed image span takes -- row 0 start, 2 newline, 3 end;
 * row 1 is the encoder's own rows and is not written. A copy, no arithmetic. */
static const RadParamSpec pDs41SpanRows[] = { P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oDs41SpanRows[] = {
    { "start", RAD_OPD_WEIGHT, 0 }, { "nl", RAD_OPD_WEIGHT, 0 }, { "end", RAD_OPD_WEIGHT, 0 },
    { "span_rows", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41SpanRows[] = {
    C_GE("n", 1), C_IN("dtype", "bf16"),
};

static const RadParamSpec pDs41EmbedPlanes[] = { P_INT("M"),     P_INT("n_embd"),
                                                 P_INT("n_vocab"), P_INT("group"),
                                                 P_STR("dtype"), P_INTO("vocab_offset"),
                                                 P_F64O("mult") };
static const RadOperandSpec oDs41EmbedPlanes[] = {
    { "tokens", RAD_OPD_IN, 0 },     { "codes", RAD_OPD_WEIGHT, 0 },
    { "scale", RAD_OPD_WEIGHT, 0 },  { "table", RAD_OPD_WEIGHT, 1 },
    { "x", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41EmbedPlanes[] = {
    C_GE("M", 1), C_GE("n_embd", 1), C_GE("n_vocab", 1), C_GE("group", 1), C_IN("dtype", "bf16"),
};

/* The fused chains. n <= 16384: the norm's row passes between the steps in LDS. One row a
 * quant_roundtrip format, as there. */
static const RadParamSpec pDs41NormRopeRt[] = {
    P_INT("M"),     P_INT("n"),        P_F64("eps"),   P_INT("n_head"),  P_INT("head_dim"),
    P_INT("rot"),   P_INT("off"),      P_STR("mode"),  P_INT("inverse"), P_INT("block"),
    P_STR("format"), P_STR("scale"),   P_STR("dtype"), P_F64O("wadd"),   P_F64O("mult"),
};
static const RadOperandSpec oDs41NormRopeRt[] = {
    { "x", RAD_OPD_IN, 0 },       { "w", RAD_OPD_WEIGHT, 0 },    { "y", RAD_OPD_OUT, 0 },
    { "cos_sin", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 },
};
#define C_DS41_ROPE                                                                                \
    C_GE("M", 1), C_GE("n_head", 1), C_GE("head_dim", 2), C_GE("rot", 2), C_DIV("rot", 2),         \
        C_GE("off", 0), C_GE("inverse", 0), C_LE("inverse", 1), C_IN("mode", "gptj neox"),          \
        C_GE("block", 1), C_IN("dtype", "bf16")
static const RadConstraint cDs41NormRopeRtFp8[] = {
    C_DS41_ROPE, C_GE("n", 1), C_LE("n", 16384), C_IN("format", "fp8_e4m3"),
    C_IN("scale", "ue8m0 f32"),
};
static const RadConstraint cDs41NormRopeRtFp4[] = {
    C_DS41_ROPE, C_GE("n", 1), C_LE("n", 16384), C_IN("format", "fp4_e2m1"),
    C_IN("scale", "ue8m0 e4m3"),
};

static const RadParamSpec pDs41RopeRt[] = {
    P_INT("M"),   P_INT("n_head"), P_INT("head_dim"), P_INT("rot"),    P_INT("off"),
    P_STR("mode"), P_INT("inverse"), P_INT("block"),  P_STR("format"), P_STR("scale"),
    P_STR("dtype"),
};
static const RadOperandSpec oDs41RopeRt[] = {
    { "x", RAD_OPD_INOUT, 0 }, { "cos_sin", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 },
};
static const RadConstraint cDs41RopeRtFp8[] = {
    C_DS41_ROPE, C_IN("format", "fp8_e4m3"), C_IN("scale", "ue8m0 f32"),
};
static const RadConstraint cDs41RopeRtFp4[] = {
    C_DS41_ROPE, C_IN("format", "fp4_e2m1"), C_IN("scale", "ue8m0 e4m3"),
};
#undef C_DS41_ROPE

static const RadParamSpec pDs41CollapseNorm[] = { P_INT("M"),     P_INT("n"),     P_INT("hc"),
                                                  P_F64("eps"),   P_STR("dtype"), P_F64O("wadd"),
                                                  P_F64O("mult") };
static const RadOperandSpec oDs41CollapseNorm[] = {
    { "h", RAD_OPD_IN, 0 },     { "pre", RAD_OPD_IN, 0 }, { "x", RAD_OPD_OUT, 0 },
    { "w", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41CollapseNorm[] = {
    C_GE("M", 1), C_GE("n", 1), C_GE("hc", 2), C_LE("hc", 8), C_IN("dtype", "bf16"),
};
static const RadOperandSpec oDs41PostCollapseNorm[] = {
    { "x", RAD_OPD_IN, 0 },   { "h", RAD_OPD_IN, 0 },  { "post", RAD_OPD_IN, 0 },
    { "comb", RAD_OPD_IN, 0 }, { "out", RAD_OPD_OUT, 0 }, { "pre", RAD_OPD_IN, 0 },
    { "xc", RAD_OPD_OUT, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
/* mhc_post_bf16's range (cMhcPlain). */
static const RadConstraint cDs41PostCollapseNorm[] = {
    C_GE("M", 1),  C_LE("M", 65535), C_GE("n", 1),          C_LE("n", 16384),
    C_GE("hc", 2), C_LE("hc", 8),    C_IN("dtype", "bf16"),
};

#elif defined(R4D_DS41_SCHEMAS)

    SCHEMA("token_map", pDs41TokenMap, oDs41TokenMap,
           "mapped[m] = table[ids[m]], -1 for an id outside [0, n_vocab): ids through an id table."),
    SCHEMA("stream_gate_add", pDs41StreamGate, oDs41StreamGate,
           "Per row and stream s: dot = sum h w k * rsqrt(mean h^2 + eps) rsqrt(mean k^2 + eps) / sqrt(n), "
           "gate = sigmoid(copysign(sqrt(max(|dot|, gate_eps)), dot)), out = h + gate v; f32, rounded "
           "once. v is one [n] row a token shared by the hc streams; w is f32 [hc*n]."),
    SCHEMA("ds41_rope_rows", pDs41RopeRows, oDs41RopeRows,
           "rope from a rope_table plane over columns [off, off+rot) of each head_dim-wide head of x, in "
           "place: gptj pairs adjacent columns, neox i with i+rot/2; (a,b)(c+is) in f32, rounded once; "
           "inverse 1 conjugates the rotation. A negative or out-of-table position leaves its row."),
    SCHEMA("quant_roundtrip", pDs41QuantRoundtrip, oDs41QuantRoundtrip,
           "Block-scaled quantise then dequantise in place: per `block` values of a row, amax, a scale "
           "(fp8_e4m3: amax floored at 1e-4, ue8m0 = power of two at or above amax*f32(1/448), f32 = "
           "amax*f32(1/448); fp4_e2m1: ue8m0 from amax floored at 6*2^-126 times f32(1/6), e4m3 = "
           "fp8(max(amax, 6*2^-9)/6)), codes of clamp(x/scale) (fp8 RNE saturating, e2m1 ties to the "
           "even code), times scale in f32, rounded once to dtype."),
    SCHEMA("mhc_mix", pDs41MhcMix, oDs41MhcMix,
           "mhc_pre without the collapse: the same RMS-scaled projection, gates and Sinkhorn mixing, with "
           "the pre gates written to `pre` (f32 [M, hc]) instead of collapsing h. For models that collapse "
           "a sublayer's input with gates an earlier sublayer computed (DeepSeek-V4.1); mhc_collapse "
           "applies them."),
    SCHEMA("mhc_collapse", pDs41MhcCollapse, oDs41MhcCollapse,
           "x[i] = sum over hc streams of pre[stream] * h[stream, i], accumulated in f32 and rounded once "
           "to dtype. pre is f32 [M, hc]; with every pre 1/hc it is mhc_contract."),
    SCHEMA("softmax_pool", pDs41SoftmaxPool, oDs41SoftmaxPool,
           "Channel-wise softmax pooling of `ratio` rows: stage [M*ratio, 2n] f32 holds each entry's "
           "values then scores; out[w,c] = sum_i softmax_i(score[i,c]) value[i,c] in f32, rounded once. "
           "An entry whose optional page is negative is not written."),
    SCHEMA("entry_store", pDs41EntryStore, oDs41EntryStore,
           "x[w] (n values) into the whole of block page[w] of a paged cache [blocks, heads, block_size, "
           "slot], heads*block_size*slot == n, rounded once to the cache dtype; a negative page writes "
           "nothing. Declared right after its group, its block_size sets the group's. format "
           "e4m3_e4m3_16 / e4m3_ue8m0_32: each entry packed as block-scaled E4M3 (rad_kv_fp8.h) into an "
           "E4M3 cache, heads*block_size*slot its bytes; exact for quant_roundtrip's output."),
    SCHEMA("router_scored_topk", pDs41RouterScored, oDs41RouterScored,
           "router_sigmoid_topk with the score function a required name: sigmoid, sqrtsoftplus "
           "(sqrt(softplus(logit)), softplus linear above 20) or softmax over all n_expert logits."),
    SCHEMA("router_scored_topk_vl", pDs41RouterScoredVL, oDs41RouterScoredVL,
           "router_scored_topk with two correction biases: rows with mask 1 (an image span) select "
           "by score + bias_vl, other rows by score + bias; weights are the unbiased scores."),
    SCHEMA("ds41_vit_rope2d", pDs41VitRope, oDs41VitRope,
           "DeepSeek-V4.1 tower rotary: rotate-half on q and k of qkv [M, 3*H*D], both axes on ONE "
           "theta^(-2j/(D/2)) ladder -- half the rotary width tracks coord[0], half coord[1]; v is "
           "unchanged; f32 products rounded once to the stored dtype."),
    SCHEMA("ds41_pack3x3", pDs41Pack, oDs41Pack,
           "Pack each complete raster patch grid into 3x3 zero-padded blocks: the grid is padded to "
           "a multiple of merge on both axes with zeros, each cell of the padded grid becomes one "
           "row of merge*merge patch rows in (dy, dx) order; a cell under the padded edge is zero."),
    SCHEMA("ds41_span_rows", pDs41SpanRows, oDs41SpanRows,
           "Copy three learned rows into a span table: span_rows[0] = start, [2] = nl, [3] = end; "
           "row 1 (the encoder's own rows) is not written."),
    SCHEMA("embed_lookup_planes", pDs41EmbedPlanes, oDs41EmbedPlanes,
           "A gather from a table stored as single-level affine planes, read in place: x[m, i] = v * "
           "scale[id, i / group] * mult, v = table[code] when a table is given, else the code. Codes are i8, "
           "u8, or 4-bit elements of a bit stream along the row, lowest bits first; a table is indexed by an "
           "unsigned code and holds every value. Padding, bounds and vocab_offset as embed_lookup."),
    SCHEMA("rmsnorm_rope_roundtrip", pDs41NormRopeRt, oDs41NormRopeRt,
           "rmsnorm (x, w -> y), then rope_rows over y, then quant_roundtrip over y: y as the three ops "
           "leave it, each step rounded to dtype where that op stores. n = n_head * head_dim. A key "
           "normalised, rotated and quantise-dequantised before its cache (DeepSeek-V4.1)."),
    SCHEMA("rope_rows_roundtrip", pDs41RopeRt, oDs41RopeRt,
           "rope_rows then quant_roundtrip over the n_head * head_dim columns of x, in place: x as the "
           "two ops leave it, each step rounded to dtype where that op stores."),
    SCHEMA("mhc_collapse_rmsnorm", pDs41CollapseNorm, oDs41CollapseNorm,
           "mhc_collapse (h, pre -> x), then rmsnorm (x, w -> y): x and y as the two ops leave them, x "
           "rounded to dtype before the norm reads it."),
    SCHEMA("mhc_post_collapse_rmsnorm", pDs41CollapseNorm, oDs41PostCollapseNorm,
           "mhc_post (x, h, post, comb -> out), then mhc_collapse (out, pre -> xc), then rmsnorm (xc, "
           "w -> y): out, xc and y as the three ops leave them, each rounded to dtype before the next "
           "step reads it. out must not alias h."),

#elif defined(R4D_DS41_ROWS)

/* ---------------------------------- DeepSeek-V4.1's elem ops (r4d_ds41_elem.hip) -------------
 * R4D_PRIO_ONLY throughout: nothing else in this plugin implements them. */
{ "token_map_i32", "token_map", "embed",
  "ids through an id table, -1 outside [0, n_vocab)", "any M; one thread an id",
  "i32 ids, table and mapped", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41TokenMap), NOTUNE,
  nullptr, nullptr, r4d_rad_token_map, nullptr, nullptr, nullptr, r4d_shape_token_map },
{ "stream_gate_add_bf16", "stream_gate_add", "norm",
  "Engram's normalised-dot sigmoid gate, out = h + gate v per stream",
  "any M and n, hc 1..16; one workgroup a (row, stream)",
  "bf16 h, k, v and out (rows may be pitched column slices), f32 w and arithmetic",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41StreamGate), NOTUNE,
  nullptr, nullptr, r4d_rad_stream_gate_add, nullptr, nullptr, nullptr,
  r4d_shape_stream_gate_add },
{ "ds41_rope_rows_bf16", "ds41_rope_rows", "attn",
  "rotary from a rope_table plane over columns [off, off+rot) of each head, in place",
  "any M, heads and head_dim, gptj or neox, inverse 0/1; one thread a pair",
  "bf16 x, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41RopeRows),
  NOTUNE, nullptr, nullptr, r4d_rad_ds41_rope_rows, nullptr, nullptr, nullptr, r4d_shape_rope_rows },
{ "quant_roundtrip_fp8_bf16", "quant_roundtrip", "quant",
  "block-scaled E4M3 quantise-dequantise in place, ue8m0 or f32 scale",
  "any M, n a multiple of block; a lane group a block up to 32 (powers of two), else a thread",
  "bf16 x, f32 arithmetic", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41QuantFp8), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_quant_roundtrip },
{ "quant_roundtrip_fp4_bf16", "quant_roundtrip", "quant",
  "block-scaled e2m1 quantise-dequantise in place, ue8m0 or e4m3 scale",
  "any M, n a multiple of block; a lane group a block up to 32 (powers of two), else a thread",
  "bf16 x, f32 arithmetic", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41QuantFp4), NOTUNE,
  nullptr, nullptr, r4d_rad_quant_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_quant_roundtrip },
{ "mhc_mix_bf16", "mhc_mix", "norm",
  "mHC RMS-scaled projection, gates and Sinkhorn mixing, pre gates out",
  "any M and n, hc 2..8, iterations 1..256; 1024-column chunk partials in scratch, then a wave a "
  "row (one wave a row walking the chunks without scratch, the same bits)",
  "bf16 h, f32 fn, scale, base, pre, post and comb", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41MhcMix), NOTUNE, nullptr, nullptr, r4d_rad_mhc_mix, r4d_rad_mhc_mix_scratch, nullptr,
  nullptr, r4d_shape_mhc_mix },
{ "mhc_collapse_bf16", "mhc_collapse", "norm", "x = sum over streams of pre[j] h[j]",
  "any M and n, hc 2..8; eight columns a thread on 16-byte rows", "bf16 h and x, f32 pre",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41MhcCollapse), NOTUNE,
  nullptr, nullptr, r4d_rad_mhc_collapse, nullptr, nullptr, nullptr, r4d_shape_mhc_collapse },
{ "softmax_pool_bf16", "softmax_pool", "attn",
  "channel-wise softmax pooling of ratio rows into one entry",
  "any M and n, ratio 1..64; one thread a (entry, channel)", "f32 stage, bf16 out, i32 page",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41SoftmaxPool), NOTUNE,
  nullptr, nullptr, r4d_rad_softmax_pool, nullptr, nullptr, nullptr, r4d_shape_softmax_pool },
{ "entry_store_bf16", "entry_store", "attn", "one entry a block into a paged cache",
  "any M, n and block_size; one thread an element (format e4m3_*: a block of an entry, packed by "
  "rad_kv_fp8.h, libref's bytes)", "bf16 x, bf16 or E4M3 cache, i32 page",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41EntryStore), NOTUNE,
  nullptr, nullptr, r4d_rad_entry_store, nullptr, nullptr, nullptr, r4d_shape_entry_store },
{ "router_scored_topk_f32", "router_scored_topk", "moe",
  "correction-bias routing over sigmoid, sqrt(softplus) or softmax scores",
  "single group, experts <= 512, top_k <= 32; one wave a row",
  "f32 logits, bias and weights; i32 ids", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41RouterScored), NOTUNE, nullptr, nullptr, r4d_rad_router_scored_topk, nullptr, nullptr,
  nullptr, r4d_shape_router_scored_topk },
{ "router_scored_topk_vl_f32", "router_scored_topk_vl", "moe",
  "correction-bias routing with a second bias for image-span rows",
  "single group, experts <= 512, top_k <= 32; one wave a row",
  "f32 logits, bias and bias_vl and weights; i32 mask and ids", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41RouterScoredVL), NOTUNE, nullptr, nullptr, r4d_rad_router_scored_topk_vl, nullptr,
  nullptr, nullptr, r4d_shape_router_scored_topk_vl },
{ "ds41_vit_rope2d_bf16", "ds41_vit_rope2d", "attn",
  "DeepSeek-V4.1 tower shared-ladder 2-D rotary, q and k of a fused qkv",
  "one workgroup a row; head_dim a multiple of 4",
  "bf16 or f32 qkv in place; i32 coordinates", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41VitRope), NOTUNE, nullptr, nullptr, r4d_rad_ds41_vit_rope2d, nullptr, nullptr, nullptr,
  r4d_shape_ds41_vit_rope2d },
{ "ds41_pack3x3_bf16", "ds41_pack3x3", "vision",
  "3x3 zero-padded pack of complete raster patch grids",
  "one workgroup an output row; merge 2 or 3",
  "bf16 or f32 x and y; i32 coordinates", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41Pack), NOTUNE, nullptr, nullptr, r4d_rad_ds41_pack3x3, nullptr, nullptr, nullptr,
  r4d_shape_ds41_pack3x3 },
{ "ds41_span_rows_bf16", "ds41_span_rows", "vision",
  "learned span rows into the typed-row table",
  "n <= 16384; one thread an element",
  "bf16 weights and table", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41SpanRows), NOTUNE, nullptr, nullptr, r4d_rad_ds41_span_rows, nullptr, nullptr, nullptr,
  r4d_shape_ds41_span_rows },
{ "embed_lookup_planes_bf16", "embed_lookup_planes", "vocab",
  "embedding gather from a table's affine planes, read in place (no layout hook)",
  "any M, n_embd a multiple of group; one thread an element of a gathered row",
  "i8 u8 i4 u4 codes; f32 f16 bf16 fp8_e4m3 e8m0 scale; optional i8 u8 f32 f16 bf16 fp8_e4m3 "
  "table; i32 tokens; bf16 x",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41EmbedPlanes), NOTUNE,
  nullptr, nullptr, r4d_rad_embed_lookup_planes, nullptr, nullptr, nullptr,
  r4d_shape_embed_lookup_planes },
/* The fused chains: one launch for two or three of the rows above (and rmsnorm_bf16), the same
 * bytes (`replaces`, r4d_fuse.cpp). */
{ "rmsnorm_rope_roundtrip_fp8_bf16", "rmsnorm_rope_roundtrip", "norm",
  "rmsnorm, rope_rows and an E4M3 quant_roundtrip of a row in one pass",
  "any M, n <= 16384; rmsnorm_bf16's workgroup a row (r4d_act_threads), the row in LDS between steps",
  "bf16 x and y, f32 or bf16 w, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41NormRopeRtFp8), NOTUNE, nullptr, nullptr, r4d_rad_rmsnorm_rope_roundtrip, nullptr,
  nullptr, nullptr, r4d_shape_rmsnorm_rope_roundtrip, nullptr, r4d_fuse_rmsnorm_rope_roundtrip },
{ "rmsnorm_rope_roundtrip_fp4_bf16", "rmsnorm_rope_roundtrip", "norm",
  "rmsnorm, rope_rows and an e2m1 quant_roundtrip of a row in one pass",
  "any M, n <= 16384; rmsnorm_bf16's workgroup a row (r4d_act_threads), the row in LDS between steps",
  "bf16 x and y, f32 or bf16 w, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41NormRopeRtFp4), NOTUNE, nullptr, nullptr, r4d_rad_rmsnorm_rope_roundtrip, nullptr,
  nullptr, nullptr, r4d_shape_rmsnorm_rope_roundtrip, nullptr, r4d_fuse_rmsnorm_rope_roundtrip },
{ "rope_rows_roundtrip_fp8_bf16", "rope_rows_roundtrip", "attn",
  "rope_rows then an E4M3 quant_roundtrip, in place, in one pass",
  "any M; a thread a column, the gptj partner by a lane shuffle (head_dim and off even, block a "
  "power of two <= 32); else rope_rows and quant_roundtrip as two launches",
  "bf16 x, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41RopeRtFp8),
  NOTUNE, nullptr, nullptr, r4d_rad_rope_rows_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_rope_rows_roundtrip, nullptr, r4d_fuse_rope_rows_roundtrip },
{ "rope_rows_roundtrip_fp4_bf16", "rope_rows_roundtrip", "attn",
  "rope_rows then an e2m1 quant_roundtrip, in place, in one pass",
  "any M; a thread a column, the gptj partner by a lane shuffle (head_dim and off even, block a "
  "power of two <= 32); else rope_rows and quant_roundtrip as two launches",
  "bf16 x, f32 cos_sin, i32 positions", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41RopeRtFp4),
  NOTUNE, nullptr, nullptr, r4d_rad_rope_rows_roundtrip, nullptr, nullptr, nullptr,
  r4d_shape_rope_rows_roundtrip, nullptr, r4d_fuse_rope_rows_roundtrip },
{ "mhc_collapse_rmsnorm_bf16", "mhc_collapse_rmsnorm", "norm",
  "x = sum over streams of pre[j] h[j], then y = rmsnorm(x), in one pass",
  "any M and n, hc 2..8; rmsnorm_bf16's workgroup a row (r4d_act_threads)",
  "bf16 h, x and y, f32 pre, f32 or bf16 w", RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41CollapseNorm), NOTUNE, nullptr, nullptr, r4d_rad_mhc_collapse_rmsnorm, nullptr,
  nullptr, nullptr, r4d_shape_mhc_collapse_rmsnorm, nullptr, r4d_fuse_mhc_collapse_rmsnorm },
{ "mhc_post_collapse_rmsnorm_bf16", "mhc_post_collapse_rmsnorm", "norm",
  "out = post[k] x + sum_j comb[j,k] h[j], xc = sum_k pre[k] out[k], y = rmsnorm(xc), in one pass",
  "M 1..65535, n 1..16384, hc 2..8; rmsnorm_bf16's workgroup a row (r4d_act_threads)",
  "bf16 x, h, out, xc and y, f32 post, comb and pre, f32 or bf16 w", RAD_DOMAIN_DEVICE,
  R4D_PRIO_ONLY, ROW(cDs41PostCollapseNorm), NOTUNE, nullptr, nullptr,
  r4d_rad_mhc_post_collapse_rmsnorm, nullptr, nullptr, nullptr,
  r4d_shape_mhc_post_collapse_rmsnorm, nullptr, r4d_fuse_mhc_post_collapse_rmsnorm },

#endif
