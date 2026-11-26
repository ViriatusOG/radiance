/* r4d_ds41_sparse.inc -- DeepSeek-V4.1's sparse ops in r4d_rows.cpp, included three times:
 * R4D_DS41_SPECS (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (entries of
 * kSchemas) and R4D_DS41_ROWS (entries of kKernels). No include guard, on purpose. */
#if defined(R4D_DS41_SPECS)

/* libref's schemas (libref/ref_registry.cpp), field for field. */
static const RadParamSpec pDs41AttnSparseSink[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("window"), P_INT("ratio"), P_INT("topk"), P_F64("scale"), P_STR("dtype"),
    P_STRO("latent_format") };
static const RadOperandSpec oDs41AttnSparseSink[] = {
    { "q", RAD_OPD_IN, 0 },         { "window_cache", RAD_OPD_IN, 0 }, { "window_bt", RAD_OPD_IN, 0 },
    { "window_used", RAD_OPD_IN, 0 }, { "comp_cache", RAD_OPD_IN, 1 }, { "comp_bt", RAD_OPD_IN, 1 },
    { "sel", RAD_OPD_IN, 1 },       { "positions", RAD_OPD_IN, 0 },  { "cu_seqlens", RAD_OPD_IN, 0 },
    { "sinks", RAD_OPD_WEIGHT, 0 }, { "out", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pDs41IndexScore[] = { P_INT("M"), P_INT("heads"), P_INT("dim"),
    P_INT("ratio"), P_INT("width"), P_F64("wscale"), P_STR("dtype"), P_INTO("row_part"),
    P_STRO("key_format") };
static const RadOperandSpec oDs41IndexScore[] = {
    { "q", RAD_OPD_IN, 0 },  { "w", RAD_OPD_IN, 0 },         { "index_cache", RAD_OPD_IN, 0 },
    { "bt", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 }, { "cu_seqlens", RAD_OPD_IN, 0 },
    { "score", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pDs41IndexCand[] = { P_INT("M"), P_INT("width"), P_INT("block"),
    P_INT("topk_blocks"), P_INT("ratio") };
static const RadOperandSpec oDs41IndexCand[] = {
    { "score", RAD_OPD_IN, 0 }, { "positions", RAD_OPD_IN, 0 }, { "keep", RAD_OPD_OUT, 0 },
};
static const RadParamSpec pDs41IndexTopk[] = { P_INT("M"), P_INT("width"), P_INT("topk"), P_INT("ratio"),
    P_INTO("block") };
static const RadOperandSpec oDs41IndexTopk[] = {
    { "score", RAD_OPD_IN, 0 }, { "keep", RAD_OPD_IN, 1 }, { "positions", RAD_OPD_IN, 0 },
    { "sel", RAD_OPD_OUT, 0 },
};

/* r4d_ds41_sparse.hip's launch checks. The latent is 512 wide (16-byte loads of 512/ratio-wide
 * compressed slots: ratio <= 64), at most 1024 window and 2048 selected keys a row (the per-row
 * key list and scores live in LDS). M is any: the grid loops over rows. */
static const RadConstraint cDs41AttnSparseSink[] = {
    C_GE("M", 1),       C_GE("heads", 1), C_EQ("dim", 512), C_GE("window", 1), C_LE("window", 1024),
    C_GE("ratio", 0),   C_LE("ratio", 64), C_GE("topk", 0),  C_LE("topk", 2048),
    C_IN("dtype", "bf16"),
};
/* 128-wide index keys held in registers, a slot dim/ratio wide read 16 bytes at a time. */
static const RadConstraint cDs41IndexScore[] = {
    C_GE("M", 1),     C_GE("heads", 1), C_EQ("dim", 128),      C_GE("ratio", 1), C_LE("ratio", 16),
    C_GE("width", 1), C_LE("width", 1073741823), C_IN("dtype", "bf16"),
};
static const RadConstraint cDs41IndexCand[] = {
    C_GE("M", 1),          C_GE("width", 1), C_LE("width", 1073741823), C_GE("block", 1),
    C_GE("topk_blocks", 1), C_GE("ratio", 1),
};
static const RadConstraint cDs41IndexTopk[] = {
    C_GE("M", 1), C_GE("width", 1), C_LE("width", 1073741823), C_GE("topk", 1),
    C_LE("topk", 1073741823), C_GE("ratio", 1),
};

#elif defined(R4D_DS41_SCHEMAS)

SCHEMA("attn_sparse_sink", pDs41AttnSparseSink, oDs41AttnSparseSink,
       "Latent attention (key = value, dim wide) with a per-head sink over the window cache's keys "
       "at [pos-window+1, pos] (the window table rebased at first = pos(last row of s) + 1 - "
       "window_used[s]: block wbt[s][(j-first)/bs], cache [blocks,1,bs,dim]) and, with ratio > 0, "
       "the selected compressed entries c (tokens [c ratio, (c+1) ratio), block cbt[s][c ratio/cbs], "
       "cache [blocks,1,cbs,dim/ratio]). f32 scores q.k*scale, max floored at -1e30, "
       "out = sum p k / (sum p + exp(sink - max)), rounded once; a row with no key is zero. "
       "latent_format e4m3_e4m3_16 / e4m3_ue8m0_32: the compressed cache holds block-scaled E4M3 "
       "entries (rad_kv_fp8.h; E4M3 [blocks,1,cbs,bytes/ratio]), a value code x scale in f32."),
SCHEMA("index_score", pDs41IndexScore, oDs41IndexScore,
       "Per row and compressed position j: sum_h relu(q_h . k_j) w_h with k_j entry j of the paged "
       "index cache and w_h = round(w * wscale); the dot, relu(dot) w_h and the f32 head sum each "
       "rounded once to dtype. "
       "j >= (pos+1)/ratio or an unmapped block scores -inf. score is f32 [M, width]. With row_part "
       "p, score row i is step row p M + i (q, w, positions the step's; past the step -inf). "
       "key_format e4m3_e4m3_16 / e4m3_ue8m0_32: the index cache holds block-scaled E4M3 entries "
       "(rad_kv_fp8.h), a value code x scale in f32."),
SCHEMA("index_candidates", pDs41IndexCand, oDs41IndexCand,
       "Blocks of `block` positions scored by their maximum, the newest reachable block pinned to "
       "+inf, the topk_blocks highest (ties to the lower block) kept unless -inf; keep is a bit a "
       "block, i32 [M, ceil(ceil(width/block)/32)], bit b%32 of word b/32 set when block b is kept."),
SCHEMA("index_topk", pDs41IndexTopk, oDs41IndexTopk,
       "The topk highest-scoring passed positions (j < (pos+1)/ratio), inside keep when given "
       "(index_candidates' bits over blocks of `block`), never -inf, ties to the lower position, in "
       "ascending order, then -1."),

#elif defined(R4D_DS41_ROWS)

{ "ds41_attn_sparse_sink_d512_bf16", "attn_sparse_sink", "attn",
  "latent attention (key = value) over the window keys and the selected compressed entries, "
  "per-head sinks; libref's key order and roundings (glibc expf algorithm), so libref's bits, "
  "up to 48 rows; taller passes on WMMA (online softmax, P as bf16 hi + lo), not bit-exact",
  "dim 512, any heads, window <= 1024, topk <= 2048, ratio 0..64; one workgroup a (d half, head "
  "group of 2/4/8, row), the group picked by row count; > 48 rows a (16-head tile, row), four "
  "waves of 128 latent columns, 16 keys a step",
  "bf16 q, caches and out, or an E4M3 compressed cache (latent_format e4m3_e4m3_16: widened as "
  "read to the bf16 cache's words, its bits in every form; the decode band on the tiled PV); f32 "
  "sinks; i32 tables, positions, selections",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41AttnSparseSink), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_attn_sparse_sink, nullptr, nullptr, nullptr,
  r4d_shape_attn_sparse_sink },
{ "ds41_index_score_d128_bf16", "index_score", "attn",
  "indexer score sum_h relu(q_h.k_j) w_h over a paged index cache, libref's roundings and order "
  "(> 48 rows: the dot on WMMA, its summation order the only change)",
  "dim 128, any heads, ratio 1..16, row_part; a thread an entry (1, 2, 4 or 8 waves an entry "
  "group by grid size), -inf past (pos+1)/ratio without reading keys; > 48 rows and <= 32 heads a "
  "wave a row over 128 entries",
  "bf16 q, w and cache, or an E4M3 cache (key_format e4m3_ue8m0_32: widened as read, the bf16 "
  "cache's bits); i32 tables and positions; f32 score",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41IndexScore), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_index_score, nullptr, nullptr, nullptr,
  r4d_shape_index_score },
{ "ds41_index_candidates", "index_candidates", "select",
  "two-level top-k level one: block maxima, newest block pinned, exact top topk_blocks by radix "
  "select, ties to the lower block",
  "any width and block; one 1024-thread block a row",
  "f32 score, i32 positions and keep (a bit a block, a wave's ballot a word)",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41IndexCand), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_index_candidates, nullptr, nullptr, nullptr,
  r4d_shape_index_candidates },
{ "ds41_index_topk", "index_topk", "select",
  "exact top-k passed positions by radix select, ties to the lower position, ascending, -1 fill",
  "any width and topk; one 1024-thread block a row, or at <= 16 rows over >= 4 chunks of 8192 "
  "positions (topk <= 2048) a chunk-parallel pass and a merge (scratch)",
  "f32 score, i32 keep (a bit a block), positions and sel",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY, ROW(cDs41IndexTopk), NOTUNE,
  nullptr, nullptr, r4d_rad_ds41_index_topk, r4d_rad_ds41_index_topk_scratch, nullptr, nullptr,
  r4d_shape_index_topk },

#endif
