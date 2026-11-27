/* r4d_ds41_walk.inc -- the DSpark walk's op, dspark_walk_step, in r4d_rows.cpp, included three times: R4D_DS41_SPECS
 * (parameter/operand specs and constraints, file scope), R4D_DS41_SCHEMAS (its kSchemas entry) and R4D_DS41_ROWS
 * (its kKernels entry). No include guard, on purpose.
 *
 * One step of DeepSeek-V4.1's DSpark Markov walk over the whole vocabulary on one rank's vocabulary shard
 * (r4d_ds41_walk.hip says the arithmetic and the key). Declared by deepseek41 only with the `dspark-walk`
 * setting on; libr4d is its only implementation (no libref oracle: tests/r4d_emu/test_ds41walk holds the
 * kernel to a host reference walk bit for bit). */
#if defined(R4D_DS41_SPECS)

static const RadParamSpec pDs41Walk[] = {
    P_INT("M"), P_INT("step"), P_INT("steps"), P_INT("rank"), P_INT("n_vocab"), P_INT("N"),
    P_INT("vocab_off"), P_INT("world"), P_INTO("anchor_stride"), P_STR("dtype"), P_INTO("form"),
};
static const RadOperandSpec oDs41Walk[] = {
    { "logits", RAD_OPD_IN, 0 },     { "anchor", RAD_OPD_IN, 0 }, { "prev", RAD_OPD_IN, 1 },
    { "pred", RAD_OPD_WEIGHT, 0 },   { "succ", RAD_OPD_WEIGHT, 0 }, { "keys", RAD_OPD_INOUT, 1 },
    { "tokens", RAD_OPD_OUT, 0 },
};
static const RadConstraint cDs41Walk[] = {
    C_GE("M", 1), C_GE("steps", 1), C_GE("step", 0), C_GE("rank", 32), C_LE("rank", 512), C_DIV("rank", 32),
    C_GE("n_vocab", 1), C_GE("N", 0), C_GE("vocab_off", 0), C_GE("world", 1), C_LE("world", 64),
    C_IN("dtype", "f32"),
};

#elif defined(R4D_DS41_SCHEMAS)

SCHEMA("dspark_walk_step", pDs41Walk, oDs41Walk,
       "One step of DSpark's greedy Markov walk over the whole vocabulary, on one rank's vocabulary shard "
       "[vocab_off, vocab_off + N): step l's predecessor is the max of step l-1's gathered keys `prev` "
       "(16-byte slots, the u64 key first: an f32 [M, 4 * world] plane; the anchor at l = 0), written to tokens[s][l-1]; every "
       "column's score logits[s * steps + l][j] + sum_r pred[prev][r] * succ[vocab_off + j][r] (fma, r "
       "ascending) is keyed (ordered score bits << 32 | ~id: ties to the lower id, NaN never) and the shard's "
       "best max-ed into keys[s] (the u64 of a 16-byte slot, an f32 [M, 4] plane, zeroed before). step = steps only merges the "
       "last token. `form` (default 0) picks how a block reads succ (1: through LDS as whole lines); the bits are the "
       "same."),

#elif defined(R4D_DS41_ROWS)

{ "dspark_walk_step_f32", "dspark_walk_step", "dflash",
  "one step of the full-vocabulary DSpark Markov walk on a vocabulary shard: a thread a column, 128 a block, "
  "every sequence in passes of eight (their predecessor rows in LDS), the shard's best key a sequence by a wave "
  "max and one atomicMax a block; form 0 a thread reads its succ row in 16-byte pieces, form 1 the block's rows "
  "come through LDS as whole 128-byte lines (the same fma chain a thread)",
  "rank a multiple of 32 up to 512, vocab_off + N <= n_vocab, succ 16-byte aligned, keys [M, 4] / prev [M, 4 * world] "
  "f32 planes of 16-byte key slots, keys zeroed before the step",
  "f32 logits, bf16 pred / succ (widened exactly), f32 fma in rank order, u64 keys in 16-byte slots of f32 planes, i32 tokens",
  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,
  ROW(cDs41Walk), NOTUNE,
  nullptr, nullptr, r4d_rad_dspark_walk_step, nullptr, nullptr, nullptr,
  r4d_shape_dspark_walk_step },

#endif
