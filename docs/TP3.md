# Tensor parallelism at any world size

radiance shards by dividing: `n_head`, `n_head_kv` and the delta-net head counts split across the
world, and a count that does not divide is refused at declare, by name (`rad_arch.h`,
`core/format/share.cpp`). Qwen3.8-Flash-Next has 24 query heads, 2 KV heads, 16 GDN K heads and 48
GDN V heads. Four ranks work by replicating each KV head on a pair of ranks; three do not, since
none of the counts divides by 3 and replication needs a world that is a multiple of the KV head
count.

Two rules let the shipped container serve at any world size, with no second container and no
padded heads: a smaller attention world (§1) and uneven delta-net heads (§2).

## 1. The attention world

`rad_arch.h` `geom_from` handles a world above the KV head count by replication when the world is
a multiple of the count. For a plugin that passes `attn_world_ok` (qwen4exp does), it also handles
a world that is neither a divisor nor a multiple of the count: attention is served on the largest
multiple of the KV head count below the world (the **attention world**), and the ranks past it are
**attention-zero**.

Three ranks over two KV heads give an attention world of 2. Ranks 0 and 1 split the heads as a
two-rank serve does (12 query heads, one KV head, gqa 12, the same kernels). Rank 2 computes no
attention: `rad_block_attn_gated_q.h` declares no projections, no attention and no cache store on
it, and zeroes the block's output wire with one `fill` ([OPS.md](OPS.md)). The all-reduce and the
residual after it run as usual, and the zero adds nothing to the reduce. Five ranks over two KV
heads give an attention world of 4 (two replicated pairs) with rank 4 attention-zero; six give
three pairs and no zero rank. A plugin that does not pass `attn_world_ok` is refused at such a
world, by name.

KV groups stay in the same order on every rank. The scheduler sees one program
(`core/sched/batch.cpp` indexes `kv_groups[i]` by position) and one block table serves every rank,
so an attention-zero rank declares its attention group like its peers and marks it
`rad_kv_group_zero`: it carves nothing for the group, holds the shared table's block count, lends
none of it, and the tiers skip the group when they move an entry. Every rank carves the same block
count: each is planned, asked what it could page (`KVManager::paged_tokens`; a rank whose only
paged groups are zero limits nobody), and capped at the least before any rank carves.

An attention-zero rank gets pools of its own. The budget resolver derives one split for every card
from the most expensive rank, and an attention-zero rank's cache costs a fraction of its peers' per
token (at TP3 on Flash-Next it keeps only the QSA block keys: 2.3 GiB at the worst case, where an
attention rank's costs 15.2 GiB). Under `expert_tiered`, `Engine::plan` gives such a rank a KV pool
that pages exactly the common token count (`KVManager::pool_mib_for_tokens`, the inverse of
`paged_tokens`: the least whole MiB whose paged split reaches the count; exact because the split is
monotone in the budget and its alignment is among the fixed costs), and moves the difference into
its weights pool, up to all of its weights resident.

- Both readers of the resolved sizes, `Pools::configure` and the planner, take the rank's own
  split, so the grown pool is planned into. Startup refuses a rank whose plan and weights pool
  disagree, or whose re-sized KV pool pages fewer tokens than it was sized for.
- The token count, the carve, the loan and the attention ranks' pools are unchanged.
- Other placements keep the common split (under `auto` a fully resident rank would choose
  all_vram while its peers spill).
- `deterministic: true` and a device without virtual memory management re-split like any other: the
  cache does not lend there, and the bytes the rank never pages are as much its experts' as under a
  lending cache.
- The condition is the cost, not the world: a rank is re-split only when its cache costs less per
  token than the most expensive rank's, which at TP2 and TP4 no rank does. A rank that qualifies
  but keeps the common split gives the reason on its `kv: rank N's cache costs ...` line.
- The pinned host pool is not re-split: every rank allocates its whole share of `host-pool-mib`, and a
  re-split rank whose weights all fit on the card logs that its pinned host pool holds no weights.

The log line on the third rank:

```
kv: rank 2's cache costs 2.27 GiB at the worst case where its dearest peer's costs 15.16 GiB, so its KV pool pages the common 1446302 tokens in 1811 MiB instead of 10881; its weights pool grows 19747 -> 21692 MiB (every weight resident wants 21.18 GiB), and 7125 MiB stay unclaimed
```

That rank's `mover:` line then reports 0 pinned pool slots. The settle check's INFO line on that
card names the unclaimed MiB and points at the `kv: rank 2 ...` line, not only at the budget's
`unusable` line.

Under a smaller attention world, the even shard over `world` is the wrong cut for every attention
projection, so the attention ranks declare spans (`rad_weight_shard_span`, the call KV replication
uses for k/v): q|gate rows and output columns over heads `[attn_rank * n_head, +n_head)`, k/v rows
at `kv_off`. When the attention world is the world (every TP2 and TP4 deployment), no span is
declared.

## 2. Uneven delta-net heads

The delta net has no KV-head constraint: its kernels are bound by head widths (`head_k`, `head_v`,
the chunk) and take any head count. 16 K heads over three ranks are therefore 5, 5 and 6, and each
rank's V heads are the same multiple of its K heads (48 / 16 = 3: 15, 15 and 18). Every rank holds
real heads and nothing is padded.

The extra head goes to the rank without attention (`remainder_place` in `rad_arch.h`), which has
the time; a rank holding both the extra head and attention would be the step's slowest. The
experts' extra block stays on the parity rotation (`qwen4exp_fp8.cpp` gives the reason).
`qwen4exp` computes the split and `rad_block_gdn_q.h` declares it as spans of the container's
rows. Each rank's recurrent state has its own size, so each rank's own plan clears a fresh slot,
takes a snapshot and lays out a tier entry: a slot of the snapshot store or of the tiers is the sum
of the ranks' shares, each rank's share starting where the previous one ends.

`in_proj_qkv|z` stacks four projections along dim 0 (q 2048, k 2048, v 6144 and z 6144 rows), and
`rad_weight_shard_span` applies one `[lo, hi)` to every stacked part, which assumes equal parts.
`rad_weight_shard_span_parts` takes a span per part together with each part's whole extent in the
container, and `share.cpp` cuts every part by its own rows. The conv channels `[q|k|v]`, the `[a|b]`
projections, `A_log`, `dt_bias` and the output projection's columns take the same rank's heads.
When the counts divide, the even split is used.

## 3. Interfaces

- `rad_weight_shard_span_parts(b, w, lo, hi, total, n)`: a span per stacked part.
- `rad_kv_group_zero(b, g)`: this rank keeps no cache for a paged group.
- The `fill` op: libr4d's `fill_bf16`, libref's `ref_fill`.
- `Geom::attn_world / attn_rank / attn_zero`, set by `geom_from` under `attn_world_ok`;
  `GdnQ::Config::k_first / v_first`, set by the plugin.
- Both exports are additive: `RAD_ABI_VERSION` is unchanged by them, and a plugin that does not
  call them is unaffected.
- Tests: `tests/share_test.cpp` covers the per-part cut, `tests/arch_test.cpp` the three-rank
  declaration, and `tests/mem_test.cpp` the capped carve, the zero group and a tier entry moved
  across ranks with unequal shares.

Load check: the relayout pool refuses to finish when any staged weight was never delivered. A slot
nothing wrote holds whatever the card had there: zeros on a card whose driver wipes freed memory,
garbage after a reset ("VRAM is lost"). To expose such a read on any card:

    RADIANCE_DEBUG_POISON_ALLOC=255 radiance --config radiance.yaml    # model: stock.rad, tp: 3

This fills every fresh device allocation with 0xFF (NaN in every float width) before it is handed
out; the serve must answer exactly as without it. `RADIANCE_DEBUG_POISON_RANGE=lo:hi` narrows the
fill to a range of a thread's allocation ordinals, for bisecting (`core/device/dispatch.cpp`).

## 4. Serving

```yaml
# radiance.yaml
memory:
  gpu-headroom-mib: 96
models:
  qwen3.8-next-flash:
    model: qwen3.8-next-flash-iqr-v4.rad
    tp: 3
    tp-wire: exact
    max-num-seqs: 8
    max-model-len: 262144
    placement: expert_tiered
    host-pool-mib: 24576
    kv-cache-dtype: i8
    num-speculative-tokens: 3
    max-num-batched-tokens: 2048
```

```sh
radiance --config radiance.yaml
```

At three ranks the exact all-reduce is the one-shot up to 1 MiB per message (every decode step) and
the two-shot above it (every prefill chunk). The two-shot is reduce-scatter + all-gather and sends
4/3 of the message per rank instead of twice it. A message rarely divides into thirds (a
2048-wide bf16 row is 256 16-byte words), so the first `words % 3` shards take one word more. Both
fold in fp32 in ascending rank order, so the switch between them changes no bits. `tp-wire: wht6`
at three ranks is the tiered-int8 wire over the same uneven two-shot for messages from
`tp-wire-min-kb` up, and the exact one-shot below.

Startup logs the geometry:

```
radiance: world_size = 3 over n_head_kv = 2: attention on 2 rank(s), rank(s) 2..2 carry no attention and take the delta net's remainder first
```

Each rank's delta-net share is a note in its declared graph; `debug-graph: true` and the dashboard's
graph view show rank 0's:

```
rank 0 takes delta-net K heads [6, 11) of 16 and V heads [18, 33) of 48
```

Ranks 1 and 2 take K heads [11, 16) and [0, 6), and V heads [33, 48) and [0, 18).

The following was measured by a contributor on three Radeon AI PRO R9700 (gfx1201, PCIe Gen5 x16)
with the uncensored Flash-Next fine-tune, on an earlier revision of this design: the
attention-zero rank still ran its projections on zero weights, and the extra delta-net head sat on
rank 0. Hot = 1 / 4 / 8 streams, 8K prompt, 1K generated, aggregate tok/s; crowd = 4 streams at
100K context, per stream; HumanEval 164 and a 200-question GSM8K subset at temperature 0:

| container | `tp` | hot c1 / c4 / c8 | crowd | HE | GSM8K |
|---|---|---|---|---|---|
| unpadded | 3 | 191 / 309 / 353 | 42.7 | 0.878 | 0.925 |
| unpadded, the same file | 2 | 177 / 308 / 377 | 33.8 | 0.884 | 0.915 |
| padded offline | 3 | 182 / 288 / 395 | 43.5 | 0.848 | 0.935 |

At these settings three ranks were faster than two for a single stream and for the long-context
crowd, but not for the eight-stream aggregate. The quality columns are within run-to-run variation
(HumanEval moves by about 3 of 164 between runs of one container). Unpadded and padded at three
ranks gave the same answers on 40 prompts at temperature 0, bit-identical on one: the head regroup
changes the order the all-reduce sums in, and near-ties flip.

`tools/pad_tp.py` pads a checkpoint for an engine without an attention world (zero heads appended
so that 36 / 3 / 18 / 54 divide, at the cost of a second container and of computing zeros), and
`tools/pad_verify.py` verifies a pad; radiance needs neither.
