# KV streaming: host-authoritative KV with a resident VRAM window

A streamed group keeps its whole context in pinned host RAM and only a ring of blocks in VRAM; the
VRAM that frees up goes to experts. `--max-model-len` and `--max-num-seqs` size the host side,
`--kv-resident` the ring. The operator's view is [GUIDE.md](GUIDE.md) §4.4.1; source comments cite
the section numbers below.

Streaming pays only when a step reads far less than the context, which is true of radiance's
indexer-selected attention (the QSA family and GLM-5.3's MLA block) and false of dense attention
(§2). An architecture with no streamed path ignores the flag with one warning (§4.5).

Two views sit over one allocation: the **ring**, `n_slots` blocks deep and `n_layers` stripes,
addressed through a residency map, and the **stage**, the group's context deep and one stripe,
addressed by block id. The allocation is `max(ring, stage)`, never the sum (§3.1). A pass's view
follows from its phase and row shape, both already in the pass key.

---

## 1. Why the idle tiers are not the stream

The VRAM paged pool (`core/mem/kv.{h,cpp}`) is where a live sequence's KV lives. The idle tiers
(`core/mem/kvtier.*`, `core/runtime/tierexec.*`) copy a finished conversation's blocks to pinned
host memory and back on a prefix-cache hit: a hot cache whose unit is a published prefix entry,
moved between identical layouts on publish and admission, in memory no kernel dereferences.
Streaming needs many small, per-launch, selection-driven copies with a residency map, so it is a
module of its own (§4.1). It keeps the tiers' two rules: never block the host on the step path, and
order copies with events, not syncs (`spec.md` §5.4).

---

## 2. What bounds a step's read

Qwen3.8-Flash-Next (`arch/qwen4exp_fp8`) at `--tp 2`, fp8 KV, per rank; 12 QSA layers plus the
MTP head bind `kv_attn`:

| group | per 4-token block | per token | at 262 144 tokens |
|---|---|---|---|
| `kv_attn` (fp8, 2 kv heads, head_dim 256, 13 layers) | 26.00 KiB | 6.50 KiB | 1.66 GiB |
| `kv_qsa_bkey` (bf16, 1 head, head_dim 16) | 3.25 KiB | 0.83 KiB | 208 MiB |

Decode is sparse at every context: the selection keeps `topk = budget / ratio = 512` blocks a row
(`docs/QSA.md`), so a step reads about 13 MiB of `kv_attn` whatever the history. The indexer is
different: `qsa_score` reads every complete block of `kv_qsa_bkey` each step (208 MiB at 262K), so
that group is dense-read and stays resident (§3.1).

A dense family (Qwen3.8-27B, `arch/qwen35_*`) reads the whole context every step, 16 KiB a token per
rank at fp8, so a ring only subtracts from a read that is already link-bound. It declares no
streamed path.

---

## 3. The contracts

### 3.1 One id space, two depths

The paged pool is block-id indexed and layer-major: layer `i` at `base + i * layer_stride`, block
`b` at `b * layer_bytes_per_block`, and a slot is `block_table[bi] * block_size + pos % block_size`.
Nothing indirects, so a block id outside the pool corrupts the next layer with fluent text.

`KVGroupPlan` therefore carries two depths: `n_blocks`, the id space (the context), which the
allocator, refcounts, block table, prefix cache and every `id < n_blocks` guard count; and
`n_slots`, the blocks the device pool backs. `pool_blocks()` is `n_slots` (or `n_blocks`),
`streamed()` is `n_slots < n_blocks`, and `layer_stride()` uses `n_slots`. Only addressing moves.

The pool is one reservation of `max(ring, stage)` (`kv_pool_bytes`, `kv_ceiling_bytes`), and
`carve()` writes `n_blocks`, `n_slots` and `depth` together (`kvstream_apply`). A streamed group takes
its id space from the context (`kv_context_blocks`) rather than the pool split, so it does not
compete for VRAM; it never lends to the expert slab, and the idle tiers never gather it (§4.1).
Refused or charged by name:

- **a window that does not fit** the budget is refused against `--kv-resident` before the split
  (`kvstream_window_fit`), rather than leaving a neighbour with zero blocks;
- **a streamed `RAD_KV_WINDOW` group** is refused: a window bounds its own history;
- **the dense-read floor**: a group read over its whole history every step
  (`rad_kv_group_dense_read`, an indexer's block keys) cannot stream, and while a sibling streams
  its carve is `max_seqs` full contexts, since the sibling's mirror holds that many (`kvstream_dense`).
  With nothing streaming the floor is off;
- **the maps** are reserved from the KV pool before `carve()`, which the budget probe cannot do, so
  `kvstream_maps_charged()` charges them in both paths and the carve adds back what was reserved.

A ring that covers the group's context is not a streamed group: no mirror, maps, stage or resolve,
and a null `RadKVStream`, decided at startup.

### 3.2 The host mirror

The mirror is `RAD_MEM_HOST_MAPPED`, so a kernel reads it over the link in place, and laid out like
the pool: layer `l`'s row for block `b` is at `host + l * host_layer_stride + b * block_bytes`, with
`block_bytes` both the resolve's row pitch and the length of every copy into the ring. It is
`n_blocks * bytes_per_block`, zeroed at bring-up, and kept per rank, each rank with its own maps in
its own pool (`kvstream_step_view`). It is the first in-place device writer of mapped host memory,
so its coherence is pinned by a test (§6.2).

### 3.3 The writer maintains both copies

`kv_store_streamed`, and its fold into the prologue `qk_norm_rope_gate_kv_store_streamed`, writes
the host mirror always and the ring slot only when `res[block] >= 0`. A slot never goes stale, so
nothing needs invalidating, not on a rejected draft and not on a restore. The residency test must
be in the writer: at decode the block a step writes is usually the one the previous step made
resident, so a mirror-only writer would leave the ring's copy short of this step's tokens and the
resolve, seeing it resident, would not copy it. `res` is written by `kv_resolve` alone.

A layer's order is **store, then resolve, then attention**, so a block copied in from the mirror
carries this step's tokens; the indexer runs before the store and so cannot resolve. The streamed
store has E4M3 and bf16 rows; a packed cache (`fp8_v4`, `k6_v4`) has none and is ignored (§4.5).

### 3.4 The resolve

`qsa_select` writes pool page ids into `sel`, and `attn_paged` uses each entry as a raw pool index.
`sel` is the only place the sparse path names KV storage, so `kv_resolve`, between select and
attention, is the one op streaming adds: it makes every selected block resident, copying misses in
from the mirror, and rewrites `sel` as ring slots. Its operands, parameters and control block are in
docs/OPS.md ("kv_resolve"); `libref/ref_kvstream.cpp` is the serial definition.

The order is a contract, so every implementation places the same blocks in the same slots:

1. bump the epoch;
2. one lookup per valid entry: a hit stamps its slot and bumps its two-bit heat counter once a
   call; a miss claims its map entry once (`-1 -> -2`), so the map is its own set;
3. the miss list in ascending block id, never first-appearance order;
4. one victim per miss: the block's own slot when free (a context shorter than the ring stays at
   identity, and the attention's adjacent ids read adjacent slots), else a free slot, else a CLOCK
   sweep over the heat counter that never takes a slot this call stamped;
5. re-point both maps;
6. copy each placed block as bytes;
7. rewrite every valid entry as its slot.

Step 3 makes the resolve deterministic; a device implementation sorts its claim list rather than
scanning `blocks`, so a launch costs its selection, not the cache.

**Overflow.** The ring must hold the call's selection union. When it cannot, the unplaced claims
revert to `-1`, `ctl[3]` latches and their entries become slot 0, a defined address (a block id
there would be read past the ring). The engine copies each group's `ctl` behind the step's last
resolve and reads it at the commit, so the step path never waits; a newly latched overflow fails
every request prefilling in that step, naming the ring and the union, and clears the latch.

**State and recording.** The epoch, hand, heat and counters live between calls, so two launches
must never overlap (the row is not in `rad_kernel_concurrent`). The op is recordable: its addresses
and pitches are fixed from bring-up. `Ctx::pass_key` keys a streamed group by the memory its view
names (`kvstream_view_key`), never by the view's address, which the builder's rotating Out slots
change every step.

**Three implementations.** `libref/ref_kvstream.cpp` is the oracle; `libavx/avx_kvstream.cpp`
follows it step for step and vectorises only the rewrite and the copy; `libr4d/r4d_kv_resolve.hip`
is the card's row, checked against libref by `r4d_selftest --case kv_resolve` (hits, ascending
misses, eviction, free before live, second chance, saturation, overflow, 2052 claims, the copy, the
refusals).

### 3.5 Prefill: the union, and the stage

Above `qsa_exact_to = budget + ratio - 1 = 2051`, prefill is per-query sparse: every row of a chunk
selects its own 512 blocks. One launch's union (`scripts/kv-union.sh`, Flash-Next `--tp 2`):

| prompt tokens | ring that held | largest union | largest misses of one call |
|---|---|---|---|
| 3 958 | 2 048 blocks | 990 | 976 |
| 16 048 | 4 096 | 4 000 | 1 024 |
| 66 131 | 16 384 | 14 608 | 1 024 |
| 133 331 | 32 768 | 23 429 | 1 024 |

The union is 70-100 % of the history and grows with it, so no ring sized for decode (at most 513
blocks a row) serves prefill. Prefill therefore takes the **stage**: one layer's whole history at
block-id addressing, filled from the mirror per layer, with no selection awareness and no resolve.
A prompt below `qsa_exact_to` runs dense attention, which reads block ids from the table; it stages
the same way.

### 3.5.1 The aliased stage and the ordering rule

Every layer of the stage aliases offset 0 of one allocation. The operand resolution accepts that:
a `RAD_KV` operand is bounded by `off + layer_stride <= bytes`, which a one-layer pool satisfies at
`off = 0` for every layer (`a_streamed_groups_layers_may_all_alias_one_pool_slot`). The aliasing is
the binding's `depth` (1 = every layer at offset 0), not a rewritten `layer_slot`: that table is also
the host mirror's row (`rad_kv_layer_index`), so collapsing it would write every layer into one
mirror row.

The KV operand is external memory, so the buffer plan never sees it, and the hazard pass sees the
same range for every aliased layer. The order rests on the stage's barriers alone, stated beside
the code in `core/kvstream/stage.h`: the copy of layer L lands before L's store, L's store and
attention follow, and the copy of L+1 starts after every attention launch of L. What breaks it:
taking the ride before all of L's chunked attention launches, waiting after the store instead of
before it, moving the copy to lane 1 without a join, replaying a recorded pass without re-running
the arm and submit (`Ctx::tape_kv_stage` does), or a binding that moves between passes sharing a
key.

A staged pass reads the pool by block id, while `kv_store_streamed` writes the ring only where the
map names a slot (none before the group's first decode), so a staged pass also stores at identity
(`AttnGatedFP8::store_stage_identity`, GLM's plain store), after the mirror writer.

**Volume.** A layer's copy moves only the blocks the step's own block tables name, not the whole
context allocation: its list carries one range per run of those ids (`kvstream_runs`), read every
step after the step's blocks are ensured (`kvstream_step_runs`). The ranges must never miss a block
the pass reads: the attention would read whatever the previous layer left, which is wrong text, not
a fault. They cover every read because the pass reaches the stage only through the step's tables,
the stage is identity-addressed as the mirror row is, and a prefix hit's cached blocks are in the
hit's table. Gaps of up to 16 blocks are copied through, and past 256 ranges the smallest gaps are,
so a fragmented table stays one short list. Blocks the prefix cache retains for other
conversations are not copied: they often hold the low ids, and a new sequence takes ids above them,
so a bound on the highest id a step names (or on the highest held one, as before) would still copy
every retained conversation, and each later prefill paid for all of them. A step whose sequences
are not known, or whose trunk pass takes the ring, copies up to `KVManager::held_top_blocks`.

### 3.6 The ABI surface

`RadKVStream` (`abi/rad_runtime.h`) is one group's fixed view: the mirror, the per-layer residency
maps (`res`, `slot_block`, `slot_state`), the shared control block, `block_bytes`,
`host_layer_stride`, the ring's `slots` and `host_bytes`. `RadKVGroupBatch` carries a trailing
`const RadKVStream* stream`, null for a resident group: a pointer, because the view is fixed at
bring-up while the batch is rebuilt every step. The build side is `RadBuildCtx::kv_resident_blocks`
(the ring in tokens), `RadBuildCtx::kv_stream_ok` and `RadArchProbe::kv_stream_ok` (§4.5). The arch
side is `rad_kv_layer_index`, `rad_kv_stage_layer` and `rad_kv_stage_next`, with the
`kv_stream_layer`, `kv_resolve_declare`, `kv_resolve_issue` and `kv_stage_*` helpers in
`arch/common/rad_arch.h`.

### 3.7 Flags, report, metrics

`--kv-resident N` is the only switch. The KV report gains the streamed group's two-tier line, and the
startup lines name the ring (and why it was raised), the host mirror, the stage, or which context a
covering flag spans. Metrics: `radiance:kv_resident_blocks`, `radiance:kv_stream_misses_total` and
`radiance:kv_stream_bytes_total{dir="in"}`. A request's end logs each group's cumulative resolves,
lookups and misses with the largest union, misses and claims of one launch (`ctl[10..12]`).

### 3.8 The host-RAM budget and the gate

The mirror is the largest host allocation after the experts' pool: 4.69 GiB a rank for four
204 800-token sequences of Flash-Next. `kvstream_host_gate` checks the mirrors once, before any rank
pins one, against the host tiers still to be pinned after them (the activation arenas and, when
sized, the checkpoint store) plus `--host-headroom-mib`. A refusal names `--max-model-len` and
`--max-num-seqs`, which size the mirror; `--host-pool-mib` sizes the experts' tier and cannot make
room.

The prefix cache's host tier (`--prefix-cache-mib`) is turned off when a group streams, before
anything is sized from it: an entry covers every cached group, and a streamed group's rows are
addressed by the block ids its sequence used, so the entry cannot move to a host slot
(`PrefixCache::demotable`). Prefix hits still work from VRAM; a conversation evicted from VRAM is
prefilled again rather than restored. "Streams" is the engine's answer (`cfg.kv_streams`): the flag
set, the architecture's streamed ops declared and linked for the group with an fp8 or bf16 cache,
and a ring under the context. In every other case the host tier is as configured.

### 3.9 What it costs

`--kv-resident` has two floors, applied in `kvstream_ring_shape`/`kvstream_ring` and clamped to the
context, and the carve takes the larger:

- **correctness**: one decode step's selection, `max_seqs x (1 + n_spec) x 513` blocks (16 416
  tokens at two sequences and speculation 3); a request under it could not be served from the
  selection table;
- **cost**: **32 768 tokens** (`kStreamFloorTokens`). Below about 24 576 tokens the misses cost more
  than the VRAM the ring frees (at 64k of history: 24 576 runs 37.9 tok/s, 8 192 runs 37.1 with 75 %
  more misses), and a ring larger than the history buys nothing.

A flag under either is raised with one warning. The raised number is what
`RadBuildCtx::kv_resident_blocks` carries into the declare, since `kv_resolve` sizes its table from
it. A context under 32 768 tokens cannot stream.

Decode: a miss costs about 0.8 us, and a resolve launch about 4 us of device time, flat across ring
sizes; the twelve a step cannot merge (each follows its layer's store), so streaming costs about 3 %
of decode at `--tp 2` and nothing at `--tp 1`. The identity placement is worth +2 % at 16k. Warming
the ring from the last prefill chunk made decode 3.8 % slower, through slot order.

Capacity is bought in host RAM. Flash-Next fp8, two ranks, `--kv-resident 24576`:

| configuration | VRAM KV per card | host RAM per rank |
|---|---|---|
| resident, 2 x 204 800 | 1.43 GiB, one conversation retained | - |
| resident, 4 x 204 800 | 1.53 GiB, 25 % of the four conversations | - |
| streamed, 2 x 204 800 | ring 144 MiB, stage 200 MiB (pool 200 MiB), indexer 300 MiB: 618 MiB | 2.34 GiB |
| streamed, 4 x 204 800 | ring 144, stage 400, indexer 600: 1 232 MiB, all four funded | 4.69 GiB |

Per retained token: 6.00 KiB of mirror (host), 0.83 KiB of indexer and 0.50 KiB of stage (VRAM);
only the ring does not scale. Prefill comes out 3-5 % above the resident arm, because the VRAM the
cache no longer holds goes to experts that would otherwise be read over the link. The stage copy's
cost is its volume (0.6 GiB a pass at 16k and 45 GiB at 160k, within 0.3 % of the live KV, a
figure the step's own ranges keep however much the prefix cache retains), not its overlap, since it shares the link as an equal (§6.1). Capacity is bought with
`--max-num-seqs`, inside the trained window, so no rope change is needed. Perplexity: 3.994 resident,
4.015 with a covering window, 4.014 with a fraction of it.

---

## 4. Architecture

### 4.1 The module

`core/kvstream/` holds the feature; the existing files make one call at each seam.

| file | owns |
|---|---|
| `plan.h`, `.cpp` | the policy: which group streams, the ring's floors, the sizes, which ops a group needs. Arithmetic only. |
| `bind.h`, `.cpp` | the layout: both views over a plan, `layer_slot` kept as the mirror's row, the per-pass view, `kvstream_step_view`, `kvstream_view_key`. |
| `stage.h`, `.cpp` | the driver: per pass and layer, the copy of the layer's live blocks from the mirror into the stage on the DMA engine (§5.3). |
| `mirror.h`, `.cpp` | the mirror and the maps: sizes, allocation, initialisation, release, the window cap and the counters. `KVManager` holds one `KVStreamStore` and delegates. |
| `carve.h`, `.cpp` | the carve's part: windows and the dense-read floor leaving the budget first, the window-fit refusal, `n_blocks`/`n_slots`/`depth`, the maps' charge, the report line. |
| `bringup.h`, `.cpp` | the engine side: the group, the refusals (§4.5), the floors, the host-RAM gate, the startup lines. |

Only `stage.cpp` and `mirror.cpp` touch a device. Deliberately left alone: the block ids, refcounts
and free lists (the stage addresses every block identically, and a prefix hit adopts the same ids);
`TierExec` (a gather of one layer from a one-layer pool would copy the last staged layer `n_layers`
times; the mirror is the tier); the prefix cache's host tier (§3.8); the expert stager and mover; and
`qsa_select`.

### 4.2 Core changes

1. **The carve split**: `n_slots` beside `n_blocks`, `layer_stride()` over `n_slots`, ids still in
   `[0, n_blocks)`.
2. **The host mirror**: `n_blocks * bytes_per_block` of mapped pinned memory a streamed group and rank,
   gated by `lockable_room`.
3. **The maps**: device `res`, `slot_block`, `slot_state` per layer and a control block, from the KV
   pool, carried in the batch.
4. **Report, flag, metrics** (§3.7).

### 4.3 Device changes

1. `kv_resolve`: a libr4d kernel, a libavx form and the libref oracle (§3.4).
2. `kv_store_streamed` and its prologue fold (§3.3), E4M3 and bf16.
3. The stage copy, a DMA list per layer (§5.3); no kernel.

### 4.4 Arch changes

The gated fp8 attention block (`arch/common/rad_block_attn_gated_fp8.h`) declares the streamed store
beside `kv_store`, issues it instead when the batch hands the group a stream, and issues the resolve
after the store and before each sparse launch. GLM-5.3's MLA block (`rad_block_glm_mla.h`) declares
the same pair and resolves at finish, after its indexer selects.

### 4.5 The capability gate

A plugin states once, in `RadArchProbe::kv_stream_ok`, that its architecture can stream; the engine
carries it into every declare as `RadBuildCtx::kv_stream_ok`. `qwen4exp_fp8` and `glm5next` set it.
The three streamed-only ops (`kv_store_streamed`, its fold, `kv_resolve`) are declared only when the
architecture can stream and `--kv-resident` asks for a ring (`kvstream_op_gated`, consulted by
`Builder::op_declared`), so a model file never knows whether the operator asked, and every
`rad_op_declared` check falls through to the resident path. Fail-safe, with one warning each:

- a group with no `kv_resolve` or no streamed store for its geometry is ignored: without the resolve
  the attention would read block ids past the ring; without the store a miss would read zeros. A
  declaration with no kernel, or one stating another group's geometry, does not count;
- a packed cache (`fp8_v4`, `k6_v4`) is ignored the same way, naming the layout;
- an architecture whose probe says no turns the flag off at startup.

Without the flag no streamed op is declared and no budget changes. The graph is not byte-identical
to a build without the feature, though: `qsa_select` writes its entry count (`qsa_nsel`,
`selected_count`), an i32 a token, whether or not anything streams.

---

## 5. The stage copy's transport, and the ring's depth

### 5.1 The ring's depth

`--kv-resident` is off by default. When set it is never carved below 32 768 tokens, nor below one
decode step's selection when that is larger (§3.9), and never above the context; one function,
`kvstream_ring_shape`, gives the budget probe, the carve and the declared `kv_resolve` the same
depth.

`kv_pass_view` picks a pass's view: a decode pass of at most `1 + n_spec` rows a sequence takes the
ring; every other pass (prefill, a mixed step, and a decode-classified pass wider than that, such as
the serial drafter's history pass over a prefill chunk) takes the stage.

### 5.2 There is no streaming switch

`--kv-resident N` is the feature: absent or 0 and nothing streams, set and the ring is N tokens. A
ring that covers the context does not stream at all (§3.1), so a configuration whose context fits
runs the same path as without the flag.

### 5.3 The transport

The copy of layer L+1 is one `rad_xp_list` on the card's prefetch DMA queue, riding (`rad_ride`) on
the compute stream after every attention launch of layer L, so it starts once L is done with the
aliased buffer and crosses the link under L's out projection and L+1's projections, holding no CU
slot. The compute stream waits for it before L+1's store through an event the list completes
(`rad_event_dma`): the list writes 0 to the event's completion word, armed with a 1 in stream order
by the weight-table patch dispatch the ride completes on. A host-side arm could race a wait still
queued for layer L, and an event record cannot be used, since a record completes by reaching 0.

The list passes `RAD_XP_NO_YIELD`: the compute stream waits on it, so yielding the link to a hold
whose release is behind that wait would deadlock. A backend with no DMA queue copies on the compute
stream in the same order. The arch seams are `kv_stage_wait` before the store and `kv_stage_next`
after the attention (`arch/common/rad_arch.h`).

### 5.4 What is not reused

The transport is the expert prefetch's; the policy is not. The prefetch patches weight-table entries
and is sized for experts, and a late prefetch is simply unused, while a stage copy cannot be skipped
and must be waited on. They do not compete: the prefetch is decode-only and the stage prefill-shaped.
Rejected alternatives: carving the stage from the arena's lent top (a KV base inside another owner's
pool, lent per step while the carve is per bring-up); a full `n_layers`-deep buffer lent to the
experts between requests (1.66 GiB at 262K during prefill, and reclaiming it is a host wait); the
whole context resident during prefill (the group would still carve `n_layers x context`).

---

## 6. Failure modes and the tests that hold them

| failure mode | what prevents it | test |
|---|---|---|
| a block id read as an address past the ring | the resolve rewrites every entry; overflow writes slot 0, latches, fails the request | `tests/ref_test.cpp` `kv_resolve_*`; `r4d_selftest --case kv_resolve` |
| a streamed plan without a mirror or maps | `kvstream_apply` refuses | `a_streamed_group_is_refused_at_carve_without_its_host_copy` |
| a pool whose depth disagrees with its id space | `n_blocks`, `n_slots` and `depth` written together | `a_streamed_group_addresses_its_context_but_backs_only_the_resident_window` |
| a streamed window group, or a window that does not fit | refused by name | `a_streamed_window_group_is_refused_by_name` |
| a ring under one decode step's selection | raised at startup | `a_ring_below_one_decode_step_s_selection_is_raised_to_the_ceiling` |
| a covering ring paying for streaming | `streams()` false, null `RadKVStream` | `a_ring_that_covers_the_context_streams_nothing_and_the_engine_pays_nothing`; `scripts/kv-parity.sh` |
| a wide decode-classified pass resolving a ring it cannot fit | `kv_pass_view` takes the stage | `a_decode_pass_wider_than_a_verify_stages_instead_of_resolving_the_ring` |
| a mirror nothing writes | the store must resolve to a kernel for the group's geometry | `a_declared_streamed_store_with_no_kernel_is_not_a_streamed_path`; `a_packed_cache_warns_that_the_layout_has_no_writer_and_serves_non_streamed`; `a_resolve_declared_for_another_group_does_not_enable_streaming` |
| the flag honoured with no streamed path | the capability gate | `a_model_whose_architecture_declares_no_streamed_ops_ignores_the_flag`; `a_streamed_op_exists_only_for_a_capable_architecture_and_a_ring` |
| a stage reading the previous layer's bytes | the ordering rule (§3.5.1) | `a_streamed_groups_layers_may_all_alias_one_pool_slot`; `scripts/kv-stage-parity-64k.sh` |
| a stage copying less than the pass reads | the ranges of the step's own tables | `the_stage_ranges_cover_every_block_a_table_names`; `scripts/kv-stage-parity-64k.sh` |
| a stage copying what other conversations hold | the ranges of the step's own tables | `the_stage_ranges_leave_out_blocks_no_table_of_the_step_names` |
| a stage list's mark reused while the list still polls it, both cards stopped | the signal pool holds a mark until its list's engine is past the poll | `device_test` `a_dma_lists_mark_stays_out_of_the_pool_until_the_engine_has_read_it` |
| maps leaving the pool short of the probe | `kvstream_maps_charged` in both paths | `a_streamed_groups_maps_are_charged_before_the_pool_is_sized` |
| a dense-read sibling funded for one sequence | the `kvstream_dense` floor | `a_dense_read_groups_floor_holds_every_sequence_while_a_sibling_streams` |
| a recorded pass replayed against another view | `kvstream_view_key`; fixed addresses | `Ctx::pass_key`'s static assertion on `RadKVGroupBatch` |
| the copy moved to another lane or re-issued from the host | §3.5.1 and §5.3 | reviewed, not tested |

### 6.1 The link, the lanes, and the hold

At `--tp 2` the all-reduce, the stage copies and the prefill expert stager share the link. DMA lists
take their share at full rate without a CU slot; an ordinary stream copy would hold one and starve
the kernels. Because the stage list must not yield, it competes as an equal, which is why overlap
recovers only part of its cost (§3.9). A stage moved to lane 1 needs `rad_lane_join(1, 0)` before
its reader.

### 6.2 The first in-place device write to mapped host memory

`r4d_selftest --case kv_store_streamed_fp8` and `--case kv_store_streamed_bf16` put the mirror in
mapped pinned memory, run the kernel and compare the bytes the host reads after the sync with
libref's; `scripts/kv-stage-parity-64k.sh` exercises the same write in the served path.

### 6.3 Rank coverage

Every rank resolves the same shared block ids through identically built maps of its own. Only
`--tp 1` and `--tp 2` have run; `--tp 3` and `--tp 4`, the uneven split, the attention-zero ranks
and the four-card selftest cases are untested, and the per-rank view (`kvstream_step_view`) is the
path to check first.

### 6.4 The draft model

The MTP head binds the same `kv_attn` group. A verify step issues one selection a row and takes the
ring, which is why the decode bound counts `1 + n_spec` rows (§3.9), and a decode-classified pass
wider than that stages (§5.1). Whether the drafter should share the residency map with the trunk
is open.
