# QSA: Qwen4-Exp's sparse attention

The full-attention layers of Qwen4-Exp (every fourth: 3, 7, … 47) do not attend to the whole
history. A small **indexer** scores 4-token blocks of the key history, and the attention reads the
best 512 of them plus the incomplete tail, so attention cost is constant past 2051 tokens.

The declaration side is `arch/common/rad_qsa.h`. The kernels are libr4d's `qsa_block_key`,
`qsa_score`, `qsa_select`, `qsa_work` and `qsa_tail_store`; the sparse attention is the ordinary
`attn_paged`. Schemas and operand contracts are in `docs/OPS.md`.

## Shape

From the checkpoint: `indexer_n_heads` 4, `indexer_kv_heads` 1, `indexer_head_dim` 128,
`indexer_budget` 2048, `indexer_compress_ratio` 4, so `block_topk = 2048 / 4 = 512` blocks.
`rotary_dim` is the main attention's (`head_dim 256 * partial_rotary_factor 0.25 = 64`), applied to
the first 64 of the indexer's 128 dims from the layer's cos/sin table; the plugin refuses a
checkpoint where it does not fit. Weights per QSA layer:

```
model.language_model.layers.<L>.self_attn.indexer.index_qk_proj.weight   [640, 2560]
model.language_model.layers.<L>.self_attn.indexer.q_layernorm.weight     [128]
model.language_model.layers.<L>.self_attn.indexer.k_layernorm.weight     [128]
```

640 = (4 query heads + 1 key head) x 128.

## Computation

```
qk        = index_qk_proj(hidden)             [T, 640]
q, raw_k  = split(qk, [512, 128])             q -> [T, 4, 128], raw_k -> [T, 128]
q         = rope(q_layernorm(q), pos)         per query, at the query's own position

per complete block b of 4 consecutive key tokens:
  pooled  = mean over the 4 raw_k, in f32, cast back to bf16
  blk_k_b = rope(k_layernorm(pooled), pos of the block's first token)

per query:
  score_b = sum over the 4 heads of relu(q_h . blk_k_b) / sqrt(128)
  keep    = indices of the top min(512, n_blocks) scores
  visible = the 4 tokens of every kept block, plus the tail
            (the (pos+1) % 4 tokens after the last complete block, attended unconditionally)
```

The query side is `qk_norm_rope_gate`'s k side at `n_head` 0, which reads `[tokens, heads,
head_dim]` at the operand's row pitch (the query half of the `heads + 1` projection). It needs the
model's cos/sin table and is bounded by the attention block's fusion row count, above which the
table is not filled; otherwise `rmsnorm` and `rope` run over column slices.

## Block keys

A block key is independent of the query and, once its four tokens exist, permanent. The reference
recomputes block keys inside the per-query loop; the engine computes each once, when the token with
`t % 4 == 3` arrives, and caches it. The indexer's cache is **block keys**, `[n_blocks, 128]` bf16
(64 bytes per token per QSA layer), plus **a tail** of at most 3 raw keys for the next block.

12 QSA layers x 64 B = 768 B per token, about 3% of the main KV cache's 24 KiB per token
(2 heads x 256 x 2 (K and V) x 2 bytes x 12 at bf16): 192 MiB per sequence at 262144 tokens,
24 MiB at 32768.

## Exactness bound

With `pos + 1 <= budget + ratio - 1 = 2051` visible tokens there are at most 512 complete blocks,
the top-k keeps all of them and the tail is always attended, so QSA equals dense causal attention
bit for bit. Below the bound, a step with more than one query per sequence (prefill, mixed or
speculative) issues dense attention, which reads each key once per query tile where the sparse
path reads a table per query. Plain decode (one query per sequence) takes the sparse row's launch
at every context: there is no tile to amortise, and the sparse kernel stays exercised. The bound is
the test for every change here (see "Checking").

## Below the bound

At or below the bound the selection does not depend on the scores: every complete block, in
block order, then the partial page. So a step that is *entirely* at or below it scores and selects
at most once for the whole pass, and where nothing reads a selection not at all. The attention
block decides (`AttnGatedQ::qsa_use`, `arch/common/rad_block_attn_gated_q.h`) and hands the
answer to the indexer, which runs first (`QsaIndexer::step`'s `use`):

| the step, at or below the bound | attention (unchanged) | the indexer issues |
|---|---|---|
| several rows a sequence, rounded bound within the budget (a chain verify, a head's history pass, prefill, mixed) | dense, over the block table | per layer: the projection, `qsa_work`, `qsa_block_key`, `qsa_tail_store` |
| one row a sequence (plain decode, a serial head's round) | the sparse row's launch over the sequence's **block table** and `seqused` | the same four |
| several rows a sequence, rounded bound past the budget (the same steps between 1025 and 2047 tokens) | the sparse row a query, over the selection | the first full-attention layer of the pass: everything, as before; every other layer: the same four |
| anything past the bound, a streamed attention group, `RADIANCE_DEBUG_QSA_EAGER` | as before | everything, per layer |

**The attention's kernel never moves.** Which kernel a step's attention runs is decided exactly as
before (dense by the rounded bound, the sparse row otherwise): the dense and the sparse kernel read
the same keys below the bound but sum them in another order, and a verify's rows feed the next
step, so a different kernel at the same context is other text at a near-tie. Only *who makes*
the selection changes, and it is byte for byte the selection each layer made for itself.

**The stateful half always runs.** Block keys and the tail are what later steps score and build
on; only the query side and the scoring are skipped.

**The block table in place of the selection** is exact because of what `qsa_select` writes below
the bound for a row at position `p` of a sequence whose context ends there: `sel` is the table's
first `ceil((p + 1) / ratio)` entries and `seqused` is `p + 1`, the sequence's own. The launch is
the same op and kernel (`attn_paged_gate_quant`, `q_len` 1), its split count does not read the
table's width, and it sums pages in the same order, so the bytes are the selection's. It holds for
a row that is alone in its sequence and at the context's last position, which the architecture
vouches for (`qsa_table_rows`: a plain decode row and a serial head's round on Qwen4-Exp), and it
is taken only when `max_q_len == 1`.

**One selection a pass** (`kQsaOnce`) is exact because below the bound a row has at most `topk`
complete blocks, so `qsa_select` takes every block whose key is not 0 -- any block with a page and
a score that is a number -- and what it writes (the pages in block order, the partial page, the
count, `seqused`) depends on the positions and the block table alone: every layer's scores give the
same bytes. The first full-attention layer of the pass therefore issues its selection exactly as
before (query, scores, `qsa_select`) and every later layer's attention reads that one; the later
layers issue no query, no score and no select. `sel`, `nsel` and `seqused` are the model's buffers,
live across its layers, and nothing else writes them. The MTP head has one such layer and so issues
what it always did.

One consequence is deliberate: a block whose score is not a number (a NaN block key) is dropped
or kept for the whole pass as the first layer's score has it, and is always kept where no
selection is made, as the dense attention always kept it.

**Whether a step is within the bound is the group's exact ceiling.** `max_ctx_len` is rounded up to
1024 tokens so a recorded pass replays (`core/sched/batch.cpp`, `bound_bucket`); by it a 4-row
verify is "past" the budget from 1025 tokens of context on. `RadBatch::kv_used_max_h[group]` is the
exact host ceiling of every row's `position + 1` (context plus this step's rows, the furthest a
row issued ahead can be), and `qsa_within` reads it where the architecture's pass key carries
which side of the bound it is on (`qsa_exact_host`; `qwen4exp_fp8.cpp`, `pass_key`). The cost is
one more recording of each pass kind when a request crosses the bound.

Per chain verify step of Qwen3.8-Flash-Next (12 QSA layers) this removes 36 launches up to 1024
tokens of context and 33 from there to the bound, and 3 a head pass up to 1024 tokens (its rounds
up to the bound).

## Attention over a selected set

With the full-attention KV group paged at 4, a compress block is a page and the selected set is a
paged block table, so sparse attention is the existing `attn_paged` with a 512-entry table.

The page size is a kernel constant. The QSA attention units (`r4d_attn_paged_h256_gqa12.hip`,
`r4d_attn_paged_h256_gqa6_bs4.hip`) build `r4d_attn_paged_gqa8.hip` with `G8_BLOCK_SIZE 4`, and
their rows declare `C_EQ("block_size", 4)`. `RadKVGroupDecl` carries no block size: the core takes
it from the resolved attention kernel, and the plugin asks for 4 through the attention op's
`block_size` parameter. **The rows and the `#define` must agree**: a row claiming 16 against a
kernel compiled at 4 resolves at declare and fails at issue.

`attn_paged` takes the block table and `seqused_k` as ordinary operands
(`arch/common/rad_block_attn_gated_q.h`), so the sparse attention is issued with `sel` in place
of the sequence's table and `qsa_select`'s `seqused` in place of its length. Nothing in `core/`
knows about QSA.

Declaration order is the reverse of issue order. `attn_paged` declares `sel` and `seqused` as
reads, and the buffer plan comes from declared reads, so the architecture sets them on the
attention block before `attn.declare()` and declares the indexer after it.

## Block-key storage

Block keys live **in their own `RAD_KV_FULL` group, one page per compress block, indexed by page
id**. The scorer reads `bk[page_id]` through a table the KV manager already maintains, and the
top-512 output is a list of attention page ids, which is the block table the attention reads.

A sequence's block-key pages differ from its attention pages, but both groups are paged at
`ratio`, so **block index `b` means the same in both**. Each op takes the table of the group it
serves:

| op | table it takes |
|---|---|
| `qsa_score` | the block-key group's, because it gathers block keys |
| `qsa_select` | the attention group's, because its output is what the attention reads |

Swapping them scores one sequence's keys through another's pages. A `RAD_BUF_PERSIST` array
indexed by `(slot, block)` would be simpler to declare but needs a translation in three places.

A `RAD_KV_FULL` group stores `[pages, kv_heads, block_size, 2*head_dim]`. With `kv_heads = 1`,
`block_size = ratio` and `head_dim = n / (2*ratio)` (here `128 / 8 = 16`), a page holds exactly
one block key of `n` values (`2 * head_dim * ratio == n` is asserted). A group's page size comes
from the first later-declared op with a `block_size` (`Builder::find_kv_consumer`), so the group is
declared immediately before `qsa_block_key`, whose `block_size` sizes it.

## The tail

A block key needs `ratio` consecutive raw keys; at decode three come from earlier steps. The tail
is a `RAD_KV_LINEAR` group with `state_dim` `{ratio + max_spec, head_dim}`, bound to every QSA
layer including the MTP head's. The alternatives all produce fluent wrong text:

| a tail stored as… | what goes wrong | when |
|---|---|---|
| one buffer for the whole model | layer L pools layer L-1's raw keys, because `index_qk_proj` is per layer | every decode step: three of every four keys in a completing block |
| a plane indexed by the batch row | a request that outlives an earlier one inherits its keys, because the scheduler orders a step by priority and stable-sorts decode first | any step after a concurrent request retires |
| `ratio` rows indexed `pos % ratio` | a rejected draft restarts the next step at a position this one already overwrote, so a completed block is rebuilt from a future token's key | any speculative step that straddles a block boundary and does not accept everything |

The tail is read on a later step than the one that wrote it and is keyed by layer, sequence and
position; `cu_seqlens` gives only a token's sequence within the step. A KV group slot is zeroed on
hand-out and checkpointed with the linear state, so a resume onto a half-built block reads the keys
it was built from. A `RAD_KV_WINDOW` of width `ratio` holds only the step's last four tokens, which
is the completing block at decode but not within a prefill chunk.

`qsa_work` gathers the blocks completing this step into the flat list `qsa_block_key` reads, and
`qsa_tail_store` writes this step's raw keys into the state. **`qsa_work` is issued before
`qsa_tail_store`**, because the gather reads tail keys the store overwrites. Position `p` lands at
row `p % ring`. The work list has `ceil(T / ratio) + n_seq` items; items past the real count get
`page = -1` and are skipped, so it is one launch per layer per step.

## Per-query selection (prefill)

At prefill every query in a chunk selects its own set from one shared history. Both scoring ops
take `cu_seqlens`, so `m` is a query row while the block table stays the sequence's. Each query is
bounded by its own position, `n = min(nc[seq], (pos + 1) / ratio)` (at decode, `nc[seq]`).

The attention is issued **one row per query**: `attn_paged` indexes its block table and length by
row, so `T` queries issued as `T` rows of `q_len` 1, each with its own `sel` and `seqused`, is
per-query sparse attention. Launches are chunked at `max(max_seqs, 256)` rows (the gated form takes
the whole step), because the decode kernel's split-KV scratch is sized at declare from `max_seqs`
and `q_len`.

**The partial page is decided by position.** At decode `bt[nc] == -1` on a block boundary, since
nothing is allocated past the sequence, so `bt[n] >= 0` would work there. Mid-chunk, the same
query finds a later token's live page in `bt[n]`, which `seqused` would then cover. The partial
page is appended only when `(pos + 1) % ratio != 0`.

## The sparse launch at a prefill chunk

A row a query is what the op is, not how libr4d has to run it. The split-KV decode kernel takes a
launch row in one wave, and each row gathers its own 2051 keys from the paged cache: at a
2048-row chunk that is 4.3 GB of K and V a layer and rank, about 3 ms at a ninth of the card's
bf16 WMMA rate, 12 % of a Qwen3.8-Flash-Next prefill chunk past the bound. The rows of a chunk
are consecutive queries over one history and their lists overlap: neighbours share 53 % to 73 %
of their pages, and eight consecutive rows name 1000 to 1750 distinct pages where their lists sum
to 4104.

Pinned to it (`r4d_attn_usel_gqa12_pin`, r4d_selftest), libr4d runs the same op from 256
launch rows of `q_len` 1 (lists no wider than 520 pages, the bf16 cache, twelve queries a KV
head: the model at two ranks), `attn_paged`, `attn_paged_gate_quant` or
`attn_paged_gate_had_quant` with the same operands, as **tiles of eight rows over the union of
their page lists** (`libr4d/r4d_attn_usel.h`). Nothing above libr4d changes: no operand, no
schema, no graph. It is a switch and not the rule because the form is exact to rounding and not
bit for bit, and a served model shows it ("What a served model makes of it", below).

1. `r4d_usel_build_kernel`, a workgroup a tile: the union of the tile's lists, and beside each of
   its pages the rows that list it, a bit a row. A hash table in LDS collects the pages
   (compare-and-swap) and ORs the rows' bits together, so the result does not depend on which
   thread ran first. The union is **ordered by (first row, last row, place in the first row's
   list)**: a page read by rows a to b of the tile sits beside the others read from row a on and
   until row b, so the pages one row reads lie in runs. Order and bits depend on which entries of
   the lists name the same page and on nothing else: the same lists under other page ids give
   the same bytes. A row's partial page keeps its place and the number of keys the row sees of it.
2. `r4d_attn_usel_kernel`: K and V of sixteen keys of the union are staged once in LDS for the
   tile's eight rows, a wave a row (its twelve heads, as in the decode kernel). A key scores -inf
   for a row that does not list its page, so a row's softmax is over exactly its own keys, and a
   wave whose row lists none of a tile's four pages skips the tile's matmuls: on a chunk's real
   lists a row meets 36 % to 61 % of its tile's sixteen-key tiles, where its own list alone is
   30 % to 52 % of them. It stores the decode kernel's partial (O / l in f16, m, l) as the row's
   one segment.
3. `r4d_attn_decode_usel_fb_kernel`: the decode kernel itself, for the tiles the builder hands
   back: those whose rows share too little (lists that sum to less than 3/2 of their union: rows
   of unrelated sequences, which a workgroup would walk one row at a time). A tile is served by
   one kernel or the other, never both.
4. The decode launch's merge, unchanged, with one segment a row: the output row, the gate and the
   quantiser are computed from the partial as the decode launch computes them.

On a R9700, 2048 rows of a real chunk (`r4d_selftest --perfusel`, the gated merge included):

| lists | union of 8 rows | decode kernel | tiled |
|---|---|---|---|
| BetterBench's shuffled filler, the chunk at 10 K tokens | 1219 pages | 2.81 ms | 1.59 ms |
| the same at 20 K | 1511 | 2.83 ms | 1.82 ms |
| the same at 28 K | 1656 | 2.86 ms | 1.92 ms |
| the same at 35 K | 1726 | 2.87 ms | 1.98 ms |
| this repository's documentation at 10 K | 1006 | 2.80 ms | 1.49 ms |
| the same at 22 K | 1172 | 2.82 ms | 1.59 ms |
| the same at 35 K | 1268 | 2.83 ms | 1.75 ms |
| rows of unrelated sequences, 512 pages each (handed back) | 4096 | 7.15 ms | 7.39 ms |

**What is exact and what is not.** A row attends exactly the keys of its own list, each once: the
masks are the lists. Every score is the decode kernel's bit for bit (a score is a chain of WMMAs
over the head dimension and does not depend on which keys share its tile), and `p`, the rescale
rule, the partial's format and the merge are the decode kernel's arithmetic. What differs is the
order in which a row's softmax meets its keys: the decode kernel takes them sixteen at a time in
the order of the row's list, this form in the order of the tile's union. That moves two
roundings. The reference max against which every `p = 2^(s - m_ref)` is rounded to f16 is the max
of the first sixteen keys a row meets (and of each later sixteen that outgrow it by 14 octaves),
so another first tile is another rounding of every `p`, by 2^-11; and float addition is not
associative. The two forms therefore agree to that rounding, not bit for bit: on a chunk's lists
about 13 % of the bf16 output values differ, each by one unit in the last place of a value near
the row's largest (at most 1.2 % of it), and 4 % of the int8 codes of the gated merge, each by 1.
Both forms are as close to an f64 evaluation of each row's own key set as each other, and to
libref's `attn_paged` (rel_l2 2.1e-3 either way): neither is the more exact one.

**What a workgroup loads.** A row attends its own keys, but its workgroup stages the pages of
the tile's union for all eight rows; the decode kernel loads no key past its row's count (it
clamps the key index to the row's last key). A key a row does not see gets a `p` of exactly 0,
and 0 times V is nothing only while V is finite. Not every slot of a cache is: behind the last
token of the block a sequence ends in nothing has ever been written, the memory holds whatever
it held before, which read as bf16 is as often inf or NaN as a number, and 0 * inf in one row's
V product is a NaN in every row of the workgroup that walks those sixteen keys. So the form
stages no slot the decode kernel would not have read for some row of the same tile. Beside each
page of the union the builder leaves the number of its slots that **no row of the tile sees**
(the page is some rows' partial page and no row's whole one), and the kernel stages zeros for
those slots, K and V. Every other slot is read by one of the tile's own rows under the decode
form as well, so it has been written. The kernel is launched as two instances, each returning
from the other's tiles: one that reads those counts and one that does not, because the walk
has no register to spare and the masks, compiled into it, cost every tile 5 % to 9 %. Where a
launch starts on a page and a tile's rows fill their pages (a cold chunk), only the tile at the
prompt's end takes the masking instance; where a launch starts inside a page (the first chunk
of a turn appended to a session), every tile ends in a page whose other keys belong to the next
tile's rows, and all of them do.

That is the difference the dense and the sparse kernel already have below the bound ("Below the
bound"), with the same consequence: a prefill chunk past the bound may differ in the last bit of
a hidden state from what the decode kernel gave it, and a near-tie in a later greedy pick may
fall the other way. It is deterministic: the same lists give the same bytes on every run and
under every allocation of pages. A chunk's rows are tiled from the launch's first row, so a
prefix-cache hit, which resumes where a cold prefill starts a chunk, recomputes the cold run's
own tiles.

What a served model makes of it (Qwen3.8-Flash-Next at two ranks, six fixed prompts of 4.5 K to
42 K tokens, greedy, the first eight tokens' logprobs): with every tile handed back
(the pin's limit of 1) text and logprobs are the decode launch's bit for bit. Tiled, the logprob
of a picked token moved by at most 0.15 nats. At the one position of these continuations where
three candidates share the probability (" to", " respond", " answer", 0.13 to 0.49 each) a
candidate's logprob moved by up to 0.43 nats, and the greedy pick changed in two of the six
prompts (the decode launch's two best were 0.24 and 0.41 nats apart there). Tiles of sixteen
rows (the second geometry), a second reordering of the same sums, lie as far from tiles of
eight as either lies from the decode kernel, and change the pick in two prompts too, one of them
another prompt. So the moves are what this model, with an int8 quantiser behind every sparse
attention, makes of a last-bit change in 4 % of those codes, and not an error of one order; but
they are not ties in the last digit either, which is why the form is asked for and not the rule.

Where a tile's rows share one list the form is byte-identical to the decode kernel, and likewise
below the exactness bound, where a row's list is a prefix of its successor's and the union keeps
every row's own order. `r4d_selftest` checks both, the hand-back, the independence of the page
ids, each form against an f64 evaluation of every sampled row's own key set and against libref,
and prints the share of values the two forms disagree in. Two more of its checks are there
because bounded random queries over two thousand keys cannot see a mask that is one key off:
queries that single out one key (of the row's own: both forms return that key's V, bit for bit;
one the row must not read, a later token of its own block or a page only its neighbour lists:
both return the softmax over the row's own keys), and a context that ends inside a page with
inf and NaN in the slots behind its end (no output moves with them and none is not finite,
whether the launch starts on a page or inside one). Every launch of the case finds its scratch
full of ones, as an engine's is full of another launch's bytes.

A served engine runs every sparse launch on the decode kernel: on Qwen3.8-Flash-Next at two ranks
the tiled form measured no faster (cold prefill of 2000- and 6000-word prompts within 0.5%, the
same text on those prompts). The test pin's limit of 1 keeps the builder and hands every tile
back, which is the decode kernel's bytes by another road, and the second built geometry tiles
sixteen rows instead of eight (slower: more of a larger union is read by nobody). `r4d_selftest --perfusel T CTX` times the forms on its own lists, on a real chunk's
(`R4D_USEL_SEL`, `R4D_USEL_SEQU`: dumps written with `RADIANCE_DUMP_BUF=qsa_sel:2048,qsa_sequ:2048`
and `RADIANCE_DUMP_DIR`), or on rows of unrelated sequences (`R4D_USEL_KIND=2`). The four-rank
unit (six queries a KV head) keeps the decode kernel: the form has not been run at that ratio.

## Cost

The indexer is the only part of a decode step that grows with context: `qsa_score` and
`qsa_select` scale with the number of compressed blocks, once per layer. `r4d_selftest --perfqsa
CTX [M] [iters]` times them at a real context length with no model, passing the declared width and
the live context separately (launch geometry comes from the declaration, work from `nc`).

- `qsa_score` reads the key history once per 4-row query tile, matching a decode sequence's
  `1 + n_spec` rows at depth 3; a tile straddling two sequences is processed per part.
- `qsa_select` splits the block list into chunks that each find a local top-k, exact because a
  chunk's local top-k contains every global top-k member in it. The proof and the tie rule are in
  `libr4d/r4d_qsa_select_bf16.hip`.
- Wave reductions use DPP (`row_xmask` for XOR offsets 8..1, `v_permlanex16_b32` across rows), not
  `__shfl_xor`, which lowers to `ds_bpermute_b32` and queues on the LDS pipe. Same lanes and add
  order, so bit-identical (`libr4d/r4d_common.h`, `r4d_wave_sum`).

**Any two forms of the selection kernel must produce the same bytes**, not only the same set: the
attention sums its softmax over pages in table order, and float addition is not associative.

**The score plane is the one buffer whose size is a product of the deployment's bounds** — a row
of `max_ctx / ratio` floats a query, 512 MiB at a 2048-row step and 256K of context. Under
`prefill-span`, where every step that mixes sequences fits in a MoE pass
(`RadBuildCtx::max_mixed_tok`), it is declared a pass tall instead of `max-num-batched-tokens`
rows: a wider step is one sequence, scored and selected in windows of the plane's rows, each handed
as a sequence of its own rows. Every operand a query row reads is its own (query, position,
selection), so a row is scored and selected exactly as in one pass, and a row two windows share is
written twice with the same bytes.

**The caller bounds the history.** Both ops size their grid from the declared width (the live count
is device data), so a short prompt would otherwise launch for the whole `max-model-len`. The
architecture narrows the score operand's column count to a host-side ceiling from the batch, with
the pitch unchanged. **Every term of the ceiling is generous**, because an underestimate silently
drops blocks.

## Checking

There is deliberately no switch that forces dense attention above the bound, where it is not the
model the checkpoint was trained as. libref implements all five QSA ops (`ref_qsa_block_key` in
`libref/ref_attn.cpp`; `ref_qsa_score`, `ref_qsa_select`, `ref_qsa_tail_store` and `ref_qsa_work`
in `libref/ref_qsa.cpp`), so each device kernel has a row to compare against. No oracle gives the
whole-subsystem comparison, a dense run against a sparse one above the bound. Changes are checked
with:

- **`r4d_selftest` oracles**: `qsa_block_key` and `qsa_score` against f64 evaluations,
  `qsa_work` and `qsa_tail_store` against a host walk of the batch, `qsa_select` against a host
  sort, a prefill case for the per-query join, the wide-launch form against the single-workgroup
  form, and split against unsplit at bit identity on mostly-tied scores, so the tie rule decides;
- **`RADIANCE_DEBUG_QSA`**: prints `nc`, `nsel`, `seqused` and the head of the table for the first
  few steps. `nc` above `topk` is intended dropping, not a divergence. It synchronises the stream.
- **`RADIANCE_DEBUG_QSA_EAGER`**: every step scores and selects per layer, the issue sequence
  before "Below the bound" existed (`tests/arch_test.cpp`'s pinned step hashes are the old ones
  under it). Greedy text with and without it must be byte-identical, drafts included: at or below
  the bound the two differ only in what is launched, never in a byte the attention reads.

**Count the reach, not the prompt.** The bound is on the context each query sees, so a prompt a
few tokens under it crosses the bound during decode and is expected to start differing there.
