# gemma4_exl3 -- Gemma 4 dense with EXL3-V3 trunk linears

One codebook for the exl3i weights: **mad**.

The radiance architecture plugin for the dense models of the Gemma 4 family (31B, 12B, E4B) whose trunk
linears are EXL3-V3 trellis weights (plugin `exl3i`). Radiance at plugin ABI 44. It declares
the model's graph and does not run kernels itself: a container's EXL3-V3 trunk linear is the
`exl3i_dense` op through `plugin/qwen35_exl3/exl3_dense_op.h`, and everything else is
`gemma4_bf16`'s path. It registers as `(gemma4, exl3)`; a checkpoint with experts
(Gemma 4 26B-A4B) is refused by its expert count. `rad-convert` packs a Gemma 4 checkpoint into the
container the way `plugin/exl3quant/README.md`'s Gemma 4 rows map its names (`self_attn.q_proj`,
`self_attn.o_proj`).

## What it serves

- **The trunk linears**: each EXL3-V3 weight serves through `exl3i_dense` at the rate its container
  carries (any rate in 1/8 steps), codebook mad (the one codebook), and the activation mode of its
  `exl3.act.<weight>` key (f16, a8t, a8g<N>); `exl3_dense_op.h` checks each weight's encoding and
  picks its mode, and a mode no kernel serves fails the load by name. A weight the container holds
  plain serves the stock path: a mixed container (exl3i in some layers, bf16 in the rest) declares
  and serves.
- **The junctions**: the residual junctions stay, writing beside their int8 pair the bf16 norm row
  the EXL3 linears read (`into.in`). The pair is one scratch, `junction.q8` / `junction.s8`, shared
  by every EXL3 junction: its codes are written and never read, so the fused junction costs a row of
  bytes a token against the unfused chain's five dispatches between two blocks. The prefill
  overlap's slices write the bf16 row too: at tensor parallelism above 1, a slice without it would
  serve a stale row.
- **Tensor parallelism 2 and 4**: q, k, v and `ffn_gate_up` split their rows, `attn_o` and
  `ffn_down` their columns, as in `gemma4_bf16`, each plane declared at the rank's extents. A KV
  head replicated on every rank (Gemma 4 12B's one global KV head) serves whole on every rank; rows
  that are not an even share are refused at declare, by name.
- **The token table**: the checkpoint's bf16 (`embed_lookup`, sharded by vocabulary rows), or
  quantised in one of the two forms `qwen35_exl3` serves: codebook-quantised under fwht128
  (`embed_tq`; recipe `token_embd.weight rtn table=w4nl group=128 scale=bf16 transform=fwht128
  rule=search`, any 2^B values up to 8 bits) or an exl3i trellis (`embed_exl3i`; recipe
  `token_embd.weight exl3i src=<level> codebook=mad`). Every rank holds a quantised table whole.
  Neither gather scales its row, so rank 0 gathers and then multiplies by one row filled with
  bf16(sqrt(n_embd)) (`fill`, `mul`): the bits `embed_lookup` gives with that `mult`. The other
  ranks gather too and then write zeros over their rows, and the all-reduce stays, so the media
  rows that rank 0 alone holds still sum in exactly. They gather so that every rank reads the
  table, as every rank reads its shard of a bf16 one: the placement finds a table by its readers in
  each rank's program. Zeros, not the rows times 0, since a table row need not be finite. The
  assistant head embeds through the same table on every rank, with no all-reduce. A DFlash
  drafter embeds through the bf16 table, so a quantised one is refused under it, by name. The
  placement pins in host memory only a table that a token gather alone reads
  (`is_embedding_table`, which counts `embed_tq` and `embed_exl3i` as token gathers), so under
  `--embedding-placement ram` each rank pins every plane of a quantised table. The E-series'
  per-layer table (`ple_token_embd`, n_layer x n_ple wide) may be quantised the same two ways.
  Every rank gathers its whole rows, so they need no all-reduce, and they are scaled by sqrt(n_ple)
  (16 at 256) with a `fill` and a `mul`, as the token table is.
- **The stock rest**: the vision tower, the E2B/E4B audio encoder, the 12B's embedders and the
  three drafters (the assistant head, DFlash, DFlash2) are `gemma4_bf16`'s, but for the token table
  above; a step's decode rows chain as there, the EXL3 linears keeping the barrier (plugin/exl3i has
  no chained forms). A DFlash drafter's taps read the rows the junctions leave whole.
- **Multi-instance**: `RAD_ARCH_MULTI_INSTANCE`, as `gemma4_bf16`: every per-instance state is
  indexed by slot, so two engines in one process serve their own.

## The copy

`gemma4_exl3.cpp` is a copy of `arch/gemma4_bf16/gemma4_bf16.cpp`. A plugin compiles against the
installed `abi/` headers alone and cannot include the engine's sources: the copy is the one way the
trunk serves EXL3. The file's header names the location; the difference against the stock file is
the G4Linear branch, the junction path, the token table (`G4TokTable`, `G4EmbedScale`) and the
registration --

```
diff -u arch/gemma4_bf16/gemma4_bf16.cpp plugin/gemma4_exl3/gemma4_exl3.cpp
```

is the porting surface: a fix that lands in the stock file is ported here (the diff then shows it
beside the hunks this plugin owns), and nothing else of the copy should differ from the stock
file.

## Build and test

```
cmake -S plugin/gemma4_exl3 -B build-gemma4-exl3 -G Ninja -DCMAKE_PREFIX_PATH=<radiance install, plugin ABI 44>
cmake --build build-gemma4-exl3 -j
ctest --test-dir build-gemma4-exl3 --output-on-failure
```

The test (`tests/gemma4_exl3_test.cpp`) is host-only: it declares the architecture on fake
containers -- an EXL3-V3 container, where the fakes of `tests/arch_fakes.h` answer exl3i encodings
for the weights the container carries -- and checks the declared graph without a card: every trunk
linear an `exl3i_dense` op fed by the junctions, each rank's shares at tp 2 and 4, a KV head no
rank can hold refused, the 12B's whole global KV head, the overlap slices' bf16 row, the act modes,
the step order, a mixed container, two engines in one process on their own slots, the token
table (a quantised one gathered on every rank, each plane read by the gather alone, and scaled on
rank 0, zeros on the others; the assistant head's gather; the forms the gathers do not serve
refused), and Gemma 4 E4B: its KV-shared layers, tp 2, and its per-layer table quantised (gathered
whole on every rank, scaled by 16, no all-reduce) or refused. Two DFlash cases declare a drafter
beside the EXL3 trunk linears (the 31B shape at block 16, and Bolt-E4B's at block 8 on E4B at tp 1
and 2): each tap is written once a decode step, and the context and query passes run the drafter
alone, linted with the step. Gemma 4 E2B's geometry (one KV head, a double-wide MLP on the
KV-shared layers) declares at tp 1, 2, 4 and 8, every EXL3 width a multiple of 128, and tp 3 is
refused.

The second test (`tests/gemma4_exl3_embed_test.cpp`) runs the token-table path in numbers: a tq
table (4 and 5 bits) and an exl3i one (natural and R16 rows) are gathered by their host rows
(`plugin/exl3i`, linked in) and scaled as the plugin declares it, and must equal `embed_lookup` with
the same `mult` over the table the gather decodes, in every bit, for one token and a short batch
with a padding id; the same at E4B's per-layer width (10752, scale 16) for 4, 5 and 8 bits. A last
case gathers from a table that holds inf and NaN and checks that a rank past 0 writes +0 over its
rows, where the rows times 0 would be NaN. `fill`, `mul` and `embed_lookup` are libref's, loaded
from `$RADIANCE_HOME/kernels` (CMake sets it); without libref the test is skipped (exit 77).
