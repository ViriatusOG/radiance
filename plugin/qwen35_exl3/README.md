# qwen35_exl3 -- the Qwen3.5 / 3.6 / 3.8 dense architecture with EXL3-V3 weights

One codebook for the exl3i weights: **mad**.

The radiance architecture plugin for the dense text models of the Qwen3.5 / 3.6 / 3.8 family whose linear layers are
EXL3-V3 weights (plugin `exl3i`). Radiance at plugin ABI 44. It declares the model's graph (the names map, the
blocks, the ops) and does not run kernels itself; the linears are `exl3i_dense` and the token table is `embed_exl3i`.

## What it serves

- **The trunk**: the decoder layers of the family: gated attention layers, gated delta-net (GDN, linear-attention)
  layers, the gated MLP, rmsnorm, rope. The blocks are `rad_block_attn_gated_exl3.h`, `rad_block_gdn_exl3.h`,
  `rad_block_mlp_exl3.h`.
- **The linears**: each EXL3-V3 weight is served through `exl3i_dense` at the rate its container carries (any rate in
  1/8 steps), with codebook mad (the one codebook), and the activation mode of its `exl3.act.<weight>` key (f16, a8t, a8g<N>).
  `exl3_dense_op.h` checks each weight's encoding and picks its mode; a mode no kernel serves fails the load by name.
- **The token embedding**: `token_embd.weight` as an exl3i trellis table (`embed_exl3i`, codebook mad, rows
  natural or R16), as a table-quantised fwht128 table (`embed_tq`), or as a plain bf16 table.
- **The head**: `output.weight` as an exl3i head in R16 row order (the rows of the head are stored permuted), and the
  drafter's candidates rescored against it by `exl3i_rerank` (on one rank only).
- **The drafter**: DFlash2 when the container carries a `draft.` block (`arch/common/rad_block_dflash2.h`), fed by the
  trunk's taps. Its logits are rescored against the trunk's head, or against the recipe's `dflash.rerank_head`.
- **The decode step**: small ops fused where a kernel serves them (`exl3_fuse.h`: `addnorm`, `gdn`, `qk`, `gate`). The A8
  activation quantiser (AQ) feeds the linears its rows directly, also in the decode band for the mad GEMV. The test
  compares each with the chain it replaces through test pins (`Exl3TestPins`, `exl3_dense_op.h`) that nothing in a
  served engine sets.

Not served: the V1 / V2 `exl3` containers (no importer for them in this tree).

**Tensor parallel (TP2 / TP4).** The trunk's exl3i weights serve sharded: the loader cuts each plane in its own
elements (the record by 16-row groups, suh / suh.1 along K, svh along the rows), so the rank owns its rows of a
row-parallel projection (q|k|v, z, gate|up, qg) or its K of a column-parallel one (o, ssm_out, down); the row-parallel
outputs carry an `all_reduce`. The fused GDN a | b (`ssm_ab`) serves whole on every rank -- its a / b rows are not a
slice of the trellis record -- and the block reads the rank's own value heads' gate and beta as column views of the
full output. The R16 head (the rows stored permuted) likewise serves whole on every rank, over the whole vocabulary
(the engine reads rank 0); a natural-order exl3i head row-shards with the vocabulary. The trellis token table is read
whole on every rank, as at one rank. `exl3i_rerank` (the drafter's rescoring against the trunk's head) stays at one
rank.

## Files

- `qwen35_exl3.cpp`: the architecture: the names map, the declaration of the graph, the step's ops.
- `exl3_dense_op.h`: the per-linear op (the encoding check, the act modes, the shared row buffers for AQ).
- `exl3_fuse.h`: the step fusions and their switch.
- `rad_block_*_exl3.h`: the attention, GDN and MLP blocks.
- `qwen35_exl3.h`: the plugin's declarations and static structures. `RADIANCE_VERSION`: the radiance version it targets (2.2.0).

## Build and test

```
cmake -S plugin/qwen35_exl3 -B build-qwen35 -G Ninja -DCMAKE_PREFIX_PATH=<radiance install, plugin ABI 44>
cmake --build build-qwen35 -j
ctest --test-dir build-qwen35 -R qwen35_exl3_test --output-on-failure
```

The test (`tests/qwen35_exl3_test.cpp`) is host-only: it declares the architecture on fake containers (a mixed
container whose weights answer exl3i encodings) and checks the refusals (an unknown act value, a shared-input group
given two act modes, a 256-row ssm_ab) and the tensor-parallel split: the exl3i planes declared at the rank's extents
(TP2, row and column shards), the whole container at TP2 / TP4 (sharded trunk, whole R16 head with the whole-vocab
logits, whole ssm_ab and trellis table, one all_reduce a row-parallel output, the step lint clean at both edge
ranks). It needs no GPU; the device rows are checked by the exl3i tests (which cover the sliced planes through the
layout hooks and a launch).

## Numbers

The step-pass measurements (what was fused, the decode step by family) are in rad-exl3-mix under
`docs/plugins/qwen35_exl3/PERF.md`.
