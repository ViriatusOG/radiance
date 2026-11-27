# Flint on radiance

Recursive Flint (39.8B total / ~2.5B active): a 49-layer decoder with 2 dense layers followed by
47 sigmoid-routed MoE layers. The architecture plugin is `arch/flint`; the containers are built
from the BF16 checkpoint with `rad-convert`.

## Status

- **Q4** (`flint-q4.rad`, 19.72 GiB, fits one card): u4 attention, int4 experts, plain int8
  dense/shared/head. Served on this build by the same forms as `hyv3_bf16`'s W4 and W8 recipes.
- **IQ8R** (`flint-iq8r.rad`, about 38 GiB, two cards): an IQ8R trunk and rotated int8 experts,
  the forms `qwen35moe_w8a8` serves Qwen3.6-35B-A3B with. It replaces the plain int8 "Q8"
  container of the 1.17-era port, whose unrotated int8 experts this build does not serve (a
  container converted that way is refused by name; convert it again with the IQ8R recipe).

The measurements below were taken on the 1.17-era build with the forms it used then (the plain
int8 Q8 container and that build's plain int8 grouped GEMM); they have not been repeated on this
build. See [Measured](#measured).

## Geometry

| | |
|---|---|
| layers | 49: 2 dense, 47 MoE |
| attention | 32 query / 4 KV heads, head_dim 128, QK RMSNorm per head |
| sink | learned per-head `softmax_offset` on every layer (`attn_paged_sink`) |
| position | RoPE on the 35 sliding layers; NoPE on the 14 full layers |
| window | 513 on the sliding layers |
| FFN | dense `n_ff` 6144; MoE 256 experts, top-6, expert width 512, shared 1024, sigmoid routing, `norm_topk_prob` |
| vocab | 150144, untied head |
| KV | two groups: full (14 layers) and window 513 (35 layers) |

The full layers are `0, 1, 5, 9, ..., 45, 48`; the rest are sliding and carry RoPE.

## Containers and recipes

| file | contents | size |
|---|---|---|
| `flint-q4.rad` | U4 attention, I4 experts, INT8 dense/shared/head, BF16 router/norms/embed | 19.72 GiB |
| `flint-iq8r.rad` | IQ8R attention/dense/shared gate-up, rotated INT8 experts and shared down, INT8 head, BF16 router/norms/sinks/embed | about 38 GiB (not re-measured) |

Recipes: `data/recipes/flint-bf16-q4.recipe` and `data/recipes/flint-iq8r.recipe`.

The plugin reads which form a container holds from layer 0's q projection:

- **Rotated int8 trunk** (IQ8R, or the 128-group rotated int8 form): the norms write
  `rmsnorm_had_quant_i8g`'s codes (`ar_rmsnorm_had_quant_i8g` with the fused all-reduce at TP2),
  the attention output `had_quant_act_i8g`'s and the gated products `gated_had_quant_i8g`'s, and
  every linear reads them at `i8a8h` without a quantiser of its own. The routed experts must be
  stored rotated (int8, or a rotated four-bit form with an int8 row); the grouped GEMM
  (`moe_gemm_i8a8`) reads the block input's own codes for gate/up and the gated product's, at one
  scale a 128, for down.
- **Any other trunk** (bf16, the Q4 recipe's u4 attention, plain int8 linears): each linear reads
  the bf16 plane or quantises it itself (`quant_act_i8g` for a plain int8 weight), and four-bit
  experts read the block input's E4M3 pair (`moe_gemm_w4a8`).

A rotated trunk over unrotated experts, or rotated int8 experts under an unrotated trunk, is
refused at declare with the recipe to use.

Two non-obvious inputs:

- `data/flint-zero-bias.safetensors`: Flint has no router correction bias. The sigmoid router's
  bias is selection-only, so the converter merges 47 x `[256]` zero tensors under
  `model.layers.N.mlp.expert_bias`.
- `data/flint-chat-template.jinja`: the checkpoint's template contains `\x00`, which the Jinja
  parser rejects. This is the same template without that replace; pass it with `--chat-template`,
  or `/v1/chat/completions` answers 501 for want of a template.

## Convert

```bash
export RADIANCE_HOME=$PWD/build/radiance_home

# Q4
./build/bin/rad-convert "$HOME/models/flint" \
  --add data/flint-zero-bias.safetensors \
  --recipe data/recipes/flint-bf16-q4.recipe \
  --chat-template data/flint-chat-template.jinja \
  --report 20 --gpus 0,1 -o "$HOME/models/rad/flint-q4.rad"

# IQ8R
./build/bin/rad-convert "$HOME/models/flint" \
  --add data/flint-zero-bias.safetensors \
  --recipe data/recipes/flint-iq8r.recipe \
  --chat-template data/flint-chat-template.jinja \
  --report 20 --gpus 0,1 -o "$HOME/models/rad/flint-iq8r.rad"
```

`--plan-only` is the cheap pre-flight: it runs declare and resolves every op against the loaded
kernels before reading the 75 GiB checkpoint.

## Serve

```bash
# Q4, one card
./build/bin/radiance --model "$HOME/models/rad/flint-q4.rad" \
  --radiance-home build/radiance_home --max-model-len 262144 --port 8090

# Q4, two cards (long context, expert parallel)
./build/bin/radiance --model "$HOME/models/rad/flint-q4.rad" --tp 2 \
  --radiance-home build/radiance_home --max-model-len 262144 --port 8090

# Q4 with the int8 KV cache (about half the cache memory)
./build/bin/radiance --model "$HOME/models/rad/flint-q4.rad" \
  --radiance-home build/radiance_home --max-model-len 262144 \
  --kv-cache-dtype i8 --port 8090

# IQ8R needs both cards: about 38 GiB of weights
./build/bin/radiance --model "$HOME/models/rad/flint-iq8r.rad" --tp 2 \
  --radiance-home build/radiance_home --max-model-len 131072 --port 8090
```

The KV cache is bf16 by default, or with `--kv-cache-dtype i8` the int8 cache: K and V as int8
codes with one f16 scale each per token and head (`2 * head_dim + 4` bytes a token and head).
Both KV groups, the full one and the 513-token window, take it, and the sink attention, the store
and the fused prologue all have rows for both. No other cache dtype is served.

## Measured

Taken on the 1.17-era build, before this port: the "Q8" columns are the plain int8 container
(unrotated int8 linears and experts, served by that build's plain int8 grouped GEMM), which this
build no longer serves; the IQ8R container has not been measured. The Q4 container is the same
recipe, served now by this build's kernels, and has not been re-measured either.

Two R9700s, greedy, warm, prefix cache disabled. Q4 and Q8 both at TP2. The payloads are
1,325 / 8,928 / 26,073 tokens. Tokenizers differ at 26K, so the payload is not the same token
count on both sides.

| context | Q4 prefill | Q4 decode | Q8 prefill | Q8 decode |
|---|---|---|---|---|
| 1.3K | 14,488 | 161.4 | 8,036 | 112.5 |
| 9K | 14,019 | 154.5 | 8,232 | 109.8 |
| 26K | 11,850 | 136.6 | 7,621 | 100.4 |

Reference on the same box, llama-server Q8 (fresh server, `-ngl 999 --jinja`) and Q4 (the
release's GGUF): about 2,364 / 5,359 / ~5,028 tok/s prefill for Q8 at 1.3K / 9K / 26K, ~65-70
tok/s decode, and 2,197 / 5,400 / ~3,290 tok/s prefill for Q4 with 77.7 tok/s decode at 1.3K.
llama.cpp's numbers move with its build and flags; re-measure beside the run being reported.

## Kernels

- **Experts.** IQ8R: `moe_gemm_q` at dtype `i8a8h` (`moe_gemm_i8a8`), rotated int8 weights at a
  bf16 scale a (row, 128 of K) against the rotated int8 activation. Q4: `moe_gemm_q` at `w4a8`
  (`moe_gemm_w4a8`) against the E4M3 activation. Expert parallel at TP2: each rank holds 128
  whole experts a layer. On the 1.17-era build the Q4 w4a8 oracle reported the rows past
  `off[n_expert]` as a mismatch (the kernel zeros the dropped tail, the reference computes it);
  every live row agreed, and the first bad offset was `live * N`.
- **Sink attention.** `attn_paged_sink` at head_dim 128, gqa 8, on the bf16 or the int8 cache.
  A prefill-shaped launch takes the WMMA kernel with the sink folded in: the sink seeds the running
  max and one lane's denominator in the log2 domain, and the final reduction adds both lanes.
  Decode and verify launches take the scalar kernel with split-KV, whose unsplit store and merge
  fold the sink.
- **Residual and all-reduce.** `NormQuant` folds the previous block's delta and its two-rank
  all-reduce into the norm (`fuse_ar`), so only the embedding's reduce and the last block's
  remain. Flint also sets `shared_fp32_combine` 0: the gather adds the local routed and shared
  partials and one collective covers both, which is the TP1 arithmetic. The exact cross-rank
  wire is the default; under `--tp-wire wht6` both trunk forms' norms have a 6-bit fused form
  (`ar_rmsnorm_had_quant_i8g` and `ar_rmsnorm_quant_fp8` at `wire` 6).
- **NoPE prologue.** At decode heights the 14 full layers use `qkv_norm_rope_kv_store` with
  `rotary_dim` 0: the kernel copies the normed head through the rotary stage, so it is the norms
  and the store alone. Prefill heights take the standalone norms and `kv_store`, without a
  rope on those layers.

## Correctness

On the 1.17-era build (not repeated on this one):

- Needle retrieval at 8K and 32K returned `ZX-417-QQ` exactly, with BF16 and E4M3 KV, including
  after the sink-aware WMMA prefill.
- A `get_weather` tool call parsed as `tool_calls` with `{"city":"Paris"}`; reasoning uses
  `<think>...</think>`.
- A 250,696-token prompt completed on two cards. On one card, Q4 at 262K fit with that build's
  E4M3 KV cache: 19.37 GiB of weights plus an 11.45 GiB cache. The int8 cache of this build takes
  a few percent more a token (its two f16 scales a token and head).

On this build, `tests/flint_arch_test.cpp` checks the declared graph and the step of both
containers at one and two ranks on both caches, and `tests/r4d_emu/test_attn.cpp` the sink WMMA
prefill on the bf16 and the int8 cache.

## Known limitations

- **Decode attention.** The scalar sink kernel is the short-context ceiling: ~184 us for a
  513-window layer at 1.3K, against ~15 us of KV traffic. A flash-decoding decode form is the
  follow-up.
- **No draft head.** Flint has no MTP head. Prompt-lookup speculation is available at serve time
  and is workload-dependent.
