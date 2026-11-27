# exl3quant -- the `exl3i` quantiser plugin (plugin ABI 44)

One codebook: **mad**.

The quantiser the `qwen35_exl3` architecture's recipes call for EXL3-V3 weights. It does not quantise: it IMPORTS.
The EXL3-V3 quantizer (rad-exl3-mix `quantize/ourq.py`) writes a level: one directory of fractional-rate trellises in
exllamav3's layout, plus ourq's `stamp.json`. This plugin reads a level from `src=DIR` into the `exl3i` encoding
(`../exl3import/include/exl3i_encoding.h`), bit for bit. What it adds over the source is the container: the name
map, the rate, the planes, the stacked gate_up's suh.1, the R16 row order of the head, and the decode hook the oracle
(and rad-kbench) uses.

Containers of the older `exl3` export have no importer in this tree.

## `rad_quant_at(0)`: `exl3i` (EXL3-V3)

Recipe form: `blk.*.ffn_down.weight exl3i src=$LEVEL codebook=mad [rate=2.5]`.

| option     | meaning |
|------------|---------|
| `src`      | the level directory: ourq's output (or any exllamav3 layout): `config.json`, `model.safetensors.index.json` and the safetensors shards. Required. |
| `codebook` | `mad` (IQR's computed code on the int8 grid): the one codebook. Required: the trellis does not say which; any other name is refused. |
| `rate`     | K as text (`2.25`, `0.5`, `4`). Absent, K is the trellis' own. Given, every imported trellis must be at this rate. |
| `rows`     | `natural` (default) or `r16`: the rows were quantized in R16 order (the head; `ourq --head-perm`), and stay so. |

`codebook` is checked against the level's `stamp.json` when there is one: its last run's `args.codebook` must name
the same codebook (`mad`, or `hash` with the exl3i constants, = mad). Any other value has no codebook served
here and is refused with `RAD_E_DTYPE`. A codebook name the plugin does not know
at all is refused with `RAD_E_INVAL`.

The rate is the trellis' own: its last dimension is 16 K, and `rate=` checks it.

## which names

The `qwen35_exl3` architecture's (this tree) container names, mapped to the checkpoint's:

| container | source |
|-----------|--------|
| `blk.<L>.attn_qg.weight` | `self_attn.q_proj` (query and gate, one tensor) |
| `blk.<L>.attn_k.weight` / `attn_v` / `attn_output` | `self_attn.k_proj` / `v_proj` / `o_proj` |
| `blk.<L>.ffn_gate.weight` / `ffn_up` / `ffn_down` | `mlp.gate_proj` / `up_proj` / `down_proj` |
| `blk.<L>.ssm_qkv.weight` / `ssm_z` / `ssm_out` | `linear_attn.in_proj_qkv` / `in_proj_z` / `out_proj` |
| `blk.<L>.ssm_ab.weight` | `linear_attn.in_proj_ab`: the GDN's a \| b \| padding to 128 rows as one trellis (`qwen35_exl3` serves it in the qkv / z group) |
| `output.weight` | `lm_head` (the embed fallback stays plain) |
| `token_embd.weight` | `model.language_model.embed_tokens`, a trellis token table (served by the exl3i plugin's embedding gather from the stored record) |

The `gemma4_exl3` architecture (Gemma 4 dense, same checkpoint layer prefix `model.language_model.layers.<L>`)
uses the same table where the names meet, and these two beside it:

| container | source |
|-----------|--------|
| `blk.<L>.attn_q.weight` / `attn_k` / `attn_v` / `attn_o` | `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj` |
| `blk.<L>.ffn_gate_up.weight` (stacked) / `ffn_down` | `mlp.gate_proj` \| `up_proj` (gate rows first, up second, suh and suh.1) / `mlp.down_proj` |

A Gemma 4 global layer reads V from K and has no `v_proj`: no `blk.<L>.attn_v.weight` there. Embedding, head,
norms, the assistant head and the vision tower are not EXL3 weights and stay on their stock paths.

`blk.<L>.ssm_in.weight` is REFUSED by name: this model's fused q\|k\|v\|a\|b is 10416 rows, not in 128s, so the
architecture serves `ssm_qkv` (exl3) beside plain bf16 `ssm_a` / `ssm_b`. A layer may hold its a / b as one trellis
instead (`ssm_ab`, recipe `blk.*.ssm_ab.weight exl3i src=$AB codebook=mad rate=4`): rows a, b, then a's first rows
again up to 128. The MoE forms (`ffn_*_exps`) are kept for the `qwen4exp` architecture. A name that maps to
nothing declines with `RAD_E_UNSUPPORTED` and the next recipe rule decides it.

## what it checks, and what it refuses

The import is exact, and everything that is not an exl3i projection is said so: a rate the format does not have
(`RAD_E_FORMAT`: m even, 2..128), a codebook the marker or the stamp does not name, a trellis whose tile count is not
the weight's, a `rate=` that disagrees with the trellis, a stacked weight whose two halves are not one rate, and a
`.bias` on a projection. Decode (the oracle path) is the exl3i decoder, `exl3i_decode_rows`.

## build

```
cmake -S plugin/exl3quant -B build -G Ninja -DCMAKE_PREFIX_PATH=/path/to/radiance-install
cmake --build build -j4 && ctest --test-dir build --output-on-failure
```

The test is self-contained: it builds an ourq-style level in a temp directory (fractional trellises at several rates,
a stamp, a head level in R16 order) and drives the plugin through its ABI the way rad-convert does --
`encoding` / `row_block` / `quantize` / `decode` -- holding it to the exact import (the codes plane is the source
trellis through the layout, suh/svh verbatim, the decode equal to the oracle's on the same planes) and to every
refusal's status.

`exl3_decode.c`'s and `exl3i_decode.c`'s orderings are bit-exact: the build compiles them (and the test links them)
with `-ffp-contract=off`, or an FMA that fused a multiply into a Hadamard sum would move the oracle.
