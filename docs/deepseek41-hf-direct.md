# DeepSeek-V4.1-Flash: a .rad straight from the release (no GGUF converter)

The official DeepSeek-V4.1-Flash release converts with `rad-convert` alone: no llama.cpp GGUF is
built as an intermediate, and no separate drafter directory is grafted. The pieces are in the tree:

- `data/engram_tables.py` builds the Engram hash tables from the release's own `tokenizer.json` and
  `config.json`, reimplementing the tokenizer's ByteLevel decode and the normaliser sequence
  (NFKC, NFD, strip marks, lowercase, collapse spaces, sentinel) and the PCG64 multipliers and
  prime buckets. Validated against HF `tokenizers`: 0 mismatched token ids of 129280, compressed
  vocab 99092, the release's two Engram layers [1, 14]. It writes `engram-tables.safetensors`.
- `data/imatrix_merge.py` fuses the imatrix entries llama.cpp's converter writes split
  (`ffn_gate_exps`, `ffn_up_exps`) into the plugin's fused `ffn_gate_up_*`, summing the two halves
  (and `shexp`): 1184 entries carried through, 160 fused for 40 layers. IQ1_S refuses a fused
  weight without one.
- `data/recipes/deepseek41-ud-iq1s.recipe` is the UD-IQ1_S per-tensor mix, as rules for libquant's
  `ggml` quantiser, so the release converts to the same encodings the GGUF gives.
- `rad-convert --chat-template FILE` bakes a template into the container's vocab section. The
  release carries no `chat_template.jinja` and its `tokenizer_config.json` names none, so without
  this flag an HF-direct container answers `/v1/chat/completions` with 501. Trailing newlines are
  stripped as for a sibling template, and an unreadable or empty file is refused, not skipped.

## The three steps

```
# 1. the Engram tables, from the release's tokenizer (not a GGUF)
python3 data/engram_tables.py <HF release dir> engram-tables.safetensors

# 2. fuse the imatrix's split gate/up entries
python3 data/imatrix_merge.py imatrix_unsloth.gguf_file imatrix-merged.gguf

# 3. convert; the template is the model's own, baked into the container
rad-convert <HF release dir> --add engram-tables.safetensors \
    --recipe data/recipes/deepseek41-ud-iq1s.recipe \
    --imatrix imatrix-merged.gguf --tokenizer <HF release dir>/tokenizer.json \
    --chat-template chat_template.jinja -o deepseek-v4.1-flash-ud-iq1s.rad
```

Measured on the release (DeepSeek-V4.1-Flash-1017, fp8 with e8m0 per 32×32 blocks, MXFP4 experts):
plan 32701 weights / **251.07 GiB**, conversion **251.13 GiB in 2h11m** on this host, and the
container serves on 4× R9700 at TP4 with its DSpark metadata (block 5, 128 experts, targets
37/38/39) and Engram baked. Quality of the converted container, greedy, thinking off: HE 154/164 =
.939 (the native UD-IQ1_S reference .963, McNemar p = .125), GSM 194/200 = .970 (reference .960).

Serving: nothing else is needed now that the template is baked. A container built before this flag
needed `--override-chat-template FILE` at serve time; the file has to be a Radiance-renderable
template (the release's own is; llama.cpp-only jinja such as `dict()` in a GGUF's template is not).
