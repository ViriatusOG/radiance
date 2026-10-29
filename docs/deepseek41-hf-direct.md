# DeepSeek-V4.1-Flash: a .rad from the release (no GGUF converter)

The official DeepSeek-V4.1-Flash release converts with `rad-convert` alone: no llama.cpp GGUF is
built as an intermediate, and no separate drafter directory is grafted. The pieces below prepare
and convert the release. Serving the container needs the `deepseek41` architecture plugin, which
this tree does not include yet.

- `data/engram_tables.py` builds the Engram hash tables from the release's own `tokenizer.json` and
  `config.json`. It reimplements the tokenizer's ByteLevel decode, the normaliser sequence (NFKC,
  NFD, strip marks, lowercase, collapse spaces, sentinel), the PCG64 multipliers and the prime
  buckets. Against HF `tokenizers`: 0 mismatched token ids of 129280, compressed vocab 99092, the
  release's two Engram layers [1, 14]. It writes `engram-tables.safetensors`.
- `data/imatrix_merge.py` fuses the imatrix entries llama.cpp's converter writes split
  (`ffn_gate_exps`, `ffn_up_exps`) into the plugin's fused `ffn_gate_up_*`, summing the two halves
  (and those of `shexp`): 1184 entries carried through, 160 fused for 40 layers. IQ1_S refuses a
  fused weight without one.
- `data/recipes/deepseek41-ud-iq1s.recipe` is the UD-IQ1_S per-tensor mix as rules for libquant's
  `ggml` quantiser, so the release converts to the same encodings the GGUF has.
- `rad-convert --chat-template FILE` bakes a template into the container's vocab section. The
  release carries no `chat_template.jinja` and its `tokenizer_config.json` names none, so a
  container converted without this flag answers `/v1/chat/completions` with 501. Trailing newlines
  are stripped as for a sibling template, and an unreadable or empty file is refused.

## Conversion

```
# 1. the Engram tables, from the release's tokenizer (not a GGUF)
python3 data/engram_tables.py <HF release dir> engram-tables.safetensors

# 2. fuse the imatrix's split gate/up entries
python3 data/imatrix_merge.py imatrix_unsloth.gguf_file imatrix-merged.gguf

# 3. convert, with a chat template for the model baked into the container
rad-convert <HF release dir> --add engram-tables.safetensors \
    --recipe data/recipes/deepseek41-ud-iq1s.recipe \
    --imatrix imatrix-merged.gguf --tokenizer <HF release dir>/tokenizer.json \
    --chat-template chat_template.jinja -o deepseek-v4.1-flash-ud-iq1s.rad
```

From DeepSeek-V4.1-Flash-1017 (fp8 with e8m0 scales per 32x32 block, MXFP4 experts) the plan is
32701 weights, 251.07 GiB, and the container 251.13 GiB. The `deepseek41` plugin serves it on 4x
R9700 at `--tp 4` with its DSpark metadata (block 5, 128 experts, targets 37/38/39) and the Engram
tables baked in. Quality, greedy, thinking off: HumanEval 154/164 = 0.939 (the native UD-IQ1_S
reference 0.963, McNemar p = 0.125), GSM 194/200 = 0.970 (reference 0.960).

## Serving

A container converted with `--chat-template` needs no template at serve time. One converted
without it needs `--override-chat-template FILE`; the file must be a template radiance renders
(the release's own is; llama.cpp-only Jinja, such as `dict()` in a GGUF's template, is not).
