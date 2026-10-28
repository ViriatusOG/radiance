<!-- Machine-drafted by Claude (Opus 5.5, then Fable 5.1) on 2026-10-27 and 2026-10-28. Decision by the user. -->

# ADR 0001: clef's joint head runs on the GPU, inside the qwen35 plugins

Status: accepted 2026-10-27, built 2026-10-28.

## Context

Issue #166 asks radiance to serve clef and clef-flash on `/v1/systemone` with no Python sidecar. clef's head reads the final hidden state of every prompt token, not one position as d1 does. Two places fit: the GPU, inside `qwen35_bf16` and `qwen35_fp8`, or the CPU, in the server.

On threadripper, fp8 clef-flash with the head in a PyTorch sidecar served 1.43 respondents/s on 13k-token guidance prompts and 9.16 on 676-token prompts. The PyTorch head took 8.6 ms of the 95 ms short-prompt time (`bench/timings.md`). A CPU head was estimated at 60-120 ms on a short prompt and 0.4-0.9 s on a 13k-token cache miss (inferred, not measured).

## Decision

The head runs on the GPU, in the plugin step (`arch/common/rad_block_clef.h`), with these parts:

1. Every step, the final norm's rows go through the head's `hidden_norm` and into a paged FULL KV group of the head's own (`clef.kv`, one layer bound past the trunk's). Its block size is the trunk's, so the prefix cache keeps those rows with the trunk's cache. 8 KB a token on clef-flash.
2. On the step that samples a prompt's first token, the head gathers every row of its group back out (`gather_rows` over the sequence's block table, the way the KV tier packs blocks), runs the memory projection and the six cross-attention layers over all of them, averages the question and option spans, routes, runs the decoder layers and the scorer, and writes three values an option (lexical prior, cosine, residual) into the request's logits row.
3. The request's spans, option tokens and question types reach the plugin as integer words (`Request::plugin_words`, `RadBatch::plugin_words`, laid out by `abi/rad_clef.h`). This is the one ABI change: version 32, five trailing fields on `RadBatch`, and `rad_declare_plugin_words`. A pass carrying words is never recorded for replay.
4. The server reads the three values through `Request::token_scores`, as d1 does, and applies the head's three learned scalars from the container header (`decision.clef.*`, written by `rad-convert` from `joint_head.safetensors` and `joint_head_config.json`).

## What was reused, and the one new kernel

| Head piece | Existing code |
| --- | --- |
| LayerNorm with bias, linear with bias and fused exact GELU | `layernorm`, `gemm_nt_bias`, as the vision tower uses them |
| Self-attention over the field rows | `attn_dense`, `causal 0` |
| Field and option rows over every prompt token | `attn_cross`, as sdxl uses it |
| Span means, lm_head rows of the option tokens | `seg_mean`, `gather_rows`, `embed_lookup` |
| Per-token rows kept across requests | a FULL KV group, `kv_store`, the prefix cache |
| Rows back out of the group | `gather_rows` over the pool, one block a row |
| Cosine and prior | `rmsnorm_p` on both rows, then the row mean of their product (an identity) |
| Head weights in the container | `rad-convert --add joint_head.safetensors --set decision.type=clef` |
| The answer back to the server | the `token_scores` path, unchanged |

One kernel did not exist: a softmax-weighted sum over each question's options (`seg_softmax_pool`), added to libref, libavx and libr4d with its schema, shape, self-test case and `docs/OPS.md` entry. `softmax` had only host kernels, and a weightless op with no device kernel is a hole the engine refuses.

## Consequences

- The head runs on one rank: `world > 1` is refused at declare. clef 27B in bf16 needs two cards and so is not served with the head yet.
- The lm_head must stay bf16 (the head averages its rows); a recipe that quantises it is refused at declare.
- A request has at most 64 questions, 512 options and 8192 option tokens (`abi/rad_clef.h`); past that the server answers 400.
- `tests/clef_decision_test.cpp` serves a tiny clef on the CPU and compares every probability with the release's own head in float32 (`tests/mkclefgolden.py`): largest difference 0.0004 on 2026-10-28.
- bf16 through the head moves a probability by up to a few hundredths against PyTorch on the real model; the 100-respondent runs in `bench/timings.md` measure it.
