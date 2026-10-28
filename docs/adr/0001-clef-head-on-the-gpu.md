<!-- Machine-drafted by Claude (Opus 5.5) on 2026-10-27. Decision by the user. -->

# ADR 0001: clef's joint head runs on the GPU, inside the qwen35 plugins

Status: accepted, 2026-10-27.

## Context

Issue #166 asks radiance to serve clef and clef-flash on `/v1/systemone` with no Python sidecar. clef's head reads the final hidden state of every prompt token, not one position as d1 does. Two places fit: the GPU, inside `qwen35_bf16` and `qwen35_fp8`, or the CPU, in the server.

On threadripper, fp8 clef-flash serves 1.43 respondents/s on 13k-token guidance prompts and 9.16 on 676-token prompts. The PyTorch head takes 8.6 ms of the 95 ms short-prompt time (`bench/timings.md`). The CPU head is estimated at 60-120 ms on a short prompt and 0.4-0.9 s on a 13k-token cache miss. That figure is inferred, not measured.

## Decision

The head runs on the GPU, in the plugin step. The per-token part (LayerNorm, memory projection, and K and V for the 6 cross-attention layers) is written into a paged FULL KV group during the trunk step, so the prefix cache keeps it. A per-request tail pass runs the query side over those planes.

## Reuse, before any new code

| Head piece | Existing code |
| --- | --- |
| LayerNorm with bias | `layernorm` op, as in `arch/common/rad_block_vit.h` |
| Linear, with or without bias, fused exact GELU | `gemm_nt_bias` (`act "gelu"`), `gemm_nt` |
| Self-attention over the field rows | `attn_dense`, `causal 0`, head_dim 64 |
| Cross-attention over the memory rows | `attn_cross` or a paged read of the clef KV group |
| Span means, lm_head row means | `seg_mean`, `gather_rows`, `embed_lookup` / `embed_lookup_q` |
| Per-token K/V kept in the prefix cache | a FULL KV group, bound past `n_layers` as `drafter_kv_group` does |
| Head weights in the container | `rad-convert --add joint_head.safetensors --set decision.type=clef` |
| Weights optional, gated on metadata | `RadWeightDecl::optional`, gated like the vision tower |
| Result back to the server | the `token_scores` path: stage, issue, collect in `core/engine.cpp` |
| Prompt, spans, answer | `core/server/clef.cpp` (commit 82c114e) |

## New code this needs

1. A per-request payload on `Request` and `RadBatch`: question and option spans, type ids, option token ids.
2. A per-request head tail in the plugin step, and a small logits buffer the engine copies out.
3. A GPU `softmax` kernel for the option routing weights. Only CPU kernels exist.
4. The three learned scalars (`prior_logit_scale`, `joint_logit_scale`, `residual_gate`) go through `exp` and `sigmoid`. The plugin returns prior, cosine and residual per option, and the server combines them, so no `exp` kernel is needed.
5. Span rows must be fresh: prefix reuse stops at the start of the schema, so the span hidden states are always computed in the request's own step.

## Consequences

The clef KV group needs an attention consumer at block size 16, or the prefix cache silently drops it. The startup report names such a group "not cached". Cosine and dot products run in bf16, so answers can differ from PyTorch in the third decimal place. The golden test measures that difference.
