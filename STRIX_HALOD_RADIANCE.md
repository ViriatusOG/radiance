# Strix Halo support for Radiance (`libhalo`) — progress record

Last updated: 2026-10-30 (§7.15–§7.17; earlier sections are dated in their headings).

This file is the durable progress log for the task. It records what was asked,
what was decided, what is verified, and what remains. Claims marked
*(unverified)* have not been re-run since the state they describe changed.

---

## 1. Primary request and intent

1. Scout the Strix Halo cluster nodes node2 and node3 (`ssh <user>@<node>`);
   leave node1 alone. Bring down the running vLLM/Ray setup and report
   back. — **Done.**
2. Permanently bring down both `llama-rpc` services on node2/node3. — **Done.**
3. Main task: add Strix Halo support to the Radiance engine by implementing
   `libhalo` as the Strix Halo kernel library, with minimal engine-core changes
   for unified memory, supporting **all** operations that `libr4d` supports so
   all model plugins run on Strix Halo. The earlier vLLM-patched `libhalo` on
   the nodes may be used as inspiration/starting point. Architecture questions
   must be asked instead of decided unilaterally. Make each op as fast as
   possible ("everything tuned").

## 2. User constraints and preferences

- Do not touch node1. Work only on node2 and node3.
- Architecture questions must be asked, not decided unilaterally. Three
  blocking questions were asked and answered (see §4).
- Keep engine-core changes for unified memory to a minimum.
- All operations supported by `libr4d` must be supported by `libhalo`.
- User scope choice was "everything tuned", not correct-but-generic
  placeholders as the final state.
- The existing node `libhalo` from the earlier vLLM project may be used as
  inspiration/starting point.
- Durable tests matter: use the existing selftest/kbench/ctest harnesses.
  Generator scripts in `/tmp` are scaffolding, not delivered tests.

## 3. Cluster state (teardown, completed and verified)

- vLLM/Ray Docker Compose containers removed on node2/node3 via project
  tooling (`~/strix-halo-vllm`); ports 8000/8265/6379 closed; node1 untouched.
- Both `llama-rpc.service` units were disabled on node2 and node3:
  `sudo -n systemctl disable --now llama-rpc.service`, verified
  inactive/disabled, no `ggml-rpc-server` process, port 50052 closed,
  `/dev/kfd` free.
- Node reference material: the earlier vLLM `libhalo` (19 HIP units,
  `halo_wmma.h`, `halo_gdn_wmma.h`, GEMM units, `test_bf16_gemm.hip`) was
  copied to `/tmp/halo-ref/libhalo/` on the dev box.
- Hardware facts measured on node2: Radeon 8060S, `gfx1151`, 40 CUs, wave32,
  1280 wave slots, ~124 GB RAM, ROCm 7.2.3, read bandwidth ~187–230 GB/s
  (256 theoretical), launch floor ~1.87 µs batched / 3.49 µs alone, no
  full-card wave-slot cliff through 1280 slots (so `halo_cliff_rows` returns
  rows unchanged, with a rationale comment).

## 4. Decisions and rationale (user-approved)

1. **Basis — "Shared layouts, ported kernels."** Keep `libr4d` schemas and
   stored weight layouts bit-compatible so the same `.rad` containers,
   recipes, and goldens work. Port kernel bodies where the ISA allows;
   rewrite gfx12-only matrix cores using the node libhalo's gfx1151
   techniques.
   - Refinement discovered during implementation: fragment-ordered weights
     are a per-library relayout at load (spec §4.3). Many layouts are already
     shared; kernels adapt to gfx11's consecutive-K fragments by cross-half
     exchange (`__shfl_xor(..., 16)`) while keeping the `r4d.*` stored bytes
     byte-for-byte. Stored layout tags (e.g. `r4d.fp8.w.frag64`) are
     arrangement names, not library names, and are intentionally retained.
2. **Memory — "Capability flag, single ledger."** Add one `unified` device
   capability; charge VRAM + pinned pools against a single 124 GB RAM ledger
   (120 GiB GTT cap); collapse tiers to resident; host-visible device
   allocations make cross-link copies no-ops. **Not yet implemented**
   (pending core task, §8).
3. **Scope — "Single node, everything tuned."** Single-node v1 (no two-box
   tensor parallelism). All ops correct vs `libref`. No correct-but-generic
   placeholders at the end; all rows tuned toward roofline.

### Technical decisions (implementation-level)

- Plugin name `libhalo`, version `0.1.0`, `build_target` `gfx1151`. Plugin
  line at load: `libhalo 0.1.0 kernels 486 kernel(s), 245 schema(s), built
  for gfx1151` (row count differs from libr4d's 424 kernel rows because rows
  expand differently; schemas match at 245).
- `rad_kernel_chain` returns 0 until chained forms land per family.
- Declaration sources are generated from `libr4d` by a rename transform
  (`/tmp/gen_halo.py`): `r4d_→halo_`, `R4D_→HALO_`, `R4D→Halo`, `R4d→Halo`,
  `r4dchain→halochain`, `libr4d→libhalo`. Op/row names, parameter keys, and
  stored bytes are unchanged. (Corrected in §7.8: `r4dchain` is the ABI's
  chained-kernel marker, not a library prefix; the transform now maps it to
  `halo_r4dchain` — see PORTING.md.)
- Unimplemented entry points are generated stubs (`/tmp/gen_stubs.py`)
  returning `RAD_E_UNSUPPORTED`, so the library loads and every gap is
  named. The stub count is the parity scoreboard; stubs are regenerated
  after each unit lands. Progression: 580 → 521/520/517/516/512/508/501/496/
  494/491/480/475 → 465 (wpa8i8 tiled) → **464** (`gemm_bf16_nt_bias_tiled`).
  Stub count includes C entries and selftest hooks, not only kernel rows.
- gfx1151 ISA adaptations:
  - No `global_load_tr_b128`: shuffle transpose implemented in
    `halo_common.h`.
  - No fp8 conversion instructions: software `halo_fp8::cvt_pk_fp8_f32` /
    `f32_to_e4m3` in `halo_fp8_wmma.h`, proven exhaustively equal to
    `rad_f32_to_fp8e4m3` (probe: 0 mismatches over 2³¹ patterns).
  - `s_wait_storecnt` replaced with `s_waitcnt vmcnt(0)`.
  - gfx11 WMMA fragments are 16-wide consecutive K (`v16bf`/`v16h`/`v4i`) vs
    gfx12's 8+8 halves; accumulator row mapping
    `accRow(lane, e) = 2*e + (lane>>4)` from the node `halo_gdn_wmma.h`.
  - e4m3 decode: hardware `cvt_pk_f32_fp8` path where available, exact
    software fallback otherwise (`halo_fp8::e4m3_pk`).

## 5. Repository and build/test loop

- Repo (dev box): the working tree, synced to
  node2 `~/radiance` (rsync excluding `.git`, `/build/`, `/build-*`; note
  `core/build/` is a checked-in **source** directory and must be synced).
- Node build: system g++ fails; use ROCm Clang via `~/build-radiance.sh` on
  node2 (`/opt/rocm/lib/llvm/bin/clang{,++}`,
  `-DRAD_WITH_FFMPEG=OFF`, `-Wl,--build-id=sha1` on shared/module/exe
  linkers). Configure prints `radiance: HIP backend ON, targets gfx1151`.
  The dev box has no `hipcc`; **all device verification happens on node2**.
  Node3 is idle but available.
- Typical loop:
  `rsync -a libhalo/ <node2>:~/radiance/libhalo/`, then on
  node2 `~/build-radiance.sh` (or
  `cmake --build build -j32 --target libhalo` for iteration).
- Tests/tools:
  - `./build/bin/halo_selftest build/radiance_home/kernels/libhalo.so
    [--case <row>]`
  - `RADIANCE_HOME=$PWD/build/radiance_home ./build/bin/rad-info --plugins`
  - `rad-kbench --kernels libhalo:libref --record-rows
    /tmp/kbrows-libhalo.rkb`; replay with
    `rad-kbench --fixture /tmp/kbrows-libhalo.rkb --all-rows --kernels
    libhalo:libref`; timing with `--timing-only --bench-warm-ms 500`.
  - Row-fixture replay (recorded: 118 cases covering 155 of 486 rows): after
    the foundations, replay showed 142 checked/passed with failures only
    from stubs plus 2 kv_store shape refusals (`kv_store_k6v4`,
    `kv_store_fp8v4`). *(Not re-run since the GEMM units landed.)*

## 6. What is implemented and verified

### 6.1 Skeleton (landed, verified)

- `libhalo/CMakeLists.txt`, `libhalo/README.md`, top-level `CMakeLists.txt`
  wiring (`add_subdirectory(libhalo)` when HIP is found).
- Generated/renamed declaration sources: `libhalo/halo*.h`, `halo_rows.cpp`,
  `halo_layout.cpp`, `halo_fp8_layout.cpp`, `halo_shapes.cpp`,
  `halo_fuse.cpp`; `halo_selftest.cpp` derived from `r4d_selftest` with
  gfx1151 device gates; `halo_stubs.cpp` (generated); shared headers
  `halo_wmma.h`, `halo_gdn_wmma.h`, `halo_fp8_wmma.h`, `halo_common.h`,
  `halo_q8.h`.

### 6.2 Foundations (landed, verified)

Units: `halo_model_bf16`, `halo_quant_act_i8`, `halo_rmsnorm_quant_i8`,
`halo_had_quant_act_i8`, `halo_rmsnorm_had_quant_i8`,
`halo_gated_had_quant_i8`, `halo_gelu_mul_quant_i8`, `halo_quant_act_fp8`,
`halo_fused_quant_fp8`, `halo_qkv_norm_rope`, `halo_qk_norm_rope_gate`,
`halo_dequant_w4_bf16`, `halo_rowtopk_bf16`, `halo_norm_join_bf16`,
`halo_expert_scale_bf16`, `halo_kv_len_trim`, `halo_pair_argmax_f32`,
`halo_ngram_ids`, `halo_p2p_copy2d`.

- kbench check on selected foundation ops: `466 checked, 466 passed,
  0 failed`, worst rel_l2 `9.006e-04`.
- Real bug found and fixed: `cast_*` slices failed `RAD_E_DEVICE` (-15)
  because the `halo_p2p_copy2d` stub returned -1; the real unit
  (`libhalo/halo_p2p_copy2d.hip`) fixed the cast cases.
- `quant_act_fp8` tuning fix: the launcher splits rows into group chunks
  (`gridDim.y`) to avoid a one-block-per-row floor; M=1 K=25600 improved
  from ~21 µs-class to 2.73 µs (timing-only kbench). Selftest passes.

### 6.2b Attention family (started this session)

- `libhalo/halo_dt16.h` — the gfx1151 16-bit operand traits, the shared foundation of
  the whole attention port (the gfx11 counterpart of libr4d's `r4d_dt16.h`). Both
  architectures keep the score fragment's COLUMN on `lane & 15` (the query), which is
  why these kernels port at all; what changes is that a gfx1151 lane holds all sixteen
  k of a run rather than half of them. Provides 16-wide fragments and `mk()`, `wmma()`
  through `halo::mma`, bf16/f16 packs, the exact e4m3 widens through
  `halo_fp8::e4m3_pk` (no fp8 convert instruction on gfx11), `p16w()` which assembles
  the sixteen-consecutive-key P fragment across the half-pair, and `o8w()`. Compile-
  checked for gfx1151 with every template instantiated.
- `libhalo/halo_attn_dense_bf16.hip` — the first attention body, ported end to end and
  **verified numerically**: causal, head_dim 64, one head, T = 3/17/40 rows, rel_l2
  1.03e-3 / 1.34e-3 / 1.35e-3 against a float64 host reference (a standalone checker:
  the `attn_dense` row has no smoke case).
- **Bug found in shared code**: `halo_common.h`'s `load_tr_b128` emulation was wrong —
  a shuffle returns the SOURCE lane's argument, so `__shfl(w[i >> 1], base + j, ...)`
  fetched the wrong dword. Nothing used it before the attention V staging; a probe went
  from 192/256 wrong to 0/256. Fixed.
- The remaining attention bodies and their cost, measured: `prefill_h256_gqa6` 651
  lines, `paged_gqa8` 429, `decode_h256_gqa6` 1085, `paged_window` 1207, `attn_gen`
  1036, `vit_h72_bf16` 489, `sam_attn_bf16` 413, `dsa` 330; the ~15 small variant files
  are `#define` + `#include` wrappers, so those bodies cover all 127 attention stubs.

### 6.3 GEMM family (in progress)

Landed and verified (selftest unless noted):

| Unit / rows | Status |
|---|---|
| `halo_gemm_bf16_nt_m64` (gfx11 fragments from node `halo_gdn_wmma.h`) | selftest passes; M=64 still latency-bound (~183 µs at N=1280 K=5120 vs ~60 µs floor) — tuning debt |
| `halo_gemm_bf16_nt_m16` (dot2 GEMV from node `halo_gemm_bf16_nt.hip`) | selftest passes; unexpectedly slow at N=1280 (M=1 59.20 µs, 221 GB/s) — tuning unresolved |
| `halo_gemm_bf16_nt_tiled` | byte-identical cases pass; probe M=2048 N=2560 K=2560: tiled128 760 µs / 35 TFLOPS, tiled64 1018 µs, m64 1339 µs |
| `halo_gemm_f32_nt_bias` | selftest passes |
| `halo_gemm_nt_f32` (`gemm_nt_f32_bf16`) | **byte-identical** to the host arithmetic at all tested shapes (lane-strided fmaf + xor butterfly is a pinned contract) |
| `halo_gemm_w8a16_nt_m64`, `halo_gemm_w4a16_nt_m64`, `halo_gemm_w5a16_nt_m64` | selftest numeric passes (incl. W5 M=150 N=5120 K=17408) |
| `halo_gemm_w8a8_nt_m64`, `halo_gemm_w4a8_nt_m64`, `halo_gemm_w8a8_tiled`, `halo_gemm_w4a8_tiled` | selftest passes |
| `halo_gemm_w4a8_prefill` | shim over the tiled kernel (all 14 prefill arms map to correct (BM, BN) instantiations; arm 14 = new BM 256×BN 128 template); selftest passes |
| `halo_gemm_fp8a8` (`gemm_fp8a8_nt_m16`, `gemm_fp8a8_tiled`, bias variant) | selftest passes; see the FP8 notes below |
| `halo_gemm_i8a8` twins (`gemm_i8a8_nt_m16`, `gemm_i8a8_tiled`) | selftest passes (13 + 8 checks, incl. rotated `i8a8h` crossed checks) |
| `halo_gemm_fp8a16_m1` (matvec, checkpoint row-major bytes) | passes at every `rb` tune point |
| `halo_logits_gemm` (`logits_gemm_fp8`, `logits_gemm_i8`) | host-reference checks pass (i8: 28 checks; fp8 via the `dequant_w4_bf16`-gated numeric block) |
| `halo_gemm_fp8a8_gated_m16` | **byte-identical** to GEMM + `gated_quant_fp8`, including refusal of split grids |
| `halo_gemm_w2a8_nt` | gemm_case passes (zero folded into int8 fragment bytes; no asum needed) |
| `halo_gemm_wpa8i8_tiled` (`gemm_w4a8i8_tiled`, `gemm_w5a8i8_tiled`) | all numeric cases pass (11 + 7 checks, 0 FAIL; parts 1–3, zero u8/u4, M up to 2100, N up to 34816, K up to 17408); ~25 TF/s after retiling (see §7) |

FP8 GEMM notes (bugs found and fixed in `halo_gemm_fp8a8.hip`):

- Weight must be read in `halo_fp8_frag.h` frag64 order, not row-major.
- `deq16_e4m3` leaves values 2⁻¹²⁰ low; compensation `HALO_FP8_FIX`
  (1.329227995784916e36) applies to the **weight only** (both operands
  would underflow); the activation is dequantized exactly (`deq16_exact`).
- Per-accumulator-slot activation scales are indexed by output row
  `2*e + (lane>>4)`, not by fragment row.
- The int8 twins' activation is **natural row-major** (`quant_act_i8g`'s
  output), not the shuffled plane of `quant_act_i8` used by the skinny
  w4a8/w8a8 GEMMs. Two different quantizers, two different planes.
- The i8 weight scale plane is row-scale `[N][nkb]`
  (`FP8_OPS_ROW_SCALE`), not the fp8 block-scale default.
- **Split law**: replaced the home-grown `fp8_sk` with libr4d's
  `narrow_ksplit` law (adapted to whole-128-group splits), because two
  byte-level contracts read it: row slices issued with `split_n` must
  accumulate exactly as the whole GEMM (fixed; slices now byte-identical),
  and the gated row must refuse exactly the grids the law would split
  (K ≥ 512 and unsplit grid < 128 waves). Split count, group partition,
  and fold order must match between slice and whole; SK changes bytes
  because f32 addition is not associative.
- Gated epilogue: accumulators round to bf16 first, then
  `silu(gate)·up` over the rounded values, rounded again; group amax over
  |v| (an early version forgot `fabsf` — bf16 products matched, scales and
  codes did not); codes via clamp + `halo_fp8::f32_to_e4m3`, exactly
  `gated_quant_fp8`'s arithmetic. Whole fp8 numeric block:
  **1526 checks ok, 0 FAIL**.

Also landed: `halo_gemm_nt_bias` (`gemm_bf16_nt_bias_tiled`) — clone of
the bf16 tiled kernel with predicated K-tail staging (chunks past K stage
zeros; any K multiple of 8, any N) and the three-rounding epilogue
(bf16(acc+bias) → bf16(act) → bf16(res+t), `res` may alias the output).
All 4 numeric cases pass (gelu/gelu_tanh/none, res=out, f32 and bf16
bias, ragged M/N/K).

**The GEMM family is complete.** The mxfp4 family landed this session and the
whole family now passes its selftest cases with 0 failures:

| Unit / rows | Status |
|---|---|
| `halo_gemm_mxfp4a8_nt_m64` | 6 checks, 0 FAIL |
| `halo_gemm_mxfp4a8_gemv` (+ wide) | 97 checks, 0 FAIL |
| `halo_gemm_mxfp4a8_decode` (+ scratch) | 86 checks, 0 FAIL |
| `halo_gemm_mxfp4a8_tiled` (+ scratch) | 17 checks, 0 FAIL; **byte-identical to m64** (row-0 ratio exactly 1.0) |
| `halo_gemm_mxfp4a8_gated_gemv` (+ scratch) | 16 checks, 0 FAIL; **byte-identical** to gemv+gated_quant_fp8, silu and gelu_tanh |

Port bugs found and fixed in `halo_gemm_mxfp4a8_tiled.hip` (all three produced
~1e4 relative errors): the staged-weight exponent used the e4m3 table's
`d = ref - exp` where the e2m1 fragment needs `exp - ref`; the per-slab A
fragment base dropped `k0`, so every slab re-read the first four steps; and the
staged LDS weight was declared `unsigned char` but written/read as 16-bit
halves (stride 8 elements, not 16). `HALO_MXT_PAD` also moved 4 -> 8 elements
so each staged row starts 16-byte aligned for the vector stores.

The gated row was rebuilt on m64's geometry (it had been written against
r4d's 8-K-wave design): contiguous `SK` splits from
`cfg_for(K, M, 32, BR_FLAT)` — the same question the gemv row asks, so the same
split law and the same accumulation order — per-32-block `fmaf` with
`f = rf * As[element row 2e + (lane>>4)]`, ascending split reduction in LDS,
and the quantiser mirroring `quant_group` (whole-wave amax over the 128-column
group, `q8_pack4`).

Decode bugs fixed in `halo_gemm_mxfp4a8_decode.hip`: a split with too little
scratch is now refused (it was silently clamped to a smaller split, changing
the row's bits) and `mxd_reduce_kernel` now takes the output row pitch — a
partitioned weight's part width is not the row pitch, so `parts=2/3` wrote into
the wrong columns and left NaNs.

MoE/router GEMMs belong to the MoE todo.

### 6.4 The seven port lanes, merged (Round 2)

Seven lanes were built in isolated worktrees off `1a23ddfe` and merged on `main`: each lane's own
`halo_*.hip` files were taken, and `halo_stubs.cpp` / the `HALO_UNITS` list were **generated**, never
hand-merged (`python3 libhalo/gen_stubs.py`, then `--check`, exit 0 — both files are byte-stable).
The merge landed **146 entries**: 108 are plain `extern "C"` definitions the generator can see, 38 are
macro-built (token-pasted) entry points that override their weak stub at link time.

| Lane | Units | Entries | Verified |
|---|---|---|---|
| `port/infra` | `gen_stubs.py`, `PORTING.md`, `README.md`; weak stubs + sorted `HALO_UNITS` | — (generator) | generator `--check` clean; the whole merge depends on it |
| `port/attn-paged` | `halo_attn_prefill_h256_gqa6.hip` (shared body) + 9 h128/h256 paged wrappers | 35 | full selftest (was rc=-18, now numeric) |
| `port/attn-decode` | `halo_attn_decode_h256_gqa6.hip` | 11 | full selftest |
| `port/attn-window` | `halo_attn_paged_window.hip` | 21 | full selftest + `attn_window_h512_gqa8_fp8kv` |
| `port/attn-gen` | `halo_attn_gen.hip` | 4 | full selftest (`attn_prefix_self_h128_bf16`, `attn_cross_h64/h128`) |
| `port/attn-media` | `halo_attn_vit_h72_bf16`, `halo_sam_attn_bf16`, `halo_dsa`, `halo_vsa` | 11 (8 ABI + 3 helpers) | full selftest + `sam_attn_bf16`, `dsa_score_fp8`, `attn_vsa_h128_bf16` |
| `port/collectives` | 9 `halo_ar_*.hip` + `halo_ar_hc.h` / `halo_ar_seq.h` updates | 65 (31 `halo_rad_ar_*` + 34 `halo_ar_*`) | 2/3/4/8-rank bit-exact on one card as streams (lane `ar-deadlock`; both reported deadlocks were HIP's `GPU_MAX_HW_QUEUES` default of 4 — §6.6); board timings and the multi-card peer path still unmeasured |

**Duplicate definitions resolved at merge.** `halo_attn_dims` was defined by both the decode unit and
the h256/gqa6 paged unit with the same geometry (256 / 6 / 16 / 64); the paged unit owns the single
definition and the decode unit now says so in a comment (before that, the link was a multiple-definition
error). `halo_attn_qcvt` exists only in the decode unit — a `__device__` template, TU-local, nothing to
resolve. Nothing else collided. Two header updates came with the collectives lane and are required by
its units: `halo_ar_hc.h` gains its own `f2bf`/`bf2f` (round-half-away, a byte contract — it must not
inherit `halo_gdn_wmma.h`'s RTNE), `halo_ar_seq.h` takes `halo_xor_lane_u32` for the wave sum.

**Numbers (node2, RelWithDebInfo, `libref.so` present in both runs, one GPU job at a time):**

| | ok | FAIL | skip | log lines | harness summary |
|---|---|---|---|---|---|
| before (`1a23ddfe`) | 3756 | 2833 | 200 | 8561 | 203 failure(s), 188 skipped |
| after (merge) | 4110 | 2561 | 200 | 8651 | 166 failure(s), 188 skipped |

FAIL lines drop by 272 (`-272`), ok rises by 354, skip is unchanged, **zero crashes** in both. 39 case
names stopped failing and **no case name started failing**. Every remaining FAIL is an unimplemented
stub's `-18`: 2054 lines carry an explicit `rc=-18`, the other 507 are sub-checks of rows that answer
`-18` (`moe_gemm_*`, `mhc_*`, `mla_project`, `KDA`, the unimplemented attention variants). Caveat on
absolute counts: a baseline measured *without* `libref.so` next to the plugin reports 2822 ok / 2561
FAIL / 7267 lines, because 215 checks bail out with "reference plugin unavailable"; the two runs above
both had it, so they exercise the same checks.

Per-case after the merge (all rc=0 unless noted): `gemm_mxfp4a8_tiled` 17 ok, `gemm_mxfp4a8_gated_gemv`
16 ok, `gemm_mxfp4a8_decode` 86 ok, `attn_window_h512_gqa8_fp8kv` 16 ok, `attn_dense_h128_bf16` 2 ok,
`attn_vsa_h128_bf16` 56 ok (both cross-checks against `attn_dense_h128` now pass), `sam_attn_bf16` 5 ok,
`dsa_score_fp8` 62 ok; `attn_dense_gqa` 16 ok, `attn_dense_h64/h384/h512_bf16` 9/2/2 ok. There is no
case named exactly `attn_dense` — `--case attn_dense` prints `skip attn_dense -- no smoke case` and
exits 2 (`--case` matches exactly), so the `attn_dense_*` cases above stand in for it.

**Still stubbed or unverified after the merge.** 34 attention case names still answer `-18`: 25 are
variants this round did not port (`attn_decode_*_gqa2/gqa12/h128/bs4`, `attn_qrow_*`, `attn_local_rel_bf16`)
and 9 are the e4m3-WMMA legs of `attn_prefill_h256_gqa6_{fp8,fp8v4,k6v4}kv_{qk8,pv8,qk8pv8}`, which stay
stubs by design (gfx1151 has no fp8 matrix instruction). The collectives family is merged but never
executed (one card): 68 `ar_*` skip lines, 52 of them the explicit "needs >=2 gfx1151 cards" gate —
that is not a pass, and it is the largest piece of verification debt the merge adds. The same note's
verification debt still stands for the decode/`attn_window`/`attn_paged` ratio rows that have no
committed case of their own.

### 6.5 The five port lanes of Round 3, merged

Five more lanes — branched from `1a23ddfe`, i.e. from *before* the Round-2 merge — were merged onto
`main` (`ac0d93f2`). Each lane's own `halo_*.hip` files were taken with `git checkout <branch> --
libhalo/<file>`; `halo_stubs.cpp` and the `HALO_UNITS` list were **generated**, never hand-merged
(`python3 libhalo/gen_stubs.py`, then `--check`, rc=0, both files byte-stable). The merge adds **40
units**: `HALO_UNITS` 68 → 108, the weak stub block 341 → 160 definitions (**−181**, all of them this
round's entries).

| Lane | Units | Entries | Verified |
|---|---|---|---|
| `port/moe` | `halo_moe`, `halo_moe_gemm_bf16`, `halo_moe_gemm_q`, `halo_moe_mxfp4` | 15 / 16 | full selftest; `moe_gemm_bf16` 89 ok, `moe_gemm_w4a8` 602 ok / 12 skip, `moe_gemm_mxfp4a8` 623 ok. `moe_fp8_f32scale` stays a stub (needs the fp8-f32scale GEMM family) |
| `port/sample-gdn-router` | 12 units: `halo_sample_stages_f32`, `halo_sample_chain_f32`, `halo_sample_ngram_f32`, `halo_gdn_kkt_solve`, `halo_gdn_chunk_scan`, `halo_gdn_conv_w4_h128_bf16`, `halo_gdn_recurrent_update_k128_v128_bf16_fp32state`, `halo_gdn_gated_rmsnorm_h128_bf16`, `halo_gdn_gated_norm_had_quant_i8`, `halo_kda`, `halo_router_topk_bf16`, `halo_router_sigmoid_topk_f32` | 27 / 32 | full selftest + libref oracles: `router_topk` byte-identical, `router_sigmoid_topk` ids exact / weights 1 ulp, GDN/KDA vs libref (see debt) |
| `port/qsa-mla-lora-hc` | `halo_qsa_work/tail_store/block_key/score/select_bf16`, `halo_mla.hip`, `halo_lora_bf16.hip`, `halo_hc_bf16.hip`, `halo_mhc.hip` + `halo_gdn_wmma.h` | 19 / 28 | `qsa_score_bf16` 4 ok (rel_l2 5.8e-08 vs f64), `mla_attn_bf16` 51 ok, `lora_shrink_scalar` 6 ok, `hc_*`/`mhc_*` in the full selftest |
| `port/longtail` | `halo_encoder_bf16`, `halo_gen`, `halo_audio`, `halo_ltx`, `halo_video`, `halo_upscaler`, `halo_glm`, `halo_givens_rotate`, `halo_gram_accum_fp8`, `halo_ple_conv_bf16`, `halo_ple_gate_bf16`, `halo_dflash_select_bf16`, `halo_dflash_conv_t2_g16_bf16`, `halo_kv_resolve`, `halo_prefetch` | ~105 | full selftest: 75 rows ok; 25 rows have no committed case (debt below) |
| `port/ar-verify` | `halo_selftest.cpp` only — no libhalo unit | — | taken after 13 one-card runs (below); the multi-card path is still unproven |

**Duplicate definitions resolved.** (1) `f2bf`/`f2bf2`/`bf2f` were defined by *both* `halo_gdn_wmma.h`
(restored there by the qsa lane) and `halo_ar_hc.h` (written by the Round-2 collectives lane). One
definition survives — `halo_gdn_wmma.h`'s, the bit-identical round-half-away one the selftest
documents — and `halo_ar_hc.h` now includes that header instead of repeating the pair. (2) The same
three were also copied into three sample-gdn-router units (`halo_gdn_chunk_scan.hip`,
`halo_gdn_conv_w4_h128_bf16.hip`, `halo_gdn_kkt_solve.hip`), written when the header did not carry
them; once it does, each is `error: redefinition of 'f2bf'` (the first build of the merged tree
died there). Deleted from all three; `f2bf4`, chunk_scan's own four-wide form, stays. (3) Checked and
*not* duplicates: `parse_f32_list` is spelled identically in `halo_video.hip` and
`halo_upscaler.hip` but both are inside an anonymous namespace (internal linkage), and the repeated
`halo_rad_gemm_bf16_m64` / `halo_ar_gather_4rank` / `halo_cfg_*` names are pre-existing forwarders,
macros and header declarations. Nothing else collided — the link is clean.

**The `ar-verify` harness change was taken, on evidence.** The branch's `halo_selftest.cpp` change is
the only thing it owns (its collectives units are already in `main`). Run on node2 with
`HALO_SELFTEST_ONE_CARD=1` under `flock /tmp/gpu.lock timeout …`, 13 rows: `ar_rmsnorm_quant_i8_4rank`
(rc=2, 170 ok, 0 FAIL, 18 skip, <1 s — it does **not** deadlock once `st_shared_nblocks` divides by the
multiprocessor count), `ar_oneshot_8rank_exact` (15 ok, 1 skip, 6 s), `ar_twoshot_8rank_exact` (524 ok,
0 FAIL, 53 s — the 8-rank pure two-shot, bit-exact), `ar_rmsnorm_quant_i8` 50 ok, `ar_hc_write` 6 ok,
`ar_gather_hc_write` 11 ok, `ar_ln_had_quant_i8` 18 ok, `ar_twoshot_4rank_exact` and the 2/3/4-rank
`oneshot`/`gather` pure rows **skip** with their own gates. No timeout fired (`rc=124` never seen), no
FAIL, no signal. Without `HALO_SELFTEST_ONE_CARD` every `ar_*` row skips exactly as before, and a full
selftest on the branch's harness reproduces the numbers below **exactly** (7610 / 463 / 236, 49
failing rows, identical failing-name histogram, zero crash lines), so the change is behaviour-neutral
unless the one-card variable is set. What it buys: the fused 4-rank split/tail grids are skipped with
their arithmetic instead of over-subscribing the card (4 ranks × 18 blocks × 1024 threads = 73728
threads against gfx1151's 40960 slots), and 13 previously-unexecuted rows now run.

**Numbers** (`rsync -a <repo>/ <node2>:~/radiance/`, `cmake --build
~/radiance/build -j32 --target libhalo halo_selftest libref`, then `flock /tmp/gpu.lock timeout 3300
~/radiance/build/bin/halo_selftest ~/radiance/build/radiance_home/kernels/libhalo.so`; `libref.so`
beside the plugin in both runs, one GPU job at a time):

| | ok lines | FAIL lines | skip lines | failing rows | harness summary |
|---|---|---|---|---|---|
| before (`ac0d93f2`) | 4110 | 2561 | 200 | 176 | 166 failure(s), 188 skipped |
| after (this merge) | 7610 | 463 | 236 | 49 | 49 failure(s), 190 skipped |

ok rises by 3500, FAIL lines fall by 2098 (−82 %), and **129 row names stop failing**. No row that was
passing starts failing, with one case to state plainly rather than hide: `router_topk_scatter` (90 FAIL
lines) passed before because its check compares the fused scatter row against the *unfused* pair
(`router_topk` + `moe_scatter`) and both sides were stubs, so the comparison was vacuous; those two
rows exist now, so the still-stubbed scatter row is finally measured (`rc=-18` on
`router_topk_scatter_bf16`, then every comparison against the real pair differs). That is a
measurement becoming real, not a regression. `skip` rises by exactly 36, all of them
`moe_gemm_w4a8`'s aligned column-tile check declined past 512 experts (the run map holds
`kMaxExpert + 1` offsets — libr4d's rule too). Zero `child died:` / signal / page-fault lines in
either run.

**FAIL histogram after** (49 failing rows / 463 lines; 228 lines are an explicit `rc=-18` or `rc -18`,
the other 235 are sub-checks of a row that answers `-18`): router scatter family 133
(`router_topk_scatter` 90, `router_sigmoid_topk_scatter` 24, `router_topk_scatter_bf16` 18,
`router_gemm_topk_scatter` 1); `MLA` (`mla_attn_matrix`/`_rows`) 65; `hc_read_e4m3` 34 + `hc_read_bf16`
3; `lora_shrink_wmma` 32 + `lora_shrink_wmma_q` 16; `conv3d_ndhwc_bf16` 26, `conv_t1d_bf16` 10,
`conv2d_nhwc_bf16` 5; `givens_dense_quant` 20; the 9 e4m3 `qk8`/`pv8` prefill legs 18; 25 attention
variants (`attn_decode_*` gqa1/2/2w/4/12 + `bs4` k6v4/fp8v4, `attn_qrow_*`, `attn_local_rel_bf16`) 100;
`layernorm_bf16` 1. Every failing row is one of the 128 rows still weak-stubbed.

**Per-case** (node2, plugin + `libref.so`; `rc=2` means "no failures, some skips"): `moe_gemm_bf16` 89 ok
/ rc=0; `moe_gemm_w4a8` 602 ok / 12 skip / rc=2; `moe_gemm_mxfp4a8` 623 ok / rc=0; `sample_chain_f32` 3
ok; `router_topk_bf16` 12 ok + 36 FAIL, all of them `router_topk_scatter` sub-checks (the router row
itself is byte-identical); `qsa_score_bf16` 4 ok; `mla_attn_bf16` 51 ok; `lora_shrink_scalar` 6 ok;
`conv1d_f32` 26 ok (strided causal forms, rel_l2 1.4e-7..8.5e-7); `gdn_kkt_solve_k128_c64_bf16` 1 ok
(smoke — numerics in the debt below); `gemm_mxfp4a8_tiled` 17 ok; `gemm_mxfp4a8_gated_gemv` 16 ok;
`attn_window_h512_gqa8_fp8kv` 16 ok; `attn_dense_h128_bf16` 2 ok; `attn_vsa_h128_bf16` 56 ok;
`sam_attn_bf16` 5 ok; `dsa_score_fp8` 62 ok. There is no case named exactly `qsa_score`, `mla_attn`,
`lora_shrink` or `gdn_kkt_solve` (`--case` matches exactly, so those four print `skip … no smoke case`
and exit 2); the names above are the cases that cover them.

**Verification debt this merge adds.** (1) `port/longtail`: 25 rows are implemented but have **no
committed case** — `causal_conv1d`, `chan_split`, `cu_repeat`, `d2s_add`, `end_status`, `flow_euler`,
`fourier_embed`, `frame_mix_store`, `gather_rows2`, `gelu_tanh`, `l2norm_chan`, `lane_gather`,
`lane_store`, `ln_mod`, `ln_mod_quant_fp8`, `ple_conv`, `ple_gate`, `resample_nearest`,
`rms_mod_quant_i8`, `s2d_avg_add`, `scatter_add_rows`, `to_rgba8`, `win_finish`, `win_input`,
`win_prep`. Thirteen of those (`chan_split`, `cu_repeat`, `d2s_add`, `end_status`, `fourier_embed`,
`gather_rows2`, `gelu_tanh`, `l2norm_chan`, `ln_mod`, `resample_nearest`, `s2d_avg_add`,
`scatter_add_rows`, `to_rgba8`) were checked only by the lane's throwaway `longtail_check.cpp`; the
other twelve were never checked numerically at all. (2) `port/sample-gdn-router`: 12 of the 15 sampler
stage rows skip as `no smoke case` (`sample_temp/mask/penalties/dry/logprobs/topk/topp/minp/typical/
xtc/pick/merge_topk`); the lane's only numeric evidence is `libhalo/check_sample_router.cpp`, 572
lines on branch `port/sample-gdn-router` that no CMake target builds — deliberately not merged, so it
is not a deliverable. (3) The two GDN rows whose accuracy needed a judgement, `gdn_kkt_solve` A and
`gdn_chunk_scan`'s O/ht, sit at rel_l2 7.3e-3 / 9.9e-3 and 3.7e-3 / 2.6e-3 against libref, against
1e-3-class expectations for a bf16 path. Judged **accepted**: a float64 model of the *same* algorithm
reproduces the loss, i.e. the algorithm's own bf16 staging is that lossy (the scan's chunk-matrix
staging carries the error, not the gfx1151 port), and the committed cases for both rows are smoke-only,
so the judgement rests on the lane's libref-oracle runs, not on a committed case. (4) `ar_*` is still
not proven on real hardware: 13 rows ran as concurrent streams on one card with zero failures, but
2/3/4/8-rank pure rows skip (`needs N gfx1151 cards`) and the multi-card protocol is untouched.

**Still stubbed after the merge** (128 weak-stub rows): the 25 attention variants and 9 e4m3 `qk8`/
`pv8` prefill legs above; `fp8_f32scale` + `layout_fp8_f32scale` + `moe_fp8_f32scale`;
`lora_shrink_wmma(+_scratch)` / `lora_expand_tiled` / `lora_expand_mm`; `mla_attn_matrix(+_scratch)` /
`mla_attn_rows`; `hc_read(+_scratch)`; `conv2d_nhwc` / `conv_t1d` / `conv3d_ndhwc` / `layernorm` /
`grid_embed(+aa)`; `mtp_enter(+_scratch)` / `scale_rows`; the five router scatter rows;
`givens_dense_quant`; `attn_dense_small_h256` / `attn_na3d`; and the `kv_resolve_scratch` /
`layout_any` hooks.

### 6.6 The six Round-4 lanes, merged

Six lanes — `port/attn-variants` (`f1a92ae0`), `port/fp8-legs` (`ff0558ee`), `port/wmma-bodies`
(`d649dc0b`, based on `port/qsa-mla-lora-hc`), `port/conv-vit` (`ea2fc184`, based on
`port/longtail`), `port/router-scatter` (`0f737a3b`) and `port/ar-deadlock` (`c369451e`) — were merged
onto `main` (`32953e3a`, code `f9ff43ac`). Each lane's owned `halo_*.hip` / `*.h` files were taken
with `git checkout <branch> -- libhalo/<file>`; `halo_stubs.cpp` and the `HALO_UNITS` list were
**generated**, never hand-merged (`python3 libhalo/gen_stubs.py`, then `--check`, rc=0, both
byte-stable). The merge takes `HALO_UNITS` 108 → **115** and the weak-stub block 160 → **100
definitions** (−60).

| Lane | Units / headers | Entries | Verified |
|---|---|---|---|
| `port/attn-variants` | the 10 paged wrappers rewritten, including the new `halo_attn_paged_h128_gqa4`; `halo_attn_decode_h256_gqa6.hip` and `halo_attn_prefill_h256_gqa6.hip` are now **pure bodies** (their entries live in the units that own each geometry) | 36 plugin rows + 13 scratch hooks + 13 internal entries | full selftest; new committed cases `attn_decode_h256_gqa8_fp8kv` 3 ok, `attn_decode_h128_gqa4_bf16kv` 4 ok, `attn_decode_h256_gqa6_bs4_fp8kv` 3 ok, `attn_decode_gq_h256_gqa6_bs4_fp8kv` 16 ok |
| `port/fp8-legs` | `halo_attn_prefill_h256_gqa6` + `halo_attn_paged_h256_gqa6` (the 9 e4m3 `qk8`/`pv8` legs), `halo_gemm_fp8_f32scale.hip` (new) | 12 | `--case fp8_f32scale` 144 ok / 0 FAIL; the dense row is bit-identical to libref |
| `port/wmma-bodies` | `halo_lora_bf16.hip`, `halo_mla.hip` | 7 | `--case lora_shrink_wmma` 16 ok, `--case mla_attn_matrix_bf16` 142 ok; `hc_read` deliberately untouched |
| `port/conv-vit` | `halo_conv2d_nhwc`, `halo_conv_t1d`, `halo_vit_norm_embed`, `halo_scale_rows_bf16` (new); `halo_selftest.cpp` gains 2 180 lines (`cvda`/`cvdb`/`cvdc`/`cvdt` + 15 dispatch lines) | 6 | `causal_conv1d_bf16` 26 ok, `scale_rows_bf16` 14 ok, `win_input` 3 ok, `grid_embed_bf16` 14 ok, `ple_conv_bf16` 25 ok, `flow_euler_bf16` 6 ok; `conv3d_ndhwc` and `mtp_enter` stay stubbed by design |
| `port/router-scatter` | `halo_router_topk_bf16` and `halo_router_sigmoid_topk_f32` rewritten as thin wrappers over the new `halo_router_topk.h` / `halo_router_sigmoid.h`, plus `halo_router_scatter.hip` and `halo_router_gemm_scatter.hip` | 5 | `--case router_topk_scatter_bf16` 54 ok; `router_gemm_topk_scatter` keeps its 14 known logits-only FAILs at M≤16 |
| `port/ar-deadlock` | `halo_selftest.cpp` only — the ar-verify delta plus `st_shared_max_kernels()` / `st_shared_ranks_fit()`; no libhalo unit | — | full selftest reproduces the numbers below; its finding is recorded under the queue section here |

**The harness merge, one suite at a time.** `libhalo/halo_selftest.cpp` is touched by four lanes, so
it was merged in the order the lanes' changes need: `port/ar-deadlock` first (it already carries
main's ar-verify delta plus the queue gate, so it is the substrate), then `port/conv-vit` (its four
`cvd*` namespaces and 15 dispatch lines), then `port/router-scatter` (`hook()` reading
`halo_stub_hooks`, the `galloc`/`guard_moved` red zones on every scatter output, the pin-hook
notes), then `port/attn-variants` (the new attention cases and the parametrised `attn_gq_fold_case`).
Each step was applied as a patch, built (`cmake --build build -j32 --target halo_selftest`, rc=0,
zero errors, four times) and smoke-run before the next: base 1 case; conv-vit 6 of its own + 2 from
others; router-scatter 4 + 1; attn-variants 4 + 2 — 15 cases, all rc=0 with zero FAIL and zero
crashes. The merged file is 33 405 lines against the ar-deadlock base's 30 910 (+2 495 = conv-vit's
2 180 + router-scatter's net 193 + attn-variants' 122), and every side's delta survives: 8 sites of
the queue gate, 4 `cvd*` namespaces, 15 dispatch lines, 26 `galloc`/`guard_moved` sites, 3
`halo_stub_hooks` readers and 11 attention-case sites.

**Which side won each overlap.** The four regions are disjoint in the file — ar-deadlock at 503 and
24 308, conv-vit from 7 918 on, router-scatter at 113/197/207/554, attn-variants at 3 071 and 27 529 —
so no textual conflict arose and no side's coverage was dropped. Semantically, `port/router-scatter`
wins the two places where it replaces plumbing: its `hook()` supersedes the `sym()` lookups for the
pin hooks (so a stubbed hook reads as absent instead of asserting about a pin that pins nothing) and
its `galloc`/`guard_moved` supersede the plain `dalloc` for the scatter outputs; both helpers exist
only in that lane, so nothing else competed for them. The one unit file two lanes own,
`halo_attn_paged_h256_gqa6.hip`, is a union — `port/fp8-legs`' nine e4m3 legs first, then
`port/attn-variants`' appended `#include "halo_attn_decode_h256_gqa6.hip"` with its geometry's
decode entries. A three-way `git merge-file` (base `ac0d93f2`, ours fp8-legs, theirs attn-variants)
raised exactly one conflict, both sides appending at end-of-file; the resolution keeps both blocks in
that order and no markers remain. The doc edit `port/ar-deadlock` carries was **not** taken as-is —
it deleted Round-3's §6.5 and replaced it; Round-3's §6.5 stands, its Round-2 collectives row is
corrected in place, and its finding is recorded next.

**Both collective deadlocks were `GPU_MAX_HW_QUEUES`, not geometry.** Lane `port/ar-deadlock` chased
the two hangs `port/collectives` reported and could not explain, and neither is a bug in a
collective. HIP's `GPU_MAX_HW_QUEUES`, which ROCm defaults to **4 per device**, is the whole cause: a
stream whose kernel gets no hardware queue is never dispatched (the launch still returns
`hipSuccess`) and a spinning kernel is never preempted to make room, so the queued ones wait for
ever. Every collective here has each block spin on its twin on every peer rank, so one undispatched
rank hangs the whole launch. Measured on gfx1151 (ROCm 7.2.3, node3) with a 20-line probe: eight
one-block 64-thread spinning kernels on eight non-blocking streams give 4 resident and 4 queued for
ever with the variable unset, and 8 of 8 resident with `GPU_MAX_HW_QUEUES=16`. It is a *kernel*
count, not a wave count — 5, 6, 9, 12 and 40 ranks and 8 x 8 blocks x 256 threads all stop at four,
so 16 waves against the card's 1 280 change nothing. **This is engine-relevant, not just
test-relevant:** a deployment that runs more concurrent rank streams per device than the queue count
hangs on its first launch in every geometry; it must raise `GPU_MAX_HW_QUEUES` or destroy the streams
it creates. Two concrete defects fall out of it: the 8-rank pure rows hung because only four ranks
ever ran (with the variable raised all three are bit-exact on one card), and the 2-rank rows hung on
a *second* init/fini cycle because the harness creates `world` streams per cycle and leaks them, so
cycle k needs k x world queues. The ar-deadlock harness change makes this visible rather than fatal:
`st_shared_max_kernels()` reads the variable (default 4) and `st_world_devs` returns no devices above
it, so a shared-card row that cannot be resident SKIPs with the arithmetic printed instead of hanging.
`main()` already raises the variable to 16 for `HALO_SELFTEST_ONE_CARD`, so the wide rows run there.
Nothing in `halo_ar_*` changed: no collective was wrong, so there was nothing to fix there.

**Numbers** (`rsync -a <repo>/libhalo/ <node2>:~/radiance/libhalo/`, `cmake --build
~/radiance/build -j32 --target libhalo halo_selftest libref` — rc=0, zero errors — then `flock
/tmp/gpu.lock timeout 3300 ~/radiance/build/bin/halo_selftest
~/radiance/build/radiance_home/kernels/libhalo.so`; `libref.so` beside the plugin in both runs, one
GPU job at a time, 467 s wall):

| | ok lines | FAIL lines | skip lines | failing rows | harness summary |
|---|---|---|---|---|---|
| before (`32953e3a`) | 7610 | 463 | 236 | 49 | 49 failure(s), 190 skipped |
| after (this merge) | 8415 | 93 | 213 | 6 | 6 failure(s), 166 skipped |

ok rises by 805, FAIL lines fall by 370 (−80 %) and **43 row names stop failing**. 49 − 43 = 6, i.e.
no previously-passing row starts failing: every one of the 6 still-failing names is in §6.5's failing
histogram. Zero `child died:` / signal / page-fault lines in either run. `skip` falls by 23 because
rows that used to answer `-18` now run their numeric checks.

**FAIL histogram after** (6 failing rows / 93 lines): `hc_read_e4m3` 28, `conv3d_ndhwc_bf16` 26,
`givens_dense_quant` 20, `router_gemm_topk_scatter` 14, `hc_read_bf16` 3, `attn_local_rel_bf16` 2. The
first five families' 79 lines are all an explicit `rc=-18` / `rc -18` from a weak stub; the 14
`router_gemm_topk_scatter` lines are the known logits-only mismatch at M≤16 (the row's own scatter
outputs pass) — the same 14 §6.5 recorded. Nothing else in the 25-attention-variant, MLA, LoRA,
conv, router-scatter or e4m3-prefill families fails any more.

**Regression guard**, all `--case` runs under `flock /tmp/gpu.lock timeout 400`, plugin + `libref.so`
(`rc=2` = no failures, some skips): `gemm_mxfp4a8_tiled` 17 ok; `gemm_mxfp4a8_gated_gemv` 16 ok;
`attn_window_h512_gqa8_fp8kv` 16 ok; `attn_dense_h128_bf16` 2 ok; `attn_vsa_h128_bf16` 56 ok;
`sam_attn_bf16` 5 ok; `dsa_score_fp8` 62 ok; `moe_gemm_w4a8` 602 ok / 12 skip (rc=2);
`moe_gather_bf16` 34 ok; `router_topk_bf16` 54 ok; `qsa_score_bf16` 4 ok; `mla_attn_bf16` 51 ok;
`lora_shrink_scalar` 6 ok; `conv1d_f32` 26 ok; `moe_scatter_i32` 70 ok; `fp8_f32scale` 144 ok;
`lora_shrink_wmma` 16 ok; `mla_attn_matrix_bf16` 142 ok; `router_topk_scatter_bf16` 54 ok. **Zero
FAIL in all 19 runs.**

**Still stubbed after this merge** (100 weak-stub definitions): `hc_read` / `hc_read_scratch`,
`conv3d_ndhwc`, `mtp_enter` / `mtp_enter_scratch` and `attn_local_rel_bf16` remain stubs by design
(their committed cases answer `rc=-18`, which is what the 79 stub FAIL lines above measure);
`givens_dense_quant` is still a stub. The rest of the 100 are stubs with no committed case or a row
the selftest does not reach — the `attn_decode_*` gqa8/gqa12 fp8v4/k6v4 and `attn_prefill_*` h128/h256
gqa8/gqa12 fp8v4/k6v4 variants, the `attn_window_h256`/`h512` bf16/fp8 family, `attn_dense_small_h256`,
`attn_na3d`, `layout_any` and the `kv_resolve_scratch` hook. `router_gemm_topk_scatter` is *not* a stub
any more and keeps 14 known logits-only FAILs at M≤16. Verification debt unchanged from §6.5: the
`ar_*` board timings and multi-card peer path are still unmeasured, and 25 longtail rows still have no
committed case of their own (this merge commits 16 of them through `cvd*`).

### 6.7 Round 5: the last two ready lanes taken (this session)

Session `warm-regulus` resumed from the Round-4 state (`714c1c79`), re-verified
the baseline on node2 (the six failing rows of §6.6 reproduced exactly), then
took the two lane branches that were built but never merged:

| Lane | Files taken | Verified |
|---|---|---|
| `port/conv3d-mtp` (base `32953e3a`) | `halo_upscaler.hip` (conv3d_ndhwc), `halo_encoder_bf16.hip` (attn_local_rel_bf16) — both files untouched on `main` since the lane's base, taken wholesale | `conv3d_ndhwc_bf16` **45 ok / 0 FAIL**, `attn_local_rel_bf16` **2 ok / 0 FAIL** |
| `port/router-m16` | `halo_router_gemm_scatter.hip` wholesale (unchanged on `main` since its base) + its 4-line `halo_selftest.cpp` delta applied by hand (M = 2, 3 added to the fused-case width list) | `router_gemm_topk_scatter_bf16` **394 ok / 0 FAIL** — the 14 known logits-only FAILs at M≤16 are gone |

`halo_stubs.cpp` regenerated after each take (`gen_stubs.py --check` rc=0;
the conv3d take removed 2 stub definitions). Of §6.6's six failing rows, three
remain, all weak stubs with committed cases: `hc_read_e4m3` (28 FAIL),
`hc_read_bf16` (3), `givens_dense_quant` (20). Those are the next work, then
the no-case stub sweep, the unified-memory core change, and full verification.

**The three remaining case rows landed** (same session):

| Row | Unit | Verified |
|---|---|---|
| `hc_read_e4m3` / `hc_read_bf16` | new `halo_hc_read.hip`: one path at every M — grouped-norm kernel, E4M3 planes decoded to bf16 scratch, two tiled WMMA products (down with the `silu(v/hc)` epilogue), inject kernel, and a mix kernel owning one 128-channel group a row so the plain and FWHT-rotated twins close in-block. The pin hook is accepted and ignored; every pinned form runs the same launches, so the byte-identical-form checks pass by construction. One bug on the way: the group amax reduced signed values instead of `fabsf` (32 % code misses), fixed. | `hc_read_e4m3` **38 ok / 0 FAIL**, `hc_read_bf16` **5 ok / 0 FAIL** |
| `givens_dense_quant` | `halo_givens_dense.hip` (already in tree) | **20 ok / 0 FAIL** |
| `mtp_enter_bf16` / `mtp_enter_e4m3` | new `halo_mtp_enter.hip`: norm-scale + norm-apply passes, E4M3 decode, two tiled products with the embedding product broadcast-added in a finisher. No committed case existed, so `halo_selftest.cpp` gains `mtp_enter_case` — a host reference of libref's stated math (`libref/ref_hc.cpp`), rows {1, 3, 16} × `ngroup` {4, 1} × both stored forms | **12 ok / 0 FAIL** (rel_l2 vs host reference) |

With that, every row with a committed selftest case passes. What remains is
the no-case stub sweep, the unified-memory core change, and full verification.

### 6.8 Round 5 continued: the stub sweep, unified memory, full verification (this session)

**The stub sweep.** `nm -D` on the built `libhalo.so` is the ground truth (the
text scan in `gen_stubs.py` cannot see macro-generated entries): of the 92
weak text stubs, 77 attention symbols are strong definitions via `GQ_SYM`
token pasting in `halo_attn_paged_gqa8.hip`. The genuinely weak 14 were:

| Symbols | Fix | Verified |
|---|---|---|
| `halo_rad_attn_dense_small_h256` | `attn_small_h256_kernel` ported into `halo_audio.hip` (LDS-staged K/V, two-pass softmax; no gfx12-isms) | committed refusal case passes |
| `halo_rad_attn_na3d` | both `attn_na3d_kernel<64>` and `attn_na3d_bf16_kernel<64>` ported into `halo_ltx.hip` | 2 numeric cases, rel_l2 4e-5 vs bound 8e-3 |
| 12 internal GEMM entries (`halo_gemm_fp8a16_nt_m1{,_block,_max_m}`, `nt_mt`, `halo_logits_gemm_{fp8,i8}`, `halo_gemm_w4a8_prefill`, `halo_gemm_w2a8_nt_group`, `halo_gemm_mxfp4a8_decode_max_m`, `halo_gemm_fp8a8_gated_nt_m16{,_max_m}`) | defined in their owning units over the same kernels the rows launch, with libr4d's stride arguments threaded through the two matvec kernels and libr4d's negative return codes. `w4a8_prefill` maps libr4d's prefill arm numbers to the tiled arms by (BM, BN) — the numberings differ. The gated entry takes this library's fragment-ordered plane, so its `w_ld` is accepted and ignored (documented at the definition). Constant queries: block 128, fp8a16 band 8, w2a8 group 128, mxfp4 decode band 128. | `nm` weak count **0**; the fp8/logits/w4a8/w2a8/mxfp4 selftest families all pass |
| `halo_gemm_fp8a8_pin_nt` | the pin selects the gated kernel's MT where it covers M (libr4d's rule); the ungated narrow kernel has no token-tile parameter, so its bytes cannot depend on the pin — the selftest's byte-identity check runs and passes | in the full run below |

Two scratch hooks also broke declare-time sizing (found by `convert_test`,
below): `halo_rad_gemm_fp8a8_m16_scratch` and `halo_rad_gemm_i8a8_m16_scratch`
read tensors through `fp8_ops`, but the builder calls scratch hooks with no
tensors. Both now answer from parameters / unconditionally, as libr4d's do
(the kernels split K in-block; the answer is 0).

**Unified memory (core, minimal).** The decision of §2 as one capability and
one ledger, nothing more:

- `RadDeviceProps.unified` (appended; `tests/abi_layout_rows.h` gained the
  field). `core/device/hip.cpp` fills it from the integrated attribute. On
  node2 it reports `unified=1`, VRAM 128 849 018 880 B — exactly the 120 GiB
  GTT cap.
- Single ledger: `VramFacts::unified/host_pinned`; `vram_budget_resolve`
  charges the rank's `--host-pool-mib` against claimable (and the report's
  unclaimed) only when every rank is unified, and prints the charge. New
  `tests/mem_test.cpp` case pins the arithmetic (claimable drops by exactly
  the pinned bytes; unchanged without the flag).
- Tiers collapse: `engine_bringup.cpp` gives the prefix-cache host tier zero
  capacity on a unified device (the tiers stay configured for the session
  record; nothing moves).
- Copies: `rad_memcpy_async`/`rad_memcpy_2d_async` return at once when source
  and destination are the same address.
- A live server start on node2 prints the ledger line in its budget report.

**Full verification (node2, this tree).**

| Check | Result |
|---|---|
| full `halo_selftest` | **8593 ok / 0 FAIL / 164 skipped** (§6.6: 8415 / 93 / 213 across 6 failing rows; no previously-passing row regressed) |
| `rad-kbench` all-rows vs libref (re-recorded fixture) | 261 passed, 32 failed — all 32 are one fixture artifact: `--record-rows` moves a 512-byte-row fp8 cache case into the `fp8_v4`/`k6_v4` rows, whose packed rows are 400/352 bytes; the reader (byte-identical to libr4d's) refuses the malformed operand. The engine's KVManager and the selftest build the real layout and pass. Not a libhalo defect; closing it needs kbench to synthesize packed v4 caches. |
| `ctest -L fast` | fails only on the documented pre-existing set (`gbnf_test_shard2/3`, `images_test`, `videos_test` — FFmpeg off; `plugin_test` — expects `libr4d.so`). `abi_layout_test` and `convert_test` failed on this session's changes and were fixed (above). `kbench_test`/`kbench_mutant_test`, baseline failures, now pass. |
| e2e | `radiance --model flux3b-chk.rad --kernels libhalo` served on gfx1151; a 256×256 image request ran the full pipeline and failed only at PNG encode (this node build has `RAD_WITH_FFMPEG=OFF`, pre-existing). Server stopped. |
| perf sample | kbench timing: cold read bandwidth 230 GB/s (of ~256 GB/s LPDDR5x peak), launch floor 1.86 µs. Per-op tuning debt remains as recorded in §7 — correctness and coverage are done; the tuning campaign is the open item. |

## 7. Performance notes and known debt

- "Everything tuned" is the user's bar; current state is *correct first,
  tuned where measured*. Known hot spots:
  - bf16 m64 latency-bound at M>16; dot2 m16 unexpectedly slow at N=1280;
    tiled GEMMs ~35 TFLOPS vs ~207 peak.
  - fp8a8 uses exact activation dequant (slower) and a wave-per-column
    design, not LDS-tiled.
  - wpa8i8 tiled: initially 7–8 TF/s. Diagnosis via code-object metadata
    (`.hip_fatbin` → embedded ELF → `llvm-readelf --notes`): 256 VGPRs,
    **151–195 spill slots**, 608–752 B private segment. Retiling
    <2,8,4,2> → <4,4,2,2> (BM 128×BN 128) gave **~25 TF/s** (3.6×) with
    spills down to **3 / 23 slots** (w4/w5) and 16 / 88 B private. The
    SWAR unpack is *not* the bottleneck (bypass experiment: 6.7 → 7.8
    TF/s). The parent `halo_gemm_w4a8_tiled` arms also spill (up to 353
    slots on the wide arms at a 192-VGPR cap; the lean arms TM·TN ≤ 4 do
    not), so this family wants small TM·TN on gfx1151. Further tuning
    (fold per half-group, occupancy) is deferred to the tuning phase.
  - R9700-derived tunable defaults (`wv/sk/mb/npw/nt`, MT choices, split
    laws) are inherited in the rows and must be re-measured on gfx1151.
- Profiler available on node2: `rocprofv3` (`--runtime-trace` works;
  counter collection not yet used).

### 7.1 Tuning pass (this session)

Measured with `rad-kbench --timing-only --report` against two rooflines:
230 GB/s cold read bandwidth and the launch floor (1.86 µs). Most large
rows are already at or near the bandwidth roofline (e.g. `logits_gemm_fp8`
at M=1 reads 2.4 GiB in 5.57 ms). Two outliers were fixed:

| Case | Before | After | Change |
|---|---|---|---|
| `gemm_mxfp4a8_decode` M=1, N=5120 K=25600 | 6727 µs | **1692 µs (4.0×)** | one tile a wave (NPW 2→1: 160→320 waves), an MT=1 band at M≤16 (no second, empty M tile), 8 waves a block; waves/block alone (16) had shrunk the grid to 10 blocks and was *slower* (5519 µs at ks=1 vs 4587). Plus the split law now takes the *smallest* split that fills the card (the old loop took the largest: 16 launches where 1 fills all 40 CUs). Per-wave tile and K order unchanged; family selftests (216 checks) green. |
| `sample_typical_f32` n_cand=1024 | 371 µs | **102 µs (3.6×)** | the typical threshold scan is O(n²) and recomputed `expf`/`logf` per pair (~10⁶ pairs). q and distance are now staged in shared memory once, and the scan loads them as `float4`. The mass still sums in candidate order per threshold — ref defines the cut by that sum — so results are bit-identical (kbench vs libref: 2/2 pass). |

Still open for the tuning campaign: the whole mxfp4 family shares the
fragment-conversion inner loop (at 1692 µs the decode remains ~6× above
the pure-bandwidth floor, level with `gemv`/`tiled` at 1370–1470 µs); fp8a8
m16 at ~130 GB/s is wave-per-column, not LDS-tiled; the bf16 tiled/m16
shapes and R9700-inherited tunables above still want re-measurement.

**mxfp4 tunables (same session).** A sweep over the skinny GEMM's
(WV, SK, MB, NPW) at the family's hot shape (N=5120, K=25600, M=1) found
the inherited defaults wrong for gfx1151 in two places: NPW 1 → 2
(1370 → ~990 µs) and the split ladder's largest legal split (SK 20 →
8–16, flat; 20 costs ~25 %). Both are now the mxfp4 rows' untuned
defaults: `cfg_for` gained an `npw_default` (fires only when the launch
carries no explicit `npw` tunable — the shared table always materialises
its own default, so absence is tested on the args) and an `sk_cap`
(untuned path only, and only to a split the K still divides). The gated
row keeps NPW 1 by design (it borrows only SK from the law; SK equality
is what its byte-identity to gemv rests on, and that still holds with
gemv at NPW 2 — 16/16 gated checks pass). Results at the hot shape:
`gemm_mxfp4a8_gemv`/`nt_m64`/`gemv_wide` **1367 → 971 µs (1.41×)**,
decode unchanged at 1692 µs. The probe env used for the sweep was
removed again; no `MXCFG` remains in the tree.

**Regression after all tuning:** full `halo_selftest` **8593 ok /
0 FAIL / 164 skipped** — the third consecutive full pass at that count,
so no tuned change altered any result.

### 7.2 Tuning pass 2

| Case | Before | After | Change |
|---|---|---|---|
| `gemm_fp8a8_nt_m16` / `gemm_fp8a8_tiled` M=1, N=16384 K=5120 | 613 µs (131 GB/s) | **469 µs (171 GB/s, 1.31×)** | two column tiles a wave (new NT template axis, NT=2 at MT≤2; NT=4 measured equal and not taken). The activation dequant (~180 ops) and its loads are paid once for two weight fragments — the same amortisation the gated kernel gets from its gate/up pair. Per-output accumulation order unchanged. An LDS activation-staging variant was tried first and *lost* (810 µs unstaged-padded, 714 µs padded — gather + syncs cost more than the redundant dequant), and was reverted. |
| `sample_topk_f32` n_vocab=248320 | 117 µs | **94 µs (1.24×)** | the radix select's histogram is now per wave-group (8 histograms, reduced after the count) instead of one shared histogram every element atomically queues on. Counts are integers, so the reduced histogram — and every result — is identical. `sample_logprobs_f32`, which shares the select, improved 133 → 113 µs with it. Fully private per-wave histograms (32 KB) were rejected: with the candidate array's dynamic shared they would overrun the 64 KB LDS limit at n_cand=4096. |

Regression: full `halo_selftest` **8593 ok / 0 FAIL / 164 skipped**
(fourth consecutive pass at that count).

### 7.3 Tuning pass 3 (negative results, kept out)

Three measured dead ends, recorded so they are not retried blind:

- **bf16 skinny tunables are flat.** A (WV, SK, NPW) sweep over
  `gemm_bf16_nt_m64`/`tiled` at M=65, N=1280, K=5120 moved nothing outside
  noise (96 ± 10 µs / 110 µs); forcing MB=1 costs ~1.8× (weight re-read).
  These kernels are latency-bound in the fragment load chain, not
  configuration-bound. (The probe env was removed again.)
- **mxfp4 decode NPW=2: 1692 → 3322 µs.** Unlike the m64 kernel, the
  decode part has no in-block split, so its wave count *is* its
  parallelism; two tiles a wave halves it. Reverted.
- **mxfp4 decode split law at 2 splits: 1692 → 1992 µs.** 640 waves do
  not beat 320 here; the extra launches and the reduce eat the gain.
  Reverted to the smallest-filling split.

What remains genuinely open: software pipelining (prefetch the next
K-step's loads during the current step's unpack/WMMA) in the shared
skinny-GEMM loop shape used by the bf16/mxfp4/quant families, and the
gated fp8 kernel's column widening. Both are kernel restructures, not
configuration changes.

Regression: full `halo_selftest` **8593 ok / 0 FAIL / 164 skipped**
(fifth consecutive pass at that count).

### 7.4 Tuning pass 4 (pipelining: already there; cost is conversion)

The software-pipelining candidate was implemented in
`halo_gemm_mxfp4a8_nt_m64_kernel` (a register double-buffer loading the
next 16-step during the current step's unpack/WMMA). It *lost*:
971 → 1042 µs, with VGPR 112 → 136 and 112 B of scratch — the buffered
structs spilled. A plain `#pragma unroll 2` on the original loop
measured 965 µs, i.e. noise (±0.6 %): the compiler already schedules
the loads ahead. Both forms were reverted; the tree is unchanged by
this pass.

The diagnosis that follows from it: the skinny family's cost is not
load latency but fragment-conversion VALU — per 16-step, ~180 ops of
exact activation dequant plus ~150 ops of e2m1 fragment build against
one or two WMMA. And `gemm_bf16_nt_m64` at M=65 (96 µs) is already at
its DRAM roofline once the MB split's ×2 weight re-read is counted
(26 MB at 230 GB/s ≈ 113 µs), so it has no headroom either. The
remaining lever for the mxfp4/fp8 families is a cheaper exact dequant
(the exponent-placement trick is exact for subnormals too, but zero
codes need a fixup path), estimated at ~2.5× off the dominant term;
that is a redesign of `deq16_exact`/`mx_frag16`, not a loop change.

Regression: full `halo_selftest` **8593 ok / 0 FAIL / 164 skipped**
(sixth consecutive pass at that count).

### 7.5 Tuning pass 5 (A-plane for fp8a8; attention OPT sweep — a dead end)

**fp8a8 A-plane (landed, kept).** §7.4 named fragment-conversion VALU
as the skinny family's cost: in `halo_gemm_fp8a8_kernel` every wave
re-ran `deq16_exact` (~180 ops) on the same activation codes for its
own column tile — at N=16384 that is 512+ wave-columns duplicating the
same dequant. The kernel now takes an optional activation plane: when
the caller supplies scratch, `fp8a8_adq_kernel` writes `deq16_exact`'s
outputs for the whole [M, K) activation once per launch (bf16, M·K·2 B)
and the waves read fragments back instead. The plane stores
`deq16_exact`'s own values, so products are bit-identical; accumulation
order is untouched. Wiring stays inside the `libr4d` interface: the
rows' scratch hooks now ask for the plane bytes
(`halo_rad_gemm_fp8a8_m16_scratch` returns the helper's 0 + M·K·2; new
`halo_rad_gemm_fp8a8_tiled_scratch`), and the rad entries pass
`a->scratch` through. Callers that size no scratch get the old
in-kernel path (`adq = 0`). The i8 twin deliberately does not use the
plane (its codes dequantise differently). Measured on the kbench row
`gemm_fp8a8_nt_m16`, M=1 N=16384 K=5120: **469 → 447 µs (1.05×)**,
bit-exact — `rad-kbench --op gemm_nt_q` 17/17 passed, fp8 family
selector case 1266 ok / 0 FAIL. The gain is real but small: at M=1 the
activation is one row, so the redundant dequant was a minor share next
to the weight-side fragment build, which the plane cannot remove
(each weight byte is read once per launch at decode shapes).

The same trick was **not** transferred to the mxfp4 gemv/m64 family:
those rows are weight-dequant bound (the e2m1 fragment build runs on
values each wave reads exactly once), so an activation plane removes
only a minor term there. Left as future work alongside the cheaper
exact dequant of §7.4.

**Attention decode OPT sweep (measured, reverted).** The paged decode
body (`halo_attn_decode_h256_gqa6.hip`, shared by every paged
geometry) compiles at 256 VGPRs with ~500 B/thread of spill scratch
(rocprofv3 on the `attn_decode_h256_gqa6_fp8v4` selftest case,
ctx 1024/2048). Its fragment-prefetch count PF (OPT bits 10–11,
currently 2) was swept: PF=4 made the kbench `attn_paged` gqa6 fp8kv
row slightly worse (21.65 → 22.7 µs); PF=1 cut spills to ~352 B and
read ~1% faster on the same rows (21.4 µs), within run-to-run noise
and with no measurable change at ctx 2048. Neither variant was kept:
the body is shared by all paged geometries (h128/h256/h512, windowed,
gated), and a noise-level gain does not justify re-validating every
geometry. The kbench attention fixtures are launch-bound microsecond
cases (q_len=1, KiB footprints); a long-context decode redesign
(starting from the 256-VGPR spill) needs a dedicated harness and is
the remaining attention debt, recorded here rather than attempted
blind.

Regression after the A-plane change: full `halo_selftest`
**8593 ok / 0 FAIL / 164 skipped** (seventh consecutive pass at that
count; log `/tmp/full7.log` on node2).

### 7.6 Tuning pass 6 (branchless e2m1 fragment build; A-plane transfers to mxfp4)

**Branchless `mx_frag16` (landed, kept).** §7.4's remaining lever was
the e2m1 fragment build: `mx_e2m1_bf16` ran a branchy scalar decode
per code (~15 ops × 16 codes per fragment, in every mxfp4 kernel's
inner loop). It is replaced in `halo_gemm_mxfp4a8_nt_m64.hip`,
`halo_gemm_mxfp4a8_decode.hip`, `halo_gemm_mxfp4a8_gated.hip` (and as
`mxt_frag8` in `halo_gemm_mxfp4a8_tiled.hip`) by a perm-table build:
the eight magnitudes' bf16 patterns (0, .5, 1, 1.5, 2, 3, 4, 6 at
ediff 0) sit in two `v_perm_b32` tables, the per-block exponent shift
is one packed add, and magnitude 0's table entry is synthesised per
call as `-(ediff << 7)` so the add cancels it back to signed zero —
no branches, no fixup selects. Two hard-won facts are recorded in the
code: `v_perm_b32` selects bytes 0–3 from its **second** operand
(probed on gfx1151 with a dedicated kernel after the first version,
built on the documented order, produced rel_l2 ≈ 1.0 across the
mxfp4 cases), and the equivalence is proven, not sampled: a host
model of the exact integer ops matches the scalar decoder for every
code × every ediff in [-126, 126] (the full range with
0 ≤ E + ediff + 126 ≤ 255, which the relayout's reference exponent
guarantees) plus 200k random dword pairs. Measured on the kbench rows
(N=5120 K=25600): gemv 971 → **856 µs**, gemv_wide 972 → **860 µs**
(1.13×) before the A-plane below.

**A-plane for the mxfp4 m64 family (landed, kept).** §7.5 declined
this transfer on the reasoning that the family is weight-bound; with
the fragment build cheapened, the redundant activation dequant (every
wave re-running `mx_deq16_exact` on the same rows for its columns)
became the larger term, so the fp8a8 pattern was applied after all:
`mxfp4_adq_kernel` writes the [M, K] bf16 plane once per launch, the
rows (`gemm_mxfp4a8_nt_m64`, `gemm_mxfp4a8_gemv`,
`gemm_mxfp4a8_gemv_wide`) declare it through
`halo_rad_gemm_mxfp4a8_m64_scratch` (M·K·2 bytes, parameter-only so
declare-time sizing works), and callers without scratch keep the
in-kernel path. Products are bit-identical (the plane stores the
exact decoder's outputs). Measured: gemv 856 → **442 µs**, wide
860 → **444 µs** (1.94×; 2.2× against the 971 µs pre-pass baseline).
The mxfp4 decode kernel (`mxd_*`) keeps its in-kernel decode: its
scratch already carries split-K partials, and carving a plane out of
that layout is a separate change.

Verification for the pass: mxfp4 selftest cases at their recorded
counts (gemv 97, decode 86, nt_m64 6, tiled 17, wide 10, gated_gemv
byte-identical), `rad-kbench --op gemm_nt_q` **17/17 passed**, weak
`halo_*` symbols in the built library still **0**, targeted ctests
(convert/abi_layout/mem) 5/5 passed. Full regression: `halo_selftest`
**8593 ok / 0 FAIL / 164 skipped** (eighth consecutive pass at that
count; log `/tmp/full8.log` on node2).

### 7.7 Tuning pass 7 (in progress — blocked on node availability)

Both cluster nodes became unreachable (`No route to host`, 100% ping
loss on node2 and node3) during this pass, so the work below is written
locally but **not yet built, measured, or verified**; it must not be
read as landed.

- **mxfp4 decode A-plane (written, untested):** `mxd_adq_kernel` in
  `halo_gemm_mxfp4a8_decode.hip` builds the [M, K] bf16 plane behind
  the split-K partials in scratch (hook now returns
  `ks·M·N·4 + M·K·2`); `mxd_partial_kernel` reads fragments from it
  when present. Rationale: at NPW=1 the activation decode is ~⅔ of
  the kernel's conversion work and every wave repeats it. First
  action when the node returns: build, run
  `--case gemm_mxfp4a8_decode` (expect 86 ok), time against the
  1692 µs recorded baseline, keep only if faster.
- **Queued diagnoses:** the §9 bf16_nt_m16 skinny case (N=1280, M=1,
  59.2 µs — compute its GB/s from the fixture geometry before
  touching it) and `gemm_fp8a16_nt_m1` (511 µs ≈ 157 GB/s at the
  kbench shape; its inner loop is a 16-deep dependent FMA chain over
  software-decoded e4m3 pairs — check whether
  `__builtin_amdgcn_cvt_pk_f32_fp8` is hardware on gfx1151 before
  restructuring; the row is numeric-tolerance (rel_l2 < 3e-2), so a
  partial-sum split is permitted if it pays).

### 7.8 Tuning pass 8 (session `third`, 2026-10-28) — prefill tiles, offline toolchain

**Nodes unreachable again at session start**: not the nodes — the dev box's VPN tunnel
to them was down and needed the user to re-authenticate. Everything below waits on that.

**Local gfx1151 toolchain (new).** Arch's ROCm 7.2.4 packages (`rocm-llvm`,
`hip-runtime-amd`, `rocm-device-libs`, `rocm-core`, `comgr`, `hsa-rocr`) extracted —
not installed — to `~/.cache/halo-rocm-7.2.4/root` (7.1 GB on disk). A configure of the
whole tree as `build-gfx1151/` (`-DRAD_GPU_TARGETS=gfx1151`, that ROCm's clang as
C/CXX/HIP compiler with `--rocm-path`, `-DCMAKE_HIP_COMPILER_ROCM_ROOT`) builds
`libhalo` + `halo_selftest` in ~30 s. No card, so nothing runs; what it buys is compile
errors and the code-object metadata (VGPRs, spills, LDS) per kernel without the node.
Its first catch: §7.7's decode A-plane did **not compile** (`mxd_adq_kernel` used before
its declaration) — fixed, and its three `getenv("MXD_*")` sweep probes removed from the
launch path.

**Spill census** (all 2313 kernels of the built library): 430 spill. The worst are
tuner-only variants (NPW 4/8 skinny GEMMs at MT 3–4: `w4a8_nt_m64<4,8>` 1441 slots) that
the default path never takes. Three default-path families did spill and are rewritten:

| Row (prefill band) | Before | After |
|---|---|---|
| `gemm_fp8a8_tiled` (49 of 60 recipes are block-fp8: every DiT step and LLM prefill) | the narrow wave-per-column kernel at MT 4: no operand reuse, 203 spill slots | `halo_gemm_fp8a8_tiled_kernel`: bf16 tiled's block shape (128×128 / 64×64, 64-K LDS tiles, register prefetch, grouped-M order); A from the exact plane or decoded while staging, W decoded once per block; **0 spills** (231 / 90 VGPRs) |
| `gemm_i8a8_tiled` | same narrow kernel at i8 | `halo_gemm_i8a8_tiled_kernel`: bytes staged, iu8 WMMA; **0 spills** (197 / 92) |
| `gemm_mxfp4a8_tiled` | fp8 activation fragments decoded by every wave at every step (~3× the WMMA work), 116–589 spill slots in every instantiation | `halo_gemm_mxfp4a8_tiled2_kernel`: both operands as exact bf16 in LDS, m64's per-32 fold; **0 spills** (222–242 / 94–124) |

Each keeps the per-output arithmetic of the row it must match, so the contract is
**byte identity**, not a tolerance: fp8/i8 tiled == the narrow `nt_m16` row over 64-row
slices; mxfp4 tiled == `nt_m64` at `sk=1` over 64-row slices (the old tiled kernel
carried its accumulator in scale ratios and was only close). New selftest cases
`fp8_tiled_bytes_case` (fp8 and i8, both tile arms) and `mxfp4_tiled_bytes_case` (both
arms, both scale grids, ragged N) check plane / no-plane / slices bytewise.
Interface unchanged: row names, ops, constraints, tunables and the raw `halo.h` entries
(`halo_gemm_mxfp4a8_tiled`, `halo_mxfp4a8_tile_a[_bytes]`) are as before; the mxfp4
row's scratch hook is now `halo_rad_gemm_mxfp4a8_tiled_plane_scratch` (M·K·2), the
w4a8i8/w5a8i8 rows keep the fp8-fragment hook. Both plane hooks cap at 256 MiB (past it
the kernels decode while staging rather than the program reserving the plane).

A review fork desk-checked all three kernels against their references and simulated the
index maps exhaustively (block remap over every grid 1–19 × 1–39; every staged LDS position
of the fp8/i8 frag64 and mxfp4 weight maps, both tile widths): **no bugs found**.

**Interface parity with libr4d, measured (new).** A second local tree, `build-both`
(`RAD_GPU_TARGETS="gfx1151;gfx1201"`), builds libr4d beside libhalo, and the new
`halo_parity` tool (`libhalo/halo_parity.cpp`, ctest `halo_parity`, label `fast`, present
whenever one build has both plugins) compares the two tables: **245 schemas identical
(params, types, required, roles, operands), 486 rows identical (name, op, domain,
constraints in order, tunables with values and defaults), concurrency promises identical
(414 rows), collective promises identical (34 + 22)** — 0 interface differences; the
negative control (libref in libhalo's place) reports 2133. Reported, not failed: 4 rows
declare a scratch hook libr4d's do not (the activation planes: `gemm_fp8a8_tiled`,
`gemm_mxfp4a8_nt_m64/gemv/gemv_wide`; every one runs without scratch too), and the chain
promise (below).

The raw `halo.h` surface was checked the same way (`nm -D`, demangled, through the rename
transform) and had drifted in five places, all fixed to libr4d's exact signatures:
`halo_gemm_mxfp4a8_nt_m64` (had gained `scratch, scratch_bytes` — the plane variant is now
the additive `halo_gemm_mxfp4a8_nt_m64_plane`), `halo_gemm_fp8a8_nt_m16` /
`halo_gemm_i8a8_nt_m16` (had gained `sk_want` — now internal `fp8_nt_m16` / `i8_nt_m16`),
`halo_gemm_fp8a8_tiled` (declared with libr4d's signature but defined only with two extra
parameters, so libr4d's symbol did not exist), and `halo_gemm_w2a8_nt` (declared extern "C"
with `asum`/`NT`, defined as a C++ overload without them — now libr4d's C-linkage signature,
`asum` unread since the zero is folded into the fragment bytes). Remaining export
differences are libr4d-internal helpers that no row and no header names
(`r4d_attn_decode_gqa{1,2,4,6,8,12}_scratch_bytes`, `r4d_rad_attn_decode_gq_h256_gqa{4,8}_*`)
plus libhalo's additive entries.

**Chained launches: the marker was renamed away (fixed, promise still off).** The AQL
backend recognises a chained kernel by `"r4dchain"` in its name (`core/device/aql.cpp:515`,
RadKernelChainFn); the rename transform had turned every chained form into `halochain_*` /
`halo_chain_attn_*` / `halo_win_chain_*`, so none could ever chain. All 413 chained kernels
are renamed to `halo_r4dchain*` (and `PORTING.md`'s transform now keeps the marker).
`rad_kernel_chain` still returns 0 — libr4d promises 45 rows — so nothing changes at run
time yet; chaining is only on for a speculative drafter whose plugin measured a gain
(Gemma 4 DFlash) on the AQL backend, and promising it needs each row's chained path audited
(every block reaches `halo_chain_done` once) and an e2e chained run on the node.

The audit is done (a fork traced each of libr4d's 45 chained rows through libhalo at
`chain = 1`): **39 rows are safe to promise as they stand** — every kernel on their chained
path is a `halo_r4dchain*` form wrapping its body between `halo_chain_wait` and
`halo_chain_done`, early returns inside the wrapped body, no protocol bug found; plain
kernels on a chained path occur exactly where libr4d has them (the backend keeps their
barrier). **6 rows have no chained form yet** (`gemm_bf16_nt_m16`, `gemm_fp8a8_nt_m16`,
`gemm_fp8a8_gated_nt_m16`, `gemm_i8a8_nt_m16`, `logits_gemm_fp8`, `logits_gemm_i8`):
promising them is safe but buys nothing until each gets a wrapper (the fp8/i8/logits bodies
return early, so their bodies must move into `__device__` functions first; under `chain`
the fp8 row must skip its A-plane kernel or chain it). The promise stays **off** this
session on purpose: the selftest launches through HIP and can never engage a chain record,
no chain-paying model (Gemma 4 DFlash) is on hand for an engine run, and a protocol slip
would show as a hang at decode. Turning it on = the 39-row list in `rad_kernel_chain`
after one speculative e2e run on node2; caveat to check then: a chained kernel whose
argument block passes byte 992 silently runs unchained (`aql.cpp:1121`).

Also measured offline: gfx1151 has **no** `__builtin_amdgcn_cvt_pk_f32_fp8`
(`__has_builtin` false; gfx1201 true) — §7.7's question; the fp8a16 matvec's e4m3 decode
stays software.

**Host test tier, local build (no card):** the whole tree builds with the extracted ROCm;
`ctest -L fast`: 100/105 pass; the five failures are the documented set (`gbnf_test_shard3`,
`images_test`/`videos_test` — FFmpeg off, `plugin_test` — expects `libr4d.so`, which a
gfx1151-only build does not make) plus `kbench_mutant_test`'s one device-timing check,
which needs a card (it passes on node2, §6.8).

**Card-less execution: `tests/halo_emu` (new, durable).** tests/r4d_emu's SIMT emulator
(every GPU thread a fiber, cross-lane ops as wave barriers) gained gfx11 forms —
`v_wmma_f32_16x16x16_{bf16,f16}` and `v_wmma_i32_16x16x16_iu8` on halo_wmma.h's lane map
(sixteen consecutive k a lane, D at `[2e + (l>>4)][l & 15]`), `cvt_pkrtz`, the scheduling
hints, `HALO_DYN_LDS`, `hip_bf16.h`/`hip_fp16.h` — and `tests/halo_emu/test_gemm.cpp`
(ctest `halo_emu_gemm`, labels `halo_emu;fast`, 2.4 s) runs the plugin's own sources for
`halo_gemm_fp8a8`, `halo_gemm_mxfp4a8_{tiled,nt_m64,decode}` on the host through their
`halo_rad_*` entries. To make those units emulator-clean their five `<<<>>>` launches became
`hipLaunchKernelGGL` and their `extern __shared__` arrays `HALO_DYN_LDS` (code-object
metadata of every kernel in the three units unchanged). Results, all **0 differing
elements**: fp8 tiled (plane and no plane) == `gemm_fp8a8_nt_m16` over 64-row slices on both
tile arms; i8 tiled == `gemm_i8a8_nt_m16` likewise; mxfp4 tiled == `gemm_mxfp4a8_nt_m64`
at `sk=1`, both scale grids, ragged N; mxfp4 **decode with its §7.7 activation plane ==
without it at ks 1/2/4** (so §7.7 is now verified for bytes too). The narrow fp8 row
against a float64 reference reads rel_l2 1.7e-3, which pins the emulator's gfx11 lane map
to the arithmetic. Mutants of the staging maps (fp8/i8 second k-run offset, mxfp4 fold
cadence) fail the test; the one surviving mutant (dropping the 64-wide mxfp4 tile's
duplicate-store guard) is equivalent — the upper threads would write the same bytes. All
19 r4d_emu tests still pass with the shared emulator extended.

**Rows no check had ever reached, now checked.** A static census (row names against the
selftest's literals and the committed kbench coverage lists) found the libr4d ops that have
*no libref oracle by design* — the sandwich-norm joins, `expert_scale`, `pair_argmax` —
had no committed case in libhalo either, so rad-kbench never checked them and nothing else
did; the joins run in every layer of every Gemma 4 trunk. r4d_emu's tests of those ops
(the unfused chain bit for bit, padding untouched, plus a host arm) port by the rename
transform: `halo_emu_norm`, `halo_emu_escale`, `halo_emu_pick` (units
`halo_norm_join_bf16`, `halo_expert_scale_bf16`, `halo_pair_argmax_f32`; the emulator
gained `HALO_KLAUNCH`/`HALO_EMIT`). Results: `norm_add_norm`, `norm_add_norm_route`
(with its fp8 group quant) and `join_add_scale` **0 differing elements from the chain** at
every width/pitch tested, a zero group included, padding untouched; `expert_scale` 32
cases, `pair_argmax` 80 cases, 0 failures. `ctest -L halo_emu`: 4/4, 2.4 s.

**Every r4d_emu test now has a libhalo twin (19 ctests, label `halo_emu`).** The rest of
r4d_emu's suite ported by the rename transform onto the libhalo ports of the same units:
windowed paged attention (`halo_attn_paged_window`: 3 shards + the seeded sweep; by hand
`sweep 200`: 200 shapes, bf16/fp8 KV, causal/SWA, ragged, split path 57 and WMMA 67,
**0 failures**, rel_l2 ≈ 1.7e-3 vs the libref port), QSA selection, the LoRA pair (held to
libref itself: 1.62 M outputs, worst |err|/bound 0.995), GLM's F32 routing projection and
MoE decode fusions, mHC, the F32-scale E4M3 GEMM, KDA (against libref: 4/4 in bounds), the
MXFP4 expert GEMM, MLA, kv_resolve and DSA's split — **all pass**. Two copies no longer
assert *which* libr4d kernel form ran (`test_fp8f32`, `test_moemx`: libhalo's units are
gfx1151 designs with their own forms — one `mx_moe_kernel` serves every MoE-MXFP4 shape);
every output comparison stands, and every output matched to the last word. The emulator
gained what these sources need (`__builtin_amdgcn_fdot2`, `__fmaf_rn`, `__shfl_sync`,
`__any`); `halo_gemm_nt_f32.hip`'s one `<<<>>>` became `hipLaunchKernelGGL`. Labels: 17
`fast` (~25 s), `halo_emu_attn_sweep` and `halo_emu_moemx` (~160 s, libhalo's single MoE
kernel is slow to emulate) `slow`.

**The longtail rows, held to libref (`halo_emu_ref`, new).** The census's other gap: the
image/video/audio/residual helpers no model fixture of rad-kbench reaches and no selftest
case names. `tests/halo_emu/test_ref.cpp` runs each libhalo row on the emulator and libref's
row for the same op on the host, on operands drawn from **libref's own operand
descriptions** (RadShapeFn) at a hand-picked geometry (the schema is shared), and compares
every written operand **byte for byte** — 22 cases: `gelu_tanh`, `cast` (f32→bf16,
bf16→f32), `scatter_rows`, `cu_repeat`, `randn_lanes` (bf16, f32 pitched — the Philox
stream matches), `to_rgba8`, `d2s_add`, `s2d_avg_add`, `ln_mod`, `l2norm_chan`,
`gather_rows2` (signed index over both sources; libref declines to describe it, so the test
builds it), `scatter_add_rows`, `chan_split`, `resample_nearest`, `fourier_embed`,
`end_status` ×2, `hc_enter` — all **bit-identical**; `l2norm_chan` with silu and
`hc_write` within rel_l2 5.2e-4 / 2.8e-4 (a 32-lane norm against libref's loop; their
bound is 1e-2). A 0.01 % change to gelu_tanh's cubic coefficient fails it (19 bytes); an
all-zero reference output counts as a failure, not a pass. Units added to the emulator
build: `halo_gen`, `halo_audio`, `halo_encoder_bf16`, `halo_model_bf16`, `halo_hc_bf16`,
`halo_p2p_copy2d` (`halo_gen.hip`'s one `extern __shared__` → `HALO_DYN_LDS`; the emulator
gained `__f{add,sub,mul,div,sqrt}_rn`, `float2`, `uchar4`, `multiProcessorCount`, an exact
`cvt_pk_f32_fp8`, `__threadfence_system`, `hipMemcpy2DAsync`, and zero-argument
`hipLaunchKernelGGL`). `ctest -L halo_emu -LE slow`: **18/18 in 19 s**.

**The paged attention bodies on the emulator (`halo_emu_paged`, new) — and three emulator
fidelity bugs it exposed.** `tests/halo_emu/test_paged.cpp` drives every block-size-16
`attn_paged` decode and prefill row (36 rows: head dim 64/128/256, GQA 1–8, bf16 and fp8
caches; the table `paged_rows.inc` is generated from `halo_rows.cpp` with each row's launch,
scratch hook and q_len/window/causal limits) through test_attn's cache construction (NaN in
every slot past a context and every unnamed block) and its port of libref's
ref_attn_paged: decode at one token and verify heights, ctx 1/17/300/2049 (split-KV),
ragged batches, windows, non-causal, fp8 with per-head descales; prefill over ragged causal
batches. **166 cases, 0 failures** (rel_l2 ≈ 1.6–2e-3). Getting there found no kernel bug
but three emulator defects, each of which had made a correct kernel look wrong:
1. **Integer shuffles converted through float.** The emulator defined `__shfl`,
   `__shfl_xor`, `__shfl_down` for `float` only; HIP's int/unsigned forms are bit moves.
   libhalo's gfx11 `load_tr_b128` shuffles dwords of two packed bf16, which kept 24
   significant bits — the low value lost its mantissa (even head dims coarse, odd exact).
   Bit-exact int/unsigned overloads added; `__shfl_sync` is now a bit-cast template.
2. **A wave exchange demanded all 32 lanes.** The prefill body returns its dead query rows
   (`if (!live) return;`) before the row-sum `swap16`, whose partners are always both live
   or both dead — legal on the card, a reported deadlock here. Exchanges now complete over
   the lanes still running (a returned lane's slot is poisoned, `0x7fbadbad`, so reading
   it shows); a WMMA still requires all 32 lanes, checked at arrival, and a lane returning
   while others wait at a WMMA it skipped is reported.
3. **"Absent" was decided at read time.** A partner that posted its value, let the exchange
   complete, and returned before a slower lane read it had its value overwritten with the
   poison (-245.6 and NaN in the last head dims: the halves of `0x7fbadbad`). Absence is
   now the set of lanes already returned when the exchange *completes*.
All 19 r4d_emu tests pass on the corrected emulator. Also: `halo_attn_decode_h256_gqa6.hip`'s
one `extern __shared__` → `HALO_DYN_LDS` (all 473 paged-attention kernels' metadata unchanged);
`cfg_host.cpp` gained `halo_cfg_split`.

**Review fixes (second review of the non-kernel changes).** `test_fp8f32`: with libr4d's
kernel-name assertions gone, the dense cases compared one kernel at two pitches — now each
dense shape is also held to a float64 restatement (13 checks; 43 cases pass). `test_gemm`:
the i8 tile has no plane, so its two lines are one. The custom-command (non-clang) route
links `-ldl` for the dlopen tests. `halo_parity` now also compares **row priority and
declaration order per op** (they decide selection on a tie): identical, still 0 interface
differences.

**Perf candidates, prepared as patches (emulator-verified for bytes, not yet timed)** in the
session scratchpad `cand/` with a README and `node_ab.sh cand` A/B stage: P1 fp8 narrow
16-byte weight loads (56 → 40 load instructions a group), P2 the activation plane for the
gated fp8 row (new scratch hook, as §7.5 did for the ungated row), P3 mxfp4 half-wave loads
(each half loads its own 16 dwords, `v_permlanex16` swaps — the two halves used to fetch the
same 64 B), P4 the int8 twin's 16-byte weight loads (16 `b64` → 8 `b128` a group), P5 the
gated row's epilogue amax as a DPP reduction over the wave's 16 columns plus 8 LDS reads
(the epilogue's LDS loads 504 → 32 a tile; `fmaxf` is exact and order-free). All five apply
to the baseline and on top of each other in order; with all five applied the emulator GEMM
suite passes. Each lands only if node2 measures it faster. The census of load widths that
motivated P1/P3/P4: the one decode GEMM at ~96 % of bandwidth (`bf16_nt_m16`) issues only
16-byte loads, the fp8/i8 narrow kernels streamed their weight in 8-byte and mxfp4 in
4-byte loads with both half-waves fetching the same bytes (w4a8/w8a8/w4a16/w5a16/w8a16,
fp8a16 and the logits kernels already use 16-byte loads). Two permanent contract tests came
out of this: `gemm_mxfp4a8_decode` at one split == `gemm_mxfp4a8_nt_m64` at `sk=1`, and
**`gemm_fp8a8_gated_nt_m16` == `gemm_nt_q` + a host restatement of `gated_quant_fp8`**
(codes, scales and the bf16 product, byte for byte — the row table's own contract, now
checked without a card); both pass on the baseline.

`ctest -L halo_emu`: **21/21** (19 `fast`, ~35 s; `attn_sweep` and `moemx` `slow`). Since then
`test_gemm` also holds **`gemm_bf16_nt_tiled` == `gemm_bf16_nt_m64`** at the same shape (both
arms, split K, ragged N; `halo_gemm_bf16_nt_m64.hip`'s `extern __shared__` → `HALO_DYN_LDS`,
metadata unchanged).

**Decode attention, the spill analysed (not changed).** Every paged decode instantiation at
head dim 256 spills (`halo_attn_decode_kernel<NW,16,256,2,…>`: 211–237 slots, ~800 B/lane).
Cause: gfx11 fragments are 16 elements, so the loop-invariant Q fragments libr4d keeps in
registers (`qf[NKS]`) cost 16 × 8 = 128 VGPRs, beside 128 of accumulators and, at one wave a
block, 64 of V staging (`vreg[16]`). The spilled values are most likely exactly those Q
fragments, i.e. the "spill traffic" is a cached reload of Q per 16-key tile (~8 KB a wave-tile
against ~16 KB of K/V). Moving Q to LDS would free the registers but costs 8 KB (NW 1) to
32 KB (NW 4) of LDS a block, and LDS already bounds occupancy for these one-wave blocks — the
trade is a measurement, not a reasoning exercise. First node experiment for it:
`halo_selftest --perfpaged <q> <ctx>` at ctx 8k/32k for the baseline. **Tried offline and
discarded:** Q fragments in LDS (written once by each row's h = 0 lane, read just before their
WMMA) — bytes unchanged (`halo_emu_paged` 166/166) but the spills got *worse* (177–321 VGPR
slots plus up to 186 SGPR spills, against 188–237): the Q fragments were not the pressure the
arithmetic suggested; the loop's schedule is. A real fix is a restructure measured on the card.

A host-side tangent noted for the record: node2's bare-metal build runs `RAD_WITH_FFMPEG=OFF`
because `core/mm/mp4.cpp` does not compile against the distribution's FFmpeg 6 (the Docker
image builds its own FFmpeg, so the shipped path is unaffected); image/video models on a
bare-metal node2 build cannot encode their output. Not attempted this session.

**Status: written, compiled, ISA-checked, emulator-verified for bytes; NOT yet run on the
card (speed unmeasured).** First actions when the tunnel is back: build on node2, `--case gemm_fp8a8_tiled`, `gemm_i8a8_tiled`,
`gemm_mxfp4a8_tiled`, `gemm_mxfp4a8_decode` (the §7.7 plane), then A/B timings
(`perf_row`, `--perfmxd`) against the pre-change library, then the full selftest.

**End of session (17:30):** the tunnel never came back (node2 "No
route to host"), so none of §7.8 has touched a card. The whole node session is one script:
the scratchpad `node_run.sh` (copy kept in `~/.cache/halo-session-backup/20261028-1052/`);
its stage `save` must run before `sync` so the A/B base is the pre-change library. Candidates
P1–P5 are kept only if measured faster; P1–P5 are not applied in the working tree.

**On the card (same day, VPN back at ~17:40).** §7.8's kernels on node2: every gate case
0 FAIL; full selftest **8615 ok / 0 FAIL**. Timed against the pre-session library (selftest
perf modes, N K M): fp8 tiled 16384·5120·2048 74.5 → **8.1 ms (9.2×)**, 3072·3072·4096 15.6 →
2.1 ms; i8 tiled 15.5 → 8.2 ms (1.9×); mxfp4 tiled 5120·25600·2048 128.9 → **21.0 ms (6.1×)**;
mxfp4 decode M=1 1534 → 1002 µs (§7.7's plane). P1/P3/P4 measured no gain beyond noise, P2 no
longer applies, P5 was gated on the wrong case: none applied (patches stay in the backup).

### 7.9 Tuning pass 9 (2026-10-28 evening): profile-driven, every row timed on the card

Method: a whole-library `rad-kbench --timing-only` on node3 over the committed fixtures ranked
the rows by distance from their floor (bandwidth 230 GB/s cold, ~40 TFLOP/s for the f16/bf16
WMMA); the worst ones were fixed, each A/B-timed on node2/node3 and checked by its selftest
cases, the emulator and kbench against libref. Two of the six items (logits, paged prefill) were
done by delegated agents in isolated copies on node3 and merged as patches.

| Row / shape | Before | After | Change |
|---|---|---|---|
| `gemm_fp8a8_gated_nt_m16` N=34816 K=5120, M=64 / 32 | 27.2 / 11.3 ms | **1.22 / 1.15 ms** | past M=16 the row runs the pair it is byte-identical to: the tiled GEMM into the op's scratch (new scratch hook, M·N·2 + the A plane), then `gated_quant_fp8`. The fold launched N/512 blocks whatever M was and spilled at MT 3–4 |
| `gemm_fp8a8_nt_m16` (and the tiled row) at 17 ≤ M ≤ 64 | 4.96 ms (N=34816 M=64) | **1.23 ms** | the LDS-tiled kernel where the split law gives SK 1 (where its bytes are the narrow kernel's); 2.5–7.4× across the fixture shapes; ≤16 rows unchanged (the narrow kernel wins there) |
| `gemm_mxfp4a8_decode` M > 64 | 3.23 ms (5120·25600, M=128) | **1.34 ms** | routed to the tiled row's kernel; the row's bits stay M-invariant through 64 rows (selection already changes rows at 64 → 65) |
| `gemm_w8a8_tiled` / `gemm_w4a8_tiled` / `gemm_w4a8_prefill` | 11.7 / 9.7 TFLOP/s (default) | **~33 / ~29** | libr4d's `__launch_bounds__(…, 8)` capped a lane at 192 VGPRs (default tile spilled 158–165 slots); dropped, and the untuned default arm is now arm 59's 4×1 tile (BM 128 × BN 128). Every arm accumulates in the same K order: new selftest check `every tile arm == the default` (15 arms each, also passes on the old libraries) |
| `logits_gemm_fp8` / `_i8`, n_embd 5120, M = 8 / 16 / 32 / 64 | 8.1 / 16.2 / 32.3 / 65.0 ms | **5.7 / 5.8 / 5.9 / 6.3 ms** | (agent) M ≥ 2 (i8: ≥ 4) on a bf16-WMMA kernel: 4 waves × 16 vocab rows a block share an LDS x tile of up to 64 rows, each head code decoded once; M = 1 keeps the bandwidth-bound matvec (5.56 ms). Not byte-identical to the old kernel (≤ 4.4e-7 of max|y|); kbench vs libref 26/26 |
| paged decode attention, head 256 (36 rows), 32 MiB KV, q_len 1 / 4 / 8 | 717 / 420 / 642 µs | **213 / 269 / 456 µs** | split-D: two waves a 16-row tile, each holding half of Q and of O (64 + 64 VGPRs instead of 128 + 128), partial S tiles summed through LDS in slice order. Spills 180–452 → ≤ 11 slots. 60 kbench decode cases: 9.5 → 5.8 ms total; h128 rows compile as before |
| paged prefill attention (37 rows), q_len 11 / 132 / 512 over ~32 MiB | 6.44 / 9.21 / 23.7 ms | **0.62 / 2.90 / 9.06 ms** | (agent) split-KV across workgroups with a merge kernel (new scratch hooks on 33 prefill rows), empty-row warps skip the m-tile loop, and `O_KLATE` dropped from the gqa6 option set (it took fp8/k6_v4 from ~90 to ~300 spill slots). 60 kbench prefill cases 153.7 → 43.5 ms; kbench vs libref 240/240 |

**Final state on node2 (all of the above merged):** full `halo_selftest` **8617 ok / 0 FAIL /
164 skipped**; `rad-kbench` against libref over both fixtures **2690 checked, 2690 passed**;
whole-library kbench timing against the pre-session library: 1133 libhalo cases **882.9 →
278.2 ms (3.17×)** — fp8 tiled 7.2×, logits 3.2×, paged prefill 3.5×, mxfp4 tiled 5.7×, fp8
narrow 1.7×, fp8 gated 5.6×, mxfp4 decode 5.8×, paged decode 1.6×; no changed row slower (the
only >5% "regressions" are unchanged sampling/gdn_conv kernels of 50–380 µs, run-to-run noise).
Merge note: the prefill agent's patch was cut against a copy that predated the gated row's
scratch hook and silently reverted those two lines (`halo_rows.cpp`, `halo_plugin.h`); caught by
the whole-library timing (gated M=32 back at 9.2 ms) and restored.

Interface: `halo_parity` **0 interface differences** (245 schemas, 486 rows); the new scratch
hooks show only as informational hook differences (38). Emulator: all 21 `halo_emu` suites
pass (`halo_emu_paged` now 202 cases); the emulator links a refusing stub for
`halo_gated_quant_fp8` (`tests/halo_emu/cfg_host.cpp`) since it does not build
`halo_fused_quant_fp8.hip`; the gated pair is covered on the card (42 byte checks in
`--case dequant_w4_bf16`, which is where the fp8 numeric suite runs).

Measured and **not** changed: the decode split law (256/512/1024/2048 workgroup targets all
within noise or worse); an mxfp4 tiled zero-C first step (byte-neutral, no gain).

**Open (measured, not fixed)** — most of this list is addressed in §7.10: the spills, mxfp4
tiled and mid-M, and the sampler at small M. The subnormal note stands:
- `gemm_mxfp4a8_tiled` runs ~26 TFLOP/s against fp8 tiled's ~40 (its per-32 fold and e2m1
  unpack); at mid M the gemv_wide/decode rows (M 33–64) are 15–35% slower than the tile but are
  held by the gemv rows' tested M-invariance to 64 rows.
- `sample_topk_f32` / `sample_logprobs_f32` at M=64 cost ~4× M=32 (multi-pass radix reads of
  64 × 1 MiB rows overflow the 32 MiB MALL); < 1% of a batch-64 decode step.
- `gdn_chunk_scan` (all 8 instantiations spill 357–379), prefill attention at head 256 still
  ~90 spill slots (the agent's half-Q attempt lost to the scheduler), the h128 windowed decode
  forms (77–92 slots).
- **Subnormal fp8 (reported by the logits agent, measured on the card, not independently
  re-verified):** the gfx11 bf16 WMMA flushes subnormal inputs, so `deq16_e4m3`'s E4M3
  subnormal codes (|x| < 2^-6 before scale) count as zero in the fp8a8/moe kernels — §7.4's
  "exact for subnormals too" does not hold on the card. Effect is below the selftest/kbench
  tolerances (the narrow/tiled byte contracts hold, both decode alike); the logits kernel
  avoids it with an f16-route decode. Fixing the shared decoder changes bytes across the fp8
  family and wants its own pass.

### 7.10 Tuning pass 10 (2026-10-28/29 night): the remaining distance-from-floor rows

Same method as §7.9: ranked by total fixture time and distance from the floor, then fixed with
an A/B on the card for each change. Five items were done by delegated agents, each in its own
copy (node3, or node2 under the shared GPU lock) and merged as a patch cut against the current
tree. All of them verified again after the merge.

| Row / shape | Before | After | Change |
|---|---|---|---|
| `gemm_fp8a8_tiled` / `gemm_i8a8_tiled` (and nt_m16 past 16 rows) at a ragged M | M=129: 2287 µs, 258: 3576, 65: 1208 (34816·5120) | **1623 / 2808 / 1016** | When the M edge cuts the last row of blocks, a two-body kernel lets those blocks skip the WMMAs and folds of padding fragments past M. One launch keeps the tail band on the same weight tile through L2. 1.15–1.4× on the 129/258/65 tails; whole bands unchanged. New byte checks at M=129 for fp8 and i8 |
| `gemm_fp8a8_nt_m16` narrow kernel, 2–4 row tiles | 1024·5120 M=64: 311 µs | **146** | At MT ≥ 2 the 8-step k unroll hoisted every step's loads and spilled (MT 4: 203–346 VGPRs to scratch). The unroll is now 2 at MT ≥ 2, giving 0 spills at MT 2–3 and 31 at MT 4. 2.0–2.2× on the small-N projections at M 36–64. The gated kernel gets the same fix (MT 3: 25 → 0, MT 4: 92 → 39) |
| `gemm_fp8a8_nt_m16` / `gemm_i8a8_nt_m16`, one row tile, small N | 2048·4096 M=1: 60 µs | **55** | Below 16 blocks the block narrows from 16 waves to 8 or 4, which never changes the split. Gains are a few µs on the MoE model's projections. Taller tiles keep 16 waves |
| `gemm_bf16_nt_m64` M 17–64 | 2048·4096 M=64: 416 µs; 4608² 455; 1280·5120 200 | **194 / 285 / 106** | The k walk is now a register ring D steps deep. Previously only the next step was in flight, so 256 dependent steps each waited a round trip. From 64 column tiles on, one block covers all row tiles (MB = ceil(M/16), which is the tunable's documented default). Bits are unchanged |
| `sample_topp_f32` / `sample_pick_f32` (1024 candidates) | 130 / 60 µs (M=1) | **28 / 25** | The serial walk (its f32 rounding is the contract) read a dependent global load per candidate. Now the block stages expf(v − max) into LDS and wave 0 reads 32 at a time through readlane. The ISA keeps the separate mul and add. Bit-exact vs libref (rel_l2 0) |
| `sample_topk_f32` / `sample_logprobs_f32`, M < 16 | 103 / 113 µs (M=1) | **83 / 91** | Four float4 loads in flight per thread, on grids under 16 rows only; at 32–64 rows the extra traffic cost 9%. Bits unchanged |
| `gemm_w8a8_tiled` / `gemm_w4a8_tiled`, ragged M | 16384·5120 M=129: 1367 µs, M=258: 1975 | **1095 / 1660** | Same tail form as the fp8 tile, on the default arm only, plus the grouped M-fastest block order on every arm (M=2048 −2%). The arm byte check at M=257 covers it. w4a8 dense has no perf mode and was not timed separately |
| `gdn_chunk_scan` (agent) | M=512: 820 / 493 µs; one chunk 115 / 80 | **153 / 118; 28 / 20** | Spills 357–379 → 0. The gfx12 register plan had doubled on gfx11's replicated fragments. Fixes: a transposed K copy in LDS, `accPackX` via `v_permlanex16` in place of a bpermute/branch, select-based masking, triangular tiles stopped at the diagonal, h0/ht staged through LDS. Output is **bit-identical** (in-process A/B, 10 cases incl. varlen/fp16 state) |
| `moe_gemm_fp8a8` (agent) | 26.8 ms over the fixture; gate_up M=1–32 at 71–180 GB/s | **21.4 ms**, decode band 209–226 GB/s | Each tile's K loop starts at a rotated 128-block, because every block reading the same slice of 2 KB rows had starved most DRAM channels. Also: a two-pass register ring with LDS-only barriers, unconditional clamped loads, full-line fp8 loads, a 32-column decode tile and a per-run down-projection launch. **Not byte-identical to the previous build** (sum reassociated by whole 128-blocks; every form of the row shares the rotation and agrees bit for bit). w4a8 MoE rows faster too |
| `gemm_mxfp4a8_*` (agent) | gemv_wide 34816·5120 M=64: 2677 µs; gemv M=1 638; tiled M=512 7401 | **992; 519; 5311** | tiled2 generalised to any split count (chain + running total reset at split boundaries = nt_m64's order), so gemv_wide (M > 16, N ≥ 3072) and decode at ks = 1 route to the LDS tile bit-identically. Also: the ring prefetch fixed (LDS-only barriers, clamped loads), M-tail PART, an e2m1 unpack with ~40% fewer VALU (exhaustively host-checked), and gated_gemv on an activation plane in scratch. New byte checks: gemv_wide via tile == skinny == nt_m64 at SK 4/10/16 |
| paged decode, verify widths (agent) | gqa6 16K: q1 209, q4 275, q8 450, q10 545 µs | **166 / 173 / 208 / 208** | K is staged once per workgroup into LDS (it had been read and widened per row-tile wave), and V staging is a real prefetch (the `load_tr_b128` emulation waited on its own load). At head 256, 1 and 3 row tiles launch on the 2- and 4-tile kernels, and the split target is 192 at ≤ 32 rows. h128 windowed 2.0–2.7× and its spill gone. Bit-identical where the split count is unchanged |
| paged prefill, head 256 (agent) | 60 prefill cases 43.9 ms; gqa6 q512 16K 9143 µs | **33.4 ms; 7607** (6.6 ms in quieter sessions) | The query is held once, not twice (QX: a half-swapped copy via 8 `v_permlanex16`), cutting spills 96 → 24. Paired-row A reads halve LDS bandwidth, and the fp8 → f16 staging decode is ~2.5 VALU a value (bit-identical over all 256 codes). At h256 the QK sum order moved (dims 2i/2i+1 per half), so not byte-identical to the previous build |

**WMMA lane semantics (probed on the card by the prefill agent):** the low half-wave computes D's
even rows from its own lanes' A and B, the high half the odd rows from its lanes. A row r is
read from lane r if r is even, lane 16 + r if odd. The emulator's gfx11 WMMA
(`tests/r4d_emu/emu`) took A/B from lanes 0–15 only and now reads them per half-wave. Kernels
that replicate their operands (all others) see no difference; the iu8 form is unchanged
(not probed).

**Final state (node2, everything merged):**
- Full `halo_selftest`: **8651 ok / 0 FAIL / 164 skipped** (30 new checks).
- `rad-kbench` vs libref, both fixtures: **2690 checked, 2690 passed**.
- `halo_parity`: **0 interface differences** (245 schemas, 486 rows; the same 38 hook / 45 chain informational differences).
- All 18 `halo_emu` suites pass.
- Local ctest: the same 5 baseline failures. GPU-labelled tests fail without a card, and `oot_plugins` fails on configure (no HIP root for the local toolchain).
- Whole-library kbench timing, 1133 libhalo rows:
  - this pass: **278.2 → 233.8 ms (1.19×)**;
  - against the pre-session library: **881 → 234 ms (3.77×)**.
  - No changed row is slower. The only > 10% case move is `sample_mask_f32` M=16 (22.6 → 33.5 µs), an untouched kernel at the 16 MB cache boundary.

**Open (measured, not fixed):**
- `gemm_mxfp4a8_gated_gemv`: on gfx1151 the fused row (~760 µs at 34816·5120, M ≤ 8) now costs more than the gemv + `gated_quant_fp8` pair it replaces (~535 µs). The fusion claim (`replaces`) is still declared, so a fused graph is slower. Dropping it changes what the engine selects; that is a decision for the owner.
- mxfp4 tiled at M=512 is ~29 TF/s (fp8 tiled ~37). gemv_wide at M=9–16 still runs the skinny kernel (the 16-row tile spilled).
- `gemm_bf16_nt_tiled` at N=2048, K=4096: 64 blocks of 128² or 32 of 64², 4–15 TF/s. Neither arm wins across M; it wants a deeper-pipelined small-grid form.
- `sample_typical_f32` stays quadratic (~114 µs). Single-row `sample_topk` is one block a row (~83 µs); a multi-block select is the next step. `rowtopk_bf16` stage 2 is chain-bound by design.
- `gdn_conv_update` at q_len ≥ 16 is a kbench sweep artifact (real windows ≤ 8 run the preload path).

### 7.11 End to end: Qwen3.8-Flash-Next on one node (2026-10-29)

**Bring-up.** `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe` (113.6 GiB `.rad`, sha256
`0af5e962…aa4d20`, checked on both nodes) in `~/models/` on node2 and node3. The nodes share a
~14 MB/s uplink, so it was fetched once on node2 with 16 ranged connections and mirrored to node3
chunk by chunk over the LAN. One node serves it at `--tp 1`: placement `all_vram` (unified ledger),
weights 64.6 GiB (experts 59.2), token embedding pinned in host memory, the 47.7 GiB n-gram table
on disk (`--ngram-placement disk`; `vram` is refused — the experts would get 49 GiB of the 59 they
need — and `ram` needs a memlock limit above 15.6 GiB). Driver: `fn1.sh` (in `../pass4/scripts/`, copied
to `~/dltool/` on both nodes); benchmarks are the repo's `scripts/cstep2.sh` and `scripts/fnpf.sh`.

**Bugs the e2e run found (none visible to selftest or kbench, which pass `M` = the row count):**
- `hc_read` took the band parameter `M` as its row count and refused a 65-row step in a 2048-row
  band. Rows now come from `h` (libr4d's `hc_common`); new selftest cases run 3, 65 and 114 rows
  in a band of 2048.
- `logits_gemm_{i8,fp8}` the same (refused M=1 in a band of 16); rows and pitches from X/Y now.
  The same pattern was fixed pre-emptively in `gemm_fp8a16_m1` and the fp8 gated GEMM.
- `hc_read` and `mtp_enter` scratch hooks sized the E4M3 decode buffers only when the weight
  tensor was present; the builder calls hooks without tensors, so the issue-time check refused.
  Both read the row's `mix`/`fc` parameter when the operand is absent.

**Kernel work this pass (all verified on node3; numbers are standalone unless marked e2e):**

| kernel | before | after | what |
|---|---|---|---|
| `hc_read_e4m3`, decode (M ≤ 16) | 285 µs | 41 µs | new fast path: norm scales; split-K down product (codes → f16 WMMA fragments, group scale folded per column) with the inject dots; partial sum; up product with mix and both twins closed in-block (`halo_hc_f16.h`) |
| `hc_read_e4m3`, prefill 2048 | 4.5 ms | 1.66 ms | W_up decoded once to a scaled f16 plane; W_down's fragments decoded from LDS (2 row tiles a wave); rows in 1024-row slices so `h` stays in the 32 MiB MALL between passes |
| `mtp_enter_e4m3`, M=1 | 303 µs | 92 µs | split-K decode path over the stored E4M3 rows; x rounded once (the reference's order) |
| `gemm_w2a8_nt` (MTP draft head, 248320×2560) e2e | 2.05 ms | 1.09 ms | `cfg_for` was given no tile count, so the auto split was the largest legal (20 at K 2560); with it, no split |
| `gemm_i8a8_nt_m16`, N 16384 / 12288, M ≤ 16 | 276 / 194 µs | 230 / 180 µs | four-way K split on wide one-tile launches (2-way measured slower; no gain at N 2560); B loads merged to 16 bytes |

**End to end (node2; node3 within 2%), one stream, MTP depth 3, fp8 KV, 2048-token steps:**

| | first run | now |
|---|---|---|
| decode, 1 sequence | 65.6 ms/step, 3.08 tok/step (47 tok/s) | 60.8 ms/step, 3.06 tok/step (50 tok/s) |
| decode, 4 sequences | 72.7 ms/step, 10.6 tok/step (146 tok/s) | 68.7 ms/step, 10.7 tok/step (156 tok/s) |
| prefill, 8.9K-token prompt | 1232 tok/s | 1325 tok/s |

Measured, not changed: speculation depth 3 is the best (1: 39, 2: 44.5, 3: 49.8, 4: 48.4 tok/s);
4096-token prefill steps are no faster (1200 tok/s; the MoE prefill GEMMs are compute-bound at
2048, not weight-bound).

**Where a decode step goes** (`--profile-ops`, 50 steps, one stream; profiled times run ~10% high):
routed experts gate_up 14.8 ms + down 7.2 ms (at DRAM bandwidth for ~40 distinct experts a layer);
int8 trunk GEMVs ~20 ms (floor ~13.5); `hc_read` ~4 ms; lm head 2.9 ms (at floor); draft head
3 × 1.09 ms; `gdn_conv_recurrent_update` 36 × 74 µs (floor ~30–45); router 47 × 32 µs (floor ~12);
the n-gram disk gather ~1 ms on the host. **Prefill** (8.9K tokens): MoE GEMMs 33% (~26 TF/s),
`hc_read` ~9%, QSA sparse attention 9% (~12 TF/s), int8 GEMMs 15% (~55–63 TOPS), n-gram disk
gather 4%.

**Verification (node3, this tree):** full `halo_selftest` **8654 ok / 0 FAIL / 164 skipped** (8651
plus the three band cases); `rad-kbench` at the container's shapes (skeleton of the real `.rad`,
`--op mtp_enter --op hc_read --op gemm_nt_q --op logits_gemm`) **528 checked, 528 passed**, worst
rel_l2 5.7e-4; `halo_parity` **0 interface differences** (245 schemas, 486 rows). The whole-model
kbench (all 63 graph ops) passed 2479 of 2479 checked cases before this pass; its 248 "failures"
are two kbench limits, not defects: `moe_gemm_w4a8` refuses kbench's w4nl64 operand (the layout
hook assigns the plugin dtype; kbench does not), and `router_gemm_topk_scatter` needs the instance
kbench never creates (identical in libr4d).

**Measured dead ends this pass:** f16 W_down plane read from L2 (2.5 ms vs 1.15 decoded from LDS);
decoding W_down to f16 at staging (more LDS traffic, two blocks a WGP); non-temporal int8 weight
loads (−4..15% at N ≤ 2560, +37% at N 16384, not kept); the TLB/page-scatter hypothesis for
`hc_read` prefill (rows aliased to one row: no change).

### 7.12 End-to-end kernel pass 2 (2026-10-29): routed-expert prefill, int8 GEMV, hc_read decode

Scope set by the user: kernels only, at the container's formats (no NPU, no lower-bit weights or
activations). Harnesses added: `halo_selftest --moetp1 --perfmoe M` (the routed-expert GEMM at
Flash-Next's tp1 shape; the old harness timed rank 0 of tp2); `--cold MiB` now matters for
`--perfhc` timing (it always flushed; profile averages had mixed in the warm-up launches).

| kernel | before | after | what |
|---|---|---|---|
| `moe_gemm_w4a8` prefill, gate_up (2048 tok, harness) | 8.30 ms | 6.10 ms | run-aligned blocks of 64 rows (was 32): one block per run of ≤ 64 rows, waves of dead 16-row tiles skip; exact f16 staging for both operands (`e4m3x4`, subnormals included, 2^16 in the weight scale, replaces the bf16 path that left weights 2^-120 low); w4nl codes straight to f16 by one permute per byte plane (sign = bit 3 over eight magnitudes, 1.75 VALU/value vs 3.75); 2×2 tiles a wave (LDS reads per WMMA 1.5 → 1); live M tiles decided once a pass (the per-step test let the compiler hoist every step's fragments: 300-VGPR spill) |
| same, down (2048 tok) | 4.57 ms | 4.01 ms | the same, on the 256-column two-half form (kept: 10–20% faster than the 128-column form) |
| same, 4096 tokens | 15.0 / 8.13 ms | 10.4 / 6.4 ms | |
| `gemm_i8a8_nt_m16` M ≤ 16, weights cold: N 16384 / 12288 / 1280 (K 2560) | 255 / 190 / 31.7 µs | 214 / 161 / 21.5 µs | one-group-ahead prefetch at one row tile (weights, activation codes, scales); four-way K split for every K ≤ 4096 whose groups divide (K 6144 stays unsplit) |
| `hc_read_e4m3` decode ≤ 16 rows, weights cold | 72.7 µs | 60.2 µs | `hcf_normdec_kernel` (a row a block, its four streams side by side, the stats kernel's sums in its order) writes N once and closes the inject gates; `hcf_down_dec1_kernel` reads its A fragments straight from N with every group's loads issued up front (down 33 → 17.8 µs) |
| `gemm_w2a8_nt` (MTP draft head 248320×2560, M 4, cold) | 1091 µs | 848 µs | the 2-bit unpack: the stored order puts each fragment dword's four codes at one shift, so a shift and a mask, and the zero comes off every byte through a 0x80 bias (no carry, zero ≤ 15); identical bytes (host emulation of both over 2.56M (words, step, zero) cases), 211 GB/s |

Numerics: the routed-expert forms stay byte-identical to each other (selftest), and the engine's op
oracle on real activations (`RADIANCE_OP_ORACLE=moe_gemm_q`) shows the same agreement with libref
before and after (5–32 of 10^5 elements one bf16 step apart, rel ≤ 1.4e-3). One prompt's draft
acceptance moved 71 → 59% (its greedy text diverged at bf16 rounding); over eight varied prompts
(`accbench.sh`, in `../pass4/scripts/`) it is 53.5% before and 52.9–54.0% after. int8 cases at the
container's shapes: kbench rel_l2 0 (integer accumulate). `hc_read_e4m3` selftest 41/0.

**End to end (node2, MTP 3, fp8 KV, 2048-token steps):** decode 1 sequence 60.8 → **56.1 ms/step**
(3.05 tok/step, 54 tok/s); 4 sequences 68.7 → **63.2 ms/step** (175 tok/s); prefill 8.9K
1325 → **1441 tok/s**. 4096-token steps are still no faster than 2048 end to end. Full
`halo_selftest` on node3 after the MoE, int8 and hc_read changes: 8654 ok / 0 FAIL / 164 skipped.

Measured and left: the int8 GEMV at N 2560, K 6144 is fastest unsplit (90 µs cold; 2-, 4- and
8-way splits slower); a 32-row K split of `hc_read`'s decode down product (16 → 8/10 slices) was
slower.

Next by size: the QSA prefill attention (11% of prefill, ~9 ms a call: every query re-reads its
2048 selected keys, ~4.3 GB a call through L2/MALL; a kernel sharing K/V over consecutive queries'
union would cut that ~10x but sums in a different order than the per-query table order);
`hc_read` prefill (12%); the routed-expert down GEMM (12.5%, 27 TF/s against gate_up's 36); at
decode the engine's per-attention-layer idle gaps (~5 ms/step, engine core), GDN recurrent update
(36 × 67 µs, no harness yet), router (47 × 27 µs).

**Comparison engines (read, not run):** `warpfront/hipfire` and `peonist-ai/halogen-flash-server`
serve the same model. Their ~1,900–2,000 tok/s prefill is GPU-side (hipfire's NPU adds 7–10% at
≥ 8K and needs the IOMMU on, which halogen measured at −13–16% prefill); both rely partly on 4-bit
trunk weights and (hipfire) int4 activations, out of scope here. Applicable ideas not yet done:
prompt-lookup drafting and confidence-gated draft depth (engine core), the decode-step idle gaps
(~5 ms/step at the attention layers, host replay; engine core).

### 7.13 Kernel pass 3 (2026-10-29): the small items, all bit-identical

Same scope. Every change below leaves the op's output bytes unchanged: checked against a build of
the pass's starting tree (`halo_selftest --dumpdir DIR` now writes the outputs of the
`hc_read_e4m3` cases, the `gdn_conv_recurrent_update` fold cases and the new perf harnesses, and
the two builds' files compare equal: 78 for hc_read, 34 for GDN), the router against its unfused
launches (selftest, 394 ok), and end to end three greedy completions (800, 800, 600 tokens) are
byte-identical between the two builds. New harnesses: `--perfgdn SEQS T NACC` (the GDN decode
update at Flash-Next's shape, anchor state) and `--perfrgs M` (the fused router at its served
geometry).

| kernel | before | after | what |
|---|---|---|---|
| `hc_read_e4m3` prefill, 2048 rows (cold) | 1601 µs | 1453 µs | down product: codes decoded to f16 once at staging with the column scale folded in (both waves of a fragment decoded it before; VALU 22.5M → 10.2M instructions), 351 → 296 µs a 1024-row slice; up product: the gate epilogue one row tile a pass through a 16 KB buffer (32 KB before), six blocks a WGP instead of four, 313 → 284 µs |
| `hc_read_e4m3` decode, 4 rows (cold, whole call) | 60.0 µs | 55.5 µs | `hcf_dsum_kernel` folded into the up product (each block sums the partials for its rows in the same order) and that kernel at sixteen waves, two quarters of W_up in flight |
| `gdn_conv_recurrent_update`, 1 seq × 4 tokens (cold) | 58.0 µs | 45.1 µs | no non-temporal hint on gfx1151 (below); barriers after the anchor publish wait for loads and LDS but not stores (`__syncthreads`' release fence waited for the 64 KB state write before the token loop); q and k norms for every token before the loop (two of its four dependent chains); k before q, which ends a 13-VGPR spill (256 → 241) |
| same, 4 seq × 4 tokens | 188 µs | 128 µs | |
| `router_gemm_topk_scatter`, M 1 / 4 / 8 (cold, kernel) | 20.1 / 21.2 / 24.9 µs | 18.5 / 19.5 / 22.2 µs | the m16 phase a column pair a block (one item a wave) instead of a 16-column tile (eight items a wave in turn); M > 8 keeps the tile |

The GDN non-temporal threshold (`RU_NT_MB`) was a gfx1201 measurement; on gfx1151 an interleaved
engine A/B (two runs each) gave 55.7 / 64.7 ms/step without the hint against 56.2 / 65.4 with it,
so it is 0 here.

**End to end (node3, MTP 3, fp8 KV; the pass's start against its end, interleaved, two runs
each):** decode 1 sequence 57.0 / 56.7 → **55.7 / 55.4 ms/step**; 4 sequences 65.6 / 65.6 →
**64.7 / 65.1 ms/step**; prefill 8.9K ~1447 → ~1456 tok/s.

Measured and left: a fused norm-scale pre-pass in the prefill down product (each wave reducing
eight rows in the stats kernel's order) was slower, 445 against 296 + 125 µs, since nothing else
runs while its DRAM reads land; a wave-a-row stats kernel measured no faster than the block-a-row
one (both near DRAM speed); 48- and 64-row up-product blocks (306, 390 µs against 284); loading
the up epilogue's h ahead of the product (+30% at 192 VGPRs at prefill, nothing at decode); three
or four W_up quarters in flight at decode (slower than two; the loads alone run at ~190 GB/s);
preloading every router item's weights at the old 16-column tile (slower at M = 4). The bf16
skinny GEMM at N 96 and 640 (GDN's `ba` and attention's gate projections) is within ~3 µs of its
floor at every split it has.

### 7.14 QSA prefill attention: rows grouped over the union of their pages (2026-10-29)

The user accepted a different summation order for this kernel (it cannot be bit-identical: a row's
softmax now runs over its keys in another order). At prefill past the exactness bound the sparse
attention is `attn_paged_gate_quant` with one row per query, each with its own 513-entry selection;
the decode kernel served a row alone, 12 live rows of a 32-row tile pair, and widened every selected
page for those 12 rows: 13.6 ms a 2048-row call on an 8.9K prompt, and with 4 KB of K/V a page,
~4.3 GB a call through the caches.

Measured on real selections (`RADIANCE_DUMP_BUF=qsa_sel:2048`, a 9K prompt): consecutive queries
share most of their pages. The union of R consecutive rows' selections is 1.41 x one row's at
R = 4, 1.68 at 8, 2.0 at 16; page ids within a row are not ascending (block order through the
pool's free list), so the union is built by lookup, not by a merge.

`g8_union_launch` (gqa12 unit, `G8_UNION`): from 64 rows (q_len 1, no descales, no window), five
rows a group (60 of the decode workgroup's 64 rows):

1. `g8_union_plan_kernel`, a block a group: the union in a fixed order (each row's new pages in its
   table order, numbered by a block scan; an LDS hash maps page -> union index), and a word a union
   page holding four bits a row: the tokens of that page the row attends to (4, its last page's own
   count, or 0). ~0.1 ms a call.
2. `halo_attn_decode_kernel` under the new `O_UNION` bit, over (group, kv head), one split, 32-key
   tiles: the group is the "sequence" (cu_q), its union the block table, and the mask replaces the
   causal limit. 221 → 232 VGPRs at tile 32, no spill.
3. The gated merge, unchanged, one partial a row.

| 2048-row call, 8.9K prompt (engine kernel trace) | before | after |
|---|---|---|
| attention kernel | 13.6 ms | 6.5 ms (tile 16; tile 32 ~1% better end to end) |
| plan + merge | 0.4 ms | 0.5 ms |

Prefill 8.9K tokens on node3: ~1463 → ~1590 tok/s (+9%; the A/B is the pass-3 start against the
end, so ~0.6% of it is §7.13).

Numerics (`--case attn_decode_gq_h256_gqa12_fp8kv`, new case; `--perfqsaattn ROWS` times it): at
Flash-Next's QSA geometry with overlapping selections through a shuffled page map, partial pages,
short rows and a padded row, both forms against an f64 evaluation of the gated output: mean
relative error 1.57e-4 for both, max 2.3e-2 (union) against 2.0e-2 (per row) at 2048 rows, 2.43e-2
both at 130; ~5% of bf16 outputs differ between the forms. libref has no `attn_paged_gate_quant`
row, so the engine's op oracle cannot check this op. End to end: three needles at 2K, 15K and 30K
characters of an 11K-token prompt are all found by both builds; draft acceptance over six 9K-token
prompts (500 tokens each) 55.0% before, 59.6% after (per-prompt spread ±14 points); generated text
diverges from the previous build within the first sentences, as a summation-order change does.

Not pursued: the kernel now keeps the matrix units ~40% busy (each row scores the whole union, of
which ~69% is its own), so a dedicated multi-query kernel could gain perhaps 1.5x more on ~4.6% of
prefill GPU time.

### 7.15 Kernel pass 4 (2026-10-29/30): six lanes, merged, NOT yet verified — continue here

**State in one line:** six patches written and measured lane by lane on node3; merged and
compiled, but the merged result has had no selftest, no byte A/B and no end-to-end run. **None of
it is in this working tree.** A VPN outage (the nodes' whole subnet dark for ~12 h)
cut every lane off during its final checks; when node3 returned the user asked to wrap up.

**Where everything is (durable copies; the session scratchpad under /tmp is not):**
- `../pass4/` (beside the repo, outside it) and node3
  `~/pass4/`: `merged_all.patch` (all six lanes, applies with `patch -p1` at the repo root, checked
  by dry run on 2026-10-30), the per-lane patches `gather hc moe i8 gdn qsa .patch`, the fallbacks
  `hc_nodirect.patch` (hc without the unverified decode DIRECT form) and `moe_wip.patch` (decode
  expert follow-up on top of moe.patch, never run), `ru_spec.hip` (gdn: a speculative state load,
  ~1 µs, bit-identical, left out), and `scripts/` (the lane brief `p4_brief.md`, `agent_common.md`,
  the profile `profile_now.txt`, `trace2.sh`/`ktab.py`/`kwin.py`, and the A/B scripts below).
- node3 `~/p4_merge/`: the merged tree, configured (`-DRAD_WITH_FFMPEG=OFF`) and built
  (`libhalo` done; `halo_selftest libref rad-kbench` were building when stopped — re-run
  `cmake --build build -j24 --target libhalo halo_selftest libref rad-kbench`).
- node3 `~/base_p4/radiance_home/`: the pass-start plugin (md5 `30479e6b…0f`), the A side of
  every comparison. The pass-start sources are this working tree as it is now.
- node3 `~/dltool/`: `vab.sh` (every lane's dump harness, base vs merged plugin under the merged
  selftest binary, then a byte compare; expect only `qsa_union_*` files to differ), `ab4.sh RH_A
  RH_B [ROUNDS]` (interleaved e2e: decode 1 and 4 sequences, 8.9K prefill), `tc4.sh RH_A RH_B`
  (three greedy completions, byte compare), `vg2.sh` (VGPR/spill of a build: `VGB=~/p4_merge/build`).

**To finish (in order):**
1. node3: finish the build above; full `halo_selftest` on `~/p4_merge` (0 FAIL); `~/dltool/vab.sh`.
2. qsa first-run checks (the final kernel never ran): `--case attn_decode_gq_h256_gqa12_fp8kv`
   (mean/max error vs f64 at the §7.14 level: mean ~1.5e-4, max ~2.5e-2; the new "repeats itself bit
   for bit" check), `--qsasel ~/qsadump/3- --perfqsaattn 2048 20` under rocprofv3 (base: 7215 µs
   attention + 134 µs plan + ~400 µs merge).
3. Retime `--perfgdnpf 2048 1`: the final `chunk_scan` spills 3 VGPRs on ROCm 7.2.3 (0 before).
4. e2e: copy `~/radiance/build/radiance_home` to e.g. `~/rh_p4`, put the merged `libhalo.so` in its
   `kernels/`, then `ab4.sh ~/base_p4/radiance_home ~/rh_p4 2` and `tc4.sh` (short prompts do not
   reach the QSA union path, so their greedy text should stay identical). Draft acceptance with
   `accbench.sh`. Then apply `merged_all.patch` here and record the numbers in this section.
Since §7.16 (2026-10-30), node3's `~/radiance/build` has FFmpeg on and its `libhalo.so` carries
   the `sam_attn` change, so it is no longer the pass-start plugin; `~/base_p4` still is (and it
   lacks only that change, which no Flash-Next op uses). `merged_all.patch` still applies to this
   tree (dry run after §7.16: offsets only).
5. If a lane fails, drop its patch (they touch disjoint files except `halo_selftest.cpp`, whose
   hunks are self-contained) or take `hc_nodirect.patch` for hc.

**The lanes** (all bit-identical to the pass-start build by their dump compares, except qsa,
whose order change the user accepted; times are standalone on node3, `--cold 256` unless noted;
"share" is of a 2048-token prefill step's 1275 ms of kernels or a 56 ms decode step):

| lane | files | change | before → after | est. share | checked before the outage |
|---|---|---|---|---|---|
| gather | `halo_moe.hip`, `halo_moe_gather.h`, `halo_router_gemm_scatter.hip`, `halo_quant_act_fp8.hip` | `gather_wide<8>`: 1-D grid over (token, piece), a block's waves read one token's slot rows whole (was 512 B of each 5 KB row a pass); decode gather: all loads issued first; router M ≤ 16 phase: DPP `halo_wave_sum`, two columns a wave at MT ≤ 8; `quant_act`: one 8-byte load when aligned | gather 2048 tok 980–1050 → 563–615 µs (222 GB/s, floor); quant_act_i8g 2048×6144 156–175 → 141–148; router M 1/4/8 21.6/23.0/25.7 → 20.7/21.2/22.4; decode gather M 4 cold 9.6 → 7.0 | prefill ~1.8% | full selftest 8655/0/164; dumps 100/0 differing; kbench 128/128 rel_l2 0; emu moefuse 105/0 |
| hc | `halo_hc_bf16.hip`, `halo_hc_read.hip` | `hc_write_v8_kernel<4>` (8 channels a thread, every load before the first add; hc 4, 16-B aligned; else the old kernel); prefill slices tail-first so the first norm pass reads `hc_write`'s rows from MALL; `hcf_down_pf` gains/inject loaded a tile ahead (189 → 233 VGPRs); decode up product DIRECT form `hcf_upmix_kernel<1,4,320,false,32,true,true>` (W_up rows straight to registers, partials first) | hc_write 2048 367 → 330, 4 rows 6.5 → 5.3; write+read pair 1866 → 1770 (before the down_pf change); down_pf slice 309 → 293; decode upmix (rotated weights) 23.1 → 21.3 | prefill ~1.6%, decode ~0.3 ms | hc_read_e4m3 41/0, 78/78 dumps equal (before DIRECT); hc_write 7/7. **DIRECT never byte-checked; no full selftest.** Under rocprof the decode hc_write looked slower warm (9.2 vs 6.6 µs) — check e2e |
| moe | `halo_moe_gemm_q.hip` | one load pipeline per block across its 128-column groups (next group's first loads under the current group's last passes; group state set when the issue cursor enters a group); `moeq_down_ny`: ≥ 16 passes a block (4 groups at K 640); down spill 19 → 0 | down 2048 tok 3990 → 3223–3273 µs (256: 2437 → 1947; 4096: 6481 → 5419); gate_up 5926 → 5979 (+1%) | prefill ~2.7% | form-1 hash = base at gate_up/down 2048 and down 256–4096; moe selftest cases 0 FAIL on a build that still had probes |
| i8 | `halo_gemm_fp8a8.hip` | decode: `halo_gemm_i8a8_gpar_kernel` (M ≤ 16; a wave a 128-group, int32 sums to LDS, wave 0 folds in the old order at the same SK); prefill: `i8_tiled_block2` for whole bands (a stage = a whole 128-group, 2 barriers a group not 4, LDS double buffer, scales prefetched), PART body unchanged; grids of 61–80 128² blocks take the 64² tile | M 4: 16384×2560 211.6 → 200.6 (floor 192.8), 12288×2560 160.7 → 148.0, 2560×6144 89.0 → 79.3, 512×2560 12.7 → 11.7; prefill 2048: inz ~4583 → ~4443, out ~1744 → ~1590, k/v 177 → 157 | decode ~1.8%, prefill ~1.5–2% | 48 decode dumps = base; block2 = base at M 65/129/708/2048 (before the PART/tail routing); emu gemm passes. **Final routing not byte-checked; no full selftest** |
| gdn | `halo_gdn_kkt_solve.hip`, `halo_gdn_chunk_scan.hip`, `halo_gdn_recurrent_update_…hip` | kkt: grid k-heads fastest (chunks fastest put every resident group on one 256-B offset of each 4 KB row: ~25 GB/s), K tile staged once in LDS; chunk_scan: 4-deep q ring pinned by `sched_barrier`, next chunk's prefetch after the main loop; decode: `amdgpu_waves_per_eu(6)` (264 → 202 VGPRs: gfx1151 allocates in 24-VGPR granules, 8 of 48 head groups ran a second round) | kkt in sequence ~455 → ~150 µs; chunk_scan ~945 → ~830; decode 1×4 cold 45.3 → 38.9 (2×4 75.9 → 66.8; 4×4 127 → 129, at bandwidth) | prefill ~1.2%, decode ~0.4% | dumps equal (`--perfgdnpf 2048 1`, five `--perfgdn` shapes). **Final cleaned source not rebuilt/run; chunk_scan now spills 3** |
| qsa | `halo_attn_paged_gqa8.hip` (`G8_UNION` block only) | `g8_union_attn_kernel<5,32,GQ>`: a wave owns 16 score rows over the full 256-d head (prefill core, no split-head barriers); plan kernel sorts union pages by which rows use them (stable), a wave skips 16-key slices none of its rows attend (matrix work 1.36x → 1.22x of useful); gated merge folded into the epilogue (no `splitkv_combine_vec_gq`, no partials) | intermediate version: 7215 → 5627 µs + merge ~400 → 0 (1.35x on the op) | prefill ~2%+ | intermediate version vs f64: max 2.53e-2, mean 1.54e-4 (per-row form 2.53e-2 / 1.55e-4). **Final kernel never ran.** Other gqa12/gqa8/gqa6_bs4 kernels compile to identical code (local). The `O_UNION` code in the decode body is no longer instantiated (left; `halo.h` refers to it) |

Estimated total if all hold: prefill +9–11% (~1590 → ~1750 tok/s), decode ~2.5%. Not measured end
to end.

**Findings worth keeping (measured):**
- The int8 prefill GEMM is not at the iu8 WMMA peak as §7.11 implied: ~38–42 TOPS against ~55 for
  a pure iu8 WMMA loop; the bound is the per-group f32 scale fold (removing it: −13%; removing the
  global fetch: 0). `<4,2,true>` is not slower than `false` — its lower average is the prompt's last,
  shorter chunk.
- WMMA and VALU do not overlap on a gfx1151 SIMD (split across waves the time is the serial sum).
- Routed gate_up at 2048 tokens is ~70% DRAM-bound (865 MB of expert weight a call, floor 4.1 ms,
  measures 5.93); ~390 VALU instructions a wave-pass against 32 WMMA.
- The MoE "two passes ahead" ring never worked: an early `break` in the unrolled slot loop makes the
  wait-count pass merge a half-done path, so every stage waits on the newest slot (one deep).
  `moe_down_dec_kernel`'s `__syncthreads` + fence waits for the load it just issued.
- **Decode routed experts are 27–34% above their floor** (harness M 4, 38 distinct experts):
  gate_up 410 µs (157 GB/s, floor 306), down 200 µs (floor 158). The `T <= 80` rule for two K
  blocks a pass was tuned at tp2 and is wrong at tp1 (one block a pass: 334 µs). `moe_wip.patch`
  (whole-step K loop, two-ahead ring, `moe_down_dec` two-slot ring) is the follow-up; it compiles,
  never ran, and switches 16-bit scale loads to `uint16_t` (d16 loads, which `halo_common.h` warns
  about — check or revert). This is the largest decode item left (~5 ms a step at best).
- gfx1151 VGPR allocation granule is 24: 241–248 VGPRs round to 264 (5 waves/SIMD).
- `--cold`'s memcpy flush leaves small working sets in the MALL; rotate weight copies (hc's
  `--perfhcw` does 16) to time decode kernels honestly.
- At the floor already: `gated_quant_fp8` (66 MB), `quant_act` (38 MB), GDN `conv_prep` and
  `gated_rmsnorm`, prefill `hc_write`, the int8 GEMVs (within 1–8 µs of a plain read).
- Router (~21 µs cold): ~6.6 µs is launch + arrival atomic (an empty kernel is 5.1 µs cold), the
  top-k/sort tail ~2.5 µs (shared headers `halo_router_topk.h`, `halo_moe_sort.h`), the GEMM ~14
  against a 12.5 µs byte floor.

**Measured dead ends:** gather_wide/gated-quant 8-byte loads (noise); router: preload all of a
split's weights (125 → 165 VGPRs, slower at every M), four columns a wave; hc: plane kernel after
the first stats (down_pf 769 µs) or after dsum (pair 1795); moe: 2 groups a block for gate_up, one
block over the full width, per-pass group-state recompute (+8%), in-loop store at HC=1 (+8%, spill),
lambda K-loop tail (scratch); i8: two groups a wave (−10%), activation staged in LDS, the block2
body for partial bands (spills, 7–17% slower), activation scales hoisted to registers (spill); gdn:
chunk_scan LDS double buffer (+4%), 16 waves of one tile (+18%), A fragments in the ring (+3%),
gate prefetch and head permutations (neutral), conv_prep head-fastest (neutral); qsa: 8-row
groups (no gain; the hash cannot hold the worst-case union), 4-row groups and 16-key tiles in the
old staging (spills). The remaining chunk_scan limit is LDS (64.5 KB → two groups a WGP, 96 groups
in three rounds); a third group a WGP needs ≤ 43 KB.

**Not touched (engine core, needs the user's approval):** the decode step's ~5–7 ms of host-side
idle between kernels, which is now larger than all remaining kernel headroom at decode.

### 7.16 Three more models on libhalo (2026-10-30): EmbeddingGemma 2, Unlimited-OCR, Qwen-Image 2.1

All three serve on node3 with `--kernels libhalo` alone (every op resolves to libhalo), and all three
are on both nodes' disks (`~/models/`, sha256 checked on each). Downloaded once on node3, copied to
node2 over the LAN at ≤ 150 MB/s (`rsync --bwlimit`, node2 pulling from node3 under `ssh -A` in the
foreground — a backgrounded copy loses the forwarded agent). node2's production server was not touched.

| model | container | source | sha256 (container) |
|---|---|---|---|
| EmbeddingGemma 2 | `embeddinggemma2-bf16.rad` (1.4 GiB) | converted on node3 from `google/embeddinggemma-2` @ `f9972ac22e69` (`rad-convert ~/models/src/embeddinggemma-2 -o … --home radiance_home`, no recipe; the source is kept in `~/models/src/`) | `073ddae3…efb35c2` |
| Unlimited-OCR | `unlimited-ocr-bf16.rad` (6.2 GiB) | `Aijuus/unlimited-ocr-bf16` | `ee661b7a…55324c` |
| Qwen-Image 2.1 | `qwen-image-2.1-fp8.rad` (16 GiB) | `StillDeadcode/radiance-qwen-image-2.1-fp8` | `cabd2b5c…93be9` |

**Media support on node3 (a node build change, no engine code).** The node builds ran with
`-DRAD_WITH_FFMPEG=OFF` (the distribution's FFmpeg 6 does not build `core/mm/mp4.cpp`), which refuses
every image/audio input and cannot encode an output picture — all three models need it. FFmpeg
8.1.3 (the Dockerfile's version and sha256) is now built on node3 in `~/opt/ffmpeg` with the
Dockerfile's configure line minus dav1d, OpenH264 and x86 asm (not installed; AV1 decode and H.264
encode are not available), and `~/radiance/build` is configured with
`PKG_CONFIG_PATH=$HOME/opt/ffmpeg/lib/pkgconfig -DRAD_WITH_FFMPEG=ON` (the binary carries the
rpath). node2's build is unchanged (still FFmpeg off): to serve these models there, repeat both steps.

**Harness** (`../models3/` and node3 `~/m3/`): `srv.sh` (M, KERN = `libhalo` | `libhalo,libref` |
`libavx` with `EXTRA_ENV=ROCR_VISIBLE_DEVICES=99` for the host backend, EXTRA, WRAP for rocprofv3),
`stop.sh` (never `pkill -f bin/radiance` from an ssh command line: it matches the shell itself),
`emb.py`/`embcmp.py`/`emb_st.py`/`embperf.py`, `ocr.py` (renders a page of known text with
DejaVu, `MT=` caps the reply), `img.py`, `psnr.py`; `dl3.sh` the downloads. Python on node3:
numpy, Pillow, torch 2.15 (CPU), torchvision, transformers 5.22, sentence-transformers 6.2 in
`~/.local` (`pip --user --break-system-packages`).

Note on references: `--kernels libref` alone is not a GPU reference — libref is a host library, so
the device ops still resolve to libhalo (a "libref,libhalo" run is libhalo; its outputs matched to
the bit for that reason). The references used: the host backend (libavx, no card visible), the op
oracle (`RADIANCE_OP_ORACLE=*` with `--kernels libhalo,libref`; libhalo serves, libref checks), and
for EmbeddingGemma 2 sentence-transformers itself.

**Correctness:**
- **EmbeddingGemma 2** (`--max-num-seqs 64 --max-num-batched-tokens 8192`): seven texts × three
  prompt forms (query, document, none), three drawn images, two audio clips. libhalo against the
  host backend: cosine ≥ 0.99988 on every vector (max |Δ| 2e-3); against sentence-transformers
  (CPU, the checkpoint's bf16): ≥ 0.99987 on every text vector. Retrieval sense is right (query
  "capital of France" ranks the Paris passage first).
- **Unlimited-OCR** (the compose file's flags, `gundam`): a rendered A4 page (title, four paragraphs,
  a 4 × 3 table) comes back word- and digit-exact; the host backend gives identical text, the boxes
  differing by ≤ 2 of 1024 (bf16 drift). Op oracle: 7,642 checks over the DeepEncoder tower, the
  decoder prefill and decode (MoE GEMMs, router, gather, norms, rope, sampling) — all agree; decode
  attention (`attn_paged_qrow`) has no libref row (covered by the text match).
- **Qwen-Image 2.1 fp8** (defaults): 1024² generations are clean (`../models3/qi_fox.png`); text
  rendering exact (`qi_sign.png`, "RADIANCE BAKERY / Fresh bread daily"); an edit adds a scarf and
  keeps the scene (`qi_edit.png`); the same seed gives a byte-identical PNG. Op oracle: 1,200 checks
  over the text encoder and the DiT (fp8 GEMMs, quant/modulation fusions, qk-norm-rope, attention)
  and 219 over the VAE (conv2d_nhwc, l2norm_chan, d2s_add, attn_dense h128/h1152) — all agree.

**Performance (node3) and the one fix:**

| model | measure |
|---|---|
| EmbeddingGemma 2 | 40K tok/s at ~200-token docs (196 docs/s), 34K tok/s at ~1000 tokens. GEMMs are 61% of kernel time at ~16 TF/s (hidden 512: four 128-deep K passes a tile); elementwise at bandwidth; ~40% of wall is host-side between steps (tokenising, JSON). Nothing obvious left in kernels. |
| Unlimited-OCR | one page (907 prompt + 284 generated tokens): TTFT 0.78 → **0.55 s**, request 2.7 → **2.4 s** after the fix below; decode ~6.5 ms/token against a ~5.5 ms byte floor (~1.15 GB a token at 210 GB/s; experts at ~200 GB/s, head at bandwidth) |
| Qwen-Image 2.1 fp8 | 1024², 40 steps: **107 s** (8 steps 20.5 s), 1024×768 72 s. GPU busy throughout; the fp8 DiT GEMMs are 74% at ~30 TF/s (every shape: qkv, gate_up 24576, out/down), the int8-score attention 15% at ~17 TF/s, VAE ~0.6 s. No obvious fix; the GEMM is within ~20% of the best fp8 GEMM measured here. |

The fix: **`sam_attn` global path on the matrix units** (`halo_sam_attn_bf16.hip`,
`sam_attn_wmma_kernel<4>`). The DeepEncoder's SAM global attention (window 0) was the RDNA4 port's
SIMT form — a thread a query row, 4096 keys of elementwise dots at the 64 grid — at ~2 TF/s, 28.3 ms
a layer call (0.23 s of a page). Now halo_attn_dense's core (S^T = K.Q^T, f16 operands, f32
accumulate, P through the C→B fragment) with the decomposed rel-pos term: a block first computes its
64 query rows' `q.Rh` and `q.Rw` dots (f32, the SIMT kernel's `tab_dot` with its resampling) into
LDS in log2 units, then adds `relh[kh] + relw[kw]` to each score. 28.3 → **5.2 ms** a call; 249
VGPRs, 51 KB LDS, no spill. Taken for 16-byte-aligned rows with gh + gw ≤ 128 (the tower's 64 and 40
grids); else the SIMT forms run. Not bit-identical to the SIMT kernel (f16 operands and P, another
summation order) — within the same house tolerance against the host port; four aligned global
geometries were added to `sam_attn_case` (16, 20 with a partial tile and rows wrapping inside a
16-key step, 40 on resampled tables, 64 over two segments). The windowed SIMT kernel (3.5 ms × 2 a
page) is unchanged. Verified: `sam_attn_case` 8/8 geometries ok (1,757,184 values, 0 outside
tolerance; the four aligned global ones run the new kernel, confirmed by trace); the op oracle on a
real page, 24 `sam_attn` issues against libref (global 40 and 64 grids on the new kernel, windowed
42 and 70) — all agree; OCR text still identical to the host backend's; full `halo_selftest` on
node3 **8659 ok / 0 FAIL / 164 skipped**. (`rad-kbench --op sam_attn` checks nothing: the committed
fixtures carry no sam_attn case.)

### 7.17 Upstream 1.14 merged; libhalo beside libr3d (2026-10-30)

The work is on branch `strix-halo/libhalo` of the internal repository (one squashed commit on
the 1.12 base, then a merge of upstream 1.14.0). Upstream added **libr3d**, the RDNA3 kernel
library (`^gfx11`, so gfx1151 included; 44 schemas, 78 rows). The user's decision: both exist,
libhalo preferred on Strix Halo only.

- **Preference, no core change.** With no `--kernels` list the loader takes kernel libraries in
  file-name order (`libavx`, `libhalo`, `libr3d`, `libr4d`; `libref` pinned last) and the first
  library with a row supplies an op, so on gfx1151 libhalo serves everything it has and libr3d
  only what libhalo lacks. `rad_plugin_open` (new, `halo_rows.cpp`) declines any machine whose
  KFD topology lists no `gfx_target_version 110501`: on another RDNA3 card libhalo is not loaded
  and declares no schema, and libr3d serves alone (the loader would drop libhalo's device rows
  there anyway, its code objects being gfx1151's; declining also keeps its schemas out). Where the
  topology cannot be read the library stays. Top-level `CMakeLists.txt` adds both subdirectories;
  `libhalo/README.md` says the same. Checked on node3: `rad-info --plugins` lists libhalo at 1,
  libr3d at 2.
- **Interface re-synced to 1.14's libr4d** for every op both declare (`halo_parity` against the
  1.14 libr4d, local `build-both`): the `rms_mod`, `rms_mod_quant_fp8`, `rms_mod_quant_i8` schemas
  gained the optional `colmap` operand (one i32 a group mapping it to its modulation-table row;
  MiniMax-H3's modulated vision pads), ported into `halo_video.hip` as libr4d has it; the twelve
  dense integer GEMM rows gained `group EQ 128` and the w4/w5/w8/w2 layout hooks refuse another
  group (`RAD_E_DTYPE`), as libr4d's do (a Q8_0 weight at group 32 had been read as 128).
- **Still missing (a port round):** the 27 rows 1.14 added to libr4d have no libhalo row, and
  libr3d has none of them either, so on Strix Halo these ops are unsupported: tree attention for
  DFlash tree drafting (`attn_paged_qrow_tree`, 18 rows), `dflash_tree_bf16`, the M2XFP4 GEMMs
  (`gemm_m2xfp4a8_decode`/`_tiled`), `gemm_mxfp4a8_gated_gemv_wide`,
  `qkv_norm_rope_kv_store_h128_bf16`, the tiled video encoder (`vtile_plan`/`_crop`/`_join`) and
  `cond_aug_f32`. `halo_parity` reports exactly these (96 differences: the rows, their
  concurrent/collective flags, six schemas and the row-order lists that contain them), so its
  ctest fails in a build that produces both libr4d and libhalo until they are ported.
- **Verified on node3 (merged tree, `~/r114`, FFmpeg on):** `halo_selftest` 8659 ok / 0 FAIL / 164
  skipped; `r3d_selftest` on gfx1151: 78 rows clean. Flash-Next: every device op band on libhalo
  (libr3d supplies none), correct answers, prefill 8.9K **1596 tok/s**, decode 4 sequences 63.7
  ms/step. EmbeddingGemma 2 embeddings, the Unlimited-OCR page and a seeded Qwen-Image picture
  are byte-identical to the pre-merge build's.

## 8. Pending work (in todo order)

> Latest (2026-10-30): kernel pass 4 is merged but unverified and not in this tree — §7.15
> has the files, the checks still owed and the order to run them in.

> Status note (supersedes the list below, which is the pre-completion
> record): every family in items 1–6 is implemented and verified
> (§6.8 — zero weak stubs, 8593 ok / 0 FAIL). What remains is the
> per-op tuning campaign (item 7, §7.1–§7.5) and a commit decision;
> all work is still uncommitted in the working tree.

1. **GEMM family (in progress):** only the mxfp4 family remains.
   Regenerate stubs after each unit; delete stubs as units land.
2. **Encoder family:** `halo_encoder_bf16` (relu/silu/clamp/act_mul/
   qk_norm_rope and relatives). Known full-selftest gaps: these rows
   currently return success with wrong/empty output because the generic
   `run_case` uses the first operand as expected output, and
   `qk_norm_rope_bf16` fails rc=-18 (its real shim lives in
   `r4d_encoder_bf16.hip`, not the ported qkv unit).
3. **Attention family:** paged/dense/gen/vit/sam/vsa, QSA, MLA, DSA. The
   two kv_store fixture refusals (`kv_store_k6v4`, `kv_store_fp8v4` —
   constraint/shim shape mismatch) belong here.
4. **GDN/KDA recurrences, MoE, sampling.**
5. **Core: unified-memory capability + single ledger (minimal).** Not
   started. Risk: planner/mover/KV-tiering assumptions may need more than
   minimal changes. First inspect `core/device/hip.cpp`, the memory pools,
   and residency/mover paths before editing.
6. **Collectives, LoRA, HC/MHC, PLE, gen/video/audio families.**
7. **Full verification:** full `halo_selftest`, `rad-kbench` all-rows vs
   libref, ctest fast tier (accounting for baseline failures, §9), e2e
   model runs, per-op perf tuning/report until the stub count is zero and
   all rows are tuned.

### Immediate next step

The GEMM family is done. Next per the todo order is the **attention family**
(127 stubs — the largest block): `r4d_attn_gen.hip` (1036 lines),
`r4d_attn_decode_h256_gqa6.hip` (1085), `r4d_attn_paged_window.hip` (1207),
`r4d_attn_prefill_h256_gqa6.hip` (651), `r4d_attn_paged_gqa8.hip` (429),
`r4d_attn_dense_bf16.hip` (430), `r4d_attn_vit_h72_bf16.hip` (489),
`r4d_attn_paged_h256_gqa6.hip` (477, the 27 decode launches),
`r4d_sam_attn_bf16.hip` (413), `r4d_dsa.hip` (330), and the small
`r4d_attn_paged_h128_*`/`h256_gqa*` variants (14-89 lines each).
Port with `/tmp/gen_unit.py <r4d_unit>`, then hand-adapt to gfx1151: bf16
forms use `halo::mma` (bf16 WMMA, present); fp8 forms need the exact-bf16
dequant emulation the fp8 GEMMs use (there is no fp8 WMMA on gfx11). Start
with the bf16 decode/prefill forms a text model actually runs.

Stub scoreboard this session: 364 generated stub functions / 316 distinct
launch rows (was 464 in the last record; the mxfp4 units landed). The full
`halo_selftest` run over all rows was executed as a baseline: every failure
is `rc=-18` (`RAD_E_UNSUPPORTED`) from a stub — **zero crashes, page faults or
signal deaths**, so the plugin's linkage and dispatch are sound.

## 9. Errors, failed attempts, and baseline failures (record)

- Node baseline build with system g++ failed (317 errors); fixed by ROCm
  Clang. FFmpeg 6 header incompatibility in `core/mm/mp4.cpp` is a
  pre-existing node limitation, worked around with `-DRAD_WITH_FFMPEG=OFF`
  (images/videos tests then fail or skip).
- An accidental local `rm -rf build build-node` revealed `core/build/` is
  a checked-in source directory; restored by rsync. Full build then exit 0.
- Baseline fast-tier ctest failures on node2 (pre-existing, before libhalo):
  `gbnf_test_shard2/3`, `glm5next_test`, `images_test`, `kbench_mutant_test`,
  `kbench_test`, `plugin_test`, `videos_test`. Notes: GNU build-id (lld)
  fixed by `-Wl,--build-id=sha1`; `plugin_test` expects `libr4d.so` (not
  built on gfx1151); `gbnf_test_shard2`/`glm5next_test` pass in isolation
  (flaky under `-j8`); `gbnf_test_shard3` /
  `json_schema_walks_write_only_valid_instances` remain suspect/pre-existing.
- Stub-generator pitfalls fixed along the way: declaration regex missed
  aligned declarations/default args; extern "C" selftest hooks
  (`halo_*_pin_*`, `*_stats`, `halo_record_dispatches`) needed explicit
  handling; an undefined `halo_p2p_copy2d` initially failed `dlopen` at
  `RTLD_NOW`.
- `halo_gemm_bf16_nt_m16` port: selftest passes but kbench timing is
  unexpectedly slow (N=1280, M=1: 59.20 µs). Not yet diagnosed; possible
  grid/SK-law or A re-read issue.

## 10. Durable todo list

- [x] Deep-dive libr4d construction + core device/mem + test harnesses
- [x] Fetch node libhalo sources; set up node2 build/test loop for gfx1151
- [x] libhalo skeleton: CMake, plugin exports, shared schemas/rows/layouts,
      halo_selftest
- [x] Kernels: elementwise/norm/quant foundations (`halo_model_bf16` +
      quant units)
- [x] Kernels: GEMM family (bf16/f32, w4/w8/w2/w5, fp8, mxfp4) — complete,
      all units pass their selftest cases
- [x] Kernels: attention family (paged/dense/gen/vit/sam/vsa, QSA, MLA, DSA) —
      all rows live; zero weak stubs in the built library (§6.8)
- [x] Kernels: GDN/KDA recurrences, MoE, sampling
- [x] Core: unified-memory capability + single ledger (minimal) (§6.8)
- [x] Kernels: collectives, LoRA, HC/MHC, PLE, gen/video/audio/encoder
      families
- [x] Full verification on node: selftest 8593 ok / 0 FAIL, kbench parity
      (32 fixture artifacts, §6.8), ctest at baseline, e2e model run,
      perf sample
- [ ] Per-op tuning campaign toward roofline (§7 debt) — the remaining item
      under the user's "everything tuned" bar

## 11. Open risks

- Core unified-memory work has not started; the "minimal" bound is
  unproven until the device/memory code is read.
- Remaining parity is large: ~465 generated stubs (incl. hooks) across
  attention, GDN/KDA, MoE, sampling, collectives, LoRA, HC/MHC, PLE, and
  gen/video/audio/encoder families.
- Fragment-order adaptations rely on cross-half exchange reasoning and
  asymmetric test matrices; byte-identity claims (tiled vs m64, gated vs
  pair, slices vs whole) must keep being checked as kernels change.
- Node2-only development so far; node3 idle. No two-box TP in v1, per the
  user's scope decision.
- Two commits now exist: `checkpoint: libhalo session-1 state ...` and
  `libhalo: mxfp4a8 tiled+gated kernels, decode scratch/pitch fixes`.
