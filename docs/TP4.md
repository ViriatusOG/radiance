# Four ranks on a PCIe fabric: what pays, what doesn't, and how to tell

A measured map of every TP4 lever tried on 4×R9700 (PCIe P2P, no NVLink),
with the receipt that decided each one — for anyone attacking the same
terrain on a different card or fabric. The shipped combination here is
span (#78) + the link-hold release (#94) + the four-rank harness (#93);
everything else in the table is either reverted or still open.
All numbers: paired within-session windows, bar ≥0.5% + sign test, geometry
`--prefill-span 2 --tp-wire wht6 --placement all_vram` iq4r MoE unless stated.
Every "dead" row is a *measured* dead — the mechanism, not a guess.

Shorthand used in the table and the prose (glossed once, used everywhere):
`pp` prefill; `c1/c2/c8/c16` concurrency (live streams) 1/2/8/16; `M` rows in a collective
(batch tokens at a step, rows in a MoE launch); `I` `--checkpoint-interval`; `T` tokens
(`T<=25` at most 25 tokens); `ti8` the int8 peer wire (`--tp-wire wht6`'s coding); `FK` the
MoE fast band; `W4` the 4-bit weight format; `iq4r` the shipped MoE quantisation; `hc_write`
the fused all-reduce + residual-write row; `pp64K` a 64 K-token prefill; `QSA` the sparse
indexer (docs/QSA.md); `L0..L3` kill levels, defined under *Kill levels*; `W#`/`R#` the
experiment rows of this table.

## The table

| # | Element | What we tried | Result (measured) | Status / PR |
|---|---------|--------------|-------------------|-------------|
| 1 | `--prefill-span 2` (upstream **#78**) | overlap the two half-chunk all-reduces with compute via ti8 lane rows; needed `I=2048` + `all_vram` (placement flips its sign!) | +5…11% pp span-on under `all_vram`; repro of upstream's numbers | **SHIPPED**; receipts posted on #78 |
| 2 | **R2** = `ar_*_4rank` link-hold release (kernel gate `link_gate/early/gate_ctr`) | 4-rank rows answered "keep holding" → all yielding DMA (stager copies, expert prefetch) idled behind every collective; wire the self-release like the 2-rank rows do | **−3.66% c8 when removed** (W16, 2 rounds, per-req agrees) — the whole R-suite carrier | **SHIPPED**, PR **#94** with the number |
| 3 | **R6** 4-rank selftest harness | shipped TP4 width had zero test coverage; wide-collective + fused-write cases at world 4 (+fixed gather oracle indexing) | test-only; it's what lets anything else be proven at TP4 | **SHIPPED**, PR **#93** |
| 4 | **R1** MoE FK four-class band (20→80) | widen expert-GEMM fast band for mixed-class tiles | **inert for W4=4 (iq4r)** — cap only bites 3-bit formats; TP2: provable no-op | REVERTED; PR **#95 closed** in favor of upstream #79 |
| 5 | **R3** fused AR+hc_write prefill rows (`wire==1` band) | fuse residual write into exact-wire AR epilogue | **0 selections** in 121k-line debug-graph boot (wht6 → lossy band; fused row is exact-only); removing whole suite = noise | REVERTED; PR **#96 closed** w/ receipts — *never isolated: measured only inside the exact-wire bundle (row 9), its own territory un-timed* |
| 6 | **W15** MoE per-class grid split | give lossy/exact class tiles their own launch grids | **non-constructible**: "a run's class is not a property of `blockIdx.y`" — all y-siblings of a (tile,run) serve one run | DEAD; its selftest arm (`R4D_PERFMOE4`) and mixed-class cases are on the author's fork, not in this tree |
| 7 | **W17/#100** ti8 fused two-shot + hc_write (`ar_twoshot_hc_write_4rank_ti8`) | delete the standalone hc_write launches from the lossy decode/verify band (new `ti8 1` key, M=26..max_tok) | byte-for-bit ✅ on GPU; row selected ✅; **c8 −0.43%, c16 sign-split — no win**. Launches are cheap here (~4µs floor); epilogue serializes behind AR spin | PR **#100** (closed, branch + byte-for-bit harness kept on the fork), honest-null description; MTP-head declare finding inside |
| 8 | **W18** AR grid 16→32→64 blocks (`R4D_AR_NB_DESIGN`) | hypothesis: 16-block design starves the P2P link | **GB/s FALLS with blocks**: M=2048 34.5→33.7→31.4; M=128 18.4→12.4→7.4. Handshake-bound, not launch-starved | DEAD rungs (images kept); microbench killed it in 3 min |
| 9 | exact wire (`--tp-wire exact`) vs wht6 | exact AR everywhere incl. fused rows | loses −7.6…−8.3% at span-off to wht6; 128 KiB wire floor proven; exact only lives at T≤25 decode | rejected; upstream informed |
| 10 | F1 rank-load balancer | rebalance shard columns to equalize rank work | ~+1.3% at 64K only; **bit-changing output** | not shipped (bit-identity beats 1.3%) |
| 11 | step-graph capture (launch overhead) | replay decode passes as HIP graphs (`libr4d/isa/graphrate.hip` arms) | trunk submit = **backpressure** (device-bound; host hidden) → capture irrelevant for trunk; draft-pass gap real but hidden by other requests at c8+ | closed for trunk; draft-loop variant unmeasured (pays only at c1/c2) |
| 12 | **W19** clean lane | revert R1+R3, keep R2+R6, re-gate | **parity** (c8 +0.45%, c16 −0.08%, noise band) → deletions cost nothing | **SHIPPED = prod** |
| 13 | ti8 gather row (a hypothetical `ar_gather_hc_write_4rank_ti8`; the fused gather row that exists, `ar_gather_hc_write_4rank`, is exact-wire only) | hypothesis: the MoE gather is the last exact-wire (2 B/elem) collective; move it to ti8 like the AR rows | **premise collapsed by code reading**: on the lossy band the MoE boundary runs the unfused triple `moe_gather + ar_twoshot_4rank_ti8 + hc_write` — the gather's all-reduce already rides ti8. The `wire==0` fused row serves exact territory only; what a ti8 gather row would buy there is launch fusion, measured null (row 7) | demoted: companion of the (closed) fused-epilogue line, exact-wire configs only |
| 14 | codec inner-loop (the "22%") | decomposed the ti8 two-shot with `--perfarx` probe kernels: encode-only / encode+push / fold-only, vs the full row, M=2048/4096 | **dead, and the reframe is the prize**: encode = 4% of the row (36 µs of ~900); the P2P push runs at ~60 GB/s = **PCIe Gen5 x16 line rate** — the fabric is saturated, ti8 is already optimal per byte; fold ~200 µs is local-memory work pp doesn't wait on (row 12/13 showed pp isn't link-bound) | CLOSED — no codec win exists; “22% tax” was line-rate physics |

## Kill levels — which dead ends are actually closed

A dead end is only closed if the code ran where it was meant to run, did the right
thing, and still lost. Rows below say which level killed them; **L0/L1 kills are
implementation gaps, not physics** — a differently-coded version of those is still
fair game, and this is the list to re-attack from.

| level | meaning | rows |
|---|---|---|
| **L3 mechanism dead** | ran, correct, lost to a measured mechanism | W18 grid (GB/s falls with blocks — handshake-bound), W17 fusion (ran + byte-exact + net-flat; the saved launches are at the 4 µs floor), step-graph capture (trunk submit = backpressure — *but this one is synthetic-evidence only, see below*) |
| **L2 territory dead** | the config that would use it loses as a bundle | exact wire −7.6…−8.3% (row 9) — but it bundles rows 5/13, see L1 |
| **L1 never activated** | declared, never observed serving; never benchmarked in its own territory | **R3 fused rows (row 5): 0 selections on the lossy lane; its exact-wire territory was only ever measured inside W11's bundle, never isolated** · **fused gather row: same story — on the lossy lane `gather_taken()` skips it by design; nobody has booted an exact-wire config and timed it** (`ar_gather_hc_write_4rank`) |
| **L0 never implemented** | the idea died before code | W15 per-class grid (single-launch version non-constructible: "class is not a property of `blockIdx.y`" — a **two-launch split version was never tried**; it only matters for mixed-class formats, not W4) · ti8 one-shot band for 128 KiB–512 KiB (flagged upstream, nobody built it) · draft-pass graph capture (real host gap, unmeasured, pays only at c1/c2) |

Two caveats on the L3 list, since L3 is the only level that forbids re-trying:
step-graph closure rests on the `graphrate` synthetic ring, not an engine capture —
the engine has no capture path to test, so treat it as strong-suspect, not closed.
And W17's net-flat is solid *at the geometry it ran*; it does not falsify fused
epilogues for configs that sit host-bound (very low concurrency + slow host).

## Instruments (not lanes, but they decide lanes)

`--debug-graph` boot (2 min): which row the config *actually selects* —
killed R3 and W15 premises, and pinned R2's sites — note it lists declared bands, so a row can match on paper while a different row serves the step; confirm against the running step. `--perfar/--perfarx`:
cross-rank row us/launch + GB/s, arms interleaved one loop (drift rule); these two
`r4d_selftest` modes are on the author's fork, not in this tree (which has `--perfops`, `--perfhc`
and the GEMM/MoE ones).
`rad-kbench`: kernel-vs-reference replay. BetterBench (one fixed standard profile):
promotion truth; quick concurrency ladders within windows.

## Open levers, ranked

1. **Decode-side config**: big steps (interval 4096 / budget 8192) measured
   +2…3% c8 and −2…4% pp — a two-flag promotion if the serving mix is
   decode-heavy. No new code; already paired twice (rows 12 + W22).
2. exact-wire configs only: the fused gather/write rows (row 13) still have
   territory there — on a lossy wire they are launch-fusion rows, not wins.
3. **span_min sweep** — the floor of eight checkpoint intervals (16384 tokens at
   I=2048) applies only where routed experts are read from host memory, and was
   priced on Flash-Next at TP2; under `all_vram` there is none and every lone
   prefill spans. At TP4 with tiered experts the floor is unmeasured. One
   boot-flag arm + window.
4. ti8 **one-shot band** for 128 KiB–512 KiB lossy messages (currently every
   such message pays two-shot 1.5N + 2 syncs) — flagged upstream on #78.
5. Draft-pass graph capture — only if low-concurrency latency ever matters.
6. **The dense pipelined prefill at TP4** (`qwen35_fp8`, `qwen35_w8a8`, `gemma3`; GUIDE §4.5): a
   prefill step of one sequence in two pieces, every block boundary's all-reduce on the second lane
   at `overlap` 1, which at four ranks would take this table's `*_4rank_*_lane` rows (row 1). It
   runs at two ranks only; enabling it here (`g.world == 2` in both plugins' declare) needs a
   timing against the serial step at 18K-64K prompts and the chunk served.

## Interaction diagram

Where each element lives in one decode/prefill step (per layer, TP4):

```
                     ENGINE LAYER (per step)
   span (PR78): prefill step = 2 chunks; chunk A's AR on lane-2
   beside chunk B's compute.  stager DMA + expert prefetch run
   CONCURRENTLY with kernels — unless the link gate is held:
   ┌────────────────────────────────────────────────────────────┐
   │ link gate (R2/#94): collective must release it itself,     │
   │ else every yielding DMA waits out the whole collective.    │
   │ any NEW collective row MUST honor it or re-lose the 3.7%        │
   └────────────────────────────────────────────────────────────┘
                     KERNEL LAYER (per layer ×8 + MTP head ×2)
  attn out ──► AR ──► hc_write ──► router ──► MoE gather ──► hc_write
              │          │                       │
              │          │                       └─ MoE boundary: on the lossy wire
              │          │                          this is the UNFUSED triple (gather +
              │          │                          ar_twoshot_4rank_ti8 + hc_write) — the
              │          │                          gather's AR already rides ti8. The fused
              │          │                          exact rows (R3: 0 selects; ti8 gather:
              │          │                          launch fusion = row 7 null) serve exact
              │          │                          W17 fused ti8+write, no win
              │          └─ tiny op, 4µs launch floor (perfops) — not attackable
              │
              ├─ ar_twoshot_4rank_ti8        (T≥26 & ≥128KiB: the shipped row)
              │    └─ codec (r4d_ar_ti8.h): 4-lane groups, 38B records,
              │       own-shard exact + int8 peers.  the "22%" is the push at
              │       PCIe line rate (row 14, appendix E): no codec lever
              │    └─ grid: nb_for rule, design=16 — W18: raising it LOWERS GB/s
              ├─ exact rows (T≤25 / <128KiB): 1-sync oneshot — already cheapest
              └─ one-shot gap: 128KiB–512KiB lossy has no 1-sync form (upstream)
                     HOST LAYER
  no step graph; trunk submit = backpressure (graphrate) → launch-count
  fusion structurally unimportant (why W17 measured flat); draft passes
  have real host gaps, invisible at c8+ concurrency.
```

Value stack, multiplicative where marked: PR78 overlaps the ARs · R2 unblocks
DMA around them · on the lossy wire the AR and the gather already share the one
codec. The bytes lever is spent, and the codec's push runs at line rate (row 14).

## Appendix A — cross-rank row microbench (`--perfar` on the author's fork, idle cards, 300 iters, arms interleaved one loop)

```
ti8 M=16   nb=rule   33.1 us  14.8 GB/s   (under the 131072 B floor: exact one-shot folds it)
ti8 M=128  nb=16     63.4 us  18.4 GB/s
ti8 M=128  nb=32     94.5 us  12.4 GB/s
ti8 M=128  nb=64    156.9 us   7.4 GB/s   (64 = R4D_AR_WIDE_MAX_BLOCKS)
ti8 M=2048 nb=16    540.8 us  34.5 GB/s   (shipped)
ti8 M=2048 nb=32    554.7 us  33.7 GB/s
ti8 M=2048 nb=64    594.9 us  31.4 GB/s
exact M=2048 rule   830.4 us  37.9 GB/s   (control: 2x the bytes, 1.53x the time)
```

Reads: more blocks LOSE (handshake-bound, not launch-starved); the ti8 wire
buys 2x bytes for ~1.53x time — the ~22% fold/push tax lives inside the codec
(appendix E decomposes it).

## Appendix B — host submit cost (`libr4d/isa/graphrate.hip`, 45-dispatch pass)

```
individual      62.4 us   (1.39 us a dispatch, +GetLastError churn)
graph            3.6 us   (0.08 a dispatch — the floor, unreachable: passes aren't static)
graph+set       21.1 us   (~10 moved operand pointers/step)
recap+update    89.5 us   (capture 0.48/node + ExecUpdate 63.7 — LOSES vs plain)
full ring       52.9 us   == idle ring => trunk submit is BACKPRESSURE, host is hidden
```

Reads: step-graph capture is irrelevant for a trunk that runs ahead of the
device; it can only matter where the host sits on the critical path (MTP
draft passes), and there other requests hide the gap at ≥8 concurrency.

## Appendix C — small-op launch floor (`r4d_selftest --perfops 16 2560`, 500 launches, one sync per stretch)

```
rmsnorm_quant_fp8  4.43 us | rmsnorm_quant_fp8(res) 4.19 | gated_quant_fp8 3.70
quant_act_fp8      4.64 us | rmsnorm_bf16 4.64       | add_bf16 4.03
```

Every elementwise op costs the same ~4-4.6 us regardless of payload: pure
launch/teardown floor at 26-88 GB/s. Fusing two such ops saves the launch,
and a launch saved here demonstrably moves nothing end-to-end (see W17).

## Appendix D — the config surface (`chunk = checkpoint_interval x span`, clamped BEFORE the budget)

The step budget is not the chunk: a prefill chunk is `min(budget, interval x span)` (the engine
prints the effective chunk at boot — read that line before sweeping anything). Measured, paired,
lossy wire, chunk-4096 baseline = shipped:

| arm | pp16K | pp64K | pp128K | c8 | c16 |
|---|---|---|---|---|---|
| chunk 8192 (`I 4096 x span 2`, budget 8192) | -1.7% | -3.7% | -3.0% | +2.0% | +1.0% |
| step 8192 (budget up, chunk clamped unchanged) | -4.1% | -1.3% | -1.3% | +3.1% | +1.0% |

Both directions agree: **bigger steps/chunks trade pp for decode** — on this box pp time hides
behind compute, decode time hides behind stalls, so nothing on the link side moves pp. The step
band is sized by the AR mailbox (`numel` instance bound = budget x width), so raising the budget
extends the device band — but the arch sizes a QSA score plane `budget x max_ctx / 4` and refuses
when it grows past its cap; the flags are coupled, the refusal is named and loud.

## Appendix E — the ti8 codec decomposed (`--perfarx` probe kernels on the author's fork, M=2048 nb=16)

One launch of the shipped row = 909 us. Phase probes: encode into local mailbox **36 us (4%)**;
encode+P2P-push+flag 210 us — the push alone runs ~60 GB/s = **PCIe Gen5 x16 line rate**; fold
(reduce + region-B + out + assemble) 196 us of local-memory work. At the band top (M=4096)
nb=32 beats the rule's nb=16 by 2.3% — inside the don't-bother band.

Reads: the "22% link-efficiency tax" of the lossy wire vs exact is **line-rate physics, not an
implementation defect** — the wire is saturated and per-byte optimal. Encode is irrelevant; the
fold is real but off the critical path (see D: pp isn't link-bound). There is no codec patch to
write; proposals for codec-kernel link wins should be pointed at the push-rate line.
