# exl3i -- EXL3-V3 linears and token embeddings on gfx1201

One codebook: **mad**.

The radiance kernel plugin for EXL3-V3 weights: a trellis-coded weight at any rate in 1/8 steps (K = m/16, from 0.125
to 8 bits a weight), decoded on the GPU in the GEMM itself. Radiance at plugin ABI 44, interface version 5
(`include/exl3i_abi.h`). Device code is gfx1201 only; the host parts build and test without a GPU.

## What it serves

Rows of the op `exl3i_dense` (radiance picks the path by the launch's real rows and the act mode):

| row | act | what it does |
|-----|-----|--------------|
| `exl3i_gemm_nt` | f16 / bf16 | the f16 WMMA kernel. Below 512 rows it is the whole linear; from 512 rows a two-pass prefill (T2: the decoded plane once per launch). |
| `exl3i_gemm_nt_a8` | `a8t`, `a8g128`, `a8g256`, `a8g512`, `a8g1024` | the A8 path from 512 real rows: rows rotated and quantised to int8 once (actq), the trellis decoded once per column chunk into an int8 plane (wdec8), then radiance's i8a8 GEMM. Below 512 rows it falls back to the f16 row. mad at `a8g128` with 1 to 8 rows takes the byte-sum GEMV (`exl3i_mad.hip.h`). |
| `exl3i_actq` | (producer side) | the A8 quantiser for up to four consumers of one x (AQ, interface version 5): optional pre stage (rmsnorm, silu_mul, mul), residual add, packed rows. |
| `exl3i_rerank` | (head) | logit rerank against an exl3i head for the drafter's candidates (up to 256 a row). |
| `embed_exl3i_gather` / `ref_embed_exl3i` | (token table) | the token embedding gathered from an exl3i trellis table: device row and host row (the reference). |
| `embed_tq_gather` / `ref_embed_tq` | (token table) | the token embedding from a table-quantised fwht128 table: device and host rows. |

Codebook: **mad** (IQR's computed code on the int8 grid, byte-sum form; the linears, the A8 path and the token table). Any other
codebook name is refused by name (`RAD_E_DTYPE` at the op's parse, `RAD_E_FORMAT` at the encoding
read), so a container that names one fails at load.

Per-weight activation modes come from the container's `exl3.act.<weight>` key (`include/exl3_act.h`): `f16`, `a8t`,
`a8g<N>` (N a multiple of 128). A value the parser does not know is refused, never read as f16.

Not in this plugin: the int4-activation (A4) path and the V1 / V2 f16 kernels.

## Files

- `rows.c`: the rows, their constraints (codebook, act, K and N in 128s), and the shape hook.
- `nt.c`: the host side of `exl3i_gemm_nt` and `_a8`: operands, dispatch, the scratch arenas, the record's layout hooks.
- `exl3i_nt.hip.h`, `exl3i_nt_mad.hip`: the f16 kernel (mad). `exl3i_rate.hip.h`: the decode
  policies (integer and fractional rates, the lane plans in `exl3i_plan*.h`).
- `exl3i_a8*.hip`, `exl3i_8a8.hip.h`, `exl3i_a8.h`, `exl3i_a8_dev.h`: the A8 path. `exl3i_mad.hip.h`: the mad GEMV.
- `exl3i_rerank.hip`, `rerank.c`, `rerank_ref.c`: the rerank row and its host reference.
- `embed_exl3i.*`, `embed_tq.*`: the two token-embedding gathers.
- The oracle (every decoder is checked against it) is `../exl3import/common/exl3i_decode.c`; the record's encoding is
  `../exl3import/include/exl3i_encoding.h` (transform `<codebook>/h128/k<rate>`, `/r16` for the head's row order).

## Build

```
cmake -S plugin/exl3i -B build-exl3i -G Ninja -DCMAKE_PREFIX_PATH=<radiance install, plugin ABI 44>
cmake --build build-exl3i -j
```

With HIP and `RAD_GPU_TARGETS` including gfx1201 the library and the device tests build; without it the library is
left out and the host tests still build. `EXL3I_BUILD_ID` sets the build id the plugin reports.

## Test

- Host (no device): `ctest --test-dir build-exl3i -L host` -- `exl3i_plan_test`, `exl3i_plan_tab_test`,
  `exl3i_twopass_host_test` (the A8 arithmetic, the arena, dispatch and refusals with the launchers stubbed),
  `exl3i_rerank_host_test`, `embed_exl3i_test` (the embedding host rows against the oracle, bit for bit, mad).
- Device (gfx1201): `ctest --test-dir build-exl3i -L gpu` -- `exl3i_gpu_test`, `exl3i_twopass` (and
  `exl3i_twopass_nofuse`, the A8 path's two passes), `exl3i_ks`, `exl3i_r16`, `exl3i_aq`, `exl3i_rerank`,
  `embed_exl3i_gpu_test` (the executables `test_twopass`, `test_ks`, `test_r16`, `test_aq`, `test_rerank`).
- Timing: `bench_twopass EXL3I.so --rate 4 --codebook mad --modes f16,a8,a8x,phases` (a driver, not a test).

In radiance's tree the library builds with the engine when its GPU targets name gfx1201 and installs as
`kernels/exl3i.so`; the tests above are part of the suite (`ctest -R 'exl3i|embed_exl3i'`), the host ones in the fast
tier, the device ones under `-L gpu`. `ctest -L oot` rebuilds the library out of tree as above and compares its device
code with the tree's.

The path a launch takes follows from its shape and act mode alone. The device tests and the bench compare each fast
form with the form it replaces through test pins (`exl3i_pin.h`, `exl3i_test_pin`, reached by `dlsym`): the A8
threshold and chunk, the decode form's K split, the mad GEMV, the one-row DOT form, the two-pass prefill form, the
residual rmsnorm's split and the fused A8 GEMM. A served engine never sets them.

## Numbers

The full records (kernel tables, the decode and prefill measurements, the A8 design) are not in this tree: they are in
rad-exl3-mix under `docs/plugins/exl3i/` (PERF.md, TWOPASS.md; A4.md for the W4A4 path).
