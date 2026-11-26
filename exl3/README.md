# exl3 — EXL3 trellis experts

radiance-exl3 (Apache-2.0, `LICENSE` and `NOTICE` here), imported from
[mikoskinen/r9700-exact-serving](https://github.com/mikoskinen/r9700-exact-serving) branch
`exl3/current` at ce9e7a6d (15.11.2026), built in this tree. Upstream pinned radiance 1.17
(plugin ABI 37); here it builds against this tree's ABI (44).

| Directory | What it is |
|---|---|
| `kernels/exl3r4d` | gfx1201 kernel library: `moe_exl3_gate_up`, `moe_exl3_down` and the ops in `docs/OPS-exl3.md` |
| `kernels/exl3ref` | host reference library for the same ops (a reference: ranked last) |
| `quantizers/exl3` | the `exl3` quantiser, which imports packed EXL3 planes into a `.rad` |
| `common/` | the canonical EXL3 decode and plane layout, and their tests |
| `include/` | interface headers: op schemas, operand positions, `EXL3_IFACE_VERSION` (10) |

`arch/common/exl3/` carries copies of `include/exl3_abi.h`, `include/exl3_encoding.h` and
`common/exl3_decode.h` for the architecture plugins; ctest's `exl3_vendor_check` keeps them equal.
`arch/common/rad_block_moe_q.h` serves a layer whose routed experts carry the `exl3` encoding
through `moe_exl3_gate_up` / `moe_exl3_down` (whole experts on their rank: expert parallel or one
rank).

Left out of the import: the Qwen architecture plugin, the device tests, the calibration and
profiling tools, and the engine-side levers of upstream's DeepSeek branch (expert tensor parallel,
the host-pool copy ring, the decode-band `_dk` ops). The kernel library still exports its fn4 and
host-pool hooks (`rad_fn4_switch`, `rad_host_pool_ranges`); this engine does not call them, so they
stay at their defaults.

`ctest -L host` runs the host tests (the Python ones need `python3` with `numpy`). The device rows run
through the same harness and NumPy transcription (`exl3r4d_cases`, `exl3r4d_band_tops`, label `gpu`,
on a gfx1201 card). `exl3_schemas_agree` checks the exl3 libraries' schemas, ABI and layout
fingerprint against the tree's kernel libraries with `rad-schemas`.
