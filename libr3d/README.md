# libr3d

libr3d is radiance's kernel library for RDNA3 and RDNA3.5, the gfx11xx family: `gfx1100`,
`gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, and `gfx11-generic`, the code object for a
gfx11 card the build did not name. It is HIP kernels behind a plain C ABI ([`r3d.h`](r3d.h)) plus
the kernel-plugin rows the engine selects from (`r3d_rows.cpp`); `libr3d/CMakeLists.txt` names
every unit and its flags.

It claims `^gfx11` where [`libr4d`](../libr4d/README.md) claims `^gfx12`; a card gets one device
library, chosen by the rows, not by the name. The kernels are not ports of libr4d's. The matrix
work runs on the dot2 path (RDNA3's 16x16 wave-matrix family uses a fragment layout these lane
maps do not, and gfx11 has no bf16 pack and no VOPD), so every arithmetic path is its own:
`v_dot2acc_f32_f16` for products, software codecs for every format, and weights read in the order
the container stores them. The contract is shared with libr4d: the same op schemas (copied slot
for slot from libref, as the loader requires), the same numerics rules and the same geometry in
constraints. Parts of the family differ in CU count and clock, never in these instructions, so one
code object per family suffices. Numbers in this document are quoted with their geometry, and with
the class of part where the part's size matters.

## Naming

An entry point is `r3d_<op>_<dtype>[_<the geometry it is compiled for>]`: `gemm_nt_bf16_m16`
serves `gemm_nt` at bf16 for M <= 16 and refuses a larger M; `r3d_rope_bf16` serves bf16 QKV in
place and refuses anything else. A dimension in a name is a compile-time constant, and the entry
point rejects a mismatch instead of running. Dimensions shared by the whole library stay out of
the names: wave32, the attention rows' head size (128), and the target architecture, which
`rad_plugin_info`'s `build_target` reports, spelled as a device reports its arch, because the
loader warns when a plugin claims a card the running device is not.

Rows are `RadKernelInfo` entries in a table the engine reads from the dlopened `.so`; the plugin
links the ABI headers and nothing of the engine's.

**Schema equality at load.** The first library to declare an op fixes its schema, and a second
library declaring it with a different schema is refused by name, with every row it carries.
Arguments are positional, so a mistyped operand role costs the whole library, not one row.

**The tail order is the ABI's:** `init, fini, launch, scratch, layout, relayout, opd_shape,
unrelayout, replaces`. `replaces` is last on purpose: kernel tables are positional aggregate
initialisers, a field inserted mid-struct shifts every later one, and `nullptr` converts to any
function pointer, so part of such a shift would still compile.

There is no launch adapter: every entry point is the row's `launch`, `(const RadArgs*,
RadStream)`, named `r3d_rad_<op>` in [`r3d_plugin.h`](r3d_plugin.h). Behind it each unit has two
layers. The **launcher** (`r3d_<op>(...)` in `r3d.h`) takes pointers as longs, checks what a
`__global__` cannot, computes the grid and issues one kernel on the given stream. The **shim**,
named by the row, reads operands with [`r3d_args.h`](r3d_args.h), takes the dtype from the tensor
and the row count from the operand, checks strides and returns a RAD status. Neither allocates and
neither synchronises.

The table groups the rows by family; `r3d_rows.cpp` and `r3d.h` hold the full set.

| family | rows |
| --- | --- |
| `norm` | `rmsnorm_bf16` (optional residual add and multiplier), `rmsnorm_f32`, `layernorm_bf16` |
| `elem` | `add` (with `div`, and the broadcast band), `mul`, `silu`, `gelu`, `sigmoid`, `act_mul_{silu,gelu,sigmoid}`, `softmax`, `scale`, `fill` (bf16/f32/i32/i64), `gather_rows`, `scatter_rows`, bf16 and f32 where the op has both, plus seven `cast` pairs and `silu_mul_bf16` |
| `attn` | `rope_bf16` (neox and gptj, in place over packed QKV), `kv_store_{bf16,f16,fp8}`, `attn_paged_{bf16,fp8}_{decode,prefill}`: head_dim 128, split-KV combined in the same launch |
| `gemm` | `embed_lookup_bf16`, `gemm_nt_bf16_{m16,tiled}`, `gemm_nt_q_fp8a16_m16`, `gemm_nt_q_fp8a8_{m16,tiled}`, `gemm_nt_q_mxfp4a8_{m16,tiled}` (OCP-MXFP4 weight: e2m1 codes, an e8m0 scale per 32 of K), `gemm_nt_q_w2a8_{m16,tiled}` (2-bit and 4-bit grids on the 128x128 tile), `gemm_nt_q_w4a8_m16` (signed int4 weight, decode band), `gemm_nt_q_w8a8_m16` (signed int8 weight, decode band), `gemm_nt_q_gated_fp8a8` |
| `vocab` | `logits_gemm_bf16`: the head, read once per token, as bf16 or as E4M3 codes with a bf16 scale per 128x128 block |
| `quant` | `quant_act_fp8_bf16`, `quant_act_i8_bf16`, `quant_act_i8g_bf16`, `gated_quant_fp8_bf16`, `rmsnorm_quant_fp8_bf16` (norm and quantiser in one pass), `dequant_fp8`, `dequant_e5m2`, `dequant_i8` |
| `sample` | pre-filters `sample_{penalties,dry,ngram_ban,mask,logprobs}_f32`; spine `sample_{argmax,temp,topk,pick,topp,minp,typical,xtc}_f32` |
| `moe` | `row_topk_{bf16,f32}` (the reference files the op under `moe`, so the row does too) |
| `draft` | `dflash_select_bf16`: the drafter's accept step, a walk over the draft block's candidate lattice |

The table holds 78 rows over 44 schemas; `r3d_selftest` checks every schema slot for slot against
the `libref.so` beside the plugin. GEMM rows carry tunables (`sk`, `wv`, `nt`); the rest have none.

There are two int8 activation rows because the reference has two ops and they are not
interchangeable. `quant_act_i8` writes one scale per row, with codes in the order an int8 matrix
operand is read (`i8_step_pos`, bits 2 and 3 of the column swapped); that is what
`w8a8`/`w4a8`/`w2a8` consumers expect. `quant_act_i8g` writes one scale per (row, group) with
row-major codes; that is what this library's `i8a8` GEMM reads and `dequant` inverts. The step
order belongs to the op: the oracle compares the plane element by element, and it is the one place
in libr3d where a store address carries a permutation.

## Constraints as data

Constraints are data in the rows the engine reads, because a further copy of a geometry in each
caller would drift. `r3d_selftest` checks every constraint key against the op's declared
parameters, so a key renamed in one place only is caught instead of producing a row that matches
nothing. Three engine rules decide what goes in a table:

* **A constraint on a key the band does not carry does not hold.** A band carries the parameters
  its graph set and nothing else, so a row that constrains an optional parameter serves only
  graphs that state it: binding `rotary_dim` (optional on `rope`) would resolve the row to nothing
  for every graph that rotates a whole head, and binding `pb1`/`pb2` on the quantised GEMMs would
  send a whole model's bands to the host. The selftest's reachability phase names every row that
  narrows itself this way.
* **A band's `dtype` is the activation's.** A row whose dtype lives in the weight does not
  constrain it: `logits_gemm` carries no `dtype` constraint, because the head's dtype is a
  property of the operand, checked on the tensor. `gemm_nt_q` is the opposite case: there `dtype`
  is the operand scheme, so `C_IN` filters on it. Each table states which case it is.
* **A row's floor is its sibling's declared ceiling + 1.** The tiled rows begin at `M >= 17`
  because `gemm_nt_bf16_m16` declares `C_LE("M", 16)`, read from the constraint and not from the
  name. A floor reasoned from prose ("the decode heights") leaves 17..64, the batched-decode
  heights, unserved.

What the kernel cannot read goes in the constraints as well as the launch check: `n_vocab >= 4` is
the store's granularity (rows are written in 16-byte runs); `n_cand <= 1024` is the candidate
kernels' LDS budget (five f32 arrays of `n_cand` = 20 KiB of the 64 KiB); `K % 32` and `K % 128`
are the tile's k-chunk and the container's scale block.

## Building

Out of tree, against an installed radiance, the way a third-party plugin is built:

```
cmake -S libr3d -B build -G Ninja \
      -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX" \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DRAD_GPU_TARGETS="gfx1100;gfx11-generic"
cmake --build build
```

`rad_plugin_gpu_targets(R3D_GPU_TARGETS "^gfx11")` filters the requested targets to the family,
and the project returns without a target when the filter is empty, so a build for gfx12 machines
never produces this library. `gfx11-generic` is the object for a gfx11 card the build did not
name, which is how one `.so` serves the family. In tree, `add_subdirectory(libr3d)` sits beside
libr4d's under the same `RAD_HIP_FOUND` guard.

`R3D_UNITS` lists the units as `"file|extra flags"`, each named after the entry point it provides;
the shared headers (`r3d_common.h`, `r3d_codecs.h`, `r3d_args.h`, `r3d_launch.h`, `r3d_tiling.h`,
`r3d_fp8_dequant.h`) provide no kernel. Every unit is compiled `-ffp-contract=off`, which
`rad_hipify` puts on all HIP sources; it is a correctness flag (see [Numerics](#numerics)).
`r3d_rows.cpp` and `r3d_shapes.cpp` include neither `r3d.h` nor the HIP runtime, so the tables
compile without ROCm. Occupancy is set per kernel: `__launch_bounds__` and `amdgpu_waves_per_eu`
(`R3D_WAVES`) are written against this family's register file, not RDNA4's.

## Adding a kernel

1. Write the `.hip`, named after the entry point it provides, geometry included.
2. Declare the launcher in [`r3d.h`](r3d.h) and the shim in [`r3d_plugin.h`](r3d_plugin.h).
3. Add the unit to `R3D_UNITS` in `libr3d/CMakeLists.txt`, with any per-unit flags.
4. Add a `RadKernelInfo` row in `r3d_rows.cpp` with its constraint table. If its `op` is one
   libref declares, copy libref's schema slot for slot (names and doc strings are free, roles are
   not); if the op is new, declare the schema in the same file.
5. Add the operand description to `r3d_shapes.cpp` if the row's operands are not the schema's
   plain reading, and a geometry to the selftest's list.
6. The selftest walks the table, so a new row runs with no new test code; its geometry comes from
   its constraints.
7. Measure it: `r3d_selftest <home>/kernels/libr3d.so --case <row>`, then
   `rad-kbench --home "$RADIANCE_HOME" --kernels libref:libr3d --op <op>`.

A row naming an entry point that does not exist is an undefined symbol at `dlopen`, which refuses
the library and every row in it. A row therefore lands in the same change as its kernel, and a
constraint array whose kernel has not landed stays out of the table.

## What the family provides, and what it lacks

These facts come from emitted assembly for the gfx11 targets (`__has_builtin` under
`--offload-device-only`, `llvm-mc -show-encoding` per target) with the ROCm this tree builds
against; a claim checked on one chip names it. None of them differs between gfx110x and gfx115x,
which is what makes the family one code object.

**The dot2 path, and the wave-matrix instructions it does not use.** RDNA3 has six wave-matrix
instructions, all 16x16x16 wave32: f16, bf16, int8 and int4 inputs, accumulating in f32, f16, bf16
or i32, and the toolchain emits them on every gfx11 target. They are not a separate matrix unit:
the ISA states that the WMMA ops run over multiple cycles on the DOT pipeline. A MAC-core
measurement (512 waves, K depth 4096, operands preloaded) agrees: f32-acc f16 19.11, f32-acc bf16
19.24, f16-acc f16 19.44, i32-acc i8 19.29 and paired `v_dual_dot2acc_f32_f16` 18.86 TFLOP/s, all
within three percent. The i32-acc i4 tile reaches 25.68 TFLOP/s, but it takes i4 in both operands,
and the grids this library serves pair i8 activations with i4 or i2 weights; RDNA3 has no
mixed-precision integer WMMA, so that pipe is unreachable from the w4a8 and w2a8 rows. At the
decode band the integer rows are bound by weight bytes, not arithmetic: w8a8 m16 runs 0.850 TFLOP/s
and w4a8 m16 0.789 (M=16, N=11008, K=4096, weights in device memory), about 22x below the i8
tile's MAC rate, so a re-tile onto WMMA would not move them.

The kernels therefore run the dot2 path: one wave per weight row, a k-split per lane, the group's
exact sub-sum folded with one shfl. Moving to the WMMA fragment layout (lanes 0..15 replicated into
16..31, A one row per lane, B one column per lane, accumulator row i at lane (i%2)*16+j VGPR i/2)
would be a re-tile, not a drop-in. Products go through `__builtin_amdgcn_fdot2` ->
`v_dot2acc_f32_f16` on every gfx11 target, paired as `v_dual_dot2acc_f32_f16` where the scheduler
can, about a quarter of RDNA4's matrix rate. With no fragment layouts there are no repacked
weights, and no layout hook except where the container's own planes require one.

**No int8 dot.** `__builtin_amdgcn_s32sd4`, `__builtin_amdgcn_s32ud4`, `__dp4a` and
`__builtin_amdgcn_madocx_i32_i8` / `_u32_u8` are all undeclared for gfx1102. Codes widen to f16
and the dot runs on `dot4_f16x4`. For 2-bit codes the widening uses the fact that the f16 bits of
`128 + x` are `0x5800 + 8x` for `0 <= x < 8`: a code becomes an f16 with a mask, two shifts and
an add, and the dot's result carries `128 * sum(a)`, which is the plane the op already has for the
zero point, so the de-bias is one exact f32 subtraction per (row, group).

**`v_fdot2_f32_f16` carries about 22 bits of mantissa on integer operands, not 24.** "Products
and partials are integers below 2^24, so the dot is exact" holds for f32 and not for this
instruction. `probe_dot_kernel` in `r3d_gemm_nt_q_w2a8.hip` measures it: one 64-code dot per lane,
its f32 result beside the exact integer. Over 262144 dots, a bias of 1024 leaves 71027 inexact
with worst error 0.125 (every one a dot whose magnitude reached 2^20); a bias of 128 leaves 70115
with worst error 0.015625. The worst error relative to magnitude is about 2^-22, so a smaller bias
shrinks the error without removing it. The same work as four `__fmaf_rn` per quad in f32 is exact
on all 262144 samples. `gemm_nt_q` at `w2a8` therefore ships with a documented one-ulp tail.

**No fp8 conversion instructions.** On gfx1102 a build calling `v_cvt_pk_f16_fp8` contains no such
instruction; it expands to software, about 100 instructions per 8 codes against the four that widen
bf16. `v_cvt_pk_fp8_f32` / `v_cvt_pk_f32_fp8` need `fp8-conversion-insts`, which no gfx11 target
has; their builtins are declared for gfx11, so a `__has_builtin` guard passes, and they fail at
instruction selection. Both directions of e4m3 are bit maths in [`r3d_codecs.h`](r3d_codecs.h),
mirroring `rad_plugin.h` line for line and checked over every code.

**The E4M3 decode is a table.** The arithmetic decode costs about ten VALU per code, and a
weight-stationary row decodes every code it multiplies (twice where a lane owns two chains). A
256-entry f16 table in LDS (512 bytes), built once per block by the arithmetic path and therefore
exact, costs two `ds_read_u16` and a pack per pair, at no register cost: the gated row's tall arm
is 109 registers per lane, four waves per SIMD, no spills. On the gated band (N=12288 K=2048
group=128, codes resident) the table takes M=64 from 8395.3 us to 2786.5 us. Two invariants:

* The table builder's codec argument is a pair (low code in the low byte), not an index. Passing
  the index decodes code 0 into every odd entry, which zeroes half the products without producing
  a NaN (`rel_l2` 9.6e-01 on the code-enumeration case).
* Decoded pairs go from the table into the dot inline. Materialising a step's eight pairs into an
  array on the way miscomputes on the tall arm (`rel_l2` 6.6e-01 at M=7, NaN at M=64) even though
  the table itself is exact on the device. Do not restore the array form.

**`f16x2_pack` is `v_cvt_pkrtz_f16_f32`: it rounds toward zero and flushes subnormal f16.** Only
values f16 represents exactly may go through it (widened bf16, widened E4M3), never a code times a
scale. Folding `code * scale` through the pack biases every product toward zero (`rel_l2` 2.8e-4
against the fold's law, 53% of outputs within 2 ulps). Folding with `v_mul_f16` (round to nearest,
subnormals kept) is one instruction and one correct rounding (4.5e-06, all outputs within 2 ulps,
none beyond six). Every scale fold here uses `v_mul_f16`.

**`v_cvt_pk_bf16_f32` does not exist** on gfx1100, gfx1102 or gfx1151: a bf16 pair is rounded by
hand and packed with `v_perm_b32`.

**The dot's bf16 selector does not select bf16.** `__builtin_amdgcn_fdot2` takes a `sel` operand
for the bf16 form; the compiler accepts `sel = true`, emits the f16 instruction, and reads bf16
operands as f16, silently producing wrong products. `dot2_f16x2` always passes `false`. f16 is the
compute type: bf16 and E4M3 widen to it exactly, with the edges handled in the codecs (above 65504
a widening saturates; below 2^-14 the f16 subnormal grid is coarser than the source's and
truncates).

**Wave32, a smaller register file than RDNA4.** A SIMD has 1024 VGPRs for wave32, in granules of
8, against RDNA4's 1536; LDS is 64 KiB per workgroup on both. Occupancy constants do not carry
across the generation, so `R3D_WAVES` is set per kernel. A CU is 4 SIMDs x 16 wave32s = 64 wave
slots, and `multiProcessorCount` reports CUs on gfx11 (WGPs on gfx12), so a row-per-block kernel
with one wave per block reaches 64 waves per CU exactly at the row count where tail waves start
waiting. A wave64 build is refused
at compile time (`r3d_common.h`): it would compute garbage rather than run slower.

**No vector f64.** Consumer RDNA3 runs doubles at 1:64, in practice soft-float, which sets the
transcendental policy per unit: the sampler keeps the reference's `(float)exp((double)x)`,
because a draw must reproduce; attention cannot afford it and uses a polynomial; the gated MLP uses
`expf` into a bf16 narrowing.

**The lane ops are libr4d's:** `update_dpp` (xor and shr controls), `permlanex16`, `readlane`,
`readfirstlane`, `ballot`, `ds_bpermute`, `v_perm_b32`, `exp2f`, `rcpf`, `s_barrier`,
`sched_group_barrier`. DPP16 and `v_permlanex16` exist from RDNA1 on, and the encodings in
[`r3d_common.h`](r3d_common.h) are the ones libr4d verified on hardware.

**No transposed loads, no VOPD.** There is no `global_load_tr_b128` (RDNA4) and
`ds_read_b64_tr_b16` is rejected by this assembler, so transposes are done in registers. VOPD is
RDNA4; the tree's `-mllvm=-amdgpu-enable-vopd=false` comes from the LLVM version. LLVM
single-buffers LDS and drains before a dependent read unless told otherwise, and without a matrix
unit there is nothing to overlap a load with, so the plain barrier is the one needed.

## Stored weights: layout hooks and their writers

Four rows declare a `layout`: `gemm_nt_q_w2a8_m16`, `gemm_nt_q_w2a8_tiled`,
`gemm_nt_q_fp8a8_tiled` and `logits_gemm_bf16`. Each also declares the `relayout` that produces
the form it names, and `r3d_selftest.cpp` enforces the pair by name: a `layout` without a
`relayout`, a `relayout` without a `layout`, or any `unrelayout` fails the table.

A dot2 GEMM reads a weight in the order it is stored, so libr3d never repacks: **no
plugin-private dtype, no repack into a format of this plugin's invention, no container format
changed from inside a kernel library.** The hooks exist because the container's quantiser itself
stores some weights as separate planes a kernel cannot read in place, codes and per-block scales
(for a head, `fp8_e4m3 * bf16[128x128]`). The engine's canonical form for those is the planes
joined behind one operand, `rad.planes`, sized by `rad_planes_plan` and written by
`rad_planes_relayout`. A row that reads such a weight describes that form and produces it; the
engine does the work at load, in the loader's pool behind the container read. The rules:

* **A hook that answers is a promise.** The ABI states: *RAD_OK: out describes the stored form,
  and `relayout` will produce it*. With `relayout = nullptr` the model does not load
  (`could not produce '...' (rad.planes) from fp8_e4m3*bf16[128x128]: unsupported`).
* **A weight is stored once, so the first row to claim it owns its plan.** The engine arbitrates
  across every row of every band that reads the weight, at declare (*is wanted in two stored
  forms: 'rad.planes' by row A and '(as it is)' by row B. A weight is stored once.*). A second row
  gets the same answer by calling the first row's hook, and its kernel reads the stored container
  by role, not by the arrangement it declared.
* **`RAD_E_UNSUPPORTED` means stored as is**, the only kind of weight the mapped tier can read in
  place, and the lever the placement ladder works with. A joined weight is a resident weight. A
  band that resolves to the host pulls its whole layer group onto the tier that reads the
  container in place, where a norm weight that needs conversion at load cannot live. A missing
  device row for `gemm_nt_q` above the decode band therefore shows up as a startup failure, not a
  slow band: the bands resolve to host kernels, the layer's weights stay in the container's
  mapping, its norms group with them, and the mapped tier is asked for a widening it cannot do.
* **A container arrives as a tensor of `rank = 1`, `shape[0] = bytes`.** That is the joined form's
  stored shape (a byte count, from `rad_planes_layout`), not a plane. A consumer that reads it as a
  plane measures a scale plane's rows against `ceil(N/group)` and refuses with a geometry the
  operand never had. Decode it by role on the host, where the header lives: the tag, the entry
  table and the stored dtype (the first plane's) are host-readable, so the shim reads them once per
  call and hands the kernel device pointers.
* **A `zero` plane is refused as `E_DTYPE`, not folded.** `w2a8` is the only scheme here with an
  activation zero point, and its consumer is a row of this library (`r3d_gemm_nt_q_w2a8`).
  `fp8a8`'s quantiser is symmetric: the reference's fp8a8 arm refuses `a_sum`
  (`RAD_E_UNSUPPORTED`), the shape hook skips the operand, and every served band omits it. A row
  that folded a zero point would compute a number the oracle has no law for.

Gains follow the same rule: they are read in the dtype the checkpoint stores (a 2048-element norm
gain in f32 is 8 KiB), because a row's `dtype` band names the activation, never the weight. The
fused norm's kernel takes the gain's dtype as a parameter (`WDT`), as the plain norm does.

## Request-time rules: a plan's vocabulary is not its contents

A clean graph dump means every band has a row; it does not mean a request runs. At request time
the engine hands the shim the contents (descriptors, strides, an arena, a step's shape), and a shim
must not read the band's vocabulary as if it were those contents. These are rules of the ABI, not
of any one model.

**A band's `dtype` is the activation's; a weight's dtype comes from the operand.** The head of a
`dtype=bf16` band can be E4M3 codes with a bf16 scale per 128x128 block. A shim that trusts the
key refuses the request the plan is making.

**An optional operand with no bytes is absent.** Optional operands arrive as descriptors with a
null `data` when the graph bound none, so the test is `r3d_has(t)` (`t && t->data`). A bare
`if (t)` dtype-checks a descriptor with no bytes and answers `E_DTYPE` to a band whose dtype is
fine.

**A sizing hook runs at declare, where a descriptor has a shape and no bytes** (`data` is null for
every tensor in the plan). A scratch hook that returns 0 when `a->t[0].data` is null answers 0 for
the whole run, the engine allocates a zero-byte arena, and the split pass fails at request time
with `scratch buffer absent or too small`. Size from the shape (`rad_tensor_numel`) or the
parameters, as the GEMM hook does. Absence at launch means the plan asked for nothing; absence at
declare means the bytes come later.

**A band key is a bound, not the step's value.** A plan's geometry line reads `at M=30 [band
M<=2048] geometry {N=2560 K=2048 group=128 M=2048}`: the geometry's `M` is the ceiling the band was
built for, and the height of this step is what the operands say. Geometry comes from the operand
(`nel / N`, `qnel / n`, `r3d_rows_pitch(y, N, &M)`); the key stays the ceiling, and exceeding it is
a refusal that prints both numbers.

**The engine sizes vector operands to the step, so their geometry comes from the tensor.** A
five-token prompt with nothing behind it carries a block table of `[n_seq, 1]`, not one as wide as
the cache. The reference takes the geometry from the tensor (`max_blocks` from `bt->shape[last]`,
row pitch from `stride[rank-2]`, column stride from `stride[rank-1]`,
`jend = min(ctx, max_blocks * block_size)`), so a context longer than the table's columns is
truncated by the oracle and the kernel alike. The attention shim derives all three from the
operand and checks the one thing required: that the last reachable index
`(n_seq-1)*bt_rs + (bt_cols-1)*bt_cs + 1` lies inside the plane. Two counts are kept apart:
`cache_blocks` bounds a slot's byte offset, `bt_cols` bounds the walk.

**Serve the unoptimised form when the optimisation has no room.** One split-KV slice is the whole
walk in one block, the same keys in the same order, so the launcher lowers `splits` to what the
arena holds, down to 1, instead of refusing.

**A refusal carries a code, names its fix, and prints what it saw.** `R3D_NO(code)` prints the
site under `RAD_R3D_TRACE=1`; `R3D_STR` prints the two numbers a stride test compared;
`r3d_trace_desc` prints dtype, rank, shape and strides in elements, as the engine binds them.

**The ragged query batch is a descriptor, not a second kernel.** `cu_seqlens`, when present, is
`[n_seq + 1]` i32. `cu[0] == 0`, monotonicity and `cu[n_seq] ==` the plane's row count are
promises of the description that the engine keeps, but no row may rely on them. The reference's
`seq_of` walks `while (s + 1 < n && cu[s+1] <= row) ++s`, so rows below `cu[0]` are not excluded:
they land in sequence 0 with a negative index into its query, and the reference computes them at
`pos = ctx - qlen + i`, `cu[0]` keys into the past. So `q_len` is per sequence
(`q_len = cus ? 0 : rows / n_seq`), the total row count is the plane's, and a ragged batch has no
`q_len`. One grid covers both forms because the clamp a tile needs is per sequence; the
rectangular case is the descriptor's absence (`len_s = q_len`, `start_s = s * q_len`).

**A descriptor carries two row counts, and the store is bounded by the second.**
`len = cu[s+1] - cu[s]` is what positions count from; `span = cu[s+1] - start`, with
`start = (s == 0 && cu[0] > 0) ? 0 : cu[s]`, is how many rows the sequence's tiles carry, because
the grid covers the plane from row 0. The tile count comes from the length and the store bound
from the span. Bounding the store by the length silently drops the last `cu[0]` rows of an
in-bounds plane with status 0, which only a whole-plane read-back after a NaN pre-fill detects
(see [Numerics](#numerics)); the fixture never draws `cu[0] > 0`.

## Numerics

f32 accumulation, IEEE round-to-nearest-even narrowing, subnormals kept, no reassociation; the
oracle is `libref`.

**`-ffp-contract=off` on every HIP source** (`rad_hipify`, library-wide): a fused product-sum
rounds once where the reference rounds twice, so a contracted kernel cannot match it. The flag
also keeps the instruction counts quoted in kernel comments true.

**Accumulation order is part of the contract.** `v_dot2acc_f32_f16` folds two products into the
accumulator and rounds once, so the match cannot be bit-exact: under an ulp per pair, a random walk
of about `sqrt(K/2)` ulps over a K-long dot, about a thousandth of the bf16 a store narrows to and
inside what `rad-kbench` measures. The order must be the reference's: k ascending, one accumulator
per output, no reassociation across lanes. Lane maps are built so no lane has to break it (a group
spanning two lanes folds with one `shfl_xor`, in integers, before any float work).

**Reductions use the reference's tree:** a wave butterfly (`dpp_xor` at 16, 8, 4, 2, 1; the halves
of a 64-lane row read through `permlanex16` first where the reference sums a whole row), then wave
partials in wave order, which gives the same bracketing at any thread count. Activations, rope and
quantisation are the reference's formulas in its order (`1/(1+exp(-x))`,
`0.5*x*(1+erf(x*0.70710678118654752))`, amax/127 with ties to even), from `libref/ref_common.h`.

**Accepted divergences.** Where a row is checked against the law it implements rather than
libref's bytes, the unit that ships it names the divergence:

| divergence | reason and cost |
| --- | --- |
| an index at or past an operand's row count takes the padding path (zero row, no store); `kv_store` drops a slot past the cache's span | the reference validates index arrays on the host and refuses, and `kv_store` refuses such a slot with `E_SHAPE`; a kernel cannot, so the bound travels as an argument (`slot_max`) and a slot is lost rather than a step. A negative index is padding too and is not clamped to zero, which would duplicate row 0 across every padding row |
| rope divides the position by `scale` and evaluates the frequency ladder in double | `scale` is the position scale (the multiplier on a rotated pair is YaRN's magnitude; the shims refuse YaRN, so it is 1). `powf` leaves the reciprocal an ulp off, and at position 262144 an ulp of a frequency is a hundredth of a radian, the one place in that kernel where f32 is not enough |
| `block_sum`'s tree | wave butterfly, then partials in wave order: the reference's bracketing at any thread count, so kernel and reference agree at 256 threads and at 64 |
| `sample_logprobs` keeps a NaN out of the max and in the sum; `sample_mask` skips a row whose `mask_row` is negative | maxima are a strict `>` (`std::max`'s law), so a NaN is never adopted; the sum propagates it, because the reference's `exp(NaN - mx)` is NaN and is added. The k-list is per lane in the reference's order (descending by value, ties to the lower column), merged in the same butterfly; the pool is 64 deep and a larger k is refused, not truncated. The mask bound is the reference's test (`mask_row` is `-1` for no mask), read from the operand, not from M |
| the sampler's `exp` and `log` are `double` calls narrowed once | draws must reproduce: the device's `(float)exp((double)x)` is bit-identical to the host's over 2^20 points in [-100, 100], where `expf` differs by 1 ulp on 5.7% of them. At 1:64 doubles this is the slow path (one read of the band's logits plane takes 3.6 ms, a read that also sums double exponentials 30.4 ms), affordable on the candidate set (1024 exponentials per row) but not in attention |
| attention's `exp` is a degree-8 polynomial; the gated MLP's sigmoid uses `expf` | prefill needs tens of millions of exponentials per layer. Against `(float)exp((double)x)` over 1.31M points of [-100, 0], every point is within one f32 ulp and 76.1% are bit-identical (device `expf`: 94.0%, no better in ulps), and `exp(0)` is exactly 1. The polynomial is the only place an explicit `__fmaf_rn` is allowed, because what must be reproduced is a function. The MLP's sigmoid feeds a bf16 narrowing (ulp 2^-8 against `expf`'s ~1e-7), so the bf16 differs only on a tie |
| `w2a8` has a one-ulp tail | the pair-dot carries about 22 bits on integers: 0.1-0.2% of outputs land one bf16 ulp off, `rel_l2` 5.0e-3 at the band, nothing further. The row's text says so, because the law it transcribes is exact integers |
| the fp8 scale folds land in f16, not f32 | the tile is f16, so a scale enters as an f16 multiply at staging, one `v_mul_f16` rounding per staged value. Against the reference's exact dequantise law: `rel_l2` 2.77e-04, half an f16 ulp, 250x finer than an E4M3 code's quantum (2^-4). At a bf16 output the store's ulp (3.9e-3) dominates: 83% of outputs bit-exact, 99.93% within the fold plus one store ulp |
| the bf16 GEMM's long-K tail | each pair is folded into the accumulator with one rounding, so about 0.3% of a long-K output lands one bf16 ulp from the reference's and none further: 99.93% bit-exact at the 512-column case, 100% within two ulps |

**Proving a plane was written.** Every output plane in a comparison run is pre-filled with the
dtype's NaN (`0x7fc0` for bf16 and f16) and read back whole. A kernel that returns `RAD_OK`
without launching leaves the caller's bytes, and zeros are finite, so only a NaN fill shows the row
ran. A clean case reports `untouched 0` and `nonfinite 0`, no output beyond six ulps of the law the
kernel implements, and a whole-plane `rel_l2` within what the fusion costs (`2 * sqrt(K/2)` output
ulps, with room for the fixture's spread). Two planes (an int8 row-sum padded to a 16-row grid, a
quantiser's per-group statistics) legitimately keep the fill in their padding, because the row
writes only the rows it was given; the report names that padding, and a NaN anywhere else is a
silent no-op.

Infinities are part of the contract in the sampler and row top-k: a dropped candidate, a
narrowing stage's drop and an unfilled `row_topk` slot are all `-inf`.

## Testing

[`r3d_selftest.cpp`](r3d_selftest.cpp) dlopens the `.so`, so an undefined symbol is its failure
rather than a link failure, and runs five gates:

* **Codecs** against the ABI's host versions in `rad_plugin.h`: bf16 and f16 over all 65536
  patterns in both directions; e4m3 over all 256 codes plus the encoder's saturating, subnormal,
  signed-zero and NaN cases; int8 against `rad_round_even_sat` over every tie; the widenings over
  the range in which they are exact; the bf16 pair packer against `rad_f32_to_bf16`. kbench compares
  whole ops, where a codec disagreement shows up only as a diffuse percentage, so the codecs are
  tested here.
* **The table:** unique names, a launch on every row, an `init` with a `fini`, the device domain,
  prose on every row, a schema for every `op`, every constraint key a declared parameter, each
  tunable's default among its ascending values, and the layout pair rule.
* **Band reachability:** every constraint a row puts on an optional parameter is listed. It is not
  an error (it is how a row says it serves only the form that states the parameter), but it must be
  a deliberate choice.
* **The schemas against `libref.so`** beside the plugin, with the loader's own comparison
  (parameter count, keys, types, requiredness; operand count, roles, optionality). One mistyped
  role makes the loader refuse the whole library.
* **Every row, run** on a card: buffers from the row's own `opd_shape`, filled per its fill code,
  outputs checked whole, one child process per row so a GPU page fault names its row and the run
  continues. Without a card the static phases still run, so a build machine runs them.

The numerical authority is `rad-kbench`, replaying a recorded fixture against libref:

```
rad-kbench --home "$RADIANCE_HOME" --kernels libref:libr3d
```

`--all-rows` runs every row whose constraints hold, not only the one selection picks, which is how
tunable arms the tuned defaults never reach get measured. `--coverage DIR --coverage-write`
records the rows a fixture reaches, so a row that stops being reachable fails the run; the census
is [`kbench_rows_kernels_libr3d.txt`](../data/kbench_rows/kbench_rows_kernels_libr3d.txt), written
by the same run.

**A fixture covers a checkpoint, not the library.** A fixture is generated from the bands a
recorded checkpoint issues, so a row can be correct and absent from the census: the base fixture
has no `w2a8` band above M=16. Record one fixture per served checkpoint; `rad-kbench` records one
from a model (reading only the tables) and widens every case into the constraints of every row
under test:

```
rad-kbench -m /path/to/served.rad --record "$RADIANCE_HOME/kb-served.rkb"
rad-kbench --fixture "$RADIANCE_HOME/kb-served.rkb" --record-rows "$RADIANCE_HOME/kbrows.rkb" \
           --kernels libr3d,libref
rad-kbench --fixture "$RADIANCE_HOME/kbrows.rkb" --kernels libref:libr3d --all-rows \
           --no-cache --max-cases 2 --max-bytes 1024 --seed 42
```

The first command checks the served checkpoint's own bands against the reference on the
container's real weight bytes. The second and third take each case to the M and `q_len` values no
container selects, the edges of each row's constraints where floors and ceilings are exercised.
Rows the checkpoint's graph never uses report `no recorded case of its op`.

**Checks a tolerance cannot make.** A refactor is proved by the recorded decision lines being
bit-identical between the old and new builds (diffed with the timing column blanked). A change to
an index, a scale plane or a row mapping is proved bit for bit against the same fixture: a fixture
whose scales are statistically alike cannot tell a row-indexing error from rounding (with uniform
random data the amax of 128 samples is concentrated, every block's scale rounds to the same bf16,
and reading group 0's scale for group 7 changes nothing), so paired runs spread their activations
by orders of magnitude one group at a time. The comparison harness carries a `selfcheck` control
that hands the scorer a plane rebuilt under a known misreading and requires a divergent verdict
naming it.

The gates test a built binary, not the commit: build a clean export of the commit on the target
before trusting the gates for that commit.

## The fp8a8 GEMM by row count

The bands this family targets are 42 layers plus a drafter at `group=128 dtype=fp8a8`, band
ceiling M=2048: N=12288 K=2048 x44 (the gated MLP, two halves of 6144), N=2560 K=2048 x44 (qkv),
N=2048 K=6144 x44 (o-proj), N=2048 K=2048 x44, and N=2048 K=10240 x3 (the drafter). Three rows
cover them by row count, with boundaries read from the constraint tables: `gemm_nt_q_fp8a8_m16`
declares `C_LE("M", 16)`, so `gemm_nt_q_fp8a8_tiled` opens at `C_GE("M", 17)`, and
`gemm_nt_q_gated_fp8a8` serves M=1..64 with N even and K % 128 == 0.

**The m16 row** gives each lane one output element and streams the weight past an activation kept
in registers, with no wave reduction, so a lane's chain is the reference's and the row is
bit-comparable at the store. Above 16 rows the same operands use a 128x128 tile through LDS with
8x8 f32 accumulators per lane. The two ranges are complements, and the m16 row is compared against
the tiled row on the same bytes, where a moved boundary would show.

**The tiled row folds the scales at staging.** The tile is f16, so both scale streams enter as f16
multiplies where the codes are staged: the activation's `a_scale[m][g]` per staged row, the
weight's `b_scale[nb][g]` once per chunk (TN 128 is the scale plane's row extent, so a tile is one
scale row, and kKC 32 divides 128, so a chunk sits inside one k-block). The products are
separable, `(a*sa)*(b*sb) = (a*b)*sa*sb`, so the accumulator stays one f32 register. The
reference's law is the group sub-sum accumulated from the raw codes and scaled once per group
(`ref_gemm.h`'s `quant_dot`, the body the m16 and gated rows carry); the fold differs from it by
`rel_l2` 2.77e-04 against a bf16 store that costs 3.9e-3, so it is invisible in the output (see the
divergence table). `|y|` is not a usable denominator, since a sum of 2048 random products is often
near zero; the scale is `S = sqrt(sum (a_k b_k)^2)`, and the fold's floor is 9.8e-4 of it.

**At the band ceiling** (M=2048, container resident): N=12288 K=2048 28.171 ms (3659 GFLOP/s);
N=2560 K=2048 5.911 ms (3633); N=2048 K=6144 14.699 ms (3506); N=2048 K=2048 4.381 ms (3922);
N=2048 K=10240 29.798 ms (2883). That is 27 to 36% of the part's dense f16 rate. The bytes floor at
163 GB/s is 0.051 to 0.257 ms per band, and 2MNK/(K+N) is about 1900 flops per byte at M=2048, so
prefill is compute-bound and these numbers describe the loop, not memory.

**A grid-shape cliff.** The tiled kernels run one block per CU at 18.4 to 18.5 us per block at the
column-tile counts the graphs issue (16 and 20); 19 tiles cost 211 us per block (11.4x) and 21
tiles 385 us (20.9x), deterministic to 0.1%. Every column-tile count the served graphs issue is
even, so no band hits it. The first things to try are a block-index swizzle, then the `nt`
streaming loads.

**Verifying the crossing.** One `RadArgs`, the device row and libref's host row on the same bytes,
the whole `y` plane compared, pre-filled with `0x7fc0` so an unwritten element is visible, every
other plane poisoned, and 256-byte guards checked after the launch. The run covers every form the
served plan hands the row (rank-1 `a_scale` of `[M]`, the codes as a rank-3
`[M][ceil(K/128)][128]` plane, the weight as the container the layout hook writes) at the four
served bands and at M = 17, 21, 30, 45, 64, 65, 128 and 258, against a 5e-3 tolerance; the same
band with `b_scale=[scale,zero]` is refused with `E_DTYPE`. The harness has two plugins dlopened,
both row tables resolved the way the engine's loader resolves them, and scores a plane against nine
index laws with a hash beside it. It also covers the collapsed mapped and managed tier
arrangements (the mirror layout, the default, is the served one). The rank-1 `a_scale` binding
matters: a shim passing the wrong row pitch for it reads rows 0 and 1 correctly and everything past
the plane after that, which the row's own geometry test cannot see.

### The gated MLP row

`gemm_nt_q_gated_fp8a8` fuses the GEMM, the gate and the output quantiser, so the bf16 `[M, N]`
intermediate is never written, read or allocated. The gate couples column `i` with column `n + i`,
192 waves apart at n = 6144, out of reach of any lane exchange; so each lane computes both products
of its column, walking k ascending with two accumulators over weight rows `i` and `n + i`, the same
traffic the unfused GEMM spends on the two columns. A block is one group of columns by `256/group`
rows, which closes the group's amax inside the block: a wave max folded over the group's waves in
LDS, one scale store, one code per lane. The op is one launch with no scratch, so the row is
concurrent (`rad_kernel_concurrent(i)` is `scratch == nullptr`). Both halves are narrowed to bf16
before the gate, because the pair it replaces stores and re-reads them; the activation is not
(`gt * sg` is f32 from the narrowed gate, and the product with `up` rounds once, at the store). The
law is libref's `ref_gemm_nt_q_gated`, and its order must be kept.

The launcher picks the arm by row count. fp8a8 at M <= 8 reads the activation and both weight rows
directly, 16 bytes at a time, decodes them from the block's table and multiplies: no shared memory
between load and dot, no barrier in the k-loop, six loads in flight. M >= 9, and every fp8a16 case,
stage through LDS; at those heights the grid is 48 x M/2 blocks and occupancy feeds the stream.
On the band's stream (24 MiB of codes, resident, loop without the epilogue), staging with the
arithmetic decoder takes 497 us, direct reads with the table 187 us, and the bare loads 139 us.

Measured on the band (N=12288 K=2048 group=128, codes resident), best / median of nine launches,
staged arithmetic-decode arm against the shipped arms:

```
M = 1     987.4 / 1154.9  ->   171.3 /  181.6 us     (147 GB/s of codes; the floor for
M = 2    1120.6 / 1196.5  ->   166.2 /  177.1 us      24 MiB at 163 GB/s is 154 us, and
M = 4    1417.3 / 1581.2  ->   221.0 /  229.6 us      the same sixteen loads with no
M = 8    1874.0 / 2318.0  ->   403.4 /  412.3 us      decode and no dot run in 139 us)
M = 16   2949.3 / 3226.1  ->  1055.0 / 1160.6 us
M = 32   5123.4 / 5405.1  ->  1660.6 / 1705.0 us
M = 64   8395.3 / 9051.0  ->  2786.5 / 2877.3 us
```

The M <= 8 threshold is a tunable, chosen for decode (M=1) and draft verify (M <= 8). With the table
decoder in both arms the direct arm takes 402 us against 754 at M=8 and 955 against 1030 at M=16,
so the crossover is above 16; M=9..15 is unmeasured, and a prefill of 9..16 rows pays about 7% for
the staged arm. The load window is for the direct arm only: on the staged arm it costs 32
registers per lane, a fifth wave per SIMD, and at M=64 it costs more than it saves.

The row carries no relayout. libref reads the gated `[N, K]` weight row-major; libr4d stores fp8a8
weights in WMMA fragment order, and the two readings differ by a relative error of about 1.35, so
a row that inherited the fp8 GEMM's `unrelayout` would fail the per-op oracle. The launcher refuses
M > 64 (beyond the bands' ceiling the weight stream is a tiled kernel's problem), N odd, a group
that is not a power of two in [32, 256], `group > n` (the reference would then use n for the
quantiser's group while keeping the weight's K block at the stated width, two grids this kernel
reads as one), `K % group != 0` (a ragged K block reads past the scale plane), `sk > 1` (the
epilogue needs the whole row), and an absent `act`.

## The paged attention, decode and prefill

Two rows, chosen by `q_len`: the decode entry, which takes `splits`, and the prefill entry. The
served geometry is n_head 16 over n_head_kv 2 at head_dim 128 (gqa 8), `block_size 16`, a 4096
context, and an `fp8_e4m3` cache with its descales. The layout is the reference's:
`kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]`, K then V inside a slot, so a key's row
is a 512-byte pair and both halves are 16-byte aligned at every slot. `block_table[seq, j /
block_size]` names the block holding key j; a negative entry is an unmapped block and hides its
key (the allocator frees a sequence by writing -1, so the kernel keeps the test although the
fixture never sets it). `seqused[s]` is the sequence's total key count including this step's query
rows, so query row *i* of a `len_s`-row step sits at absolute position `seqused[s] - len_s + i`.
One kernel serves decode (q_len 1), speculative verify (q_len 7) and a prefill chunk (q_len 2048).

**A lane owns one output row, and a wave is the GQA heads of one KV head:**
`h = kh * GQA + (lane & (GQA-1))`, `tok = base + (lane >> lg GQA)` (8 heads over 4 rows at gqa 8),
so every lane of a wave reads the same cache row at the same time. Every cache read is then an LDS
broadcast, one `ds_read_b128` serving the wave, and the paged walk is a per-wave event. 32 rows per
cache read is also the register file's ceiling: a second output row per lane would need 2 x 64
VGPRs of widened query plus 2 x 128 of f32 accumulator, 384, over the 256 that four waves per SIMD
allow.

**The walk runs twice.** The reference keeps its scores, but 32 keys x 128 dims of f32 is 512 KiB
per wave against 64 KiB of LDS, so pass two recomputes them, in the same order over the same
operands, bit-identically; the op costs half again what keeping them would. A masked key is
skipped, never scored to -inf, so the denominator has exactly the reference's terms, and a row with
no visible key stores zeros (the reference's `mx == -INFINITY` case). P*V stays f32: folding the
softmax weight into V would read every weight to 11 bits, a ~1e-4 change that survives the bf16
store on a few percent of outputs. `exp` is the degree-8 polynomial from the divergence table, and
`exp(0)` is exactly 1, the term the row is normalised by.

**Workgroup width is the effective lever.** A row slot past a sequence's length computes a clamped
duplicate (walk, tile and store all run for it). `pick_waves` picks the narrowest workgroup that
carries the step's `q_len` (one wave up to 32/GQA rows, two up to 64/GQA, four above); prefill
keeps four. Against a fixed four-wave workgroup (medians of 15 per-launch event times): 294.7 vs
158.4 ms at 256 sequences x 7 rows over 4096 keys, 268.0 vs 128.4 ms with 4 splits, 4.33 vs 2.06 ms
at 8 sequences over 2048, 4.27 vs 3.97 ms at one sequence over 4096, prefill unchanged. The kernel
issues about 15% of the SIMD's slots at the band and 3.7% at eight sequences, so the gain is not
explained by arithmetic; the width is therefore chosen at run time.

**Levers that did not help:** `s_setprio` around the consume leg of the walk (|delta| <= 0.7%);
`R3D_WAVES(2)` to lift the scalar-register ceiling (`sgpr_count` 107 and about 98 SGPRs spilled at
both 4 and 2 waves per SIMD, timings unchanged: the compute-shader scalar budget is not set by
waves per SIMD the way the VGPR budget is); `__noinline__` on the staging leg (spills 98 -> 72, a
160-byte private frame, timings unchanged); a two-stage pipeline (+7% prefill, +7% decode, -5%
split-KV).

**Split-KV is a rescale.** A slice accumulates its own `(acc[128], sum, max)` into scratch and the
reducer combines them before the single narrowing, rescaling each partial by `exp(m_s - M)`. Every
term then arrives as `exp(s - m_s) * exp(m_s - M)` where the reference has `exp(s - M)`, a last-bit
difference at f32 (about 2 ulp per term, 1 ulp on the sum) that vanishes under the bf16 store
except on rounding boundaries. The row's text states the contract: `splits = 1` is bit-identical to
the reference, `splits > 1` is not. `splits` is a maximum, resolved per sequence from its own
`seqused`: `nslice = ceil(nblk / ceil(nblk / splits))` with `nblk` the sequence's block count, so a
short sequence uses fewer slices and needs no per-sequence legality check, and the reducer
recomputes `nslice` from the same two numbers, so a partial needs no header.

The split curve on the served shape (64 sequences, 8 KV heads, fp8 cache, ms per launch at splits
1/2/4/8/16/32/64): ctx 1024, 8.17 / 5.51 / 9.29 / 1.50 / 2.54 / 0.39 / 0.23; ctx 4096, 35.71 /
20.92 / 11.41 / 5.79 / 2.67 / 1.44 / 2.60. Each workgroup walks `ceil(nblk / splits)` blocks with
`nblk = ceil(jcap/16)`, so the total keys read are `jcap` at every split count, and the cost is
slices times per-slice work plus the reduce. The clamp `splits <= ceil(jcap/block_size)` is a no-op
at the served shape (`ceil(1024/16)` is 64) and applies only where splits exceed `nblk`; a grid
built from the requested count then has slices with no blocks, which costs waves but stays
correct.

**Bounds.** `jcap = min(max_ctx, cols * block_size)`: the block table is the window the kernel may
read (see [the request-time rules](#request-time-rules-a-plans-vocabulary-is-not-its-contents) for
`cache_blocks`, `bt_cols` and the last-index check). In the tile walk the descriptor's length gives
the tile count and its span the store bound, so a sequence with `cu[0] > 0` stores every row the
grid covers.

**A padding row must not write a partial.** A padding row's `row` clamps to a real row, so the
split path tests row validity before the partial store. Without it, two lanes write the same
`(row, h, slice)` record inside the declared arena, their 132-float records interleave, and the
reducer rescales a full slice by the wrong maximum, producing sums thousands of times too small. It
needs `q_len < 32/gqa` (at gqa 8, q_len 7 fills a wave and there are no padding rows), and it is a
schedule, not a sample: it fails identically at every seed. The slots a padding row leaves unwritten
are exactly those the reduce does not read; its index runs over real rows only.

**Verification** is a paired run with libref and libr3d dlopened in one process, both called with
the same `RadArgs`, absent optional operands passed as a descriptor with null `data` (never as a
shorter `n_t`), with the fp8 cache form in the same run. It covers the served shapes, the
rectangular matrix, deep contexts 512..4096, narrow block tables (`bt cols` 1/2/3/4/8, so `jcap`
16/32/48/64/128 under a ctx of 512), empty sequences (`cu = [0,0,5,5,17]`), split counts 1 and 4,
windows 1/3/17/64/512, causal and non-causal.

**Launch tracing.** A served decode step is played: the recorded pass re-issues queue packets and
never calls the plugin's shim, so a print in the shim never sees played steps. Under
`RAD_R3D_TRACE` only, `attn_kernel` writes one record per launch: its launch ordinal, the row's
`ctx` from `seqused`, the whole block-id row of the table, the six operand addresses, and FNV-1a
folds of the query row and of the first and last block it walks. Records go to a plugin-owned
`__device__` ring of 256 fixed 216-byte slots that the shim drains to stderr on a traced call; the
budget is a device global the shim writes, decremented with `atomicSub`, the counter advanced with
`atomicAdd`. A played step's record is emitted one step late, and the record's own `L` and `ctx`
restore the order. In a served process the plugin is dlopened, and the runtime's symbol stub for a
plugin-owned `__device__` global is not the address the kernel's module writes, so the ring cannot
be armed there; the served channel is a `hipMalloc`'d record slot passed as a kernel argument,
drained at the launch and armed by the same flag. With the flag unset the kernel pays one scalar
load and a not-taken branch on one thread, and the launch one cached flag read. With it set, a
traced call pays one device synchronize (the descriptor probe reads the table on the host) and one
stream synchronize (the record drain), and the first traced call allocates the record slot; that is
why the flag is a tracing switch and not a mode.

Device `printf` cannot be used instead: a kernel that calls it takes an extra implicit argument,
`hidden_hostcall_buffer`, which the AQL backend does not provide, so the engine cannot dispatch it
(the fixture refuses the row with `refused a geometry its constraints accept: device error`), while
the same kernel launched with hipLaunch from a standalone harness prints normally. Only a gate that
dispatches through the engine, here the recorded fixture, sees such dispatch-side problems.

## The sampler spine

Thirteen rows: five pre-filters, `sample_topk` feeding `sample_pick`, and the narrowing stages
between them. `abi/rad_sample.h` is the wire format, and the central design choice: every scalar
(`temp`, `top_k`, `top_p`, `min_p`, `typical_p`, the `xtc_*` pair, `seed`, `pos`, the flags, the
history window) is data, one 96-byte row per sampled position, because parameters are geometry,
geometry is frozen at declare, and two rows of one batch with different `top_p` must resolve to one
kernel. `param_rows` is that check, and it is the dtype check libref states: the plane is 32-bit
words, and a 16-bit view passes a byte-count test and then misreads every row.

**Draws must reproduce, so summation order is transcribed.** Every max is a strict `>`, which also
keeps NaN out: libref's `if (v > mx) mx = v` from -infinity never adopts a NaN, and the butterfly
merges with the same comparison. The reference's sums run over the candidates in index order, and
its CDF scan accumulates in index order, so the exponentials are computed in parallel (each an
independent value) and the accumulation is serial on one lane, in index order.

**The draw is a function of the ABI, not of a library.** splitmix64 over
`seed * 0x9E3779B97F4A7C15 + pos`, the top 24 bits over 2^24, in integer arithmetic so device and
host agree, with independent streams per purpose (the pick XORs `RAD_SAMPLE_KEY_PICK`, XTC's coin
`RAD_SAMPLE_KEY_XTC`). `ref_exp`/`ref_log` are `(float)std::exp((double)x)` and
`(float)std::log((double)x)`, transcribed as double calls rounded once (see the divergence table);
a `u < cum` comparison on an ulp boundary changes the drawn token, which fails the fixture even
when the distribution is right. The widest softmax here is the candidate set, 1024 exponentials per
row at about half a millisecond per step, where the 1:64 f64 rate is affordable; attention, at 17M
exponentials per layer, uses the polynomial.

**The top-R is one scan.** `topk_next` takes, on each of R scans, the largest value strictly below
the last pick, ties to the lower column; as a whole that is a sort by (-value, +column), and the
rule `v == *pv && i <= *pi` is "the next one within a run of equal values". One pass keeping the
best R in that order produces the same R pairs in the same order with the same tie-breaks. At the
band's row (M=1792, N=130560, R=16, bf16) the reference scans the row sixteen times (7.0 GB); this
scans it once (468 MB), in 22.0 ms bf16 and 22.4 ms f32. The pass is bound by accepted inserts
(about 55 sifts per lane at 510 elements per lane), not by memory; filtering against a block-wide
threshold instead of each lane's own tail would cut that to about 5 sifts and approach the 2 ms
bandwidth floor. Each lane keeps a 64-deep list and a merge keeps the best 64 of two; testing an
element against the lane's 64th best is one comparison for almost every element (about 12% of a
row clears it at 510 elements per lane). 64 covers R=16, the selftest's 64-wide set and every
practical `top_k`; above it `r3d_sample_topk` and `r3d_row_topk` take `ceil(k/64)` passes, the
reference's own chain, so the result is the same at any k.

**The candidate plane is the contract.** A row's set is its `top_k` largest logits, descending,
ties to the lower token id, with `cand_idx` -1 and `cand_val` -infinity for a hole. The width is
the buffer's declare-time bound: 1024 candidates is the engine's declared width (two f32 arrays of
it are 8 KiB of the 64 KiB a workgroup has), and the row's `top_k` narrows within it, so `top_k` 40
in a 1024-wide buffer is 40 candidates and 984 holes. The whole buffer is written every step,
because later stages test for holes. A greedy row takes no candidates: its token is
`sample_argmax`'s and the pick must not overwrite it. `cand_val` carries logits, not probabilities:
every narrowing stage takes its own softmax over the set, so the stages compose in any order and a
pre-filter can be moved without re-deriving anything.

**The step's row count comes from the operand** (see
[the request-time rules](#request-time-rules-a-plans-vocabulary-is-not-its-contents)). A candidate
stage that used the band key's `M` would ask one row of candidates to be 256 rows tall and refuse
every logprobs request (`refused op sample_topp at M=1 [band M<=256] geometry {n_cand=1024
M=256}`).

**Verification** is the fixture rather than a paired run: the selftest builds each row's buffers
from its own `opd_shape`, runs it and checks the plane whole in a child process, and kbench drives
the `sample_*` bands against libref on the served checkpoint's fixture. What the fixture cannot
check is the seeded draw against a recording, over the parameter rows a real request sends (see
[What is not here yet](#what-is-not-here-yet)).

## Measuring

**State the floor beside the number.** 66.8 MB of codes at a part's 22 GB/s is a 3.0 ms floor;
255 MiB at 163 GB/s is 1.64 ms; a 73.1 MB operand is a 0.449 ms read. Against those, 13.6 ms per
token or 24.151 ms at M=64 describe the loop, not memory.

**State the placement of every operand on every timing line.** On a part whose memory cannot hold
the head it is worth an order of magnitude: at M=1, one kernel, the same head in VRAM is 13.6 ms
per token; on the mapped tier (pinned host bytes read over the host link by 8-byte strided loads)
643 ms; managed and left alone 2071 ms at 122 MB/s, because managed pages do not migrate for a
read-mostly device access and the kernel faults page by page. A layout hook makes the mapped
placement legal, not fast. Benchmarks state which placement produced which number and time after a
prefetch. A container operand must be host-readable (the shim reads its header on the host), so it
is pinned or managed, never plain `malloc`, which faults. Allocation history matters too: the same
shape in the same binary measures 5.911 ms as a process's first case and 85.7 ms after a 24 MiB
managed container was allocated and freed, because the slow run reads its weight over the host
link. The row's number is the VRAM one, because the engine moves weights to VRAM at load.

**`vgpr_spill_count` is a gate:** every kernel must report zero VGPR spills. Per-group partials
scaled into the accumulator cost 64 more accumulators on an arm already at 237 VGPRs, and a runtime
index into a register array spills the array to private memory; both need a rewrite, not a tunable.
A scalar spill in a compute-shader wavefront is a slow path, not a stall (see the attention
measurements), so SGPR spills are reported, not gated.

**Per-op profile times sum to more than the token.** `--profile-ops` restores the synchronisations
that the engine's overlap removes, as the engine's startup diagnostics state
([`core/diag/checks.cpp`](../core/diag/checks.cpp)). A per-op table taken under that flag ranks the
work and does not price it; the token rate is the price. Quote a served run's shape with its
numbers (context length, prompt lengths), because a ranking is a property of the shape, not the
model.

## Notes on the kernels

Three properties are assumed throughout and not restated per kernel: **wave32 with a 1024-entry
VGPR file per SIMD**, so every lane map has a lane owning whole dwords and every budget is computed
against 1024; **64 KiB of LDS per workgroup**, which sets the candidate kernels' `n_cand <= 1024`
and the tile sizes; and **the dot2 path**, which is why the tile is a dot loop over LDS with an
8-element XOR swizzle, shared by every tiled row in [`r3d_tiling.h`](r3d_tiling.h).

**The 2-bit grid's tiled staging moves 8x the bytes it uses.** A staged chunk of 8 bytes against a
row pitch of 512 at `kKC` 32 gives 24.151 ms at M=64, N=130560, K=2048 against a 0.449 ms read
floor, 3.7 GB/s useful. The lever is a wider k-chunk (64 codes = 16 bytes, paid for in LDS); more
loads in flight moved it by 0.6%.

**The head is the largest read in a decode step,** and the sampler reads f32, so the store is
typed, not narrowed, and a 2048-row prefill band writes 1.07 GB. bf16 head: 3.29 ms per token at
M=1 (163 GB/s), 0.507 ms at M=16 (the head read once for sixteen rows), 269 ms at 2048 rows (4067
GFLOP/s). E4M3 head with a bf16 scale per 128x128 block, read as its two container planes: 13.6 ms
per token at M=1, 8x above the 1.64 ms floor. The table decoder gives 2.70x where the decoder binds
(M=8: 2.500 -> 0.924 ms per token), nothing at prefill, and nothing at M=1. The likely limit at
M=1 is load width (eight codes per lane through an 8-byte load, where the bf16 arm loads sixteen
bytes per lane), so the next step there is a 16-byte code load.

**Clear the HIP error flag before a launch check.** `hipGetLastError()` reports the first error
raised since it was last cleared, so a check that does not clear first attributes to this launch
whatever the process raised before it (in a served process, the previous op; in a comparison run,
its own setup). The symptom is a whole timing mode returning the device code on every geometry of
every row, disappearing under `HIP_LAUNCH_BLOCKING=1` (a synchronous launch clears a pending
asynchronous error before the check). Clear the flag after the arguments are staged and before the
launch, read it straight back, and on a refusal name the error and print the grid, block, waves,
splits and width the launcher chose.

**No collective rows and no chained forms.** The parts in this family are single-GPU, and an
all-reduce over the host link most of them sit on is slower than sending the bytes through the
host. Nothing here is built as a chained form, so both optional symbols are omitted, which is the
ABI's own answer.

## What is not here yet

**The 2-bit grid's staging width.** The packed-code tile at prefill heights moves 8x the bytes it
uses at `kKC` 32; the fix is a 64-code chunk (16 bytes per lane) paid for in LDS, which is the
whole gap between 24.151 ms and a 0.449 ms floor at M=64, N=130560, K=2048.

**A 16-byte code load for a one-row E4M3 head.** The decode arm loads eight codes per lane through
an 8-byte load and runs 8x above its floor at M=1; the table decoder is not the lever there.

**A gated MLP pass over rows and column groups.** Past M=64 the fused row refuses and the work is
four ops over a bf16 `[M, N]` tensor (two GEMMs, the gate, the quantiser). At M=64, the ceiling 47
of the 48 bands declare, the weight read is the limit; a pass that amortises the weight read across
rows and column groups is what is missing.

**The gated row's arm threshold.** The crossover between the direct and staged arms lies above 16
rows and M=9..15 is unmeasured; measuring it would move the threshold, not the kernel.

**`gemm_nt_q_bias` has no row.** `gemm_nt_q` has no bias operand; bias is its own op in the
registry ("gemm_nt_q with a per-column bias"), and this library declares no row for it. Every band
carrying a bias therefore resolves to the reference's host row: correct and slow, with no `E` line
in the coverage and no profile entry, because nothing refused and the op never reaches a device. A
checkpoint whose bias tensors have count 0 (no planes, no bytes) carries no bias band, by the rule
that an optional operand with no bytes is absent; `rad-info -v <model> | grep -i bias` shows
whether a model's bias tensors have planes. The row would be a bias load in the tiled rows'
epilogue plus a declared schema.

**Sliding window over a partial last tile diverges.** With `window` from 2 to 15 over a query plane
whose last tile is not full, the attention row diverges from the reference (for example 17 rows at
ctx 33 with w = 3: `rel_l2` 1.196e+00, status 0 from both rows, no non-finite values). Full
attention (`window = 0`) agrees, and served plans use a 2048 window over 16-wide blocks, so no
served band reaches it. The fix belongs in the last tile's mask, not the window's clamp.

**`sample_merge_topk`**, the cross-rank candidate merge: under a split vocabulary it holds
`world_size * n_cand` gathered pairs, and there is no row for it.

**A seeded-draw comparison for the spine.** kbench grades the `sample_*` bands against the
reference on the fixture, which covers the arithmetic; nothing yet compares the drawn token against
a captured sequence, over the parameter rows a request actually sends.

**int8 x int8.** The constraint table is written (`i8a8`, group 128, M <= 16, K % 128) and
`quant_act_i8g` produces the row-major activation it reads; the kernel has not landed, so the row
is not in the table.

**Chunked logits.** The head's f32 plane is written whole (1.07 GB at a 2048-row prefill band);
nothing emits it in chunks for the sampler to consume in passes.

**A sub-byte zero plane.** `w2a8` reads a zero point one byte at a time and refuses a zero stored
as U2/U4/U5 by name; the bit-extract path would be the same `rad_load_bits` the codes use.

**Speculation that pays.** The draft head is served end to end (its bands by the 2-bit grid, its
accept step by `dflash_select`) and the accepted text is identical. The target's head read
amortises over the draft block (5x cheaper per token at 8 rows than at 1 on the same kernel), so
the drafter's own steps have to pay for themselves, and on a 16-CU part they do not (87.6 -> 94.4
ms per token on a 5-token prompt, 91.6 -> 187.1 on a 16-token chat band with 7 drafts). The
engine switches the drafter off for a plugin whose rows measured a gain without it; what is missing
is a draft path that costs less than the head read it saves.
