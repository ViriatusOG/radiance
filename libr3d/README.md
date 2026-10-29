# libr3d

libr3d is radiance's kernel library for RDNA3 and RDNA3.5 — the gfx11xx family:
`gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, and `gfx11-generic`,
the code object for a gfx11 card a build did not name. HIP kernels behind a plain C
ABI ([`r3d.h`](r3d.h)) plus the kernel-plugin rows the engine selects from
(`r3d_rows.cpp`); `libr3d/CMakeLists.txt` names every unit and its flags.

It claims `^gfx11` where [`libr4d`](../libr4d/README.md) claims `^gfx12`: a card gets
one device library, picked by the rows and never by the name. The kernels are not
libr4d's ported: the matrix work runs on the dot2 path — RDNA3's 16×16 wave-matrix
family is a fragment layout these kernels' lane map does not use, and gfx11 has no
bf16 pack and no VOPD — so every arithmetic path is its own: `v_dot2acc_f32_f16` for
products, software codecs for every format, a weight read in the order the container
stores it. The contract is shared: the same op schemas (copied
slot for slot from libref, as the loader requires), the same numerics rules, the same
geometry in constraints. Family parts differ in CU count and clock, never in these
instructions, so one code object per family suffices — and every number here is quoted
with its geometry, and with the class of part where the part's size is part of the
fact.

## Naming

An entry point is `r3d_<op>_<dtype>[_<the geometry it is compiled for>]`:
`gemm_nt_bf16_m16` serves `gemm_nt` at bf16 for M ≤ 16 and refuses a larger M;
`r3d_rope_bf16` serves bf16 QKV in place and refuses anything else. A dimension named
in a name is a compile-time constant, and the entry point rejects a mismatch instead
of running. Shared dimensions stay out of the names: wave32, the attention rows' head
size (128), and the target architecture, reported by `rad_plugin_info`'s
`build_target` — spelled as a device reports its arch, because the loader warns when a
plugin claims a card the running device is not.

Rows are `RadKernelInfo` entries in a table the engine reads from a dlopened `.so`;
the plugin links the ABI headers and nothing of the engine's.

**Schema equality at load:** the first library to declare an op fixes its schema, and
a second declaring it with a different schema is refused by name with every row it
carries — arguments are positional, so a mistyped operand role is not a bad row, it is
no library.

**The tail order is the ABI's:** `init, fini, launch, scratch, layout, relayout,
opd_shape, unrelayout, replaces` — `replaces` last on purpose, because kernel tables
are positional aggregate initialisers: a field inserted mid-struct shifts every later
one, and `nullptr` converts to any function pointer, so part of the shift would
compile.

There is no launch adapter: every entry point *is* the row's `launch`, `(const
RadArgs*, RadStream)`, spelled `r3d_rad_<op>` in [`r3d_plugin.h`](r3d_plugin.h).

Behind it, two layers per unit: the **launcher** (`r3d_<op>(...)` in `r3d.h`) takes
pointers as longs, checks what a `__global__` cannot, computes the grid, issues one
kernel on the stream it was handed; the **shim**, named by the row, reads operands
with [`r3d_args.h`](r3d_args.h), takes dtype from the tensor and the row count from
the operand, checks strides, returns a RAD status. Neither allocates, neither
synchronises. The table lists a selection; `r3d_rows.cpp` and `r3d.h` hold the full
set.

| family | rows |
| --- | --- |
| `norm` | `rmsnorm_bf16` (optional residual add and multiplier), `rmsnorm_f32`, `layernorm_bf16` |
| `elem` | `add` (with `div`, and the broadcast band) · `mul` · `silu` · `gelu` · `sigmoid` · `act_mul_{silu,gelu,sigmoid}` · `softmax` · `scale` · `fill` (bf16/f32/i32/i64) · `gather_rows` · `scatter_rows`, bf16 and f32 where the op has both, plus seven `cast` pairs and `silu_mul_bf16` |
| `attn` | `rope_bf16` (neox and gptj, in place over packed QKV), `kv_store_{bf16,f16,fp8}`, `attn_paged_{bf16,fp8}_{decode,prefill}` — head_dim 128, split-KV combined in the same launch |
| `gemm` | `embed_lookup_bf16`, `gemm_nt_bf16_{m16,tiled}`, `gemm_nt_q_fp8a16_m16`, `gemm_nt_q_fp8a8_{m16,tiled}`, `gemm_nt_q_mxfp4a8_{m16,tiled}` (OCP-MXFP4 weight: e2m1 codes, an e8m0 scale per 32 of K), `gemm_nt_q_w2a8_{m16,tiled}` (2-bit and 4-bit grids on the 128×128 tile), `gemm_nt_q_w4a8_m16` (signed int4 weight, decode band), `gemm_nt_q_w8a8_m16` (signed int8 weight, decode band), `gemm_nt_q_gated_fp8a8` |
| `vocab` | `logits_gemm_bf16` — the head read once a token, bf16 codes, or E4M3 codes with a bf16 scale per 128×128 block |
| `quant` | `quant_act_fp8_bf16`, `quant_act_i8_bf16`, `quant_act_i8g_bf16`, `gated_quant_fp8_bf16`, `rmsnorm_quant_fp8_bf16` (norm and quantiser in one pass), `dequant_fp8` `dequant_e5m2` `dequant_i8` |
| `sample` | pre-filters `sample_{penalties,dry,ngram_ban,mask,logprobs}_f32`; spine `sample_{argmax,temp,topk,pick,topp,minp,typical,xtc}_f32` |
| `moe` | `row_topk_{bf16,f32}` — the op's stage in the reference is `moe`, so the row's is too |
| `draft` | `dflash_select_bf16` — the drafter's accept step, a walk over the draft block's candidate lattice |

Seventy-eight rows, 44 schemas, every schema checked against the `libref.so` beside the
plugin, slot for slot, by `r3d_selftest`. GEMM rows carry tunables (`sk`, `wv`, `nt`);
the rest have none.

Two int8 activation rows, because the reference has two ops and they are not
interchangeable: `quant_act_i8` is one scale per row, codes in the order an int8
matrix operand is read (`i8_step_pos`, bits 2 and 3 of the column swapped) — what
`w8a8`/`w4a8`/`w2a8` producers mean; `quant_act_i8g` is one scale per (row, group),
row-major codes — what this library's `i8a8` GEMM reads and `dequant` inverts. The
step order is the op's, not a kernel's convenience: the plane is what the oracle
compares element by element, the one place in libr3d where a store address carries a
permutation.

## Constraints as data

Constraints are data in the rows the engine reads — a third copy of a geometry in each
caller drifts — and `r3d_selftest` checks every constraint key against the op's
declared parameters: a key renamed in one place only is a row that matches nothing.
Three rules decide what goes in a table, all three the engine's.

* **A constraint on a key the band does not carry does not hold.** A band carries the
  parameters its graph set and nothing else, so a row constraining an optional
  parameter serves only graphs that spell it out: `rotary_dim` is optional on `rope`,
  and binding it resolved a row to nothing for every graph that rotates a whole head;
  `pb1`/`pb2` on the quantised GEMMs did the same to a whole model's bands, which fell
  to the host. The selftest's reachability phase names every row that narrows itself
  this way.
* **A band's `dtype` is the activation's.** A row whose dtype lives in the weight does
  not constrain it: `logits_gemm` carries no `dtype` constraint, because the head's
  dtype is a property of the operand and the row answers dtype with a check on the
  tensor. `gemm_nt_q` is the opposite case — there `dtype` *is* the operand scheme, so
  `C_IN` filters — stated at each table.
* **A row's floor is its sibling's declared ceiling + 1.** The tiled rows begin at `M
  >= 17` because `gemm_nt_bf16_m16` declares `C_LE("M", 16)`, read from the constraint
  and not from the name. A floor reasoned from prose ("the decode heights") left
  17..64 unserved and 209 batched-decode bands on the host: the shape a serving card
  spends most of its life in.

What the kernel cannot read goes in the constraints, not only in the launch: `n_vocab
>= 4` is the store's granularity (rows are written in 16-byte runs); `n_cand <= 1024`
is the candidate kernels' LDS budget (five f32 arrays of `n_cand` = 20 KiB of the 64
KiB); `K % 32` / `K % 128` are the tile's k-chunk and the container's scale block.

## Building

Out of tree, against an installed radiance, as a third-party plugin is built:

```
cmake -S libr3d -B build -G Ninja \
      -DCMAKE_PREFIX_PATH="$RADIANCE_PREFIX" \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DRAD_GPU_TARGETS="gfx1100;gfx11-generic"
cmake --build build
```

`rad_plugin_gpu_targets(R3D_GPU_TARGETS "^gfx11")` filters requested targets to the
family, and the project returns without a target when the filter is empty: a build for
gfx12 machines does not produce this library, on any path. `gfx11-generic` is the
object for a gfx11 card the build did not name — why one `.so` serves the family. In
tree, `add_subdirectory(libr3d)` sits beside libr4d's under the same `RAD_HIP_FOUND`
guard.

`R3D_UNITS` lists units as `"file|extra flags"`, each named after the entry point it
provides; the shared headers (`r3d_common.h`, `r3d_codecs.h`, `r3d_args.h`,
`r3d_launch.h`, `r3d_tiling.h`, `r3d_fp8_dequant.h`) provide no kernel. Every unit is
`-ffp-contract=off`, which `rad_hipify` puts on all HIP sources — not a style
preference (Numerics). `r3d_rows.cpp` and `r3d_shapes.cpp` include neither `r3d.h` nor
the runtime, so the tables compile without ROCm. Occupancy is per kernel:
`__launch_bounds__` and `amdgpu_waves_per_eu` (`R3D_WAVES`) are written against this
family's register file, not RDNA4's.

## Adding a kernel

1. Write the `.hip`, named after the entry point it provides, geometry included.
2. Declare the launcher in [`r3d.h`](r3d.h) and the shim in
   [`r3d_plugin.h`](r3d_plugin.h).
3. Add the unit to `R3D_UNITS` in `libr3d/CMakeLists.txt`, with any per-unit flags.
4. Add a `RadKernelInfo` row in `r3d_rows.cpp` with its constraint table. If its
   `op` is one libref declares, copy libref's schema slot for slot — names and doc
   strings are free, roles are not; if the op is new, declare the schema in the
   same file.
5. Add the operand description to `r3d_shapes.cpp` if the row's operands are not
   the schema's plain reading, and a geometry to the selftest's list.
6. Give the row a case: the selftest walks the table, so a row is run by adding no
   test code — geometry comes from its constraints.
7. Measure it: `r3d_selftest <home>/kernels/libr3d.so --case <row>`, then
   `rad-kbench --home "$RADIANCE_HOME" --kernels libref:libr3d --op <op>`.

A row naming an entry point that does not exist is an undefined symbol at `dlopen`,
which refuses the library and takes every row with it: a row lands in the same change
as its kernel, and a constraint array whose kernel has not landed stays out of the
table.

## What the family gives, and what it takes

Measured by reading emitted assembly for the gfx11 targets (`__has_builtin` off
`--offload-device-only`, `llvm-mc -show-encoding` per target) against the ROCm this
tree builds against; where a claim was taken on one chip it says which. Nothing here
changes between gfx110x and gfx115x — which is what makes the family one code object.

**The dot2 path, and the wave-matrix instructions it does not use.** RDNA3 carries
exactly six wave-matrix instructions, all 16×16×16 wave32: f16, bf16, int8 and int4
in, f32, f16, bf16 and i32 accumulating, and the toolchain emits them on every gfx11
target. They are not a separate matrix unit: the ISA states the WMMA ops run over
multiple cycles and drive the DOT pipeline internally, and the MAC-core measurement
on this part confirms it (512 waves, K depth 4096, operands preloaded: f32-acc f16
19.11, f32-acc bf16 19.24, f16-acc f16 19.44, i32-acc i8 19.29, dot2 paired as
`v_dual_dot2acc_f32_f16` 18.86 TFLOP/s — the f16/bf16/i8 tiles and the dot path sit
on the same pipe within three percent, matching the full-kernel 1.99-vs-1.97 of
r3d_common.h). The i32-acc i4 tile is the exception at 25.68 TFLOP/s, 35% above the
dot path, but that instruction takes i4 in *both* operands and the grids this
library serves pair i8 activations against i4 and i2 weights — there is no
mixed-precision integer WMMA on RDNA3, so that pipe is unreachable from the w4a8
and w2a8 rows. The i8 tile's only edge over the current w8a8 path is a decode, and
the kernel-level measurement closes the question: the w8a8 m16 row runs 0.850
TFLOP/s and w4a8 m16 0.789 (M=16, N=11008, K=4096, weights in device memory) --
twenty-two times below the i8 tile's own MAC core, weight-byte-bound at the decode
band's geometry, where no re-tile of the arithmetic moves the number. These
kernels run
the dot2 path: one wave a weight row, a k-split per lane, the group's exact sub-sum
folded with one shfl. The WMMA fragment layout (lanes 0..15 replicated into 16..31,
A one row a lane, B one column a lane, the accumulator's row i at lane (i%2)·16+j
VGPR i/2) is a re-tile, not a drop-in, and no measured variant earns it. Products
go through `__builtin_amdgcn_fdot2` → `v_dot2acc_f32_f16` on every gfx11 target,
paired as `v_dual_dot2acc_f32_f16` where the scheduler can — about a quarter of
RDNA4's matrix rate. No fragment layouts, so no repacked weights, so no layout hook
except the container's own planes force.

**No int8 dot.** `__builtin_amdgcn_s32sd4`, `__builtin_amdgcn_s32ud4`, `__dp4a`,
`__builtin_amdgcn_madocx_i32_i8` / `_u32_u8` are all undeclared for gfx1102, and all
four were tested. Codes widen to f16 and the dot runs on `dot2_f16x4`, paid for with a
trick: the f16 bits of `128 + x` are `0x5800 + 8x` for `0 <= x < 8`, so a 2-bit code
becomes an f16 with a mask, two shifts and an add, and the dot's answer comes back
carrying `128 · Σa` — the plane the op already has for the zero point, so de-bias is
one exact f32 subtraction per (row, group).

**`v_fdot2_f32_f16` carries a little under 24 bits on integers — call it 22.**
"Products and partials are integers below 2^24, so the dot is exact" is true of f32
and false of this instruction. Instrument: `probe_dot_kernel`, in the unit that uses
it — 32 lanes a block, one 64-code dot per lane, its f32 beside the exact integer.
Bias 1024: **71027 of 262144 dots inexact, worst |err| 0.125**, every one a dot whose
magnitude had reached 2^20; bias 128: 70115 of 262144, worst 0.015625. Worst error
over magnitude is ~2^-22, so shrinking the bias shrinks the tail and does not remove
it. The same work as four `__fmaf_rn` per quad in f32: 262144 samples, **0 inexact,
worst error exactly 0**. `gemm_nt_q` at `w2a8` ships a documented tail, not a claim.

**`v_cvt_pk_f16_fp8` is listed by the compiler and absent from the silicon:** on
gfx1102 the ISA of a build calling it contains zero `v_cvt_pk_f16_fp8`, counted by
grep — it expands to software, ~100 instructions per 8 codes against the four that
widen bf16. `v_cvt_pk_fp8_f32` / `v_cvt_pk_f32_fp8` need `fp8-conversion-insts`, which
no gfx11 target has; their builtins are *declared* for gfx11 — the trap a
`__has_builtin` guard falls into — and die at instruction selection. Both directions
of e4m3 are bit maths in [`r3d_codecs.h`](r3d_codecs.h), mirroring `rad_plugin.h` line
for line, checked over every code.

**No fp8 instruction means the decode is arithmetic, so the arithmetic gets a table.**
Ten VALU a code, and a weight-stationary row decodes every code it multiplies — twice
over where a lane owns two chains — which at M=1 is every code in the operand. A
256-entry f16 table in LDS, 512 bytes, built once a block by the arithmetic path
itself and therefore exact by construction, costs two `ds_read_u16` and a pack a pair
at zero register cost: the gated row's tall arm is 109 registers a lane, four waves a
SIMD, zero spills, and the table alone moves M=64 from 8395.3 us to 2786.5 us (its
band: N=12288 K=2048 group=128, codes resident, best of nine) — two thirds of that
arm's gain is the decoder. Two landmines in it, both load-bearing. The table's
argument is a **PAIR**, low code in the low byte, not an index: feeding it the index
fills the even entries and decodes code 0 into every odd one, and the code-enumeration
case names that as `rel_l2` 9.6e-01, not a NaN. And the decoded pairs
go from the table into the dot **inline**: materialising the round's eight pairs into
an array on the way — the shape this loop had, and the shape the arithmetic decoder
feeds — put the plane about a third away from the reference at every tall height (M=7
`rel_l2` 6.6e-01; M=64 `-nan`) with the table itself proven exact on the device, entry
by entry and pair by pair. The array form is the one that miscomputes; do not restore
it as a tidy-up.

**`f16x2_pack` is `v_cvt_pkrtz_f16_f32`: rounds toward zero, flushes subnormal f16.**
Only values f16 represents exactly may go through it — bf16 widened (7 mantissa bits
into 10), E4M3 widened (3 into 3) — never a code times a scale. Folding `code * scale`
through the pack biased every product toward zero: `rel_l2` 2.8e-4 against the fold's
law, 0.53 of outputs within 2 ulps. Folding with `v_mul_f16` (round to nearest,
subnormals kept) is one rounding in the right direction and one instruction: 4.5e-06,
1.000 within 2 ulps, nothing beyond six. Every scale fold here goes through
`v_mul_f16`.

**`v_cvt_pk_bf16_f32` does not exist** on gfx1100, gfx1102, gfx1151: a bf16 pair is
rounded by hand and packed with `v_perm_b32`. **The dot's bf16 selector is a lie** —
`__builtin_amdgcn_fdot2` takes a `sel` operand for the bf16 form, the compiler accepts
`sel = true`, emits the f16 instruction, and bf16 operands are read as f16s: nonsense
products, silently. `dot2_f16x2` passes `false`, always. f16 is the compute type: bf16
and E4M3 widen to it exactly, with the edges in the codecs — above 65504 a widening
saturates, below 2^-14 the f16 subnormal grid is coarser than the source's and
truncates.

**Wave32, smaller register file than RDNA4:** 1024 VGPR entries a SIMD for a wave32,
granules of 8, against RDNA4's 1536; 64 KiB of LDS a workgroup on both. Occupancy
constants do not cross the generation, so `R3D_WAVES` is per kernel. A CU is 4 SIMDs ×
16 wave32s = 64 wave slots, and `multiProcessorCount` reports CUs on gfx11 (WGPs on
gfx12), so a row-per-block kernel with one wave a block passes 64 waves a CU at
exactly the row count where tail waves start waiting. A wave64 build is refused at
compile time: it computes garbage rather than running slower.

**No vector f64:** consumer RDNA3 is 1:64 on doubles, in practice soft-float, which
sets the transcendental policy per unit — the sampler keeps the reference's
`(float)exp((double)x)` (drawing a different token than the recording is wrong at any
speed), attention cannot pay it and ships a polynomial, the gated MLP uses `expf` into
a bf16 narrowing.

**The lane ops are libr4d's** — `update_dpp` (xor and shr controls), `permlanex16`,
`readlane`, `readfirstlane`, `ballot`, `ds_bpermute`, `v_perm_b32`, `exp2f`, `rcpf`,
`s_barrier`, `sched_group_barrier`; DPP16 and `v_permlanex16` are RDNA1 onward, and
the encodings in [`r3d_common.h`](r3d_common.h) are the ones libr4d proved on
hardware. **No transposed loads, no VOPD:** neither `global_load_tr_b128` (RDNA4) nor
`ds_read_b64_tr_b16` (rejected by this assembler), so a transpose is done in
registers; VOPD is RDNA4, and the tree's `-mllvm=-amdgpu-enable-vopd=false` comes from
the LLVM version, not from here. LLVM single-buffers LDS and drains before a dependent
read unless told, and with no matrix unit there is nothing to overlap a load with, so
the barrier needed is the plain one.

## Weights are stored once: the hook, and the writer that goes with it

Four rows declare a `layout` — `gemm_nt_q_w2a8_m16`, `gemm_nt_q_w2a8_tiled`,
`gemm_nt_q_fp8a8_tiled`, `logits_gemm_bf16` — and each declares the `relayout` that
produces the form it names; `r3d_selftest.cpp` enforces the pair by name (a `layout`
without a `relayout` fails the table, a `relayout` without a `layout` fails it, any
`unrelayout` fails it).

The starting rule — *no layout hooks in this library* — came from the dot path: a dot2
GEMM reads a weight in the order it is stored, so there is nothing to repack. It
stands in every part that carries weight: **no plugin-private dtype, no repack into a
format of this plugin's invention, no container format changed from inside a kernel
library.**

What it got wrong is that the container's quantiser already stores weights where a
kernel cannot read them in place — codes and per-block scales as separate planes,
`fp8_e4m3 * bf16[128x128]` for a head — and the engine's answer is its own canonical
form: planes joined behind one operand, `rad.planes`, sized by `rad_planes_plan`,
written by `rad_planes_relayout`. A row that reads such a weight *describes* that form
and *produces* it; the engine does the work at load, in the loader's pool behind the
container read. Three laws came out of it.

* **A hook that answers is a promise** — the ABI says so: *RAD_OK: out describes the
  stored form, and `relayout` will produce it*. With `relayout = nullptr` the model
  does not load: `could not produce '…' (rad.planes) from fp8_e4m3*bf16[128x128]:
  unsupported`.
* **A weight is stored once, so the row that got there first owns its plan.** The
  engine arbitrates across every row of every band reading the weight, at declare: *is
  wanted in two stored forms: 'rad.planes' by row A and '(as it is)' by row B. A
  weight is stored once.* A second row gets the same answer by *calling the first
  row's hook*, and its kernel reads the container the builder settled — by role, not
  by the arrangement it declared.
* **`RAD_E_UNSUPPORTED` means stored as is**, the only kind of weight the mapped tier
  can read where it lies — the lever the placement ladder runs on. A joined weight is
  a resident weight, and a band that resolves to the host drags its whole layer group
  onto the tier that reads the container in place, where a norm weight needing
  conversion at load refuses to live. Hence the quantised GEMM at prefill heights as a
  startup failure and not a slow band: no device row for `gemm_nt_q` above the decode
  band → bands resolve to host kernels → the layer's weights stay in the container's
  mapping → its norms group with them → the mapped tier is asked for a widening it
  cannot perform.
* **A container arrives as a tensor of `rank = 1`, `shape[0] = bytes`.** That is the
  joined form's stored shape — a byte count, from `rad_planes_layout` — and not a
  plane. A consumer that reads it as a plane measures a scale plane's rows against
  `ceil(N/group)` and refuses with a message about a geometry the operand never had:
  the fp8a8 tiled row refused the sibling decode row's settled spelling (codes plain
  on `b`, scale joined behind `b_scale`) with `E_SHAPE` from a tensor geometry test on
  every case of one probe and at `n_b = 0`, and with `E_STRIDE` at one band — two
  codes, one expression, both naming rows and columns of a thing that was never a
  plane. Decode by role, on the host, where the header lives: the tag, the entry table
  and the stored dtype (the first plane's) are host-readable, so the shim reads them
  once per call and hands the kernel three device pointers. In that form the row
  computed the reference's answer on **192 cases, 192 agreements, worst `rel_l2`
  1.362e-03** — and the same fix carried the m16/tiled crossing to 1018 cases with no
  numeric divergence, where before it the crossing never reached the joined spelling.
* **A `zero` plane is refused as `E_DTYPE`, not folded.** `w2a8` is the only scheme
  here with an activation zero point, and its producer is a row of this library.
  `fp8a8`'s quantiser is symmetric, so the reference's fp8a8 arm refuses `a_sum`
  outright (`ref_gemm.cpp:395`, `RAD_E_UNSUPPORTED` — that arm has no grid to sum a
  zero over), the shape hook skips the operand, and every served band omits it. A row
  that folded a zero point would be inventing a number the oracle has no law for, so
  the refusal names its own fix instead: the row that computes zeros is
  `r3d_gemm_nt_q_w2a8`.

Same law, gains: read them in the dtype the checkpoint stores (a 2048-element norm
gain in f32 is 8.00 KiB), because the row's `dtype` band names the activation, never
the weight — the fused norm's kernel grew the plain norm's `WDT` leg, a dtype
parameter on the gain, rather than ask the checkpoint to be rewritten.

## The request: a plan's vocabulary is not its contents

A clean graph dump means every band has a row; it does not mean a request runs. At
request time the engine hands the shim the *contents* — descriptors, strides, an
arena, a step's shape — and the refusals between a clean dump and a first generated
token were one mistake: code that read the band's vocabulary as if it were the plan's
contents. Laws of the ABI, not of any one model.

**A band's `dtype` is the activation's; a weight's dtype comes off the operand.** The
key names the step's activations, and the head of a `dtype=bf16` band can be E4M3
codes with a bf16 scale per 128×128 block. A shim that trusts the key refuses the
request the plan is making.

**An optional operand with no bytes is absent.** Optional operands arrive as
descriptors with a null `data` when the graph bound none, so the test is `r3d_has(t)`
— `t && t->data`; a bare `if (t)` is the bug: dtype-checking a descriptor with no
bytes answers `E_DTYPE` to a band whose dtype was fine, which is what
`rmsnorm_quant_fp8` did at M=5 with an unbound `out_bf16`.

**A sizing hook runs at declare, where a descriptor has a shape and no bytes** (`data`
is null for every tensor in the plan). A scratch hook opening `if (!a->t[0].data)
return 0` answers 0 for the whole run, the engine allocates an arena of 0 bytes, the
split pass finds nowhere to write — a request-time `scratch buffer absent or too
small` from a hook that looked healthy. Size off the shape (`rad_tensor_numel`) or the
parameters, as the GEMM hook does. Absence at launch means the plan asked for nothing;
absence at declare means the bytes come later.

**A band key is a bound, not the step's value.** A plan's geometry line reads `at M=30
[band M<=2048] geometry {N=2560 K=2048 group=128 M=2048}`: the geometry's `M` is the
**ceiling the band was built for**; the height of *this* step is what the operands
say. A GEMM asking a 30-row activation for 2048 rows, and a sampler computing `rows =
(M_hint > 0) ? *M_hint : cirows` before asking the plane to be that tall, refuse every
request that asks for logprobs. Geometry comes off the operand — `nel / N`, `qnel /
n`, `r3d_rows_pitch(y, N, &M)` — and the key stays the ceiling promised not to be
exceeded. **Exceeding it is a refusal, and it prints both numbers.**

**The engine sizes vector operands to the step, so their geometry comes from the
tensor.** The attention shim required `numel(BT) >= n_seq * max_blocks` with
`max_blocks` from the band's `max_ctx` — a table as wide as the cache — while a
five-token prompt with nothing behind it carries `[n_seq, 1]`: one block, because one
block is what the sequence has. The reference takes geometry from the tensor
(`max_blocks` from `bt->shape[last]`, row pitch from `stride[rank-2]`, column stride
from `stride[rank-1]`, `jend = min(ctx, max_blocks * block_size)`), so a context
longer than the table has columns is truncated — by the oracle as much as by us, which
makes it truncation, not corruption. The shim derives all three from the operand and
checks the one thing required: that the last reachable index `(n_seq-1)*bt_rs +
(bt_cols-1)*bt_cs + 1` is inside the plane. Two conflated counts are named apart:
`cache_blocks` bounds a slot's byte offset, `bt_cols` bounds the walk.

**Serve the unoptimised form when the optimisation has no room.** Split-KV's halving
loop said "down to 1, which needs none" in its comment and returned `RAD_E_SCRATCH` at
1. One slice is the whole walk in one block, the same keys in the same order: not a
degraded answer, the answer without the split. Refusing a band because an optimisation
has no arena tells the user to go away.

**A refusal carries a code, names its own fix, and prints the descriptor it saw.**
`R3D_NO(code)` prints the site under `RAD_R3D_TRACE=1`; `R3D_STR` prints the two
numbers a stride test compared; `r3d_trace_desc` prints dtype, rank, shape and strides
— in **elements**, what the engine binds and what a reader reproducing a refusal
needs.

**The ragged query batch is a descriptor, not a second kernel.** `cu_seqlens`, when
present, is `[n_seq + 1]` of i32, and `cu[0] == 0`, monotonicity, and `cu[n_seq] ==`
the plane's row count are the *description's* promises — the served engine keeps them,
and no row may rely on them. `seq_of` walks `while (s + 1 < n && cu[s+1] <= row) ++s`,
so rows below `cu[0]` are not excluded: they land in sequence 0 with a negative index
into its query, and the reference computes them at `pos = ctx - qlen + i`, pushed
`cu[0]` keys into the past. Nothing in the oracle says the plane starts where the
descriptor starts. So `q_len` is *per-sequence* (`q_len = cus ? 0 : rows / n_seq`), the
total row count is the plane's, and a ragged batch has no `q_len` at all. One grid
covers it because the clamp a tile needs is per sequence; the rectangular case is the
descriptor's absence (`len_s = q_len`, `start_s = s * q_len`). One mapping, one mask,
one walk, one store, both forms.

**A descriptor carries two row counts, and the store is bounded by the second.**
`len = cu[s+1] - cu[s]` is what the positions count from; `span = cu[s+1] - start`,
with `start = (s == 0 && cu[0] > 0) ? 0 : cu[s]`, is how many rows the sequence's tiles
carry, because the grid covers the plane from 0. The tile count comes from the length;
the store bound comes from the span. Bounding the store by the length drops the last
`cu[0]` rows of an in-bounds plane, silently, with status 0 from both rows: 20 rows,
`cu = [3, 17]`, one sequence, ctx 24 left 6144 items — 3 rows × 2048 — at the `0x7fc0`
pre-fill while the 17 rows it wrote agreed to bf16 rounding (max abs 1.22e-04), and the
engine read its own `out` buffer back as attention output. The plane's `rel_l2` is
`nan` in that case only because the untouched slots are bf16 NaN, which is exactly why
a finite-output check passes it and a whole-plane read-back does not. Every other
served shape agreed — 233 cases — which is what makes `cu[0] > 0` the descriptor's own
law and not a geometry note: the reference reads the descriptor, the fixture never
draws one, and the ragged oracle builds its prefix sum from 0, so the shape
that broke the agreement is a shape no oracle upstream could generate.

## Numerics

f32 accumulation, IEEE round-to-nearest-even narrowing, subnormals kept, no
reassociation; the oracle is `libref`.

**`-ffp-contract=off` on every HIP source** (`rad_hipify`, library-wide): a fused
product-sum rounds once where the reference rounds twice, and a kernel diverging by a
rounding the reference does not make will not match it; the flag also keeps the
instruction counts quoted in kernel comments true.

**Accumulation order is part of the contract:** `v_dot2acc_f32_f16` folds two products
into the accumulator and rounds once, so the match is not bit-exact and cannot be —
under an ulp a pair, a random walk of ~`sqrt(K/2)` ulps over a K-long dot, a
thousandth of the bf16 a store narrows to and inside what `rad-kbench` measures. The
*order* must be the reference's: k ascending, one accumulator per output, no
reassociation across lanes, with lane maps built so no lane ever has to break it (a
group spanning two lanes folds in one `shfl_xor`, in integers, before any float work).

**Reductions are the reference's tree:** a wave butterfly (`dpp_xor` at 16, 8, 4, 2,
1; the halves of a 64-lane row read through `permlanex16` first where the reference
sums a whole row), then wave partials in wave order — the same bracketing at any
thread count. Activations, rope and quantisation are the reference's formulas in its
order (`1/(1+exp(-x))`, `0.5*x*(1+erf(x*0.70710678118654752))`, amax/127 with ties to
even), from `libref/ref_common.h`.

**The accepted divergences, and the family reason for each** — where a row is checked
against a law it implements rather than against libref's bytes, each named in the unit
that ships it.

| divergence | why, and what it costs |
| --- | --- |
| an index at or past an operand's row count takes the padding path (zero row, no store); `kv_store` drops a slot past the cache's span | the reference validates index arrays on the host and refuses, and `kv_store` refuses such a slot with `E_SHAPE` before it starts; a kernel cannot, so the bound travels as an argument (`slot_max`). A slot lost, not a step lost. A negative index is padding too and is not clamped to zero, which would duplicate row 0 across every padding row |
| rope divides the position by `scale` and evaluates the frequency ladder in double | `scale` is the position scale (the multiplier on a rotated pair is YaRN's magnitude, and the shims refuse YaRN, so it is 1). `powf` puts the reciprocal a ulp off, and at position 262144 a ulp of a frequency is a hundredth of a radian — the one place in that kernel where f32 is not enough |
| `block_sum`'s tree | wave butterfly, then partials in wave order: the reference's bracketing at any thread count, and why kernel and reference agree at 256 threads and at 64 |
| `sample_logprobs` keeps a NaN out of the max and into the sum; `sample_mask` skips a row whose `mask_row` is negative | maxima are a strict `>`, `std::max`'s law, so a NaN is never adopted; the sum propagates it, because the reference's `exp(NaN - mx)` is NaN and is added. The k-list is per lane in the reference's order (descending by value, ties to the lower column), merged in the same butterfly; the pool is 64 deep and a bigger k is refused, not truncated. The mask bound is the reference's own test (`mask_row` is `-1` for no mask), read off the operand, not off M |
| the sampler's `exp` and `log` are `double` calls narrowed once | the draws must reproduce: the device's `(float)exp((double)x)` is bit-identical to the host's over 2^20 points in [-100, 100] (0 differ) where `expf` differs by 1 ulp in 5.7% of them. Consumer RDNA3 is 1:64 on doubles, so this is the slow path by law: at the band, one read of the logits plane is 3.6 ms, a read that also sums double exponentials 30.4 ms, this kernel 38.9 ms. The candidate set is where it is affordable (1024 exponentials a row); attention is not |
| attention's `exp` is a degree-8 polynomial; the gated MLP's sigmoid is `expf` | no vector f64, and prefill is tens of millions of exponentials a layer. Against `(float)exp((double)x)` over 1.31M points of [-100, 0]: every point within one f32 ulp, 76.1% bit-identical (the device `expf` is 94.0% identical and no better in ulps), `exp(0)` exactly 1 — the only place an explicit `__fmaf_rn` is allowed, because what must be reproduced is a *function*. The MLP feeds a bf16 narrowing whose ulp is 2^-8 against the transcendental's ~1e-7, so the narrowing lands the same bf16 everywhere but on a tie |
| `w2a8` lands with a one-ulp tail | the pair-dot carries ~22 bits on integers: 0.1–0.2% of outputs one bf16 ulp off, `rel_l2` 5.0e-3 at the band, nothing further. The row's prose says so, because the law it transcribes is exact integers and the row reads like a bit-exact one |
| the fp8 folds land in f16, not f32 | the tile is f16, so a scale enters as a f16 multiply at staging — one rounding per staged value, through `v_mul_f16`. Against the reference's exact dequantise law: `rel_l2` 2.77e-04, half an f16 ulp, 250× finer than an E4M3 code's quantum (2^-4); against a model of the fold itself: 4.5e-06, the dot's fused rounding. At a bf16 output the store's ulp (3.9e-3) binds, not the fold's: 83% of outputs bit-exact, 99.93% within fold plus one store ulp |
| the bf16 GEMM's long-K tail | the pair is folded into the accumulator by one rounding, so ~0.3% of a long-K output lands one bf16 store ulp from the reference's and none is further: 99.93% bit-exact at the 512-column twin, 100% within two ulps |

**How a plane is proved to have been written.** Every output plane in the comparison
runs is pre-filled with the dtype's NaN (`0x7fc0` for bf16 and f16) and read back whole: a
kernel returning `RAD_OK` without launching leaves the caller's bytes behind, and the
caller's zeros are finite, so only a NaN fill says the row ran. A clean case prints
`untouched 0`, `nonfinite 0`, no output beyond six ulps of the law the kernel
implements, and a whole-plane `rel_l2` inside what the fusion costs — `2 · sqrt(K/2)`
output ulps, with room for the fixture's spread.

Two named planes (an int8 row-sum padded to a 16-row grid, a quantiser's per-group
statistics) keep the fill in their padding legitimately, because the row writes only
the rows it was given: the report says *pad NaN of N, the reference's grid, not the
row's work*, and a NaN anywhere else is the silent no-op the fill exists to find.

Infinities are contract, not corruption, in the sampler and row top-k: a dropped
candidate, a narrowing stage's drop, and an unfilled `row_topk` slot are all `-inf`.

## Testing

[`r3d_selftest.cpp`](r3d_selftest.cpp) dlopens the `.so` — so an undefined symbol is
its failure, not a link failure — and runs five gates.

**Codecs** against the ABI's host versions in `rad_plugin.h`: bf16 and f16 over all
65536 patterns both directions; e4m3 over all 256 codes plus the encoder's saturating,
subnormal, signed-zero and NaN cases; int8 against `rad_round_even_sat` over every
tie; the widenings over the range they are exact in; the bf16 pair packer against
`rad_f32_to_bf16`. The codecs are this library's own arithmetic, so they belong to the
tests: kbench compares whole ops, where a codec's disagreement is a diffuse
percentage.

**The table:** unique names, a launch on every row, an `init` with a `fini`, the
device domain, prose on every row, a schema for every `op`, every constraint key a
declared parameter, a tunable's default among its ascending values, the layout pair
rule.

**Band reachability:** every constraint a row puts on an optional parameter, named —
not an error, it is how a row says *I serve the form that states the parameter*, but a
decision, and a row whose author did not make it silently will not serve.

**The schemas against `libref.so` beside the plugin**, with the loader's own
comparison (parameter count, keys, types, requiredness; operand count, roles,
optionality): one role mistyped and the loader refuses the whole library, every row
with it, one line in a log.

**Every row, run** on a card — buffers from the row's own `opd_shape`, filled per its
fill code, outputs checked whole, one child process per row so a GPU page fault names
its row and the run continues. No card is not a failure: the static phase is a third
of what it checks, and a build machine runs it without one.

The numerical authority is `rad-kbench`, replaying a recorded fixture against libref:

```
rad-kbench --home "$RADIANCE_HOME" --kernels libref:libr3d
```

`--all-rows` runs every row whose constraints hold, not only the one selection picks —
how a tunable arm the tuned defaults never reach gets measured. `--coverage DIR
--coverage-write` records the rows a fixture reaches, so a row that stops being
reachable fails the run; the census is
[`kbench_rows_kernels_libr3d.txt`](../data/kbench_rows/kbench_rows_kernels_libr3d.txt),
written by the same run, so census and numbers cannot disagree.

**A fixture covers a checkpoint, not the library.** A fixture is generated from the
bands a recorded checkpoint issues, so a row can be shipped, correct, and absent from
its census: the packed 2-bit grid serves the head bands of the checkpoint it is served
on, and the base fixture has no `w2a8` band above M=16. Its gate: clean cases on both
stored arrangements, every refusal refused as stated, the writer's codes, scale and
zero regions byte-identical to the fixture's including a 73,113,920-byte head operand —
plus the fp8a8 row against the same file, 36 of 36 clean.

A row should say which of the two vouches for it.

The rule has a tool: `rad-kbench` records a fixture **from a model**, reading only the
tables, then widens every case into the constraints of every row under test.

```
rad-kbench -m /path/to/served.rad --record "$RADIANCE_HOME/kb-served.rkb"
rad-kbench --fixture "$RADIANCE_HOME/kb-served.rkb" --record-rows "$RADIANCE_HOME/kbrows.rkb" \
           --kernels libr3d,libref
rad-kbench --fixture "$RADIANCE_HOME/kbrows.rkb" --kernels libref:libr3d --all-rows \
           --no-cache --max-cases 2 --max-bytes 1024 --seed 42
```

Line 1 answers the served checkpoint's own bands against the reference on the
container's real weight bytes: **975 checked, 975 passed, 0 diverged, worst `rel_l2`
1.605e-03** — the sentence worth keeping in a log is *every resolved kernel agrees
with libref on this container's bytes*.

Lines 2–3 take each case into the M and `q_len` no container selects, the edges of a
row's constraints where its floors and ceilings get exercised: 45 cases reach 45 rows,
90 checked, 90 passed, 0 diverged, worst 9.684e-04. Rows a checkpoint's
graph never uses report `no recorded case of its op`. **Record one fixture per
checkpoint you serve:** a row green on a 27B fp8 fixture and untested at the shape
your served model walks is covered by a checkpoint, not by you.

Against that base fixture, on a 16-CU RDNA3 part with 8 GiB, built out of tree as
above: **1153 checked, 1153 passed, 0 failed, 24 skipped for the case's own memory
budget, 0 diverged, 0 capped, worst `rel_l2` 1.362e-03** against a tolerance of 8e-03
— the GEMMs' long-K accumulations and the fused norm-quantiser's E4M3 step. 436 cases
ran inside guarded allocations and 0 wrote outside one: the whole-plane write law,
tested on the fixture and in the hand-rolled comparisons.

**Two proofs a tolerance cannot give.** A refactor is proved by the recorded decision
lines **bit-identical**: the same run against the build of HEAD and the build of the new
file, diffed with the timing column blanked (`6143/6144`, `2047/2048`,
`3667126/3670016`,
`4193353/4194304`). A change to an index, a scale plane or a row mapping is proved
**bit-for-bit against the same fixture**, in a mode whose whole job is that pair test:
feeding a per-row scale plane as the first M floats of the `[M][groups]` array
*passed*, because those floats are row 0's slots, so row *m* read row *m/groups*'
scale, and on data drawn from one distribution the answer is a few percent off —
inside a `rel_l2` tolerance, outside the truth. A fixture whose scales are
statistically alike cannot tell a row-indexing error from rounding; the pair test can.

A test that cannot fail is not a test, so the comparison harness carries the control
that says so: a `selfcheck` run hands the scorer a plane rebuilt under a
known misreading and requires a DIVERGENT verdict naming it.

The first full fixture run found three bugs a finite-output check cannot find: a cast
and a gather advancing the row base by the pitch in *bytes* where it is in *elements*;
raw bf16 patterns fed to `v_dot2acc_f32_f16` (0x3F80 is bf16 1.0 and f16 1.875); a
staging loop striding by the compile-time block size, correct at eight waves a block
and silently reading unwritten LDS at one. All three are in rows a fixture selects at
M > 1, none in a row the tuned defaults select.

The gates prove a binary, not a tree. The box builds a *synced* tree, so a file that
never went to the box can reach the repo unmeasured, and one commit did exactly that:
it carried 567 lines of a unit's redesign where the built and measured file is 483,
and the added staged-arm and window paths use a constant — `kWin` — that is declared
nowhere, so the tree at tip did not compile while every gate above kept passing,
because not one of them builds from the commit. The check that catches it costs one
clean export: archive HEAD, build the archive on the target, and `build=0` is the only
answer. Two rules follow. A file enters the tree in the bytes that were measured — the
file an agent measured in its own snapshot is the file that lands, and its md5 is
quoted with its numbers. And unfinished work stays in a scratch directory, never in a
path a commit will sweep: the 567-line file is outside the repo now, with the constant
declared, for whoever finishes and measures it.

## The fp8x8 GEMM by row count

The band this family exists for is 42 layers plus a drafter at `group=128
dtype=fp8a8` with a band ceiling of M=2048: N=12288 K=2048 ×44 (the gated MLP, two
halves of 6144), N=2560 K=2048 ×44 (qkv), N=2048 K=6144 ×44 (o-proj), N=2048 K=2048
×44, and N=2048 K=10240 ×3 (the drafter). Three rows cover it by row count and the
boundary is read off the constraint tables, not off names: `gemm_nt_q_fp8a8_m16`
declares `C_LE("M", 16)`, so `gemm_nt_q_fp8a8_tiled` opens at `C_GE("M", 17)`, and
`gemm_nt_q_gated_fp8a8` runs M=1..64 with N even and K % 128. A floor reasoned from
prose, not from the sibling's *declared* ceiling, left 17..64 unserved and 209
batched-decode bands on the host. The census in [Naming](#naming) is the index; the
measured floors and the placement law that prices them are in
[Measuring](#measuring).

**The m16 row is a wave a run of columns.** One output element a lane, the weight
streamed past an activation kept in registers, no wave reduction — so a lane's chain is
the reference's chain and the row is bit-comparable at the store. Above 16 rows the same
operands want a 128×128 tile through LDS with 8×8 f32 accumulators a lane. The two
ranges are complements by construction, so neither row competes for the other's band,
and the m16 row is run against the tiled row on the same bytes, which is where a
boundary that moved would be seen.

**The tiled row folds the scales at staging, and the fold is the arithmetic.** The tile
is f16, so both scale streams enter as f16 multiplies where the codes are staged: the
activation's `a_scale[m][g]` per staged row, the weight's `b_scale[nb][g]` once a chunk
— TN 128 is the scale plane's row extent, so a tile is one scale-row, and kKC 32 divides
128, so a chunk sits inside one k-block. The products are separable,
`(a*sa)*(b*sb) = (a*b)*sa*sb`, so folding each side is the same sum formed a different
way and the accumulator stays one f32 register. The reference's law is the group sub-sum
accumulated from the raw codes and scaled once per group (`ref_gemm.h`'s `quant_dot`, the
same body the m16 and gated rows carry); the fold answers to it at 2.77e-04 `rel_l2`
against a store that costs 3.9e-3, so it is invisible under the bf16 output: 1.71e-03
overall, 83% of outputs bit-exact, 99.93% within the fold floor plus one store rounding,
nothing beyond eight fold floors and two store ulps. `|y|` is not the denominator — a sum
of 2048 random products is as often near zero as not — the scale is
`S = sqrt(sum (a_k b_k)^2)`, and the fold's measured floor is 9.8e-4 of it.

**At the band's ceiling.** M=2048, container resident, every number on a
placement line: N=12288 K=2048 28.171 ms and 3659 GFLOP/s; N=2560 K=2048 5.911 ms and
3633; N=2048 K=6144 14.699 ms and 3506; N=2048 K=2048 4.381 ms and 3922; N=2048 K=10240
29.798 ms and 2883, the drafter. 2883 to 3922 GFLOP/s is 27 to 36% of a part's dense f16
rate; the bytes floor at the 163 GB/s this class streams is 0.051 to 0.257 ms a band, and
2MNK/(K+N) is ~1900 flops a byte at M=2048. Prefill is compute-bound here, as it should
be, so these numbers describe the loop and not the wire.

**The cliff is recorded with its numbers, because it is not arithmetic.** The tiled
kernels run one block a CU at 18.4 to 18.5 us a block on every column-tile count the
graphs issue (16 and 20 measured, 4.72 and 5.911 ms over the grid); 19 tiles costs 211 us
a block, 11.4×, and 21 tiles 385 us, 20.9× — deterministic to 0.1% in two isolated
processes, clean at both. Every column-tile count the served graphs issue is even,
so no band walks into it; first try the block-index swizzle these cliffs usually answer
to, then the `nt` streaming loads.

**What proved the crossing, and what would not have.** One `RadArgs`, the device row and
libref's host row on the same bytes, the whole `y` plane compared — pre-filled `0x7fc0`
so an unwritten element is visible, every other plane poisoned, 256-byte guards checked
after the launch. In every form the served plan hands the row (rank-1 `a_scale` of `[M]`,
the codes as a rank-3 `[M][ceil(K/128)][128]` plane, the weight as the container the
layout hook writes) at the four served bands and at M = 17, 21, 30, 45, 64, 65, 128, 258:
**832 of 1600 cases computed by both sides, 0 numeric divergences, worst `rel_l2`
1.440e-03** against a 5e-3 tolerance, 86.8% of elements bit-identical. The container form
above is what let the crossing reach the joined spelling at all: in it, 192 cases, 192
agreements, worst 1.362e-03, and the full crossing comes to 1018 cases with 0 numeric
divergences; the same band with `b_scale=[scale,zero]` is 192 cases refused `E_DTYPE`.

**A fixture whose scales are alike cannot tell an indexing error from rounding.** The
paired run spreads its activation by orders of magnitude a group at a time, because with
uniform random data the amax of 128 samples is statistically concentrated, every block's
scale comes out equal to bf16, and a scale-*index* bug — reading group 0's scale for
group 7 — changes nothing. The unit's own selfcheck said so: two of its eight sabotages
scored 0.00. So the crossing is proved by the paired run and not by the selftest, and
the
planted bug is the proof of the instrument: one line in the shim's launch call — the row
pitch handed to the kernel for a rank-1 activation scale becomes `kb`, a row plane read
at a group plane's pitch, checks and geometry untouched. Against the paired run, 300
cases:
**240 divergent, worst `rel_l2` 1.389e+00**, and all 180 cases whose `a_scale` is the
served rank-1 binding diverge (9.669e-01, 43 507 of 43 520 elements differing, max abs
6.75e+01: rows 0 and 1 read real scales, every row from 2 reads the bytes past the plane
and produces zeros) while the 60 rank-2 cases are untouched and agree. In `served`, the
fixed scorer names 8 of 8 as `a_scale[m*kb+g]` — the read law, which is what the sabotage
*is*. The same bug is invisible to the row's own geometry test, which sees a rank-1 scale
of `M` rows at a stride of `groups`: a gate that cannot see a wrong scale is not a gate.

**The gated row couples column *i* to column *n + i*,** so a lane owns both products of
its column: the partner is 192 waves away at the band's n = 6144, no lane-to-lane
exchange reaches it, and staging a whole row through LDS to get it is a round trip of the
[M, N] tensor the row exists to avoid. A lane walks k ascending with two accumulators
over two weight rows, `i` and `n + i` — exactly the traffic the unfused GEMM spent on the
two columns, so the coupling is free — and a block is one group of columns by
`256/group` rows, which closes the group's amax inside the block: a wave max folded over
the group's waves in LDS, one scale store, one code a lane. Product, gate and quantiser
in one launch with no scratch, which is why the row is concurrent
(`rad_kernel_concurrent(i)` is `scratch == nullptr`) and why the bf16 row of the
intermediate is not written, read, or allocated. Both halves are narrowed to bf16 before
the gate because the pair it replaces stored them and re-read them; the activation itself
is not — `gt * sg` is f32 from the narrowed gate and the product with `up` rounds once, at
the store. The law is `ref_quant.cpp:700` and its order is the point.

**The gated band, both arms, best/median of nine launches, codes resident:**

```
M = 1     987.4 / 1154.9  ->   171.3 /  181.6 us     (147 GB/s of codes; the floor for
M = 2    1120.6 / 1196.5  ->   166.2 /  177.1 us      24 MiB at 163 GB/s is 154 us, and
M = 4    1417.3 / 1581.2  ->   221.0 /  229.6 us      the same sixteen loads with no
M = 8    1874.0 / 2318.0  ->   403.4 /  412.3 us      decode and no dot run in 139 us)
M = 16   2949.3 / 3226.1  ->  1055.0 / 1160.6 us
M = 32   5123.4 / 5405.1  ->  1660.6 / 1705.0 us
M = 64   8395.3 / 9051.0  ->  2786.5 / 2877.3 us
```

**The arm is chosen by row count, in the launcher, from the same band key.** M ≤ 8 with
fp8a8 reads the activation and both weight rows straight from memory, 16 bytes at a
time, decodes them out of the block's table and multiplies: no shared memory between a
load and a dot, no barrier in the k-loop, six loads in flight. M ≥ 9, and every fp8a16
case, stages — at those heights the grid is 48 × M/2 blocks and occupancy, not the
window, feeds the stream, and the stage pays for itself. The two numbers that chose the
shape came from the band's own stream (24 MiB of codes, resident, this loop with the
epilogue off): staged through LDS and decoded arithmetically 497 us; read straight and
decoded from the table 187 us; the read alone 139 us. The crossover and its gap live in
the launcher's comment, beside the `M <= 8` it selects on: with the table decoder in
**both** arms, N=12288 K=2048 group=128, codes resident, skinny 402 against staged 754 at
M=8 and skinny 955 against staged 1030 at M=16 — so the crossover is above 16 and
M=9..15 has never been measured. Decode lives at M=1 and a draft's verify at M ≤ 8, which
is what the shipped threshold is chosen for; a prefill at 9..16 rows would take the skinny
arm on the M=16 numbers, for about 7%. The window is a skinny-arm fix only: at M=64 it
buys its loads per lane at 32 registers, which is a fifth of a wave a SIMD given up on the
47 bands at the ceiling, and there it costs more than it saves.

**No relayout on the gated row, and that is a claim.** libref reads the gated
`[N, K]` weight row-major; libr4d stores fp8a8 weights in WMMA fragment order. The two
readings are ~1.35 apart — a relative error, not noise — so this row carries no
relayout and says so in its prose: a row that inherits the fp8 GEMM's `unrelayout`
scores 1.35. The launcher refuses M > 64 (past the bands' ceiling the weight stream is
the wall, and that is a tiled kernel's problem), N odd, a group that is not a power of
two in [32, 256], `group > n` (the reference then defaults the *quantiser's* group to
n and leaves the weight's K block at the stated width: two grids this kernel reads as
one), `K % group != 0` (a ragged K block reads past the scale plane), `sk > 1` (no K
slab closes an epilogue that needs the whole row), and an absent `act`.

**The instrument** is a paired run with two plugins dlopened, both row tables resolved
the way the engine's loader resolves them, one `RadArgs`, and a plane scored against
nine index laws with a hash beside it. It runs every stored form × the bands × the
crossing heights, the served geometry with the rank-1 activation scales the plan binds,
and the collapsed mapped and managed tier arrangements (the mirror layout, the default,
is the served one). Its numbers are the ones quoted above.

## The paged attention, decode and prefill

Two rows, chosen by `q_len`: the decode entry, which takes `splits`, and the prefill
entry. The served plan is n_head 16 over n_head_kv 2 at head_dim 128 — gqa 8 — with
`block_size 16` and a 4096 context, the cache `fp8_e4m3` with its descales attached. The
layout is the reference's: `kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]`,
K then V inside a slot, so a key's row is a 512-byte pair and both halves are 16-byte
aligned at every slot; `block_table[seq, j / block_size]` names the block holding key
j; a negative entry is an unmapped block and hides its key, which the allocator uses to
free a sequence by writing −1, so the kernel keeps that test even though the fixture
never sets it; `seqused[s]` is the sequence's **total** key count, this step's query
rows included, so query row *i* of a `len_s`-row step sits at the absolute position
`seqused[s] - len_s + i`. One kernel serves decode (q_len 1), a speculative verify
(q_len 7) and a prefill chunk (q_len 2048).

**A lane owns one output row and a wave is the GQA heads of one KV head:**
`h = kh * GQA + (lane & (GQA-1))`, `tok = base + (lane >> lg GQA)` — 8 heads over 4 rows
at gqa 8 — so every lane of a wave reads the same cache row at the same moment. Every
cache read is then an LDS broadcast, one `ds_read_b128` serving the wave, and the paged
walk is a per-wave event rather than a per-lane one. 32 rows per cache read is also the
register file's ceiling, not a preference: a second output row a lane needs 2 × 64
VGPRs of widened query plus 2 × 128 of f32 accumulator, 384, over the 256 that four
waves a SIMD allows.

**The walk happens twice, on purpose.** The reference keeps its scores; 32 keys × 128
dims of f32 is 512 KiB a wave against 64 KiB of LDS, so pass two recomputes them. The
recomputation is the same order over the same operands, so it is bit-identical the
second time, and the op costs half again what keeping them would. A masked key is
skipped, never scored to −inf, so the denominator gets exactly the terms the reference
has, and a row with no visible key stores zeros — the reference's `mx == -INFINITY`
case, as in every other kernel here. P·V stays f32: folding the softmax weight into V
reads every weight to 11 bits, a ~1e-4 change that survives the bf16 store on a few
percent of outputs, for a third of the instructions. `exp` is a degree-8 polynomial
because there is no vector f64: over 1.31M points of [-100, 0] every point lands within
one f32 ulp of `(float)exp((double)x)` and 76.1% land bit-identical, where the device's
`expf` is 94.0% bit-identical — neither is the reference, and `exp(0)` is exactly 1,
which is the term the row is normalised by.

**The width of a workgroup is the only lever this op has ever given.** A row slot past
a sequence's length computes a clamped duplicate: the walk, the tile and the store all
run for it. `pick_waves` picks the narrowest workgroup that carries the step's `q_len`
(one wave to 32/GQA rows, two to 64/GQA, four above); prefill keeps four. Medians of 15
per-launch event times, two binaries in one session so the clocks drift together — the
same session's prefill reads 44.9 for both, which is how an earlier session's 40.0 was
explained: 294.7 → 158.4 ms at 256 sequences × 7 rows over 4096 keys, 268.0 → 128.4 ms
with 4 splits, 4.33 → 2.06 ms at 8 sequences over 2048, 4.27 → 3.97 ms at one sequence
over 4096, prefill 44.95 → 44.78. The kernel issues ~15% of the SIMD's slots at the
band and 3.7% at eight sequences, so halving arithmetic on a stalling kernel "should
not" buy 46%, and it did: two waves arriving at a barrier is not twice as cheap as
four, it is four times cheaper. A measurement, not an explanation, which is why the
width is a run-time choice.

**Measured, and not the answer.** `s_setprio` around the consume leg of the walk: |Δ| ≤
0.7% everywhere, and the one −6.4% in a session's first pair is clock ramp — the same
band run second gives +0.26%. `R3D_WAVES(2)` to lift the scalar-register ceiling:
`sgpr_count` 107 and `sgpr_spill_count` ~98 at *both* 4 and 2 waves a SIMD, timings
unmoved — the compute-shader scalar budget is not set by waves/SIMD the way the VGPR one
is. `__noinline__` on the staging leg: spills 98 → 72, a 160-byte private frame, timings
unchanged. A two-stage pipeline: +7% prefill, +7% decode, −5% split-KV.

**Split-KV is a rescale, and the row says which splits are bit-identical.** A slice
accumulates its own `(acc[128], sum, max)` into scratch and the reducer combines before
the single narrowing, rescaling each partial by `exp(m_s - M)`; every term then arrives
as `exp(s - m_s) * exp(m_s - M)` where the reference has `exp(s - M)` — last-bit at f32
(~2 ulp a term, ~1 ulp on the sum), and it vanishes under the bf16 store except on
outputs sitting on a rounding boundary. The row's prose states the contract: `splits = 1`
is bit-identical to the reference, `splits > 1` is not. `splits` is a **maximum**,
resolved per sequence from its own `seqused`: `nslice = ceil(nblk / ceil(nblk / splits))`
with `nblk` the sequence's block count, so a short sequence uses fewer slices and there
is no per-sequence legality check to fail, and the reducer recomputes `nslice` from the
same two numbers, so a partial needs no header. The arena is a bound, not an error: the
launcher lowers `splits` to what the arena holds, down to 1, which needs none.

**The split curve on the served shape** — 64 sequences, 8 KV heads, fp8 cache, 15
launches, ms a launch, at splits 1/2/4/8/16/32/64: at ctx 1024, 8.17 / 5.51 / 9.29 /
1.50 / 2.54 / 0.39 / 0.23; at ctx 4096, 35.71 / 20.92 / 11.41 / 5.79 / 2.67 / 1.44 /
2.60. Each workgroup walks `ceil(nblk / splits)` blocks with `nblk = ceil(jcap/16)`, so
the total keys read are `jcap` at **every** split count: the cost is slices × per-slice
work plus the reduce, it falls to 32 and then rises. The clamp `splits ≤
ceil(jcap/block_size)` is a no-op at the served shape — `ceil(1024/16)` is 64 — and
bites only where splits exceed `nblk`, where a grid built from the requested number
leaves slices with no blocks: 12 of 12 such cases agree, so the extra waves cost, they
do not corrupt. The curve declines the lever with its numbers; the launcher does not
refuse it.

**Two counts bound the walk, and the descriptor's two counts bound the store.**
`jcap = min(max_ctx, cols * block_size)`: the block table is the window the kernel is
allowed to read, so a context longer than the table has columns is truncated — by the
oracle as much as by us, which makes it truncation, not corruption. `cache_blocks`
bounds a slot's byte offset, `bt_cols` bounds the walk, and the shim checks the one
thing required: that the last reachable index `(n_seq-1)*bt_rs + (bt_cols-1)*bt_cs + 1`
is inside the plane. In the tile walk, the descriptor's length gives the tile count and
its span gives the store bound, so a sequence whose `cu[0]` is above 0 stores every row
the grid covers: see the descriptor's two counts in
[The request](#the-request-a-plans-vocabulary-is-not-its-contents).

**A padding row must not write a partial.** A padding row's `row` clamps to a REAL row,
so the split path needs the row-validity test *before* the partial store. Unguarded,
two lanes write the same `(row, h, slice)` — inside the declared arena, so no bounds
check sees it — and their 132-float records interleave, leaving a slot with one lane's
accumulator and the other lane's `(sum, max)`. The reducer then rescales a full slice by
the wrong maximum, the combined sum comes out thousands of times too small, and `1/sum`
hands back a power of two where a logit belongs. Measured at the sliding-window band
(one sequence, 32 heads over 8 KV heads, window 2048, fp8 cache, splits 4): q_len 4 at
ctx 8192 gives `rel_l2` 1.65e-01 with a worst item of +32.0 against a reference
+1.2e-05, and q_len 6 at the same context gives 1.23e-03 where splits 1 gives 1.09e-03.
The mechanism needs `q_len < 32/gqa`, which is why the decode band (q_len 7 at gqa 8
fills a wave, so there are no padding rows) and a run of that geometry never
saw it, and why the failing fixture cases fail identically at three seeds: a schedule,
not a sample. The slots a padding row leaves unwritten are exactly the slots the reduce
does not read; its index runs over real rows only.

**The instrument** is a paired run with libref and libr3d dlopened in one process, both
called with the *same* `RadArgs`, absent optional operands passed as a descriptor with
null `data` and never as a shorter `n_t`, with the fp8 cache form in the same run: 233
served-shape cases with no divergence, the rectangular matrix, the deep contexts
512..4096, the narrow block tables (`bt cols` 1/2/3/4/8, so `jcap` 16/32/48/64/128
under a ctx of 512), empty sequences (`cu = [0,0,5,5,17]`), split counts 1 and 4,
windows 1/3/17/64/512, causal and non-causal. It is where the descriptor law above
was found. On the fixture the row scores 87 of 87 with worst `rel_l2` 7.558e-05, the
same numbers in the base build and the work build — the pair a change has to keep.

**The unit's own launch counts itself.** A served decode step is *played* — the recorded
pass re-issues queue packets and never calls the plugin's shim — so a print that lives in
the shim has never seen the played steps. The kernel does. Under
`RAD_R3D_TRACE` only, `attn_kernel` writes one record per launch — its launch ordinal,
the row's `ctx` off `seqused`, the whole block-id row of the table, the six operand
addresses, and FNV-1a folds of the query row and of the first and last block it walks —
into a plugin-owned `__device__` ring of 256 fixed 216-byte slots, and the shim drains
the ring to stderr on a traced call. Budget is a device global the shim writes; the
decrement is `atomicSub`, the counter `atomicAdd`, and with the flag unset the kernel
pays one scalar load and a not-taken branch on one thread. The record's emission is one
step late for played steps, and the record's own `L` and `ctx` restore the order.

**The served channel is a buffer, not a global.** In a served process the plugin is
dlopened, and the runtime's symbol stub for a plugin-owned `__device__` global is not the
address the kernel's module writes: the shim's arm and the kernel's read are two
different memories, so the ring's budget cannot be armed and the ring drains zero. The
served channel is therefore a plain `hipMalloc`'d record slot passed as a kernel
argument — no symbol table between writer and reader, one record per launch, drained at
the launch, armed by the same flag. The ring stays the board's channel, where the two
memories are one.

**The flag's price, stated.** A launch function serves the step and is not a place to
allocate or to synchronize. With the flag *unset* both hold: the attention launch pays
one cached flag read and nothing else. With it *set*, the traced call pays one device
synchronize (the descriptor probe reads the table on the host) and the launch one stream
synchronize (the record drain), and the first traced call allocates the record slot.
Those are the instruments working, and they are the reason the flag is a tracing
switch, not a mode.

**The law that made it a ring:** a kernel that calls device `printf` takes an extra
implicit argument — `hidden_hostcall_buffer` — and the AQL backend does not provide it,
so such a kernel is *undispatchable by the engine* (the fixture refuses the row with
`refused a geometry its constraints accept: device error`) while the same kernel,
hipLaunched by a hand-rolled board, prints every line. A board is therefore blind to
dispatch-side problems of its own kernels: only a gate that dispatches through the
engine can see them, and for this unit that gate is the recorded fixture.

## The sampler spine

Thirteen rows: five pre-filters, `sample_topk` feeding `sample_pick`, and the narrowing
stages that run between them. `abi/rad_sample.h` is the wire format and it is the point
of the design: every scalar — `temp`, `top_k`, `top_p`, `min_p`, `typical_p`, the
`xtc_*` pair, `seed`, `pos`, the flags, the history window — is **data**, one 96-byte
row per sampled position, because a parameter is geometry and geometry is frozen at
declare, and two rows of one batch with different `top_p` must resolve to one kernel.
`param_rows` is that check, and it is the dtype check libref states: the plane is
32-bit words, and a 16-bit view passes a byte-count test and then misreads every row.

**The draws must reproduce, so the summation order is transcription and not
performance.** Every max here is a strict `>`, and that is also how NaN is kept out:
libref's `if (v > mx) mx = v` over −infinity never adopts a NaN, and the butterfly
merges with the same strict comparison, so a NaN in a row of logits cannot become the
max anywhere. A sum of f32 is not order-free: the reference's sums run over the
candidates in **index order** and its CDF scan accumulates in index order, so the exps
are computed in parallel — each a separate value, order-free — and the accumulation is
serial on one lane, in index order, exactly as written. Parallel values, serial sums:
the whole design of the candidate kernels in one line.

**The draw is a function of the ABI, not of a library, and the transcendentals are the
reference's.** splitmix64 over `seed * 0x9E3779B97F4A7C15 + pos`, the top 24 bits over
2^24, integer arithmetic so it is the same number on the device as on the host,
independent streams per purpose (the pick XORs `RAD_SAMPLE_KEY_PICK`, XTC's coin
`RAD_SAMPLE_KEY_XTC`). `ref_exp`/`ref_log` are `(float)std::exp((double)x)` and
`(float)std::log((double)x)`, transcribed as double calls rounded once to f32: the
device's are bit-identical to the host's over 2^20 points in [-100, 100] — 0 differ —
where `expf` differs from the host's by 1 ulp in 5.7% of them, and a `u < cum`
comparison sitting on an ulp boundary changes the drawn token, which is a fixture
failure even when the distribution is right. This is where the 1:64 fp64 rate is
affordable: the widest softmax here is the candidate set, 1024 exponentials a row at
half a millisecond a step, so the switch stays off and attention, at 17M exponentials a
layer, ships the polynomial instead.

**The top-R is one scan because the reference's R scans are a sort.** `topk_next` takes,
on each of R scans, the largest value strictly below the last pick, ties to the LOWER
column; read as a whole that is a sort by (−value, +column), and the rule
`v == *pv && i <= *pi` is exactly "the next one within a run of equal values". So one
pass keeping the best R by that order produces the same R pairs, in the same order, with
the same tie-breaks. At the band's row — M=1792, N=130560, R=16, bf16 — the reference
scans the row sixteen times, 7.0 GB; this scans it once, 468 MB. Measured 22.0 ms bf16
and 22.4 ms f32, 21 GB/s and 42 GB/s: the same time for half the bytes, which is the
finding. The pass is bound by the accepted inserts — ~55 sifts a lane at 510 elements a
lane — not by the plane, and a lane filtering against a block-wide threshold instead of
its own tail would take ~5 sifts a lane and land near the 2 ms the bandwidth floor
allows; 22 ms is not the memory system's answer, and it is written here so nobody reads
it as one. One pass needs a per-lane list, so the list is 64 deep and a merge keeps the
best 64 of two: a lane's test against its 64th best is one comparison for almost every
element (~12% of a row clears it at 510 elements a lane), and the deep comparisons
happen only for elements that enter. Sixty-four covers R=16, the selftest's 64-wide set,
and every `top_k` a request asks for in practice; past it `r3d_sample_topk` and
`r3d_row_topk` take `ceil(k/64)` passes — the reference's own chain, so the answer is
the same at any k and only the time is the reference's.

**The candidate plane is the contract, its width is the BUFFER's, and it carries
LOGITS.** A row's set is its `top_k` largest logits, descending, ties to the lower
token id, with `cand_idx` −1 and `cand_val` −infinity for a hole. The width is the
declare-time bound — 1024 candidates is the engine's declared buffer width, and two f32
arrays of it is 8 KiB of the 64 KiB a workgroup gets — and the row's `top_k` only
narrows *within* it: `top_k` 40 in a 1024-wide buffer is 40 candidates and 984 holes,
and the whole buffer is written every step because a hole is a value the later stages
test for. A greedy row takes none: its token is `sample_argmax`'s and the pick must not
overwrite it. `cand_val` carries LOGITS, not probabilities — every narrowing stage takes
its own softmax over the set, which is what lets the stages compose in any order and is
why a pre-filter can be moved rather than re-derived.

**The band key is a ceiling and the plane is the step, and reading one as the other
killed every logprobs request.** The step's rows are the operand's row count. Preferring
the key's `M` over the plane asked one row of candidates to be 256 of them — `refused op
sample_topp at M=1 [band M<=256] geometry {n_cand=1024 M=256}` — so every request that
asked for logprobs died with `shape this kernel does not serve`. It was the third unit
in one night to read a band's bound as the step's value, after the tiled GEMM at M=30
and the block table sized to the cache, which is the reason the rule is stated as a law
and not a fix. The key keeps its job as the ceiling the plan promised not to exceed, and
the refusal that fires on a plane over it prints the descriptor it saw.

**The spine's instrument is the fixture, not a paired run.** No comparison harness pairs
these rows with libref: the selftest builds each row's buffers from its own `opd_shape`,
fills per the fill code, runs the row, and checks the plane whole in a child process,
and kbench
drives the `sample_*` bands against libref on the served checkpoint's fixture (975 of
975 across every op the run resolves, worst 1.605e-03, in the
[Testing](#testing) tally). What a paired run would add is the part the fixture cannot
check: the seeded draw compared against a *recording*, over the parameter rows a request
actually sends. That gap is in
[What is not here yet](#what-is-not-here-yet).

## Measuring

**Measure the floor first and name it in the verdict.** 66.8 MB of codes at a part's
22 GB/s is a 3.0 ms floor; 255 MiB at 163 GB/s is 1.64 ms; a 73.1 MB operand is a
0.449 ms read. Against those, "13.6 ms a token" and "24.151 ms at M=64" are statements
about the loop, not the wire. A number with no floor beside it is a feeling.

**Name the placement of every operand on every timing line.** On a part whose memory
cannot hold the head this is worth an order of magnitude, and it is a property of the
class: at M=1, one kernel, the same head in VRAM is 13.6 ms a token; on the mapped
tier (pinned host bytes read over the host link by 8-byte strided loads) 643 ms;
managed and left alone 2071 ms at 122 MB/s, because managed pages do not migrate for a
read-mostly device access and the kernel faults page by page. A layout hook makes the
mapped placement *legal*, not fast. Boards state which placement produced which
number, time after a prefetch, and note that a container operand must be host-readable
at all (the shim reads its header on the host) — hence pinned or managed, never plain
`malloc`, which faults.

A number belongs to whoever placed the bytes: the same shape in the same binary
measured 5.911 ms as the first case of a process and 85.7 ms as the second case of a
sequence that had allocated and freed a 24 MiB managed container first — 14.5×,
deterministic, clean run both ways, and at 0.2 GB/s the slow run is reading its
weight over the host link. The row's number is 5.911 ms, because the engine ships
weights to VRAM at load and reads them 44 times.

**`vgpr_spill_count` is a gate, not a note:** 44 kernels, all zero, both builds of an
A/B. Per-group partials scaled into the accumulator cost 64 more accumulators on an
arm already at 237 VGPRs, and a runtime index into a register array of candidates
spills the array to private memory; both are rewrites, not tunables. A scalar spill in
a compute-shader wavefront is a slow path, not a stall — the ablations below say so
with numbers — so VGPR spills gate and SGPR spills get reported.

**Number before verdict.** Four things that look like they should work on the
attention unit were measured to not work, and the one width that did move it is
recorded with them
[in the attention section](#the-paged-attention-decode-and-prefill); the tiled rows'
grid-shape cliff is recorded
[in the GEMM section](#the-fp8x8-gemm-by-row-count). A negative result carrying its
numbers is the cheapest documentation in a library: it is the only thing that stops
the next session re-measuring it.

**Profiled per-op times sum to more than the token, and that is the tool working.**
On the served 2B plan, 61 + 24 + 12.7 + 4.2 = 102 ms of per-op device time against a
measured 87.6 ms token: the engine overlaps its ops, and `--profile-ops` restores the
synchronisations that serialise them — the caveat the served engine prints in its own
startup diagnostics ([`core/diag/checks.cpp`](../core/diag/checks.cpp): *times every op,
which restores synchronisations … drop `--profile-ops` for anything measured*). A per-op
table taken under that flag ranks the work and does not price it, and the token rate is
the price. The pair that proves the reading: the gated row's profiled saving on that plan
was ~44 ms of the 87.6 ms token, and the token got 11.4 ms faster. Quote a served run's
shape with its served numbers — the context length it was taken at, the prompt lengths it
walked — because a ranking is a property of the shape, not of the model.

## Notes on the kernels

Three properties are assumed throughout and not restated per kernel: **wave32 with a
1024-entry VGPR file a SIMD**, so every lane map has a lane owning whole dwords and
every budget is computed against 1024; **64 KiB of LDS a workgroup**, which sets the
candidate kernels' `n_cand <= 1024` and the tile sizes; **the dot2 path** (the
wave-matrix fragment layout is a re-tile, not a drop-in for these lane maps), which is
why the tile is a dot loop over LDS with an 8-element XOR swizzle, shared by every
tiled row in [`r3d_tiling.h`](r3d_tiling.h) and proven byte-identical between them.

**A staged chunk of 8 bytes against a row pitch of 512 is the 2-bit grid moving 8× the
bytes it uses** at `kKC` 32: 24.151 ms at M=64, N=130560, K=2048 against a 0.449 ms
read floor, 3.7 GB/s useful. The lever is a wider k-chunk (64 codes = 16 bytes, paid
for in LDS), not more loads in flight: leading the codes measured 0.6%, bit-identical,
and was reverted.

**The head is the largest read in a decode step**, and the sampler reads f32 — why the
store is typed, not narrowed, and why a prefill band writes 1.07 GB. bf16 codes: 3.29
ms a token at M=1 (163 GB/s), 0.507 ms at M=16 (the head read once for sixteen rows),
269 ms at 2048 rows (4067 GFLOP/s). E4M3 codes with a bf16 scale per 128×128 block,
read as the two container planes they are: 13.6 ms a token at M=1, 8× above the 1.64
ms floor, cause named — the E4M3 decoder, ~100 instructions an eight-code group. The
decoder-as-a-table pass (256-entry f16 table in LDS, 512 bytes, eight reads a group,
exact by construction) is 2.70× where the decoder binds (M=8: 2.500 → 0.924 ms a
token), flat at prefill, and a **named non-result at M=1**: removing ~100 ALU
instructions did not move it. The suspect at M=1 is load width — eight codes a lane
through an 8-byte load, where the bf16 arm takes sixteen bytes a lane at 163 GB/s — so
the next experiment is a 16-byte code load, not more ALU. `rel_l2` between the two
binaries is bit-identical: the check that the swap moved work, not arithmetic.

**A launch check that does not clear the flag first refuses a healthy step.**
`hipGetLastError()` reports the first error raised *since it was last cleared*, so a
check that does not clear attributes to this launch whatever the process raised before
it: in a served process, the op that ran last; in a comparison run, its own setup. Both
were
seen. The symptom to recognise is a whole timing mode returning the device code on
every geometry of every row, all of it vanishing under `HIP_LAUNCH_BLOCKING=1` — which
is what a synchronous launch does to a pending asynchronous error, it clears it before
the check — and a shim's own text rendering it as something about strides, because the
launcher printed a raw code and the caller appended its own sentence. Two lines fix
it: clear the flag after the arguments are staged and before the chevron, then read it
straight back; and on a refusal, name the error and print the grid, the block, the
waves, the splits and the width the launcher chose. A refusal that does not say what
it saw cost two hours, and the failure mode is a healthy step refused for a symptom a
neighbour caused — the launch is not the suspect.

**No collective rows, and no chained forms.** The parts this family ships in are
single-GPU, and an all-reduce over the host link most of them sit on is slower than
sending the bytes through the host. Nothing here is built as a chained form, so both
optional symbols are left out — the ABI's own answer, with one less thing to keep in
step.

## What is not here yet

Stated as library work, so nobody looks for it in a row table.

**The 2-bit grid's staging width.** The packed-code tile at prefill heights moves 8×
the bytes it uses at `kKC` 32; the fix is a 64-code chunk (16 bytes a lane) paid for
in LDS — the whole distance between 24.151 ms and a 0.449 ms floor at M=64, N=130560,
K=2048.

**The skinny-load path for a one-row head.** The E4M3 decode arm loads eight codes a
lane through an 8-byte load and runs 8× above its floor at M=1; the 16-byte load (16
codes, eight pairs, half the wavefronts) is the experiment, and the decoder table has
been shown not to be the lever.

**The gated MLP's rows-by-column-groups pass.** The table decoder moved M=64 from
8395.3 to 2786.5 us (best of nine), which is most of the row's story at the ceiling
and none of its floor: the weight read is the wall there, and 47 of the 48 bands
declare the ceiling. Past 64 the fused row refuses and the work is four ops over a
bf16 `[M, N]` tensor — two GEMMs, the gate, the quantiser — so the pass that
amortises the weight read across rows *and* column groups is what stands between the
fused form and the pair it replaces, priced at M=64.

**The gated row's arm threshold is a tunable.** The crossover between the skinny and
staged arms is above 16 rows and M=9..15 has never been measured; the shipped M ≤ 8 is
chosen for the decode's M=1 and a draft's verify, and the launcher's comment says so
beside the number it selects on. Nine launches a band would move the threshold, not the
kernel, and a band that lives in 9..15 is paying the staged arm for about 7%.

**`gemm_nt_q_bias` has no row, and the gap is silent.** `gemm_nt_q` has no bias
operand: bias is its own op in the registry (`ref_registry.cpp:1361`, "gemm_nt_q with
a per-column bias"), and this library declares no row for it. So every band carrying a
bias resolves to the reference's host row — correct, slow, and invisible: no `E` line
in the coverage, because nothing refused, and no entry in the profile, because the op
never reaches a device. A model with bias'd projections runs them on the host while
the census reports a clean tree. The row is a bias load in the epilogue of the tiled
rows plus a declared schema; the gap is that nobody has to notice it.

It is a *model-shaped* gap, not a served one, and the container says so: the checkpoint
measured here declares `draft.attention_bias` and `draft.mlp_bias` with **count 0** —
no planes, no bytes — so by the law that an optional operand with no bytes is absent,
this model carries no bias band and nothing falls. `rad-info -v <model> | grep -i bias`
is the one-line census: a model whose bias tensors have planes is the model this row is
for, and on that model the fall is real and silent.

**The sliding window over a partially-filled last tile diverges.** The shape is
`window` between 2 and 16, exclusive of 1 and 16, over a query plane whose last tile is
not full: 10 of 60 cases in the row's window matrix, base and work producing
byte-identical planes for all 60 — 10 divergent each, same hashes — so it is in the
shipped row, not in a session's change. The scan: rows 17 at ctx 33 agree at w = 1 and 16 and diverge at w = 2, 3 and
15; ctx 32 at w = 3 agrees where ctx 34 diverges; rows 32 at ctx 48 — two full tiles —
agree where rows 33 at ctx 49 diverge, tile 3 holding one row; rows 18..24 at ctx + 16
diverge by two rows each. Five runs of rows 17 at ctx 33, w = 3, give the identical plane
hash (`223c70af1e9e5276`): one row of 17, 2044 of 2048 items differing, `rel_l2`
1.196e+00, status 0 from both rows, no non-finite. Served bands are full attention — all
471 `window = 0` cases agree — and the plan's windowed layers use a 2048 window over
16-wide blocks, so no served band walks into it. It is a bounded, named, reproducible
divergence in a row the plan declares, and the fix belongs in the last tile's mask, not in
the window's clamp. A dedicated comparison run reproduces it.

**`sample_merge_topk`**, the cross-rank candidate merge: under a split vocabulary it
holds `world_size * n_cand` gathered pairs, and there is no row for it here.

**A seeded-draw comparison for the spine.** kbench grades the `sample_*` bands against
the reference on the fixture, which covers the arithmetic and not the recording: no run
compares the *drawn token* against a captured sequence, over the parameter rows a
request actually sends. The draws are the part of this library a `rel_l2` cannot grade.

**int8 × int8.** The constraint table is written (`i8a8`, group 128, M ≤ 16, K % 128)
and `quant_act_i8g` already produces the row-major activation it wants; the kernel is
not landed, so the row is not in the table, because a row naming an entry point that
is not in the library is an undefined symbol at `dlopen` and takes every other row
with it.

**Chunked logits.** The head's f32 plane is written whole (1.07 GB at a 2048-row
prefill band); nothing yet emits it in chunks the sampler consumes in passes.

**A sub-byte zero plane.** `w2a8` reads a zero point a byte at a time and refuses a
zero stored U2/U4/U5 by name; the bit-extract path is the ten lines `rad_load_bits`
already is for the codes. Written for the codes and not for the zero, and the refusal
is not an approximation.

**Speculation that pays.** The draft head is served end to end — its bands by the
2-bit grid, its accept step by `dflash_select` — and the accepted text is identical.
The target's head read amortises over the draft block (5× cheaper per token at 8 rows
than at 1 on the same kernel), so what has to pay is the drafter's own steps, and on
the 16-CU part that measured it they did not: 87.6 → 94.4 ms/token on a 5-token
prompt, 91.6 → 187.1 on a 16-token chat band with 7 drafts. The engine's own rule
switches the drafter off for a plugin whose rows measured a gain without them, so the
work left is a draft path that costs less than the head read it saves, not a flag.
