# Porting a unit from libr4d to libhalo (gfx1151)

`libhalo` is radiance's kernel library for Strix Halo (gfx1151, RDNA 3.5, the Radeon 8060S in the
Ryzen AI Max+ 395): HIP kernels behind the plain C ABI `halo.h`, plus the plugin rows the engine
selects from (`halo_rows.cpp`), loaded as `radiance_home/kernels/libhalo.so`.

`libr4d/` in the same tree is the **shipped** library for gfx1201 (RDNA4, R9700) and is the source
every unit is ported *from*. Shared by design and never to be changed by a port: op schemas, row
names, parameter keys, layout tags and the stored weight bytes (`libhalo/README.md` §"What is
shared"). Everything measured is libhalo's own and must be re-measured here: priorities,
constraint bands, tunable defaults, launch geometry, kernel bodies. The shape of the part drives
most of it: ~256 GB/s of LPDDR5X shared with the CPU, 40 CUs, wave32, unified memory.

Entry points come in two flavours and both must exist before a row can select a unit:
`halo_rad_<row>(const RadArgs*, RadStream)` shims declared in `halo_plugin.h` (the row table takes
their addresses) and the plain C entries of `halo.h` the engine dlsyms. Unimplemented ones are
weak stubs in `halo_stubs.cpp` returning `RAD_E_UNSUPPORTED` (-18); see §5.

## 1. Start a unit: the rename transform (one command)

```sh
# from the repository root.  u = the libr4d unit's name without the r4d_ prefix.
u=attn_paged_gqa8
sed -e 's/libr4d/libhalo/g'   -e 's/r4dchain/halo_r4dchain/g' \
    -e 's/R4D_/HALO_/g'       -e 's/R4d/Halo/g'          -e 's/R4D/Halo/g' \
    -e 's/r4d_/halo_/g'       -e 's/"r4d\.h"/"halo.h"/g' \
    -e 's/__builtin_amdgcn_cvt_pk_fp8_f32/halo_fp8::cvt_pk_fp8_f32/g' \
    libr4d/r4d_$u.hip > libhalo/halo_$u.hip
grep -nE '\br4d\b|R4d|R4D' libhalo/halo_$u.hip   # leftovers must all be inside message strings
```

Order matters: `libr4d` and `r4dchain` first, then `R4D_` before `R4D`, then `r4d_`. A chained
kernel keeps `r4dchain` inside its name (`halo_r4dchain_rmsnorm`): it is the ABI's marker, not a
library prefix -- the AQL backend gives a chain record only to a kernel whose name contains it
(`core/device/aql.cpp`, RadKernelChainFn in `abi/rad_abi.h`), so a chained form renamed to
`halochain_*` would silently run unchained. Quoted
includes (`#include "r4d_dt16.h"`) are renamed by the `r4d_` rule; the bare `"r4d.h"` needs its
own rule, which is why it is listed. The remaining `r4d` spellings are error-message strings; the
ported units keep them (`halo_dequant_w4_bf16.hip` says `"r4d dequant_w4_bf16: ..."`) — just make
sure no *identifier* was missed. Replace the file's leading comment with the two lines every ported
unit carries:

```
/* Derived from libr4d/r4d_<unit>.hip by the libhalo rename transform (<date>),
 * then ported to gfx1151; see libhalo/README.md. */
```

A rename is not a port: the file now compiles for gfx1151 or it does not, and either way it has
not been checked. §2-§4 are the actual work.

## 2. gfx1151 fragment rules: `libhalo/halo_dt16.h` states them

Read the header comment of `libhalo/halo_dt16.h` before touching an attention body; it is the whole
story, summarised here so you know what to look for:

| | a lane's A/B fragment | the k its element `e` holds |
|---|---|---|
| gfx1201 | 8 x 16-bit (4 dwords) | `k = 8*(lane>>4) + e` |
| gfx1151 | 16 x 16-bit (8 dwords) | `k = e` (lanes 16-31 replicate) |

libr4d's attention kernels keep the score matrix transposed (`S^T = K.Q^T`) so the D column — the
query — stays on `lane & 15` on **both** architectures; that is why they port at all. What changes
is the accumulator's ROW: element `e` is key `8*(lane>>4) + e` on gfx12, key `2*e + (lane>>4)` on
gfx1151. Both spread the tile's sixteen keys over the half-pair, so the kernels' existing
`swap16` (`lane ^ 16`) row-max / row-sum reductions keep their meaning: a port changes the *key
index* and the *fragment width*, never the softmax algebra. Per body that means

- `mk()` takes the lane's **eight** dwords (16 values), not four;
- the fragment gather reads **32 bytes** at the un-split k offset (drop libr4d's `+ 8*h` term) and
  masks on key `2*e + (lane>>4)`;
- the P fragment is assembled across the half-pair (`p16w`): a lane holds one parity, keys
  `2*e + (lane>>4)`;
- the store scatters at head dim `2*e + (lane>>4)`, not at a contiguous run.

`halo_wmma.h` has the measured WMMA layouts and `halo::mma`; `halo_fp8_wmma.h` has `e4m3_pk` (an
exact e4m3 widen) and `cvt_pk_fp8_f32`. gfx1151 has **no** fp8 convert instruction
(`v_cvt_pk_fp8_f32`), **no** `v_cvt_pk_bf16_f32` (use `halo::pk_bf16`, software RTNE) and **no**
`global_load_tr_b128` (the shuffle transpose in `halo_common.h` stands in). fp8 KV forms therefore
take `halo_fp8::e4m3_pk`; bf16/f16 forms take `halo::mma` and `DT16<0>` / `DT16<1>`.

## 3. Wrapper units: `#define` + `#include`

A family body is parameterised by macros and reused by small wrapper files, as in libr4d. The
wrapper is the translation unit (it is what `HALO_UNITS` lists and compiles); the body is text:

```c
#define HALO_ATTN_GQA  1
#define HALO_ATTN_H128 1
#include "halo_attn_paged_gqa8.hip"
```

The body builds its entry names by token pasting (`GQ_SYM(halo_rad_attn_prefill_h256_gqa, _fp8)`),
so no text scan can see which symbols it defines. That is exactly why the stubs are weak (§5): a
macro-built definition overrides its stub at link time with nothing to regenerate.

## 4. Done means numerically checked

A unit is done when a numeric harness says so, never because the build passes: with weak stubs a
row whose entry is missing answers -18 and a row whose entry exists but is wrong computes garbage,
and both leave the build green.

- Row smoke/numeric checks: `halo_selftest <plugin.so> --case <row>` (row names are shared with
  libr4d, so the case names are too). Add a case to `libhalo/halo_selftest.cpp` when the row has
  none — an unwritten case is a row nobody has measured.
- A row with no case, and every kernel with a new fragment path: write a float64 reference checker
  modelled on `.../scratch/pad/attn_dense_check.cpp` — `dlopen` the `.so`, `dlsym` the C entry,
  fill small exactly-representable inputs, compute a straightforward `double` reference on the
  host, print the worst relative L2. 16-bit and fp8 paths land in the 1e-3 class; byte-identical
  contracts (see `STRIX_HALOD_RADIANCE.md` §6.3) must compare bytes, not rel_l2. Build the checker
  on the node with
  `/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1151 -std=c++17 -O2 check.cpp -o check`
  and run `./check <plugin.so> 3`.
- Then `rad-kbench --kernels libhalo,libref` for the row set, and time it against the shape's
  roofline on the card (`--perfmxd`, `--perfw8`, `--perfq`, ... in `halo_selftest`).

## 5. Stubs and `HALO_UNITS` are generated now

`python3 libhalo/gen_stubs.py` rewrites `halo_stubs.cpp` and the `HALO_UNITS` list from the tree;
`--check` fails instead of writing (use it before committing). What it derives: the entries from
every `halo_*` declaration in `halo_plugin.h` and `halo.h` plus any line already in the file whose
symbol neither header declares; what is implemented from definitions found in the sources
`CMakeLists.txt` compiles; the unit list from the `halo_*.hip` files present, minus pure bodies
(§3). So a new unit needs no edit to either file — add `libhalo/halo_<unit>.hip` and run the
generator. Never edit `halo_stubs.cpp` or the `HALO_UNITS` list by hand, and never delete a stub
line to "make room": every line is `__attribute__((weak))`, so your definition wins anyway.

## 6. Lane workflow

One lane = one git worktree on its own branch outside the shared tree, rsynced to one node;
builds and selftests run **on the node**, detached, never in the shared tree. The recipe, the node
list (node2; never node1), the per-node configure flags and the merge rules
live in the project notebook note `halo-port-lane-workflow`; the current state of the port and
what is already proven are in `strix-halo-libhalo-task` and `STRIX_HALOD_RADIANCE.md`.

## 7. Gotchas already paid for in shared code

- **`mx_frag16x` takes `exp` and `ref` and folds `exp - ref`**, not the e4m3 table's
  `d = ref - exp`. Passing the table's sign flips every staged weight exponent (~1e4 relative error, see
  `halo_gemm_mxfp4a8_tiled.hip`'s note and `halo_gemm_mxfp4a8_nt_m64.hip`'s `mx_frag16x`).
- **`load_tr_b128` in `halo_common.h` was wrong** until recently: a shuffle's argument is evaluated
  at the SOURCE lane, so the dword it fetched was the wrong one. It is fixed; re-check any kernel
  that uses the transposed load and recompile any probe you copy from an older session.
- **`core/build/` is checked-in SOURCE** — never delete it, never regenerate it, never treat it as
  build output.
- Each unit compiles `-ffp-contract=off` (see `CMakeLists.txt`): the quantised all-reduce's byte
  contracts depend on it. Do not add fast-math or let the port re-associate a reduction.
- Two int8 planes exist and are not interchangeable: `quant_act_i8`'s shuffled plane (the skinny
  w4a8/w8a8 GEMMs) and `quant_act_i8g`'s natural row-major plane (the i8a8 twins, `wpa8i8`).
- fp8 `deq16_e4m3` underflows values near 2⁻¹²⁰; the `HALO_FP8_FIX` compensation applies to the
  **weight only** (both operands would underflow) — `STRIX_HALOD_RADIANCE.md` §6.3.
