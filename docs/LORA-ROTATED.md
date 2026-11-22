# Runtime LoRA on a W8A8 (rotated) trunk

The engine refuses a runtime adapter whose target modules are never multiplied by a GEMM it can
carry, and on a W8A8-rotated trunk (qwen4exp and kin) the residual-writing projections --
`attn_output`, `ssm_out`, the shared expert `down_proj` -- are read only by fused int8 GEMMs over
`fwht128`-rotated codes. Every abliteration adapter targets exactly those three, so none bind.
This is the runtime path that lets them.

It is two small core changes plus a kernel library; the architecture stays as it is.

## 1. The core change

`core/lora/lora.cpp`, twelve lines, two ideas:

- **The site gate accepts a plain fp GEMM over a transform-basis weight.** The veto on
  `w.enc.transform` protects the *quantised* paths: they are fed `had_quant`'s rotated codes, so an
  adapter epilogue there would multiply `A` against a rotated activation. A plain fp `gemm_nt` is
  fed what the architecture hands it -- plain activations -- provided a kernel library presents the
  planes in the plain basis at load. So the veto is narrowed to the quantised GEMMs (`gq || gqb`).
  A plain kernel with no such layout hook is refused at declare by its own layout (`RAD_E_DTYPE`);
  nothing silently reads the wrong basis.
- **A `prefetch` neither carries nor invalidates an adapter.** It only stages a weight's planes for
  the GEMMs that read them; the adapter delta is applied at the site's expand op, so staging the
  base planes stays correct. `prefetch` is the only non-GEMM weight-reading op kind on this arch.

## 2. The kernel library

`plugin/lora_plain/` is a host-only kernel library (`docs/PLUGIN.md` section 1). It registers the
plain fp `gemm_nt` (bf16) with:

- a **layout** hook that accepts the affine int8 codes + per-`[1x128]` bf16 scale + `fwht128`
  transform, exactly the encoding the rotated trunk stores;
- a **relayout** run once at load: `plain_row = FWHT_128(dequant(block))`, which returns `W`
  because `(W*H/128)*H = W` for the unnormalised Hadamard (`H^2 = 128 I`), rounded once to bf16;
- **launch** rows that delegate to libr4d's own exported bf16 entries (`r4d_rad_gemm_bf16_m16`,
  `_m64`, `_tiled`) via `dlopen(RTLD_NOLOAD)`/`dlsym`. No new GEMM arithmetic.

With this loaded, the core sees a plain fp GEMM over a transform-basis weight, accepts it as a
site, and injects `lora_shrink`/`lora_expand` around it -- the existing w4a16 adapter machinery,
reused. Build it out of tree against an installed radiance (section 13), or add
`add_subdirectory(plugin/lora_plain)` beside `spechub` in the top-level `CMakeLists.txt` to build
it in tree.

## 3. Addressing

A runtime adapter is selected by the model name, `"<base>:<adapter>"` (the per-request `loras`
field is image/video only; chat routes by `model`). Strength is fixed per card at load -- not
settable per request in this build. A single resident container serves both the base and the
adapted card. Loaded adapters are children in the fleet view too -- `/admin/fleet` (and so the
built-in dashboard) lists each as id `base:adapter` with `parent` set and no bytes of its own;
an adapter is not a model one parks or unloads, the parent carries it.

## 4. Architecture note

Declaring the three target module kinds as plain fp `gemm_nt` is an *architecture* decision, and
`qwen4exp` is built into the engine, so the worked example reaches it by compiling a shadow copy of
the architecture against patched copies of four SDK arch headers (a `radiance-home` shadow -- no
engine source is touched). That step is arch-specific and is not in this change. If a plugin could
override `declare()` for named module kinds, the whole path would live out of tree with no shadow.

In tree, the shared fp8 blocks (`attn_output`, `ssm_out`, `ffn_down_shexp`) set that plain view
only when a runtime adapter targets the checkpoint source (`rad_lora_source_targeted`). With no
adapter those weights keep `gemm_nt_q` i8a8h, which is what a Flash-Next container starts on.

## 5. Cost and verification

On a 2x R9700 gfx1201 rig against an unmodified `qwen3.8-next-flash-iqr-v4` container, 96 GEMMs
over 96 weights carry adapters at the three kinds, the refusal is gone, and on a hard-refusal
prompt the base refuses while the adapter does not (the delta is behavioural, not cosmetic); math
sanity holds on both. The bf16 detour at the three sites costs an estimated low-single-digit
percent decode and about +2 GiB. Higher ranks may prefer an epilogue-fused int8 requant over the
bf16 detour.
