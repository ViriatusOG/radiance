# GLM kernel library

`glm.so` is `arch/glm5next`'s kernel library, not an architecture plugin. Its GLM-only rows
supplement libr4d; loading it first leaves other models' current kernel selections unchanged
(`plugin/glm/exports.cpp`). Device rows are built for the build's gfx12 targets (tested on gfx1201);
host-only builds carry the SwiGLU oracle.
Like libr4d, it declines a machine whose KFD topology lists cards and none of them gfx12, so its
schema cannot select a fused form there that no device row serves; with no card listed it loads.

| op | row | work |
|---|---|---|
| `swiglu_clamped_had_quant_i8g` | `glm_swiglu_clamped_had_quant_i8g` | clamped SwiGLU, bf16 rounding, fwht128 and int8 quantisation in one pass; groups 128 or 256, optional bf16 product |
| `mla_attn`, `kv_dtype=i8` | `glm_mla_attn_i8kv` | latent 512, 1-64 heads, blocks of 4; matrix form for multiples of 16 heads, wave form otherwise |

`plugin/glm/glm_mla.hip` reads 516-byte cache rows: 512 int8 codes and two f16 scales, one per
256-channel half. Both QK and PV apply those scales; split histories merge through f32 partials.
`plugin/glm/glm.cpp` declares the rows; `plugin/glm/exports.cpp` registers device and reference rows.

## Load

```yaml
server:
  kernels: glm,libr4d,libavx,libref
models:
  glm:
    model: /srv/models/glm53-flash-iqr-v4.rad
    kv-cache-dtype: i8
    num-speculative-tokens: 0
```

Keep this order: the library hierarchy wins before row priority. Libr4d's bf16 MLA rows do not
constrain `kv_dtype`, so putting libr4d first can select a row that rejects the i8 operand.
Bf16 and fp8 MLA selection stays in libr4d. The container recipe is
`data/recipes/glm53-iqr-v4.recipe`; IQ8R MTP and the MLA row form are not served
(`docs/ARCHITECTURES.md`, `arch/glm5next/glm5next.cpp`).

## Measured

Same machine and flags, TP4 on four R9700s. Rates are tok/s; C1-C4 are BetterBench's aggregate
rates at 1-4 concurrent streams, not rates per stream. The separate decode column is the
single-stream decode measurement. These are matched configuration results, not an attribution
to an individual flag or kernel.

| build / latent cache | decode | BetterBench C1 / C2 / C3 / C4 | prefill 2k / 32k / 128k |
|---|---:|---|---|
| 2.x / i8 | 61.8 | 59.9 / 99.0 / 116.2 / 129.3 | 2528 / 2590 / 2511 |
| 2.x / fp8 | 62.0 | 60.0 / 98.7 / 115.0 / 129.9 | 2543 / 2648 / 2593 |
| 1.17 / fp8 | 58.9 | 56.2 / 87.3 / 104.9 / 115.2 | 2262 / 2514 / 2456 |

Teacher-forced comparison against the 1.17 MXFP4 reference:

| 2.x latent cache | mean KL | top-1 agreement | PPL ratio |
|---|---:|---:|---:|
| i8 | 0.149 | 87.14% | 0.985 |
| fp8 | 0.151 | 87.10% | 0.989 |

These measure reference agreement, not task accuracy. The 2.x IQR-V4/IQ8R weights differ from
1.17's MXFP4 weights, so this is not an isolated cache-quantisation comparison. The i8 GPU reader
was within 0.23% of libref in the reader comparison; this does not imply byte identity.

## Tests

- `tests/glm_device_test.cpp`: `the_fused_clamped_swiglu_quantiser_is_the_libr4d_chain_byte_for_byte`
  compares the fused row with libr4d's two-op chain; `the_int8_latent_reader_agrees_with_the_host_reader`
  compares matrix/wave and split/merge cases with libref, including unequal/zero half scales,
  absent pages and blank rows. Requires a card and device plugins.
- `tests/plugin_test.cpp`: `glm_i8_latent_selection_obeys_the_real_plugin_hierarchy` loads the real
  plugins without launching kernels. Covers heads 1/8/16/64, M 1-2048, both library orders,
  bf16/fp8 selection and the fused SwiGLU row; skips when device rows are unavailable.
