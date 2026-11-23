# Wan 2.2 TI2V-5B

The `wan` architecture serves text-to-video and first-frame image-to-video through `/v1/videos`. It produces MP4 video at 24 fps. The decoder emits one frame from its first chunk and four from each subsequent chunk, giving `4 × chunks − 3` frames. This implementation allows up to 31 chunks and an output area of 1280 × 704 pixels, with sides divisible by 32 and aspect ratio up to 4. These declaration bounds still need qualification for each memory profile.

The default output is 512 × 288 with 49 frames. Generation uses the shifted flow UniPC BH2 solver, shift 5, 50 model evaluations and CFG scale 5. The API accepts `num_inference_steps`, `guidance_scale`, `negative_prompt`, and `seed`.

## Conversion

Convert the pinned Diffusers directory with the bundled recipe. It keeps the timestep projections, modulation tables, normalization gains and velocity head in FP32, and stores the UMT5 encoder, other transformer weights and VAE in BF16.

```sh
rad-convert /models/Wan2.2-TI2V-5B-Diffusers \
  --recipe data/recipes/wan-2.2-ti2v-5b-bf16.recipe \
  -o /models/wan-2.2-ti2v-5b-bf16.rad
```

The source checkpoint and inference implementation are pinned independently. [Checkpoint](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/tree/b8fff7315c768468a5333511427288870b2e9635) · [Reference implementation](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff)

## Serving

Give the generator a batch budget of at least 32768 rows to cover its largest declared pass.

```yaml
models:
  wan:
    model: /models/wan-2.2-ti2v-5b-bf16.rad
    tp: 1
    max-num-batched-tokens: 32768
```

Create a video with the normal asynchronous API, poll the returned job ID, then fetch `/v1/videos/JOB_ID/content`. An `input_reference` image selects first-frame generation. The first frame is encoded by the VAE and its latent is restored after every sampler update.

```json
{"model":"wan","prompt":"A red ceramic mug on a table, camera slowly moving closer.","size":"512x320","seconds":2,"num_inference_steps":50,"guidance_scale":5,"sampler":"unipc","seed":42}
```

The checkpoint retains its own license and attribution. Runtime and model licensing must be reviewed separately.
