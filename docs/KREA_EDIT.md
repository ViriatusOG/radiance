# Krea 2 instruction editing

An edit-enabled Krea Raw container supports text-to-image and instruction editing under one model ID. Convert the full Diffusers pipeline, including the Qwen3-VL vision tower and VAE encoder, with matching executable and plugins:

```sh
rad-convert /models/Krea-2-Raw -o /models/krea-2-raw-edit-iq8r.rad \
  --set pipeline.krea_reference_encoder=1 --recipe data/recipes/krea-2-iq8r.recipe
```

Set `image-gen-max-refs: 2` to budget for two reference images. This setting reserves memory, so leave it at one when a second reference is unnecessary.

Memory: an edit container keeps its conditioning store resident for the whole run, even for plain text-to-image requests. It holds `2 branches x lanes x store_rows x 6144 x 2 B` per rank, where `lanes` is the configured `max-seqs` (at most 8) and `store_rows = 1024 + 576 per extra reference + max_refs x max_image_tokens` (a text-only container uses 512 rows). Image tokens are `gen_max_pixels / 256`. At the full 8 lanes, with the default DiT width of 6144:

| profile | store rows | store size (8 lanes) |
|---|---|---|
| text-only container | 512 | about 96 MiB |
| edit container, 1 reference, 1 MP | 5120 | about 0.94 GiB |
| edit container, 1 reference, 4 MP | 17408 | about 3.2 GiB |
| edit container, 2 references, 4 MP | 34368 | about 6.3 GiB |

Fewer lanes scale the store down proportionally (`max-seqs: 2` is a quarter of these figures). The transient fusion rows also grow from 512 x 12 to 1024..1600 x 12 on an edit container. Lower `gen_max_pixels`, `image-gen-max-refs` or `max-seqs` on smaller cards. The primary text model and image generator can be separate entries in a Radiance fleet.

Send `POST /v1/images/edits` with the image model ID, instruction, size, and one or two ordered references. The JSON form uses `images` as an array of image URLs or data URLs. For multipart requests, use repeated `image` fields. A one-reference request remains valid when the model allows two. Text-only generation continues through `/v1/images/generations`.

```json
{"model":"krea-edit","prompt":"Change the mug to blue, preserving the rest of the scene.","images":["data:image/png;base64,..."],"size":"1024x1024","seed":42,"num_inference_steps":20}
```

Krea fits and centers each reference independently. Use references of the dimensions recommended by your adapter. Identity fidelity with specific adapters requires separate qualification. A two-reference edit should describe each reference's role explicitly. Pose-only transfer with identity preservation remains unqualified.

The optional `ref_boost` multiplies the attention weight of reference tokens. Its default of 1 leaves reference attention unchanged. Masks are rejected for this instruction-edit path. LoRAs use the existing per-request `loras` field. Stacking adapters may require increasing `max-lora-rank` beyond its default.

A seeded result is reproducible only with the same model, adapters, settings, and runtime profile. The same numeric seed in ComfyUI and Radiance can initialize different noise.

## Registered outfill

Use the Registered Outpaint adapter with one reference, an explicit `size: "1024x1024"`, and a scene description. Add `"outfill": {"horizontal": "center", "vertical": "center"}` to the edit request. The source's long edge scales to 1024. Horizontal placement accepts left, center, or right, and vertical placement accepts top, center, or bottom. Defaults are top left. After decoding, the source is blended back over a 32-pixel seam, but only on sides that border generated area: a side on the canvas boundary keeps the source pixels unchanged.

The profile uses the simple Euler schedule with eight steps by default. `guidance_scale: 0` corresponds to ComfyUI CFG 1 for Krea. Masks, multiple references, and `ref_boost` are rejected for this path.

```sh
python3 scripts/krea2_outfill.py input.png output.png \
  --url http://127.0.0.1:8000/v1 --model krea-edit \
  --prompt "A person standing in a sunlit studio"
```

The script defaults to `http://127.0.0.1:8000/v1` and model `krea-edit` and applies no adapters. Name each adapter you want with `--lora NAME[:WEIGHT]` (repeatable); the Registered Outpaint adapter's name is whatever your deployment calls it.
