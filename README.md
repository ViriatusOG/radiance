# radiance

An inference server for AMD GPUs, in C++ and HIP: language models, and image, video, speech and
music generation. Its HTTP API speaks OpenAI's chat, completions, Responses, images, audio and video
APIs, Anthropic's Messages API and llama-server's native one, so their clients and SDKs work
unchanged. Model architectures, kernel libraries and quantisers are `.so` plugins; the ones in this
tree are built and loaded the same way a third party's are ([`docs/PLUGIN.md`](docs/PLUGIN.md)).

The bundled device kernels target AMD RDNA4 (`libr4d`: `gfx1200`, `gfx1201`), RDNA3 (`libr3d`:
`gfx11xx`) and Strix Halo (`libhalo`: `gfx1151`). The serve settings
below are for two 32 GB cards (Radeon AI PRO R9700). The engine starts on any AMD GPU family;
another card needs a kernel library for it. A machine with no GPU builds the host backend, whose
AVX-512/AVX2 kernels (`libavx`) serve every model here: the small ones (EmbeddingGemma 2,
Qwen3-ASR, the 0.8B-3B models) at interactive speed on a desktop CPU, the 27B and larger far more
slowly than a card.

To support the development of radiance: [Patreon](https://patreon.com/StillDeadCode).

## Models

| model | container | get it | fleet file |
|---|---|---|---|
| Qwen3.8-Flash-Next: IQR-V4 experts, int8 trunk, MTP, vision | `qwen3.8-next-flash-iqr-v4.rad` (113 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4) | `tp1.yaml`, `tp2.yaml` |
| GLM-5.3-Flash: IQR-V4 experts, IQ8R trunk, int8 latent cache, four cards with the experts tiered to host memory | `glm53-flash-iqr-v4.rad` (159 GiB) | [download](https://huggingface.co/animusnull/glm-5.3-flash-iqr-v4) | `tp4.yaml` |
| Qwen3.8-Flash-Next abliterated (refusals removed): IQR-V4 trunk (GPTQ), codebook experts (round to nearest), MTP, vision | `qwen3.8-flash-next-uncensored-iqr-v4.rad` (113 GiB) | [download](https://huggingface.co/davetha/qwen3.8-flash-next-uncensored-iqr-v4) | — |
| Qwen3.8-Flash-Next: calibrated Q5 expert down (w5nl, group 64), BF16 PLE, vision | `qwen4exp-w5dn64-bf16ple-gptq.rad` (166 GiB) | [download](https://huggingface.co/gostar77/qwen3.8-flash-next-radiance-q5-w5dn64-bf16ple) | — |
| Qwen3.8-Flash-Next IQ5R-V4: U5 trunk, W5NL expert down, BF16 PLE and vision | `qwen3.8-flash-next-iq5r.rad` (175 GiB) | [download](https://huggingface.co/gostar77/qwen3.8-flash-next-iq5r-v4-u5-w5-bf16ple) | — |
| Qwen3.8-Flash-Next: U4 expert gate/up, W6NL expert down, BF16 PLE and vision | `qwen3.8-flash-next-u4-w6-bf16ple.rad` (170 GiB) | [download](https://huggingface.co/gostar77/qwen3.8-flash-next-u4-w6-bf16ple) | — |
| Qwen3.8-27B IQ8R (rotated int8), DFlash2 drafter, vision | `qwen3.8-27b-iq8r-dflash.rad` (29 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-iq8r-dflash) | `tp2.yaml` |
| Qwen3.8-27B IQR-V4, 2.5 to 8 bits a weight ([docs/IQR.md](docs/IQR.md)), DFlash2 drafter, vision | `qwen3.8-27b-iq2r-v4.rad` .. `qwen3.8-27b-iq8r-v4.rad` (10 to 28 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-iqr-v4) | `tp1.yaml` (IQ4R-L) |
| Hemmingway-1.5 (Qwen3.8-27B fine-tune) IQ4R-S, tool-tail GPTQ; this version serves the trunk, not the MTP block | `hemmingway-1.5-iq4r-s-tool-tail-mtp-gptq.rad` (14 GiB) | [download](https://huggingface.co/gostar77/hemmingway-1.5-iq4r-s-tool-tail-mtp-gptq) | — |
| Qwen3.6-35B-A3B IQ8R, MTP | `qwen3.6-35b-a3b-iq8r.rad` (35 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.6-35b-a3b-iq8r) | — |
| MiniCPM5-2B IQ8R, DSpark drafter | `minicpm5-2b-iq8r.rad` (3 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-iq8r) | `tp1.yaml`, `tp2.yaml` |
| MiniCPM5-2B bf16 (served without its drafter) | `minicpm5-2b-bf16.rad` (5 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-bf16) | — |
| Spark X2.6 4B bf16: thinking and tool calls, one to four cards | `spark-x2.6-4b-radiance-bf16.rad` (8.3 GiB), with DSpark `spark-x2.6-4b-radiance-bf16-dspark.rad` (10 GiB) | [download](https://huggingface.co/gostar77/spark-x2.6-4b-radiance) | — |
| Spark X2.6 4B IQ8R: rotated int8 trunk, optional bf16 head, layers 0-11 kept bf16, optional DSpark | `spark-x2.6-4b-radiance-iq8r-diverse.rad` .. `spark-x2.6-4b-radiance-iq8r-bf16-head-dspark.rad` (4.5 to 6.6 GiB) | [download](https://huggingface.co/gostar77/spark-x2.6-4b-radiance) | — |
| Hy-Memory-30B-A3B bf16 | `hy-memory-30b-a3b-bf16.rad` (56 GiB) | [download](https://huggingface.co/gostar77/hy-memory-30b-a3b-radiance) | — |
| Hy-Memory-30B-A3B W4 GPTQ: attention, dense MLP and routed experts | `hy-memory-30b-a3b-w4-gptq-full-1m.rad` (15 GiB) | [download](https://huggingface.co/gostar77/hy-memory-30b-a3b-radiance) | — |
| Gemma 4 31B: GPTQ int4 trunk, int8 activations, MTP, vision | `gemma4-31b-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-w4a8) | — |
| Gemma 4 31B w5: GPTQ int5 trunk (a scale and a zero point per 128 weights), MTP, vision | `gemma4-31b-w5.rad` (23 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-w5) | — |
| Gemma 4 31B heretic-ara (refusals removed): GPTQ int4 trunk, int8 activations, MTP, vision | `gemma4-31b-heretic-ara-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-heretic-ara-w4a8) | — |
| Gemma 4 31B: int8 trunk and lm_head (no calibration), MTP, vision | `gemma4-31b-i8g.rad` (33 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-i8g) | — |
| Gemma 4 31B heretic-ara (refusals removed): int8 trunk and lm_head (no calibration), MTP, vision | `gemma4-31b-heretic-ara-i8g.rad` (33 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-heretic-ara-i8g) | — |
| Gemma 4 31B Q8: int8 trunk and lm_head (Q8_0's grid at group 128), int8 DFlash drafter, vision | `gemma4-31b-dense-q8-df.rad` (33 GiB) | [convert](#convert-a-checkpoint) | — |
| Gemma 4 12B bf16: MTP, vision, audio | `gemma4-12b-bf16.rad` (25 GiB) | [download](https://huggingface.co/keptn/gemma4-12b-bf16) | — |
| Gemma 4 12B int8 (i8a8): MTP, vision, audio | `gemma4-12b-i8g.rad` (14 GiB) | [download](https://huggingface.co/keptn/gemma4-12b-i8g) | — |
| Qwen-Image 2.1 IQ8R: text to image and image edits | `qwen-image-2.1-iq8r.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-iq8r) | `tp1.yaml`, `tp2.yaml` |
| Qwen-Image 2.1 Turbo IQ8R: the same in 8 steps | `qwen-image-2.1-turbo-iq8r.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-turbo-iq8r) | — |
| Krea 2 Turbo IQ8R: text to image, one card | `krea-2-turbo-iq8r.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/krea-2-turbo-radiance-iq8r) | — |
| Krea 2 Raw IQ8R: the same, undistilled | `krea-2-raw-iq8r.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/krea-2-raw-radiance-iq8r) | — |
| FLUX.3 [klein] IQ8R: text to image and edits from up to ten pictures, one card | `flux-3-klein-iq8r.rad` (9.4 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-flux-3-klein-iq8r) | — |
| FLUX.3 [klein] base IQ8R | `flux-3-klein-base-iq8r.rad` (9.3 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-flux-3-klein-base-iq8r) | — |
| LTX-2.5 distilled IQ8R: video with its stereo soundtrack from text or a first frame, one card | `ltx-2.5-distilled-iq8r.rad` (37 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-ltx-2.5-distilled-iq8r) | — |
| LTX-2.5 base IQ8R | `ltx-2.5-iq8r.rad` (37 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-ltx-2.5-iq8r) | — |
| MiniMax-H3, int8 DiT: video with its stereo soundtrack from text | `minimax-h3-i8.rad` (63 GiB) | [download](https://huggingface.co/StillDeadcode/minimax-h3-i8) | `tp2.yaml` |
| FastH3 8-Step V2, int8 DiT: the same in eight steps | `fasth3-8-step-v2-i8.rad` (65 GiB) | [download](https://huggingface.co/StillDeadcode/fasth3-8-step-v2-i8) | — |
| MiniMax-Music3 bf16: songs from lyrics and a description | `minimax-music3-bf16.rad` (20 GiB) | [download](https://huggingface.co/StillDeadcode/minimax-music3-bf16) | `tp1.yaml`, `tp2.yaml` |
| Unlimited-OCR bf16: document parsing with the DeepEncoder tower | `unlimited-ocr-bf16.rad` (6.2 GiB) | [download](https://huggingface.co/Aijuus/unlimited-ocr-bf16) | — |
| Breeze-TTS-2 bf16: speech, whole, streamed or live, one card | `breeze-tts2-bf16.rad` (5.5 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-breeze-tts2-bf16) | `tp1.yaml`, `tp2.yaml` |
| TranslateGemma 27B, int8 trunk: translation | `translategemma-27b-i8.rad` (28 GiB) | [download](https://huggingface.co/StillDeadcode/translategemma-27b-i8) | — |
| Qwen3.8-27B bf16 | `qwen3.8-27b-bf16.rad` (50 GiB) | [convert](#convert-a-checkpoint) | — |
| Qwen3.8-Flash-Next bf16, a disk-streamed reference | `qwen3.8-flash-next-bf16.rad` (335 GiB) | [convert](#convert-a-checkpoint) | — |
| Stable Diffusion XL bf16 and its finetunes (Illustrious, NoobAI, Pony): text to image, one card | `illustrious-xl-v2.0-bf16.rad` (6.4 GiB), from any single-file SDXL checkpoint | [convert](#convert-a-checkpoint) | — |
| EmbeddingGemma 2 bf16: text, image, video and audio embeddings, one card | `embeddinggemma2-bf16.rad` (1.4 GiB) | [convert](#convert-a-checkpoint) | `tp1.yaml`, `tp2.yaml` |
| nomic-embed-text-v2-moe bf16: text embeddings, one card | `nomic-embed-v2-moe-bf16.rad` (959 MB) | [convert](#convert-a-checkpoint) | — |
| LiquidAI LFM3-1.7B IQ8R: rotated int8 trunk, bf16 head; text, tool calls and reasoning, one card | `lfm3-1.7b-iq8r.rad` (2.5 GiB) | [convert](#convert-a-checkpoint) with `data/recipes/lfm3-1.7b-iq8r.recipe` | — |
| LiquidAI LFM3-1.7B bf16: text, tool calls and reasoning, one card | `lfm3-1.7b-bf16.rad` (3.8 GiB) | [convert](#convert-a-checkpoint) | — |
| Meta Muse Glimmer 30B bf16: text, images and video; sliding-window rotary layers between NoPE full ones, a windowed ViT | `muse-glimmer-30b-bf16.rad` (56 GiB) | [convert](#convert-a-checkpoint) | — |
| Meta Muse Glimmer 30B w4a8: int4 trunk, images and video, one card | `muse-glimmer-30b-w4a8.rad` (16 GiB) | [convert](#convert-a-checkpoint) | — |
| Meta Muse Glimmer 30B IQ8R: rotated int8 trunk, int8 head, DFlash drafter, images and video | `muse-glimmer-30b-iq8r-dflash.rad` (36 GiB) | [download](https://huggingface.co/StillDeadcode/muse-glimmer-30b-iq8r-dflash) | — |
| Meta Muse Glimmer 30B IQ4R-V4: trellis trunk, DFlash2, BF16 vision | `muse-glimmer-30b-iq4r-v4-df2.rad` (24 GiB) | [download](https://huggingface.co/gostar77/muse-glimmer-30b-iq4r-v4-radiance) | — |
| Meta Muse Glimmer 30B IQ4R-S-V4: smaller trellis trunk, DFlash2, BF16 vision | `muse-glimmer-30b-iq4r-s-v4-df2-bf16vision.rad` (19.6 GiB) | [download](https://huggingface.co/gostar77/muse-glimmer-30b-iq4r-s-v4-radiance) | — |
| Meta Muse Glimmer 30B IQ4R-S-V4, int8 vision (needs [#564](https://codeberg.org/StillDeadcode/radiance/pulls/564)) | `muse-glimmer-30b-iq4r-s-v4-df2-i8vision.rad` (17.8 GiB) | [download](https://huggingface.co/gostar77/muse-glimmer-30b-iq4r-s-v4-radiance) | — |
| Recursive Flint (39.8B, about 2.5B active) IQ8R: rotated int8 trunk and experts, two cards | `flint-iq8r.rad` (38 GiB) | [download](https://huggingface.co/StillDeadcode/flint-iq8r) | — |
| Ling-3.0-tiny IQ8R: rotated int8 trunk and experts, one card | `ling-3.0-tiny-iq8r.rad` (7.9 GiB) | [download](https://huggingface.co/StillDeadcode/ling-3.0-tiny-iq8r) | — |
| Nemotron 3.5 Embed 1B BF16: text and image embeddings, one card | the checkpoint as it ships | [NVIDIA checkpoint](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) | — |
| Nemotron 3.5 Embed 8B BF16: text and image embeddings, one card | the checkpoint as it ships | [NVIDIA checkpoint](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) | — |
| Nemotron 3.5 Rerank 8B BF16: text/image document relevance, one card | the checkpoint as it ships | [NVIDIA checkpoint](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) | `tp1.yaml`, `tp2.yaml` |
| Qwen3-ASR 1.7B bf16: speech to text, `/v1/audio/transcriptions` and live over `/v1/realtime`, 30 languages, one card | the checkpoint as it ships (4.1 GiB) | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | `tp1.yaml`, `tp2.yaml` |
| Inkling 2 Mini: NVFP4 experts as stored, MXFP8 experts as rotated int8; text, one card ([geometry](docs/ARCHITECTURES.md#inkling2-inkling-2-mini)) | `inkling2-mini.rad` (17.6 GiB) | [convert](#convert-a-checkpoint) with `data/recipes/inkling2-mini.recipe` | — |

**Not served by this version.** A container whose weights are block FP8, MXFP4, M2XFP4 or
ParoQuant, or whose experts read E4M3 activations, is refused at startup by the name of the first
such weight. That is every FP8 and MXFP4 container of the models above, and the models that have
no int8 container yet: Ornith-1.5 (35B-A3B and 122B), Sage-A1.5,
NVIDIA-Nemotron-3.5-Bolt-74B-A9B, Ling-3.0-flash-Sante, Fish Audio drama-3 and
Anima's FP8 container (Anima converted to bf16 is served). Each returns with an int8 container.
A container with an int8 or bf16 trunk and a block-FP8 drafter (NVIDIA-Nemotron-3.5-Bolt-E4B,
the Gemma 4 31B DFlash and DFlash2 containers, MiniCPM5-2B bf16) serves without the drafter
under `num-speculative-tokens: 0`.

A container holds everything the server needs: the weights, the tokeniser, the chat template and
any drafter. Each download link's repository has the recipe and the exact command that made the
file. Download one into your models directory (the transfer resumes if interrupted):

```sh
mkdir -p /srv/models/rad && cd /srv/models/rad
curl -LC - -O https://huggingface.co/StillDeadcode/minicpm5-2b-iq8r/resolve/main/minicpm5-2b-iq8r.rad
```

The Gemma 4 container is keptn42's: GPTQ calibrated in transformers, with an int8 lm_head; its
repository has the recipe and the calibration step. To convert Google's bf16 checkpoint yourself,
`data/recipes/gemma4-31b-dense-bf16-w4a8.recipe` quantises without calibration (round-to-nearest
with a scale search, bf16 lm_head); see [Convert a checkpoint](#convert-a-checkpoint). On two R9700s
that container decodes at 103 tok/s on prose and 141 on code, against 112 and 150 for keptn42's,
and accepts 0.46 of its drafts on prose against 0.50.

TranslateGemma 27B is converted from Google's bf16 checkpoint with
`data/recipes/translategemma-27b-i8a8.recipe`, an int8 trunk rounded without calibration data.
Its chat template takes structured turns
naming the two languages and refuses the plain turn the server renders at startup, so its entry
serves a plain Gemma 3 template, `deploy/compose/gemma3.jinja` (`override-chat-template` names it;
in a compose file, mount it into the container), and a client sends on `/v1/completions` the prompt
the checkpoint's template renders.

The Qwen-Image containers serve OpenAI's images API (`/v1/images/generations`, `/v1/images/edits`):
pictures from a prompt, and edits of up to ten pictures with an optional mask, 2048x2048 by default.
On two R9700s a 2048x2048 picture takes 61 s at the base model's 40 steps and 13 s with Turbo (56 and
12 s with `tp-wire: wht6`); 1024x1024 takes 6 s and 2.6 s. Qwen-Image 2.1 is under Qwen's research
license, which is non-commercial. Krea 2 is under the Krea 2 Community License and Krea's
acceptable use policy, FLUX.3 [klein] under the FLUX Kommunity License (non-commercial use),
LTX-2.5 under the LTX-2.x Community License, and an SDXL finetune under its checkpoint's own
license.

The MiniMax-H3 containers serve OpenAI's video jobs (`/v1/videos`): a video with its stereo
soundtrack from a prompt, 5 to 14 s, up to 1344x768. On two R9700s a 5.17 s clip at 1344x768 takes
about 68 s with FastH3 and 8.8 minutes at MiniMax-H3's 50 steps; both read about 39 GiB of pinned
host memory. With LBH-123-AI's latent upscaler in the container, `"latent_upscale": 2` runs the first
steps at half the size and refines at the full size: MiniMax-H3's clip then takes 3.4 minutes
(docs/GUIDE.md §5.16). A ComfyUI node, `integrations/comfyui/ComfyUI-Radiance-MiniMax`, renders
through `/v1/videos` from a ComfyUI graph as a client of the server. MiniMax-H3's license, which
FastH3 carries, does not cover the EU, the UK, the Republic of Korea or the USA.

MiniMax-Music3 and Breeze-TTS-2 answer `/v1/audio/speech`, whole or streamed as they are made: a song
from lyrics and a description, 44.1 kHz stereo, made at about the speed it plays on two cards; and
speech in a voice the request describes, made in 0.6 of its playing time on one card, its first
streamed audio 0.21 s after the request. Breeze-TTS-2 also speaks text while it is being written,
over the `/v1/audio/speech/stream` WebSocket; it is licensed for research and non-commercial use
only. Fish Audio's drama-3 speaks a script whose speakers and direction are written in it
(`<|speaker:1|>`, `[laughs]`), each speaker in the model's voice or one cloned from a reference
clip and its transcript, at 0.3 of its playing time on two cards, its first streamed audio
0.23 s after the request; under the Fish Audio Community License a service that lets others
generate speech with it needs a licence from Fish Audio, and imitating a real person's voice needs
their consent.

## Run with Docker

The host needs Docker and an `amdgpu` driver that supports ROCm 10.1; it does not need ROCm. The
card works best with Resizable BAR on (Above 4G Decoding and Resizable BAR in the firmware setup);
without it the engine keeps kernel arguments in pinned host memory, and every dispatch reads them
over the link. The image
is [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance), with its kernels
built for `gfx1201` (RDNA4), `gfx11-generic` (RDNA3) and `gfx1151` (Strix Halo). Pull it, clone
this repository for its compose files, then start one of them:

```sh
docker pull stilldeadcode/radiance
git clone https://codeberg.org/StillDeadcode/radiance.git
cp -r radiance/deploy/compose ~/radiance-compose && cd ~/radiance-compose
cp .env.example .env                         # set RADIANCE_MODELS, RADIANCE_API_KEY, the GIDs
docker compose -f tp2.yaml up -d             # two cards; tp1.yaml for one (RADIANCE_GPU), tp4.yaml for four
docker compose -f tp2.yaml logs -f
docker compose -f tp2.yaml down
```

A file serves all its models from one container behind one port, and brings a model onto the cards
when a request names it ([`docs/MULTIMODEL.md`](docs/MULTIMODEL.md)): a request's `model` is the
entry's name (`flashnext`, `qwen3.8-27b`, `minicpm5-2b`, ...). The entries are in
`tp1.radiance.yaml`, `tp2.radiance.yaml` and `tp4.radiance.yaml` beside the compose files, the server's configuration as
it reads it. Delete the entries of the containers
you do not have; any other model from the table is added as an entry with its keys from the table
under [Run without Docker](#run-without-docker). The two files take the same port, so run one at a
time. Running without compose, and the reason for each `docker run` option, is in
[`docs/DOCKER.md`](docs/DOCKER.md).

To build the image yourself, for example for other cards, run `docker/build.sh -t 'gfx1200;gfx1201'`
in the clone (it needs `docker buildx` and tests the build as it goes). Then set
`RADIANCE_IMAGE=radiance:latest` in `.env`.

## Build from source

You need Linux, git, a C++20 compiler, CMake 3.21+ and Ninja. For the device kernels you also need
ROCm 10.1 or 7.2 (`hipcc`; set `ROCM_PATH` if it is not `/opt/rocm`).

The C++20 compiler is tested with the clang included in the ROCm 10.1 distribution; other
compilers may produce compile errors in the C++ architecture plugins. Pull requests for
cross-compiler build issues are not accepted.

Clone the repository and build in it:

```sh
git clone https://codeberg.org/StillDeadcode/radiance.git && cd radiance
./build.sh                                   # configure, build for the cards present, run the fast tests
./build.sh --no-hip                          # host backend only
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS='gfx1201'   # build for cards this machine lacks
```

To install, name the prefix when the build directory is first configured
(`-DCMAKE_INSTALL_PREFIX=/opt/radiance`), build, then run `cmake --install build`. The default
plugin path, `<prefix>/share/radiance`, is compiled in from that first configure, so
`cmake --install build --prefix DIR` alone gives a tree that looks for its plugins elsewhere. The
prefix gets `bin/`, `include/radiance/`, `lib/cmake/radiance/`, `libexec/radiance/`,
`share/radiance/` and `share/doc/radiance/`.

With ROCm 10.1 the build turns the compiler's VOPD pairing (dual-issue instruction pairs) off: its
LLVM 24 pairs in an order that follows heap addresses, so two builds of the same source can differ.
The Docker image builds its compiler with the fix and keeps the pairing, which is 1.4-3.5% faster
on prefill GEMMs; a compiler that has the same fix is declared with
`-DRAD_HIP_VOPD_REPEATABLE=ON`. ROCm 7.2 is unaffected.

## Run without Docker

The server reads its settings from a YAML file: `server:` holds the listener, `memory:` the
per-card VRAM budget, and `models:` one entry whose name is the model id requests use. Every key
is named as in the [`docs/GUIDE.md`](docs/GUIDE.md) §4.11 reference; `radiance --help` prints it.

```yaml
# radiance.yaml
server:
  port: 8000
models:
  minicpm5-2b-iq8r:
    model: /srv/models/rad/minicpm5-2b-iq8r.rad   # or a Hugging Face repo id: StillDeadcode/minicpm5-2b-iq8r
```

```sh
export RADIANCE_HOME=$PWD/build/radiance_home   # the plugins; an install finds its own
build/bin/radiance --config radiance.yaml --check   # validate the file, serve nothing
build/bin/radiance --config radiance.yaml
```

`model` also takes a Hugging Face repo id, as vLLM does. The server fetches the repo with `curl`
and `jq` (`scripts/hfget.sh`) into `~/.cache/radiance/hf` (`server.download-dir` puts it elsewhere),
serves the container it holds, and checks it for changes at every start; a start that cannot reach
the hub serves the copy it has. The Docker image carries both tools; in a container,
`server.download-dir` must name a writable mount, since the compose files mount `/models` read-only.

The server takes no other command-line options. A command line written for an earlier release
(`radiance --model ... --tp 2`) is refused with the same settings printed as YAML, and
`rad-yaml FLAGS...` prints that file without starting anything:
`rad-yaml --model x.rad --tp 2 > radiance.yaml`. [`docs/MULTIMODEL.md`](docs/MULTIMODEL.md) §4
describes the file's levels.

Each row holds the keys a model is served with in a file of its own (the Flash-Next rows leave out
`prefix-cache-mib`). They go under the model's entry, except `gpu-headroom-mib`, which goes under
`memory:` and in a file of several models holds for all of them. `tp1.radiance.yaml` and
`tp2.radiance.yaml` carry their models' entries as set for sharing the cards; a model added to either takes its row from
here.

| model | keys |
|---|---|
| Qwen3.8-Flash-Next | `tp: 2, tp-wire: wht6, max-num-seqs: 8, max-model-len: 200000, placement: expert_tiered, host-pool-mib: 24576, gpu-headroom-mib: 96, kv-cache-dtype: i8, num-speculative-tokens: 3, max-num-batched-tokens: 2048` |
| Qwen3.8-Flash-Next, one card | `tp: 1, max-num-seqs: 4, max-model-len: 200000, placement: expert_tiered, host-pool-mib: 43000, gpu-headroom-mib: 96, kv-cache-dtype: i8, num-speculative-tokens: 3, max-num-batched-tokens: 4096, checkpoint-interval: 4096` |
| Qwen3.8-Flash-Next, four ranks | `tp: 4, max-num-seqs: 8, max-model-len: 200000, placement: expert_tiered, host-pool-mib: 49152, gpu-headroom-mib: 96, kv-cache-dtype: i8, num-speculative-tokens: 3, max-num-batched-tokens: 2048` |
| Qwen3.8-Flash-Next, three ranks (the shipped container as it is, [`docs/TP3.md`](docs/TP3.md)) | `tp: 3, max-num-seqs: 8, max-model-len: 200000, placement: expert_tiered, host-pool-mib: 24576, gpu-headroom-mib: 96, kv-cache-dtype: i8, num-speculative-tokens: 3, max-num-batched-tokens: 2048` |
| GLM-5.3-Flash | `tp: 4, placement: expert_tiered, host-pool-mib: 65536, max-num-seqs: 4, max-model-len: 262144, max-num-batched-tokens: 2048, kv-cache-dtype: i8, num-speculative-tokens: 0, prefill-interleave: 4, prefix-cache-mib: 32768, gpu-headroom-mib: 96`, and `server.kernels: glm,libr4d,libavx,libref` |
| Qwen3.8-27B IQ8R and IQR-V4, Qwen3.6-35B-A3B | `tp: 2, max-num-seqs: 32, max-model-len: 200000, kv-cache-dtype: i8` |
| Qwen3.8-27B bf16 | `tp: 2, max-num-seqs: 8, max-model-len: 200000, kv-cache-dtype: i8` |
| MiniCPM5-2B IQ8R and bf16 | `tp: 2, max-num-seqs: 32, max-model-len: 131072, kv-cache-dtype: i8, gpu-headroom-mib: 160, max-num-batched-tokens: 512` |
| Qwen3.8-Flash-Next bf16 | `tp: 2, max-model-len: 32768, placement: expert_tiered, weights-disk-tier: true, host-pool-mib: 24576, gpu-headroom-mib: 768, deterministic: true, max-num-seqs: 32, max-num-batched-tokens: 32768, num-speculative-tokens: 0` |
| Qwen-Image 2.1 and 2.1 Turbo | `tp: 2, tp-wire: wht6` |
| MiniMax-H3 and FastH3 | `tp: 2, vision-tower-placement: ram` |
| MiniMax-Music3 | `tp: 2`, or `tp: 1` (20.5 GiB of weights on the card) |
| Breeze-TTS-2 | `tp: 1` |
| Gemma 4 31B (GPTQ int4) | `tp: 2, max-num-seqs: 8, max-model-len: 65536, kv-cache-dtype: i8, num-speculative-tokens: 3, max-num-batched-tokens: 2048, temp: 1.0, top-p: 0.95, top-k: 64` |
| Gemma 4 31B w5, heretic-ara int4, int8 and heretic-ara int8 | `tp: 2, max-num-seqs: 8, max-model-len: 65536, kv-cache-dtype: i8, vision-tower-placement: ram, num-speculative-tokens: 3, max-num-batched-tokens: 2048, temp: 1.0, top-p: 0.95, top-k: 64` |
| Gemma 4 31B Q8 | `tp: 2, max-num-seqs: 8, max-model-len: 65536, kv-cache-dtype: i8, num-speculative-tokens: 0, max-num-batched-tokens: 2048, temp: 1.0, top-p: 0.95, top-k: 64` |
| Gemma 4 12B bf16 | `tp: 2, max-num-seqs: 8, max-model-len: 65536, kv-cache-dtype: bf16, num-speculative-tokens: 3, max-num-batched-tokens: 2048, mm-max-patches: 1120, temp: 1.0, top-p: 0.95, top-k: 64, chat-template-kwargs: {"enable_thinking":false}` |
| Gemma 4 12B int8 | `tp: 1, max-num-seqs: 8, max-model-len: 65536, kv-cache-dtype: bf16, num-speculative-tokens: 3, max-num-batched-tokens: 2048, mm-max-patches: 1120, temp: 1.0, top-p: 0.95, top-k: 64, chat-template-kwargs: {"enable_thinking":false}` |
| NVIDIA-Nemotron-3.5-Bolt-E4B bf16, int8 and 4-bit, without the drafter | `tp: 2, max-num-seqs: 8, max-model-len: 65536, kv-cache-dtype: bf16, num-speculative-tokens: 0, mm-max-patches: 2520, vision-tower-placement: ram, temp: 1.0, top-p: 0.95, top-k: 64, chat-template-kwargs: {"enable_thinking":false}`; `tp: 1` on one card |
| TranslateGemma 27B | `tp: 2, max-num-seqs: 8, max-model-len: 16384, kv-cache-dtype: i8`, and `override-chat-template` naming `deploy/compose/gemma3.jinja` |
| Unlimited-OCR | `tp: 1, max-num-seqs: 8, max-model-len: 32768, kv-cache-dtype: i8, gpu-headroom-mib: 96`, and `override-chat-template` naming `deploy/chat-templates/unlimited-ocr-parser.jinja` (`/chat-templates/unlimited-ocr-parser.jinja` in the compose files' container) |
| Krea 2 Turbo and Raw, FLUX.3 [klein] and its base, Stable Diffusion XL | `tp: 1` |
| LTX-2.5 distilled and base | `tp: 1, vision-tower-placement: ram` |
| EmbeddingGemma 2 | `tp: 1, max-num-seqs: 64, max-num-batched-tokens: 8192` |
| Qwen3-ASR 1.7B | `tp: 1, max-num-seqs: 16` |

The 27B in IQ8R and IQR-V4, the 35B-A3B, MiniCPM5 and the Flash-Next bf16 reference run at up to
32 sequences a step. The other Flash-Next rows and the 27B in bf16 keep 8, the one-card Flash-Next
row 4: on a hybrid model each sequence holds its own recurrent state, and 32 of them take that
memory from the attention cache and, on Flash-Next, from the expert slab. See
[`docs/GUIDE.md`](docs/GUIDE.md) §4.4.

On one card a Flash-Next prefill step copies every pooled expert across the link once, so a
4096-token step shares that copy among twice the tokens of a 2048-token one (26K-token prompts:
2545 against 1413 tok/s); the checkpoint interval matches the step because a prefill chunk never
crosses a checkpoint. At 4 sequences that row's cache holds 301K tokens. The Flash-Next bf16 row
is a reference to calibrate and compare against, a step taking tens of seconds whatever its size;
its 32K of context keeps QSA's score plane for a 32K-token step at 1 GiB.

At `tp: 4`, decode and verify steps carry the exact all-reduce in the gated residual's write, with
the MoE gather in front of it; a prefill chunk runs the unfused chain. That chain's exact
all-reduce is one-shot up to 1 MiB a message and two-shot above, which sends half the bytes on a
prefill chunk; both produce the same bits. The four-rank row runs exact. At three and four ranks
`tp-wire: wht6` selects a different lossy wire, tiered int8 over the two-shot, above
`tp-wire-min-kb`. Eight ranks have no lossy wire, so `tp: 8` with `tp-wire: wht6` stops at startup
naming the all-reduce.

Four ranks is a capacity configuration: its per-step rate against `tp: 2` on the same cards has
not been measured, and its tiering keys are the two-rank recipe carried over. As at two ranks, each
rank owns a slice of every expert's width: each parity's 128-column blocks are cut contiguously over
the ranks and the odd block alternates with the parity, so every rank's share is whole blocks and
every block has one owner. The resident expert bytes per rank are smaller, so the elastic window
has a different shape: read the boot log's `elastic` line and re-derive `host-pool-mib` before
trusting it.

`host-pool-mib` is what a model pins for its weights in all, each rank an equal share: the
four-rank row pins 48 GiB of RAM (12 GiB a rank) that nothing else can use and the kernel cannot
swap out. The amdgpu driver caps what all processes
together pin at the machine's RAM less 1/64 and 1.5 GiB; the pool is locked user memory, which
`ttm.pages_limit` does not count. A 64 GiB machine serves the one-card Flash-Next row alone, with
`prefix-cache-mib: 4096`, at about 49 GiB pinned ([`docs/GUIDE.md`](docs/GUIDE.md) §7). In
`tp1.yaml` its entry carries `prefix-cache-mib: 1024` so its host pins fit such a host beside the
8 GiB the fleet keeps free; there it is unloaded when another model loads, and on a 61 GB host it
answered its next request after 17 to 19 s, loaded from its container again.

The Gemma 4 rows sample as the checkpoint's `generation_config` states, and a request switches
thinking (`chat_template_kwargs`). `vision-tower-placement: ram` keeps the vision tower in host
memory, so its VRAM goes to the KV cache. The 31B takes its full 262,144-token context at
`max-model-len: 262144, max-num-seqs: 2`. The 12B keeps a bf16 cache because an 8-bit one costs it
more than it saves; on Bolt-E4B an E4M3 cache cost 9x the KL and the int8 cache is unmeasured. The
Bolt-E4B row serves without speculation because the published containers' drafters are block FP8.

EmbeddingGemma 2 takes an input in one step, so its `max-num-batched-tokens` is the longest input
as well as the batch. A live Qwen3-ASR session holds one sequence, so its `max-num-seqs` is the
most sessions served at once. On one card LTX-2.5's adapter slots fit at `max-lora-rank: 16,
max-loras: 1` (433 MiB).

In a file of several models, Qwen-Image's grant counts two sequences of `max-model-len` tokens of
cache, which at the model's own 33600 tokens is 55.8 GiB on one card and 32.6 GiB a card on two,
more than the cards. `tp1.yaml` sets `max-model-len: 8192` (it needs 29.7 GiB; pictures up to
about 1.9 megapixels, 1024x1024 being 4096 tokens) and `tp2.yaml` `20480` (25.4 GiB a card; the
default 2048x2048 is 16384 tokens). Alone in a file it loads at its own length, its cache taking
what the cards have left. It sizes its buffers itself and warns that it does not take
`image-gen-max-pixels` or `image-gen-max-refs`.

Add `server.api-key: KEY` (or `server.api-key-env: VAR`, which reads it from the environment) to
require `Authorization: Bearer KEY` (or `X-Api-Key: KEY`). `debug-graph: true` on the model
prints every op and the kernel it resolved to, then exits without serving. A chat request:

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"minicpm5-2b-iq8r","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

An image model answers OpenAI's images API instead, with base64 pictures:

```sh
curl -s localhost:8000/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"a lighthouse on a cliff at sunset, oil painting","size":"1024x1024"}' \
  | python3 -c 'import sys,json,base64; open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

`http://localhost:8000/` is the dashboard. [`docs/GUIDE.md`](docs/GUIDE.md) covers the API, memory
budgets, long context, speculative decoding and operating a server.

## Convert a checkpoint

`rad-convert` turns a Hugging Face checkpoint into a container, quantising what a recipe in
`data/recipes/` names. Download checkpoints with `scripts/hfget.sh <org/repo> <dir>`.

```sh
# Spark X2.6 uses the checkpoint's BF16 weights, tokenizer and chat template; no recipe required.
build/bin/rad-convert /srv/models/hf/XHToken-Spark-X2.6-4B -o spark-x2.6-4b-bf16.rad --max-ctx 4096
build/bin/rad-convert /srv/models/hf/Qwen/Qwen3.8-27B -o qwen3.8-27b-bf16.rad \
    --recipe data/recipes/q38-27b-bf16.recipe
build/bin/rad-convert /srv/models/hf/Qwen/Qwen3.8-Flash-Next -o qwen3.8-flash-next-bf16.rad --max-ctx 4096
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-w4a8.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-w4a8.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/google/gemma-4-31B-it-assistant
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-dense-w4a8-df.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-w4a8-df.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/z-lab/gemma-4-31B-it-DFlash
build/bin/rad-convert /srv/models/hf/google/gemma-4-31B-it -o gemma4-31b-dense-q8-df.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-q8-df.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model /srv/models/hf/z-lab/gemma-4-31B-it-DFlash
# Meta's Muse Glimmer 30B, as it ships (bf16), with an int4 trunk and int8 attention values, or
# with an IQ8R trunk and an int8 head
build/bin/rad-convert /srv/models/hf/meta-models/Muse-Glimmer-30B -o muse-glimmer-30b-bf16.rad \
    --recipe data/recipes/museglimmer-30b-bf16.recipe
build/bin/rad-convert /srv/models/hf/meta-models/Muse-Glimmer-30B -o muse-glimmer-30b-w4a8.rad \
    --recipe data/recipes/museglimmer-30b-w4a8.recipe \
    --set quantization_config.quant_method=compressed-tensors
build/bin/rad-convert /srv/models/hf/meta-models/Muse-Glimmer-30B -o muse-glimmer-30b-iq8r-dflash.rad \
    --draft-model /srv/models/hf/meta-models/Muse-Glimmer-30B-assistant \
    --recipe data/recipes/museglimmer-30b-iq8r.recipe \
    --set quantization_config.quant_method=compressed-tensors
build/bin/rad-convert /srv/models/hf/ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AWQ -o swift15-27b-w4a16.rad \
    --recipe data/recipes/q38-27b-w4a16.recipe
# Any single-file SDXL checkpoint (SDXL 1.0, Illustrious, NoobAI, Pony, a civitai finetune), with
# CLIP's tokenizer.json from openai/clip-vit-large-patch14
build/bin/rad-convert /srv/models/hf/OnomaAIResearch/Illustrious-XL-v2.0/Illustrious-XL-v2.0.safetensors \
    -o illustrious-xl-v2.0-bf16.rad --recipe data/recipes/sdxl-bf16.recipe \
    --tokenizer /srv/models/hf/openai/clip-vit-large-patch14/tokenizer.json
```

With Docker, use `docker run --rm -v /srv/models:/models --entrypoint rad-convert stilldeadcode/radiance ...`.
In the image the recipes are in `/opt/radiance/share/radiance/recipes/`. A checkpoint served at its
own precision needs no container: its directory is the entry's `model`
(`model: /srv/models/hf/Qwen/Qwen3.8-27B`). A release someone else quantised to an integer grid is
written at its own numbers where the kernels read them (an int4 W4A16 release, compressed-tensors'
AWQ and GPTQ or AutoRound's, on the 4-bit GEMMs with `data/recipes/q38-27b-w4a16.recipe`). Block
FP8, MXFP4, NVFP4 and ParoQuant releases are not served: convert the model's bf16 checkpoint with
an int8 recipe. [docs/GUIDE.md](docs/GUIDE.md) names the recipes of record
with their `rad-convert` commands.

Spark X2.6 runs BF16 at `tp: 1`, `2`, `3` or `4` -- and `8`, the worlds libr4d has an exact
all-reduce for -- without a drafter or quantisation recipe. Its checkpoint's chat template enables
thinking by default; replies return reasoning in `reasoning_content`. Set
`chat_template_kwargs.enable_thinking` to `false` for a non-thinking request. Tool calls use the
checkpoint's tagged format and are returned as OpenAI `tool_calls`.

## Test

```sh
ctest --test-dir build -L fast -j8         # the inner loop, under 15 s; build.sh runs it
ctest --test-dir build -LE 'gpu|oot' -j8   # everything that runs without a card (about 5 minutes)
ctest --test-dir build -L gpu              # the card tests; they need the cards to themselves
ctest --test-dir build -L oot              # rebuilds every bundled plugin out of tree against an install
```

`CMakePresets.json` names the same runs (`ctest --preset fast`, `host`, `gpu`, `full`, `asan`), and
[`docs/GUIDE.md`](docs/GUIDE.md) §1.3 describes the tiers and the test binaries' options.

## Documentation

| | |
|---|---|
| [`docs/GUIDE.md`](docs/GUIDE.md) | installing, converting, serving, the HTTP API, operating a server |
| [`docs/TOOLS.md`](docs/TOOLS.md) | every flag of every tool, and the recipe syntax |
| [`docs/DIAGNOSE.md`](docs/DIAGNOSE.md) | `diagnose`: the engine measures itself, and what makes a run a measurement |
| [`docs/DOCKER.md`](docs/DOCKER.md) | the image and the compose files |
| [`docs/PREPARING_YOUR_RIG.md`](docs/PREPARING_YOUR_RIG.md) | a two-R9700 machine from firmware setup to a serving model |
| [`docs/PLUGIN.md`](docs/PLUGIN.md) | writing a kernel library, architecture or quantiser plugin, including one for another AMD card |
| [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md) | the supported model families and their geometry |
| [`docs/TP3.md`](docs/TP3.md) | any world size: the attention world and uneven delta-net heads; three ranks |
| [`docs/OPS.md`](docs/OPS.md), [`docs/KERNELS.md`](docs/KERNELS.md) | the op vocabulary, and what `rad-kbench` last measured |
| [`spec.md`](spec.md) | the design |

## License

[Apache-2.0](LICENSE), copyright Deadcode and the radiance contributors. Keep [`NOTICE`](NOTICE)
with any copy or derivative. It credits the authors and lists the MIT-licensed code the tree
includes. The model weights are not covered: each keeps the license of the model it was converted
from. The Docker image also holds third-party libraries under their own licenses, GPL and LGPL
among them; `/usr/share/doc/THIRD-PARTY` in the image lists each one with its version and source.
