# ComfyUI-Radiance-MiniMax

One ComfyUI node, **Radiance MiniMax**, that renders a clip on a radiance server serving
MiniMax-H3 (`POST /v1/videos`) and returns it. The node is an HTTP client: the model runs in the
radiance server process, on the same machine or another one, and ComfyUI holds no model and no
VRAM for it. Several nodes can drive several servers from one ComfyUI.

Install it by copying this folder into `ComfyUI/custom_nodes/` and restarting ComfyUI. The node's
description (the **?** in its header) documents every input.

## The server

    radiance --model minimax-h3-fp8.rad --radiance-home ... --tp 2 \
        --host 0.0.0.0 --port 8133 --lora-dir /path/to/loras

`--host 0.0.0.0` makes the server reachable from other machines. The node sends no API key, so
it drives a server without `--api-key`, which takes any request: keep that server on a trusted
network. The node fetches the adapter lists from the browser, which needs the server's CORS
headers (on unless `--no-cors`).

`--lora-dir` lets a request name any `NAME.safetensors` under the directory; the node's two
adapter lists show them (`GET /v1/loras`). Pointing it at ComfyUI's `models/loras` serves the
same files ComfyUI shows. ComfyUI's Load LoRA node does not apply to this node, since the server
applies adapters to the weights it serves.

The server reserves every adapter slot in VRAM at startup and logs what a slot costs. With
`--lora-dir` alone it reserves two (`--max-loras`). On two 32 GB cards serving MiniMax-H3 a
rank-16 slot costs about 265 MiB a card, and the second one fits only when nothing else holds
VRAM on those cards. When it does not, startup stops with `vram_kv resolved to 0 MiB`: serve
with `--max-loras 1` and leave the node's second adapter at `none`, or run ComfyUI on other
cards.

## Media

| task | media used |
|---|---|
| `t2va` | none |
| `i2va` | `image_1`: the first frame |
| `l2va` | `image_1`: the last frame |
| `fl2va` | `image_1`: the first frame, `image_2`: the last frame |
| `ref2va` | `image_1..9`, `video_1..3`, `video_audio_1..3`, `audio_1..3` |

A task ignores media it does not use, with a warning on the console. For `ref2va`,
`video_audio_N` is the soundtrack of `video_N`, and `audio_N` is standalone reference audio such
as a voice. Each kind is filled in order, since the input's number is the reference's number in
the request. Pictures are sent as PNG, audio as WAV and clips as MP4.

## Sizes

The resolution tier is the short side and the aspect gives the long side (480p 16:9 is
864x480); the node's title shows the size. MiniMax-H3 takes sides that are multiples of 32 px
and at least 256, an aspect of at most 4:1 and an area of at most 1344x768, so 720p 16:9 is
1312x736. A `custom` size is adjusted to the nearest size the model takes, with a note on the
console. Every task takes every size.

`latent_upscale` (advanced) runs the early steps at half or a quarter of the size and the rest
at full size, with a container converted with the latent upscaler. Every task takes it.

## Outputs

The clip is saved under `radiance/` in ComfyUI's output folder and returned as a `VIDEO`, with
its path and the server's job id (`GET /v1/videos/<id>`). While the job runs, the node drives
the progress bar and prints a console line for each finished step:

    [RadianceMiniMax] video_x: 45%, step 12/26, 5.2 s/step, eta ~1m 07s and the decode, elapsed 55s

A refused or failed job raises with the server's message; `timeout` (advanced) bounds the wait.
