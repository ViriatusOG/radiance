"""Radiance MiniMax: a ComfyUI node that renders a clip on a radiance server serving MiniMax-H3.

The node is an HTTP client of POST /v1/videos: the model runs in the radiance server process, on
this machine or another, and the node holds no model and no VRAM. The node's description (its ?
pane) documents every input; README.md documents the server side.
"""

import io as _io
import json
import logging
import os
import re
import tempfile
import time
import wave

import requests

import folder_paths
from comfy_api.latest import io, ui, Types
from comfy_api.input_impl import VideoFromFile

POLL_SECONDS = 3.0
KEEPALIVE_SECONDS = 30.0
LOG = "RadianceMiniMax"

# The size law the minimaxh3 plugin declares (arch/minimaxh3/minimaxh3.cpp): sides multiples of
# 32 px and at least 256, an aspect of at most 4:1, an area of at most 1344 x 768. The server
# enforces it; the node applies it first so the title can show the size a request will carry.
SIZE_MULT, MIN_SIDE = 32, 256
MAX_AREA, MAX_ASPECT = 1344 * 768, 4.0

# A tier is the short side; the aspect gives the long side. 720p is the last tier whose canvas
# fits the area at 16:9.
TIERS = {"144p": 144, "240p": 240, "360p": 360, "480p": 480, "540p": 540,
         "576p": 576, "720p": 720}
ASPECTS = ["5:4", "4:3", "3:2", "16:9", "2:1", "1:1", "1:2", "9:16", "2:3", "3:4", "4:5"]

# The keyframe tasks and the form field each of their images goes in.
KEYFRAMES = {"i2va": ["first_frame"], "l2va": ["last_frame"],
             "fl2va": ["first_frame", "last_frame"]}

DESCRIPTION = """Renders video and audio on a **radiance server serving MiniMax-H3**
(`POST /v1/videos`). The node is an HTTP client: the model runs in the server process and the
node holds no VRAM.

## Media by task

| task | media used |
|---|---|
| `t2va` | none |
| `i2va` | `image_1`: the first frame |
| `l2va` | `image_1`: the last frame |
| `fl2va` | `image_1`: the first frame, `image_2`: the last frame |
| `ref2va` | every media input |

A task ignores the media it does not use, with a warning on the console. For `ref2va`:

- **image_1..image_9**: reference pictures (subjects, settings, style).
- **video_1..video_3**: reference clips (motion, camera).
- **video_audio_1..video_audio_3**: the soundtrack of `video_N`; `video_N` must be connected.
- **audio_1..audio_3**: standalone reference audio, such as a voice.

Fill each kind in order, without gaps: the input's number is the reference's number in the
request (`<Picture 2>`, `<Video 1>`).

## Inputs

- **prompt**: a widget and an input; a connected string replaces the typed text. The model is
  trained on MiniMax's structured prompt (`subject_definitions`, `summary`,
  `retention_analysis`, `detailed_description`, `<Picture N>` and similar tags, timed shots,
  `<d>[English] ...</d>` dialogue); a one-line prompt makes poor audio.
- There is **no negative prompt**: MiniMax-H3 is guidance-distilled, one forward pass a step,
  and the server refuses a `negative_prompt`.
- **lora 1, lora 2**: adapters the server offers (`GET /v1/loras`, files under its
  `--lora-dir`), each with its own strength; `none` is off. The lists are fetched when the
  node is created or loaded, when `server_url` changes, and by **refresh lora list**. ComfyUI's
  Load LoRA node does not apply here: the server applies adapters to the weights it serves.
- **resolution, aspect**: the tier is the short side and the aspect gives the long side,
  480p 16:9 is 864x480; the node's title shows the result. Sides are multiples of 32 px, at
  least 256, within an area of 1344x768, so 720p 16:9 is 1312x736. `custom` takes any WxH and
  adjusts it to the nearest size the model takes, with a note on the console.
- **seconds, seed, num_inference_steps, sampler, shift, audio_shift**: as the server takes
  them; `sampler` is `euler` or `res_multistep`. The server reads the seed as a double, so the
  seed stops at 2^53 - 1, the largest it holds exactly.
- **latent_upscale** (advanced): with a container that has the latent upscaler, the early steps
  run at half or a quarter of the size and the rest at full size. The keyframe tasks do not
  take it, since a keyframe is pinned on the full-size grid.
- **server_url, model**: the server (`http://<host>:8133`) and the model id it serves
  (`GET /v1/models`).

## Outputs

The clip is saved under `radiance/` in the output folder and returned as a `VIDEO`, with its
path and the server's job id. While the job runs, the node drives the progress bar and prints a
line to the console for each finished step, with the measured seconds a step and an estimate of
the time left; `timeout` (advanced) bounds the wait."""


def _image_png(image):
    """The first image of a ComfyUI IMAGE batch ([B, H, W, C] in 0..1) as PNG bytes."""
    import torch
    from PIL import Image
    arr = image[0].cpu().clamp(0.0, 1.0).mul(255.0).to(dtype=torch.uint8).numpy()
    buf = _io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return "image.png", buf.getvalue(), "image/png"


def _audio_wav(audio):
    """The first clip of a ComfyUI AUDIO ({'waveform': [B, C, N] in -1..1, 'sample_rate'}) as
    16-bit WAV bytes."""
    import torch
    pcm = (audio["waveform"][0].transpose(0, 1).clamp(-1.0, 1.0) * 32767.0).cpu().to(
        dtype=torch.int16).contiguous().numpy()
    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(audio["sample_rate"]))
        w.writeframes(pcm.reshape(-1).tobytes())
    return "audio.wav", buf.getvalue(), "audio/wav"


def _video_mp4(video):
    """A ComfyUI VIDEO as MP4 bytes, through its own save_to (a stream copy when the source is
    an MP4 already)."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        video.save_to(path, format=Types.VideoContainer.MP4)
        with open(path, "rb") as f:
            return "video.mp4", f.read(), "video/mp4"
    finally:
        os.unlink(path)


def _leading(items, prefix):
    """The number of connected inputs at the front of `items`; a gap is refused, since the
    input's number is the reference's number in the request."""
    n = 0
    while n < len(items) and items[n] is not None:
        n += 1
    for i in range(n, len(items)):
        if items[i] is not None:
            raise ValueError(f"{prefix}_{i + 1} is connected but {prefix}_{n + 1} is not: "
                             "fill the inputs in order")
    return n


def _fit(w, h):
    """The largest size at about the w:h ratio within the size law, sides rounded down to the
    size multiple."""
    s = (MAX_AREA / float(w * h)) ** 0.5
    fw = max(MIN_SIDE, int(w * s) // SIZE_MULT * SIZE_MULT)
    fh = max(MIN_SIDE, int(h * s) // SIZE_MULT * SIZE_MULT)
    while fw * fh > MAX_AREA:  # the 256 floor can lift a narrow ratio over the area
        if fw >= fh and fw > MIN_SIDE:
            fw -= SIZE_MULT
        elif fh > MIN_SIDE:
            fh -= SIZE_MULT
        else:
            break
    if max(fw, fh) > MAX_ASPECT * min(fw, fh):  # the rounding can tip a 4:1 ratio over
        if fw >= fh:
            fw = int(MAX_ASPECT * fh) // SIZE_MULT * SIZE_MULT
        else:
            fh = int(MAX_ASPECT * fw) // SIZE_MULT * SIZE_MULT
    return fw, fh


def _snap(v):
    """The nearest multiple of the size multiple, halves up, at least the smallest side."""
    return max(MIN_SIDE, (v + SIZE_MULT // 2) // SIZE_MULT * SIZE_MULT)


def _custom_size(text):
    """A typed WxH adjusted to the size law: each side to the nearest multiple, then scaled back
    at the same ratio when the aspect or the area is over."""
    m = re.match(r"(\d+)\s*[xX×*]\s*(\d+)", text.strip())
    if not m:
        raise ValueError(f'"{text}" is not a size: give WIDTHxHEIGHT, such as 512x512')
    w, h = int(m.group(1)), int(m.group(2))
    lw, lh = _snap(w), _snap(h)
    if lw * lh > MAX_AREA or max(lw, lh) > MAX_ASPECT * min(lw, lh):
        lw, lh = _fit(lw, lh)
    if (lw, lh) != (w, h):
        logging.warning(f"[{LOG}] size {w}x{h} adjusted to {lw}x{lh}: sides multiples of "
                        f"{SIZE_MULT} px and at least {MIN_SIDE}, an aspect of at most "
                        f"{MAX_ASPECT:.0f}:1, an area of at most {MAX_AREA} px")
    return f"{lw}x{lh}"


def _tier_size(tier, aspect):
    """The tier's short side and the aspect's long side, both snapped, the pair scaled back when
    the area is over; WxH in the aspect's orientation."""
    a, b = (int(x) for x in aspect.split(":"))
    hi, lo = max(a, b), min(a, b)
    short = _snap(TIERS[tier])
    long_ = max(short, _snap(int(short * hi / lo)))
    if long_ * short > MAX_AREA:
        long_, short = _fit(long_, short)
    return f"{long_}x{short}" if a >= b else f"{short}x{long_}"


def _resolve_size(resolution, aspect, custom=""):
    """The request's size. ComfyUI hands a DynamicCombo over as a dict of the chosen option and
    its own inputs."""
    if isinstance(resolution, dict):
        custom = resolution.get("custom_size", custom)
        resolution = resolution.get("resolution", "")
    if resolution == "custom":
        return _custom_size(custom)
    if resolution not in TIERS:
        raise ValueError(f"unknown resolution {resolution!r}: one of {', '.join(TIERS)} or custom")
    return _tier_size(resolution, aspect)


def _fmt_secs(s):
    """Whole seconds, or minutes and seconds from 90 s up."""
    s = int(max(0.0, float(s)) + 0.5)
    return f"{s}s" if s < 90 else f"{s // 60}m {s % 60:02d}s"


class _Pace:
    """Seconds a step, measured from the job's steps_done between polls. A poll every 3 s sees a
    longer step as a single increment, so the time between two polls is not a step's time: the
    rate is taken over at least two steps, then smoothed so it follows a change of pace, such as
    a latent-upscaled job moving to the full-size grid."""

    def __init__(self):
        self.t = None
        self.done = None
        self.sps = None

    def update(self, job, now):
        done = job.get("steps_done")
        if not isinstance(done, int):
            return self.sps
        if self.t is None:
            if done > 0:  # from the first finished step, so the encoders' time is not a step's
                self.t, self.done = now, done
        elif done - self.done >= 2:
            rate = (now - self.t) / (done - self.done)
            if rate > 0:
                self.sps = rate if self.sps is None else 0.4 * rate + 0.6 * self.sps
            self.t, self.done = now, done
        return self.sps


def _progress_line(job, elapsed, sps):
    """The console line for a poll: progress, steps, pace, time left and time taken."""
    if job.get("status") == "queued":
        return f"queued, elapsed {_fmt_secs(elapsed)}"
    p = job.get("progress")
    p = float(p) if isinstance(p, (int, float)) else 0.0
    parts = [f"{p:.0f}%"]
    done, total = job.get("steps_done"), job.get("steps_total")
    steps = isinstance(done, int) and isinstance(total, int) and total > 0
    if steps:
        parts.append(f"step {done}/{total}")
    if sps:
        parts.append(f"{sps:.1f} s/step")
    if steps and sps and 0 < done < total:
        parts.append(f"eta ~{_fmt_secs((total - done) * sps)} and the decode")
    elif steps and done == total:
        parts.append("decoding")
    parts.append(f"elapsed {_fmt_secs(elapsed)}")
    return ", ".join(parts)


def _progress_bar(node_cls):
    """ComfyUI's progress bar for this node over 0..100, or None outside ComfyUI's executor."""
    uid = getattr(getattr(node_cls, "hidden", None), "unique_id", None)
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(100.0, node_id=uid)
    except Exception:
        return None


class RadianceMiniMax(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        images = [io.Image.Input(f"image_{i}", optional=True) for i in range(1, 10)]
        videos = [io.Video.Input(f"video_{i}", optional=True) for i in range(1, 4)]
        tracks = [io.Audio.Input(f"video_audio_{i}", optional=True) for i in range(1, 4)]
        audios = [io.Audio.Input(f"audio_{i}", optional=True) for i in range(1, 4)]
        images[0].tooltip = ("ref2va: reference picture 1; i2va and fl2va: the first frame; "
                             "l2va: the last frame")
        images[1].tooltip = "ref2va: reference picture 2; fl2va: the last frame"
        for im in images[2:]:
            im.tooltip = "ref2va: a reference picture"
        for i, v in enumerate(videos):
            v.tooltip = (f"ref2va: reference clip {i + 1}; its soundtrack goes in "
                         f"video_audio_{i + 1}")
        for i, a in enumerate(tracks):
            a.tooltip = f"ref2va: the soundtrack of video_{i + 1}"
        for i, a in enumerate(audios):
            a.tooltip = f"ref2va: standalone reference audio {i + 1}, such as a voice"
        return io.Schema(
            node_id="RadianceMiniMax",
            display_name="Radiance MiniMax",
            category="video/radiance",
            description=DESCRIPTION,
            inputs=[
                io.String.Input("server_url", default="http://127.0.0.1:8133",
                                tooltip="the radiance server, http://<host>:<port>"),
                io.String.Input("model", default="minimax-h3",
                                tooltip="the model id the server serves (GET /v1/models)"),
                io.Combo.Input("loras", options=["none"], default="none", display_name="lora 1",
                               tooltip="an adapter the server offers (GET /v1/loras); 'none' "
                                       "is off"),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               display_name="lora 1 strength"),
                io.Combo.Input("lora2", options=["none"], default="none", display_name="lora 2",
                               tooltip="a second adapter; the server needs --max-loras 2, "
                                       "its default with --lora-dir"),
                io.Float.Input("lora2_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               display_name="lora 2 strength"),
                io.Combo.Input("task", options=["t2va", "i2va", "l2va", "fl2va", "ref2va"],
                               tooltip="which media inputs are used: see the node's description"),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="a connected string replaces the typed text"),
                # DynamicCombo.Input takes no default; web/radiance_size.js sets new nodes to 480p.
                io.DynamicCombo.Input(
                    "resolution",
                    options=[
                        *(io.DynamicCombo.Option(t, []) for t in TIERS),
                        io.DynamicCombo.Option("custom", [
                            io.String.Input("custom_size", display_name="WIDTHxHEIGHT",
                                            default="512x512"),
                        ]),
                    ],
                    tooltip="the short side; the aspect gives the long side, and the node's "
                            "title shows the size. custom takes any WxH and adjusts it to the "
                            "model's sizes"),
                io.Combo.Input("aspect", options=ASPECTS, default="16:9",
                               tooltip="width:height for a tier; a custom size ignores it"),
                io.Int.Input("seconds", default=12, min=1, max=120, step=1),
                io.Int.Input("seed", default=1234, min=0, max=2**53 - 1, step=1,
                             tooltip="up to 2^53 - 1, the largest seed the server holds exactly"),
                io.Int.Input("num_inference_steps", default=25, min=1, max=200, step=1),
                io.Combo.Input("sampler", options=["euler", "res_multistep"]),
                io.Float.Input("shift", default=12.0, min=0.0, max=100.0, step=0.5),
                io.Float.Input("audio_shift", default=3.0, min=0.0, max=100.0, step=0.5),
                io.Combo.Input("latent_upscale", options=["off", "2", "4"], default="off",
                               advanced=True,
                               tooltip="run the early steps at 1/2 or 1/4 of the size, with a "
                                       "container that has the latent upscaler"),
                *images, *videos, *tracks, *audios,
                io.Int.Input("timeout", default=1800, min=30, max=7200, step=30, advanced=True,
                             tooltip="seconds to wait for the job"),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("file", display_name="file",
                                 tooltip="the saved clip's path in the output folder"),
                io.String.Output("job_id", display_name="job id"),
            ],
        )

    @classmethod
    def validate_inputs(cls, loras, lora2):
        """The adapter lists are the server's, filled in by the frontend, so a chosen adapter is
        not among the schema's options. ComfyUI checks an input against its options only when
        validate_inputs does not take it; the server checks the names."""
        return True

    @classmethod
    def execute(cls, server_url, model, task, prompt, resolution, aspect, seconds, seed,
                num_inference_steps, sampler, shift, audio_shift, timeout=1800,
                latent_upscale="off", loras="none", lora_strength=1.0, lora2="none",
                lora2_strength=1.0, **media):
        size = _resolve_size(resolution, aspect, media.pop("custom_size", ""))
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("the prompt is empty")

        images = [media.get(f"image_{i}") for i in range(1, 10)]
        videos = [media.get(f"video_{i}") for i in range(1, 4)]
        tracks = [media.get(f"video_audio_{i}") for i in range(1, 4)]
        audios = [media.get(f"audio_{i}") for i in range(1, 4)]

        parts = []  # (form field, (file name, bytes, media type))
        if task == "ref2va":
            n_img = _leading(images, "image")
            n_vid = _leading(videos, "video")
            n_aud = _leading(audios, "audio")
            for i, a in enumerate(tracks):
                if a is not None and videos[i] is None:
                    raise ValueError(f"video_audio_{i + 1} is the soundtrack of video_{i + 1}, "
                                     "which is not connected")
            parts += [(f"ref_image_{i + 1}", _image_png(images[i])) for i in range(n_img)]
            parts += [(f"ref_video_{i + 1}", _video_mp4(videos[i])) for i in range(n_vid)]
            parts += [(f"ref_video_audio_{i + 1}", _audio_wav(a))
                      for i, a in enumerate(tracks) if a is not None]
            parts += [(f"ref_audio_{i + 1}", _audio_wav(audios[i])) for i in range(n_aud)]
            if not parts:
                raise ValueError("ref2va needs at least one picture, clip or audio")
        else:
            fields = KEYFRAMES.get(task, [])
            unused = [f"image_{i + 1}" for i, x in enumerate(images)
                      if x is not None and i >= len(fields)]
            unused += [f"{k}_{i + 1}" for k, xs in (("video", videos), ("video_audio", tracks),
                                                    ("audio", audios))
                       for i, x in enumerate(xs) if x is not None]
            if unused:
                logging.warning(f"[{LOG}] {task} does not use {', '.join(unused)}")
            parts += [(f, _image_png(images[i])) for i, f in enumerate(fields)
                      if images[i] is not None]

        data = {
            "model": model, "task": task, "prompt": prompt, "size": size,
            "seconds": str(seconds), "seed": str(seed),
            "num_inference_steps": str(num_inference_steps), "sampler": sampler,
            "shift": str(shift), "audio_shift": str(audio_shift),
        }
        if latent_upscale != "off":
            data["latent_upscale"] = latent_upscale
        adapters = [(name, strength) for name, strength in ((loras, lora_strength),
                                                             (lora2, lora2_strength))
                    if name and name != "none"]
        if len(adapters) == 2 and adapters[0][0] == adapters[1][0]:
            raise ValueError(f"lora 1 and lora 2 are both {adapters[0][0]}")
        if adapters:
            data["loras"] = [name if strength == 1.0 else f"{name}:{strength:g}"
                             for name, strength in adapters]

        # Media go as a multipart form, whose list fields are JSON text; without media, JSON.
        base = server_url.rstrip("/")
        s = requests.Session()
        try:
            if parts:
                form = {k: json.dumps(v) if isinstance(v, list) else v for k, v in data.items()}
                r = s.post(f"{base}/v1/videos", data=form, files=parts, timeout=300)
            else:
                r = s.post(f"{base}/v1/videos", json=data, timeout=300)
        except requests.ConnectionError as e:
            raise RuntimeError(f"cannot reach the radiance server at {base}: {e}") from e
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", {}).get("message") or r.text[:400]
            except ValueError:
                msg = r.text[:400]
            raise RuntimeError(f"the server refused the job ({r.status_code}): {msg}")
        job_id = r.json()["id"]

        bar = _progress_bar(cls)
        t0 = time.time()
        pace = _Pace()
        last_key, last_line = None, 0.0
        while True:
            jr = s.get(f"{base}/v1/videos/{job_id}", timeout=60)
            jr.raise_for_status()
            j = jr.json()
            status = j.get("status")
            if status == "completed":
                break
            if status in ("failed", "cancelled"):
                err = j.get("error")
                err = err.get("message", "") if isinstance(err, dict) else str(err or "")
                raise RuntimeError(f"job {job_id} {status}: {err or 'no reason given'}")
            p = j.get("progress")
            if isinstance(p, (int, float)) and bar is not None:
                bar.update_absolute(min(max(float(p), 0.0), 100.0))
            # A line when a step finishes or the status changes, and every 30 s otherwise.
            now = time.time()
            sps = pace.update(j, now)
            key = (status, j.get("steps_done"))
            if key != last_key or now - last_line >= KEEPALIVE_SECONDS:
                logging.info(f"[{LOG}] {job_id}: {_progress_line(j, now - t0, sps)}")
                last_key, last_line = key, now
            if now - t0 > timeout:
                raise RuntimeError(f"job {job_id} still {status} after {timeout} s")
            time.sleep(POLL_SECONDS)

        cr = s.get(f"{base}/v1/videos/{job_id}/content", timeout=600)
        cr.raise_for_status()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            "radiance/Radiance", folder_paths.get_output_directory(), 0, 0)
        file = f"{name}_{counter:05}_.mp4"
        path = os.path.join(folder, file)
        with open(path, "wb") as f:
            f.write(cr.content)
        if bar is not None:
            bar.update_absolute(100.0)
        return io.NodeOutput(
            VideoFromFile(path),
            os.path.join(subfolder, file) if subfolder else file,
            job_id,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )
