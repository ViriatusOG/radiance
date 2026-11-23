#!/usr/bin/env python3
"""Regenerate golden.json -- the reference numbers arch/minimaxh3/rad_ref2va_pack.cpp is tested
against (tests/ref2va_pack_test.cpp).

    gen_golden.py /path/to/ComfyUI/comfy/ldm/minimax/model.py [out.json]

The reference implementation lives in the ComfyUI tree; importing its package pulls the whole
device stack, so this extracts exactly the layout's source -- the four constants, the grid helpers
and the PackedLayout class -- with `ast`, execs them on torch alone, and runs the cases below.
Nothing here is a re-derivation: every number in golden.json is computed by the reference code
itself, at float64, from the file it is given. When a new version of the reference changes
golden.json, the port changes with it.
"""

import ast
import json
import sys

NEED = ["_axis_from_sqrt_area", "_frame_grid", "_video_t_spans", "_video_t_grid", "_ref_t_span",
        "_audio_grid", "_video_grid", "PackedLayout"]
NEED_ASSIGN = ["FRAME_PER_TOKEN", "FRAME_RESCALE", "VISUAL_COND_TIMESTEP", "AUDIO_COND_TIMESTEP"]


def dump(v, f, ind=0):
    """JSON with an object a key a line and every list on one line, so a golden diffs by key."""
    if isinstance(v, dict) and v:
        pad = " " * (ind + 1)
        items = list(v.items())
        f.write("{\n")
        for i, (k, x) in enumerate(items):
            f.write(pad + json.dumps(k) + ": ")
            dump(x, f, ind + 1)
            f.write(",\n" if i + 1 < len(items) else "\n")
        f.write(" " * ind + "}")
    else:
        f.write(json.dumps(v, separators=(", ", ": ")))


def extract(path):
    """The reference's layout source, exec'd with torch/math only."""
    import math
    import torch
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    picked, got = [], set()
    for node in tree.body:
        names = ([t.id for t in node.targets if isinstance(t, ast.Name)]
                 if isinstance(node, ast.Assign) else
                 [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [])
        if any(n in NEED or n in NEED_ASSIGN for n in names):
            picked.append(ast.get_source_segment(src, node))
            got.update(names)
    missing = (set(NEED) | set(NEED_ASSIGN)) - got
    if missing:
        sys.exit(f"{path} does not define: {sorted(missing)} -- reference moved?")
    ns = {"math": math, "torch": torch}
    exec("\n\n".join(picked), ns)
    return ns


CASES = {
    # plain t2va: the pack keeps the three segments [text | audio | video] exactly
    "t2va": dict(text_len=5, target=(3, 8, 10, 4), keyframes=[], refs=[]),
    # fl2va keyframes at frame 0 (shares the target origin) and mid-timeline, video-only and both
    "fl2va": dict(text_len=4, target=(3, 8, 8, 3),
                  keyframes=[dict(frame=0, video_t=1, audio_t=2), dict(frame=7, video_t=2, audio_t=0)],
                  refs=[]),
    # every reference kind in one request: a photo, a standalone soundtrack, a silent clip, and a
    # clip with sound (its audio packs before its video, at the clip's own width extremes); a
    # second photo after them; audio_t=0 targets the silent-video edge
    "ref2va_all": dict(text_len=6, target=(2, 8, 8, 0),
                       keyframes=[],
                       refs=[dict(kind="image", latent_h=6, latent_w=10, latent_t=0, ref_audio_t=0),
                             dict(kind="audio", latent_h=0, latent_w=0, latent_t=0, ref_audio_t=3),
                             dict(kind="video", latent_h=4, latent_w=6, latent_t=2, ref_audio_t=0),
                             dict(kind="video_audio", latent_h=5, latent_w=7, latent_t=3,
                                  ref_audio_t=4),
                             dict(kind="image", latent_h=4, latent_w=4, latent_t=0, ref_audio_t=0)]),
    # a request at plausible sizes: a keyframe, a large photo and a clip with sound
    "chain": dict(text_len=32, target=(7, 16, 24, 6),
                  keyframes=[dict(frame=0, video_t=1, audio_t=2)],
                  refs=[dict(kind="image", latent_h=8, latent_w=12, latent_t=0, ref_audio_t=0),
                        dict(kind="video_audio", latent_h=4, latent_w=6, latent_t=5,
                             ref_audio_t=20)]),
    # a song guide (MiniMaxH3AddGuide's audio alone) mid-video after a first frame, with
    # references: its rows sit on the video's clock, 5/3 a frame after every reference's span
    "guide": dict(text_len=12, target=(7, 4, 6, 20),
                  keyframes=[dict(frame=0, video_t=1, audio_t=0), dict(frame=7, video_t=0, audio_t=8)],
                  refs=[dict(kind="image", latent_h=4, latent_w=4, latent_t=0, ref_audio_t=0),
                        dict(kind="video_audio", latent_h=4, latent_w=6, latent_t=2, ref_audio_t=3),
                        dict(kind="audio", latent_h=0, latent_w=0, latent_t=0, ref_audio_t=2)]),
    # degenerate prompt: text_len 0, nothing but targets
    "no_text": dict(text_len=0, target=(2, 4, 6, 1), keyframes=[], refs=[]),
}


def main():
    import torch
    model_py = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "golden.json"
    ns = extract(model_py)
    PackedLayout = ns["PackedLayout"]
    golden = {}
    for name, c in CASES.items():
        t, ah, aw, at = c["target"]
        refs = [dict(kind=r["kind"], latent_h=r["latent_h"], latent_w=r["latent_w"],
                     latent_t=r["latent_t"], ref_audio_t=r["ref_audio_t"]) for r in c["refs"]]
        # the class reads only the latents' shapes (video .shape[2], audio .shape[-1]); zeros do
        kfs = [dict(resolved_frame_index=k["frame"],
                    latent=torch.zeros(1, 24, k["video_t"], ah, aw) if k["video_t"] else None,
                    audio_latent=torch.zeros(1, 32, 2, k["audio_t"]) if k["audio_t"] else None)
               for k in c["keyframes"]]
        lay = PackedLayout(c["text_len"], t, ah, aw, at, keyframes=kfs or None, refs=refs or None)
        golden[name] = {
            "text_len": c["text_len"], "latent_t": t, "latent_h": ah, "latent_w": aw,
            "audio_t": at,
            "keyframes": c["keyframes"], "refs": c["refs"],
            "seq_len": lay.seq_len,
            "segments": [[s, e, k] for (s, e, k) in lay.segments],
            "positions": [[float(x) for x in row] for row in lay.position_ids.tolist()],
            "img_update": [bool(b) for b in lay.img_update.tolist()],
            "audio_update": [bool(b) for b in lay.audio_update.tolist()],
            "img_pos": [int(i) for i in lay.img_pos.tolist()],
            "audio_pos": [int(i) for i in lay.audio_pos.tolist()],
        }
    with open(out, "w") as f:
        dump(golden, f)
        f.write("\n")
    print(f"{out}: " + ", ".join(f"{n} {g['seq_len']} rows" for n, g in golden.items()))


main()
