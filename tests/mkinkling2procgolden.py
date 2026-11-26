"""mkinkling2procgolden.py OUT.h -- goldens for tests/mm_test.cpp's Inkling 2 processor cases,
every number from the reference itself: transformers' own Inkling2ImageProcessor (PR #51050), run
on synthetic splitmix pictures (mm_test's pix), plus its convert_to_rgb on a picture with alpha.

  * the picture table: each picture's grid in merged squares (image_grid_thw), the first 16 values
    of its patch plane and the plane's FNV-1a. The plane is the processor's f32 pixel_values cast
    to bfloat16 and laid out as radiance stages it: one zero row (the class token's), then a row
    per patch in merge-block order, each row (y, x, channel);
  * the alpha case: a picture whose alpha is pix(1, y, x, 0), composited onto the reference's
    checkerboard; the RGB bytes' FNV-1a;
  * the flat case: a 50 x 90 picture whose every pixel is (37, 128, 255); the three channels'
    values and the padding's;
  * the file cases: tests/data/media's quad-alpha.png (alpha kept, as the processor takes a PIL
    picture with transparency) and quad.png (opaque) through the whole processor; each plane's
    FNV-1a.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

U = np.uint64


def pix(f, h, w):
    y = np.arange(h, dtype=U)[:, None]
    x = np.arange(w, dtype=U)[None, :]
    c = np.arange(3, dtype=U)[None, None, :]
    with np.errstate(over="ignore"):
        z = (U(f) * U(0x9E3779B97F4A7C15) + y[:, :, None] * U(0xBF58476D1CE4E5B9) +
             x[:, :, None] * U(0x94D049BB133111EB) + c * U(0x2545F4914F6CDD1D) + U(12345))
        z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
        return ((z ^ (z >> U(31))) & U(255)).astype(np.uint8)


def fnv(data, width):
    h = 0xcbf29ce484222325
    for v in np.asarray(data).ravel(order="C"):
        for b in range(width):
            h = ((h ^ ((int(v) >> (8 * b)) & 0xFF)) * 0x100000001b3) & ((1 << 64) - 1)
    return h


proc = Inkling2ImageProcessor()
P, M = 20, 2


def plane_of(img):
    """The reference's pixel_values in bf16, as radiance stages them, and the grid."""
    out = proc(images=[img], return_tensors="pt")
    pv, grid = out["pixel_values"], out["image_grid_thw"][0].tolist()   # [N, 40, 40, 3] f32
    bf = pv.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
    rows = [np.zeros(3 * P * P, dtype=np.uint16)]
    for t in range(bf.shape[0]):
        for mh in range(M):
            for mw in range(M):
                rows.append(bf[t, mh * P:(mh + 1) * P, mw * P:(mw + 1) * P, :].reshape(-1))
    return np.concatenate(rows), grid


pics = []
for f, h, w in ((0, 33, 61), (0, 40, 40), (2, 81, 120), (3, 1, 1), (4, 7, 95)):
    img = torch.from_numpy(pix(f, h, w)).permute(2, 0, 1).contiguous()   # uint8 CHW
    plane, grid = plane_of(img)
    pics.append((f, h, w, grid[1], grid[2], plane[3 * P * P:3 * P * P + 16], fnv(plane, 2)))

media = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "media")
file_fnv = [fnv(plane_of(Image.open(os.path.join(media, n)))[0], 2)
            for n in ("quad-alpha.png", "quad.png")]

# alpha: rgb pix(0), alpha pix(1)'s channel 0, through the reference's convert_to_rgb
ah, aw = 37, 29
rgba = np.concatenate([pix(0, ah, aw), pix(1, ah, aw)[:, :, :1]], axis=2)
t = torch.from_numpy(rgba).permute(2, 0, 1).contiguous()
rgb = proc.convert_to_rgb(t).permute(1, 2, 0).contiguous().numpy()
alpha_fnv = fnv(rgb, 1)

# flat: 2 x 3 squares, the last one's bottom right corner is padding
flat = torch.tensor([37, 128, 255], dtype=torch.uint8)[:, None, None].expand(3, 50, 90).contiguous()
fv = proc(images=[flat], return_tensors="pt")["pixel_values"]
fv = fv.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
flat_in, flat_pad = fv[0, 0, 0], fv[-1, -1, -1]

with open(sys.argv[1], "w") as o:
    o.write("/* inkling2_proc_golden.h -- written by tests/mkinkling2procgolden.py; do not edit. */\n#pragma once\n#include <cstdint>\n\n")
    o.write("struct GoldInk2Image { int32_t f, h, w, rows, cols; uint16_t head[16]; uint64_t fnv; };\n")
    o.write("static const GoldInk2Image kGoldInk2Image[] = {\n")
    for f, h, w, r, c, head, fv in pics:
        o.write(f"    {{ {f}, {h}, {w}, {r}, {c}, {{ {', '.join(str(int(v)) for v in head)} }}, {fv}ull }},\n")
    o.write("};\n")
    o.write(f"static const int32_t kGoldInk2AlphaH = {ah}, kGoldInk2AlphaW = {aw};\n")
    o.write(f"static const uint64_t kGoldInk2AlphaFnv = {alpha_fnv}ull;\n")
    o.write(f"static const uint16_t kGoldInk2FlatIn[3] = {{ {', '.join(str(int(v)) for v in flat_in)} }};\n")
    o.write(f"static const uint16_t kGoldInk2FlatPad[3] = {{ {', '.join(str(int(v)) for v in flat_pad)} }};\n")
    o.write(f"static const uint64_t kGoldInk2FileAlphaFnv = {file_fnv[0]}ull;\n")
    o.write(f"static const uint64_t kGoldInk2FileOpaqueFnv = {file_fnv[1]}ull;\n")
