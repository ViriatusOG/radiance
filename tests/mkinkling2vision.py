"""mkinkling2vision.py -- a float32 reference of Inkling 2 Mini's vision tower.

    mkinkling2vision.py tiny OUT.safetensors [--seed N]
        A tiny random checkpoint with the released tensor names (model.visual.*): width 64, 2
        blocks, 4 heads, a 6 x 6 position grid of 4 x 4 patches, merge 2, text width 48. Its
        weights are stored bf16, as the released container stores them.

    mkinkling2vision.py golden CKPT OUT.h [--seed N]
        The golden of tests/inkling2_vision_test.cpp: the reference forward of CKPT over a random
        image of the checkpoint's grid, written as tests/inkling2_vision_golden.h. The header holds
        the normalised image, the tower's tokens after its last block (CLS removed), the merged
        cells the projector reads, and the projector's output. A second image on a grid other than
        the table's (RESAMPLE_GRID patches a side, rectangular) follows, with the same three
        arrays, its position table resampled to that grid (see posemb_resample).

The reference follows llama.cpp's graph (tools/mtmd/models/inkling2v.cpp, and the helpers it calls
in mtmd-graph.cpp: build_vit, build_rope_2d, build_patch_merge_permute) op for op, including the
ggml memory order of the patch merge and of the 2-D rotary, which is where a port most easily goes
wrong. The merge is written as the reshape, permute and contiguous calls the graph makes, on the
row-major arrays those calls see, so its feature order is read off the graph rather than assumed.
Nothing here runs llama.cpp.

Run by hand, like the other mk*golden.py; the build and the suite never call it.
"""
import argparse, math, os, sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

torch.set_grad_enabled(False)
F32 = torch.float32

# The tiny geometry. Each value is the released one's shape rule applied to a small width:
# head_dim is width / heads, the rotary's two halves each take head_dim / 4 pairs (llama's
# build_rope_2d with n_dim / 2), and the projector's hidden width is 4 * width as 4096 is 4 * 1024.
TINY = dict(depth=2, width=64, heads=4, patch=4, side=6, merge=2, text=48, eps=1e-5, theta=1e4)
PREFIX = "model.visual."


def tiny_state(cfg, seed):
    """Random weights with the released names, shapes and parameter roles."""
    g = torch.Generator().manual_seed(seed)
    W, p, side, ff = cfg["width"], cfg["patch"], cfg["side"], 4 * cfg["width"]
    ph, text = 4 * W, cfg["text"]

    def rnd(*shape, scale=None):
        fan = shape[-1]
        s = scale if scale is not None else 1.0 / math.sqrt(fan)
        return torch.randn(*shape, generator=g) * s

    def near(*shape, base=1.0, s=0.05):
        return base + torch.randn(*shape, generator=g) * s

    sd = {}
    B = PREFIX + "backbone."
    sd[B + "embeddings.class_embedding"] = rnd(W, scale=0.5)
    sd[B + "embeddings.cls_position_embedding"] = rnd(W, scale=0.5)
    sd[B + "embeddings.patch_embedding.weight"] = rnd(W, 3 * p * p)
    sd[B + "embeddings.position_embedding"] = rnd(side * side, W, scale=0.5)
    sd[B + "ln_pre.weight"] = near(W)
    sd[B + "ln_pre.bias"] = rnd(W, scale=0.05)
    for i in range(cfg["depth"]):
        L = B + f"layers.{i}."
        sd[L + "ln_1.weight"] = near(W)
        sd[L + "ln_1.bias"] = rnd(W, scale=0.05)
        sd[L + "ln_2.weight"] = near(W)
        sd[L + "ln_2.bias"] = rnd(W, scale=0.05)
        sd[L + "ls_1.gamma"] = rnd(W, scale=0.3)
        sd[L + "ls_2.gamma"] = rnd(W, scale=0.3)
        for nm in ("q_proj", "k_proj", "v_proj", "out_proj"):
            sd[L + f"self_attn.{nm}.weight"] = rnd(W, W)
            sd[L + f"self_attn.{nm}.bias"] = rnd(W, scale=0.05)
        sd[L + "mlp.c_fc.weight"] = rnd(ff, W)
        sd[L + "mlp.c_fc.bias"] = rnd(ff, scale=0.05)
        sd[L + "mlp.c_proj.weight"] = rnd(W, ff)
        sd[L + "mlp.c_proj.bias"] = rnd(W, scale=0.05)
    sd[PREFIX + "proj_in.weight"] = rnd(ph, ph)  # no bias: projector_bias is false
    sd[PREFIX + "proj_out.weight"] = rnd(text, ph)
    return {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}


def load_weights(path):
    """The checkpoint as float32, the values the reference and the C++ port both start from."""
    return {k: v.to(F32) for k, v in load_file(path).items()}


def layer_norm(x, w, b, eps):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def rope_2d(x, pos_w, pos_h, theta):
    """llama's build_rope_2d on [n, heads, head_dim]: two ggml NORMAL rotaries over the halves of
    each head. The first half turns by pos_w, the second by pos_h, each over adjacent pairs
    (2j, 2j + 1) at theta^(-2j / (head_dim / 2))."""
    n, nh, hd = x.shape
    half = hd // 2
    out = x.clone()
    for h, pos in ((0, pos_w), (1, pos_h)):
        for j in range(half // 2):
            f = theta ** (-2.0 * j / half)
            ang = pos.to(F32) * f
            c, s = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
            i0, i1 = h * half + 2 * j, h * half + 2 * j + 1
            x0, x1 = x[..., i0], x[..., i1]
            out[..., i0] = x0 * c - x1 * s
            out[..., i1] = x0 * s + x1 * c
    return out


def posemb_resample(table, S, th, tw):
    """The S x S position table [S * S, W] as a th x tw table of its own: the posemb_resample op's
    formula, written out. Per axis the coordinate is S * (i + 0.5) / T - 0.5 in f32, clamped to
    [0, S - 1]; the taps are floor(coord) and the next one clamped at S - 1, weighted by the fraction
    and its complement. The four taps are summed in f32 as top-left, top-right, bottom-left,
    bottom-right, each weighted by the product of its row and column weights. align_corners=False
    with the edge taps clamped; torch's F.interpolate(bilinear, align_corners=False) takes the same
    coordinates. Row i_h * tw + i_w."""

    def axis(T):
        i = torch.arange(T, dtype=F32)
        c = (S * (i + 0.5)) / T - 0.5
        c = c.clamp(0.0, float(S - 1))
        f = torch.floor(c)
        lo = f.long()
        hi = (lo + 1).clamp(max=S - 1)
        w1 = c - f
        return lo, hi, 1.0 - w1, w1

    lh, hh, wh0, wh1 = axis(th)
    lw, hw, ww0, ww1 = axis(tw)
    t = table.reshape(S, S, -1)

    def tap(hi_, lo_):
        return t[hi_[:, None], lo_[None, :]]          # [th, tw, W]

    wh0, wh1 = wh0[:, None, None], wh1[:, None, None]
    ww0, ww1 = ww0[None, :, None], ww1[None, :, None]
    y = ((wh0 * ww0) * tap(lh, lw) + (wh0 * ww1) * tap(lh, hw)
         + (wh1 * ww0) * tap(hh, lw) + (wh1 * ww1) * tap(hh, hw))
    return y.reshape(th * tw, -1)


def merge_ggml(tok, th, tw, s):
    """llama's build_patch_merge_permute on tokens [th * tw, W] in raster order. Each step is the
    graph's call on the row-major array it sees: numpy-style axes are [y, x, c] with c fastest."""
    W = tok.shape[1]
    A = tok.reshape(th, tw, W)                      # ggml [c, x, y]
    B = A.reshape(th, tw // s, s * W)               # ggml_reshape_3d(n*s, W/s, H): [y, x/s, (dx, c)]
    B = B.permute(1, 0, 2)                          # ggml_permute(0, 2, 1, 3): [x/s, y, (dx, c)]
    C = B.contiguous().reshape(tw // s, th // s, s * s * W)      # ggml_cont_3d: [x/s, y/s, (dy, dx, c)]
    D = C.permute(1, 0, 2)                          # ggml_permute(0, 2, 1, 3): [y/s, x/s, k]
    return D.contiguous().reshape((th // s) * (tw // s), s * s * W)


def vit(sd, cfg, img):
    """The tower up to llama's encoder_out: returns tokens [th * tw, W] (CLS removed), for the grid
    of img [th * p, tw * p, 3]."""
    W, p, side, eps, theta = cfg["width"], cfg["patch"], cfg["side"], cfg["eps"], cfg["theta"]
    nh = cfg["heads"]
    hd = W // nh
    th, tw = img.shape[0] // p, img.shape[1] // p
    B = PREFIX + "backbone."
    # build_inp: the patch weight stored HWC is converted to a conv2d kernel [out, 3, p, p] (the
    # converter's reshape and permute), then applied at stride p.
    wk = sd[B + "embeddings.patch_embedding.weight"].reshape(W, p, p, 3).permute(0, 3, 1, 2)
    x = F.conv2d(img.permute(2, 0, 1)[None], wk, stride=p)[0]        # [W, th, tw]
    x = x.reshape(W, th * tw).t()                                    # raster patches [n, W]
    table = sd[B + "embeddings.position_embedding"]
    if (th, tw) == (side, side):
        x = x + table                                                # the table's own grid
    else:
        x = x + posemb_resample(table, side, th, tw)                 # the table on this grid
    cls = (sd[B + "embeddings.class_embedding"] + sd[B + "embeddings.cls_position_embedding"])
    seq = torch.cat([cls[None], x], 0)                               # [n + 1, W]

    # Positions are 1-indexed on the patch grid and the CLS sits at (0, 0), as the mtmd encoder fills
    # pos_w and pos_h.
    idx = torch.arange(th * tw)
    pos_w = torch.cat([torch.zeros(1), (idx % tw + 1).float()])
    pos_h = torch.cat([torch.zeros(1), (idx // tw + 1).float()])

    seq = layer_norm(seq, sd[B + "ln_pre.weight"], sd[B + "ln_pre.bias"], eps)
    for i in range(cfg["depth"]):
        L = B + f"layers.{i}."
        h = layer_norm(seq, sd[L + "ln_1.weight"], sd[L + "ln_1.bias"], eps)
        q = h @ sd[L + "self_attn.q_proj.weight"].t() + sd[L + "self_attn.q_proj.bias"]
        k = h @ sd[L + "self_attn.k_proj.weight"].t() + sd[L + "self_attn.k_proj.bias"]
        v = h @ sd[L + "self_attn.v_proj.weight"].t() + sd[L + "self_attn.v_proj.bias"]
        n = seq.shape[0]
        q = rope_2d(q.reshape(n, nh, hd), pos_w, pos_h, theta)
        k = rope_2d(k.reshape(n, nh, hd), pos_w, pos_h, theta)
        v = v.reshape(n, nh, hd)
        att = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd), dim=-1)
        o = torch.einsum("hqk,khd->qhd", att, v).reshape(n, W)
        o = o @ sd[L + "self_attn.out_proj.weight"].t() + sd[L + "self_attn.out_proj.bias"]
        seq = seq + sd[L + "ls_1.gamma"] * o
        h = layer_norm(seq, sd[L + "ln_2.weight"], sd[L + "ln_2.bias"], eps)
        m = F.gelu(h @ sd[L + "mlp.c_fc.weight"].t() + sd[L + "mlp.c_fc.bias"])  # exact erf
        m = m @ sd[L + "mlp.c_proj.weight"].t() + sd[L + "mlp.c_proj.bias"]
        seq = seq + sd[L + "ls_2.gamma"] * m
    return seq[1:]


def reference(sd, cfg, img):
    """The whole tower: returns (tokens, merged, out). img is [H, W, 3] normalised to [-1, 1]."""
    tok = vit(sd, cfg, img)
    merged = merge_ggml(tok, img.shape[0] // cfg["patch"], img.shape[1] // cfg["patch"], cfg["merge"])
    hidden = F.gelu(merged @ sd[PREFIX + "proj_in.weight"].t())     # mm_1, GELU_ERF
    out = hidden @ sd[PREFIX + "proj_out.weight"].t()              # mm_2, no bias
    return tok, merged, out


def normalised_image(cfg, seed, th=None, tw=None, stream=1):
    """A random 8-bit image on the grid, normalised the processor's way (rescale, mean and std 0.5).
    The grid defaults to the table's side square; `stream` picks the random stream."""
    side, p = cfg["side"], cfg["patch"]
    th, tw = th or side, tw or side
    g = torch.Generator().manual_seed(seed + stream)
    u8 = torch.randint(0, 256, (th * p, tw * p, 3), generator=g, dtype=torch.int64)
    return ((u8.to(F32) * (1.0 / 255.0)) - 0.5) / 0.5


def fmt(vals):
    return ", ".join(f"{v:.9g}" for v in vals.reshape(-1).tolist())


# The second golden's grid: rectangular, and not the table's side, so the table is resampled.
RESAMPLE_GRID = (8, 10)


def write_golden(path, cfg, img, tok, merged, out, rimg=None, rtok=None, rout=None):
    side, p, W, text = cfg["side"], cfg["patch"], cfg["width"], cfg["text"]
    lines = [
        "/* inkling2_vision_golden.h -- GENERATED by tests/mkinkling2vision.py golden from the tiny",
        " * checkpoint tests/data/golden/inkling2_vision_tiny.safetensors (float32 reference); do not edit. */",
        "#pragma once",
        "#include <cstdint>",
        "",
        f"static const int32_t kGoldInkVisSide = {side}, kGoldInkVisPatch = {p}, kGoldInkVisMerge = {cfg['merge']};",
        f"static const int32_t kGoldInkVisWidth = {W}, kGoldInkVisHeads = {cfg['heads']}, kGoldInkVisDepth = {cfg['depth']};",
        f"static const int32_t kGoldInkVisText = {text}, kGoldInkVisTokens = {out.shape[0]};",
        f"static const int32_t kGoldInkVisImageH = {img.shape[0]}, kGoldInkVisImageW = {img.shape[1]};",
        "",
        "/* The normalised image, row-major [H, W, 3]. */",
        f"static const float kGoldInkVisImage[{img.numel()}] = {{ {fmt(img)} }};",
        "",
        "/* The tower's tokens after its last block with the CLS removed, [side * side, width]. */",
        f"static const float kGoldInkVisTokensOut[{tok.numel()}] = {{ {fmt(tok)} }};",
        "",
        "/* The merged cells the projector reads, [side * side / merge^2, 4 * width]. */",
        f"static const float kGoldInkVisMerged[{merged.numel()}] = {{ {fmt(merged)} }};",
        "",
        "/* The projector's output, [side * side / merge^2, text]. */",
        f"static const float kGoldInkVisOut[{out.numel()}] = {{ {fmt(out)} }};",
        "",
    ]
    if rimg is not None:
        gh, gw = rimg.shape[0] // p, rimg.shape[1] // p
        lines += [
            "/* A second image on a grid other than the table's: the table is resampled to it (the",
            " * posemb_resample op, align_corners off), so these are the resampled-grid golden. */",
            f"static const int32_t kGoldInkVisRGridH = {gh}, kGoldInkVisRGridW = {gw};",
            f"static const int32_t kGoldInkVisRImageH = {rimg.shape[0]}, kGoldInkVisRImageW = {rimg.shape[1]};",
            f"static const float kGoldInkVisRImage[{rimg.numel()}] = {{ {fmt(rimg)} }};",
            "",
            "/* The tower's tokens on that grid after its last block, CLS removed, [gh * gw, width]. */",
            f"static const float kGoldInkVisRTokensOut[{rtok.numel()}] = {{ {fmt(rtok)} }};",
            "",
            "/* The projector's output on that grid, [gh * gw / merge^2, text]. */",
            f"static const float kGoldInkVisROut[{rout.numel()}] = {{ {fmt(rout)} }};",
            "",
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("out")
    t.add_argument("--seed", type=int, default=7)
    g = sub.add_parser("golden")
    g.add_argument("ckpt")
    g.add_argument("out")
    g.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cfg = dict(TINY)
    if a.cmd == "tiny":
        save_file(tiny_state(cfg, a.seed), a.out)
        print(f"wrote {a.out}")
        return 0
    sd = load_weights(a.ckpt)
    img = normalised_image(cfg, a.seed)
    tok, merged, out = reference(sd, cfg, img)
    rth, rtw = RESAMPLE_GRID
    rimg = normalised_image(cfg, a.seed, rth, rtw, stream=2)
    rtok, _, rout = reference(sd, cfg, rimg)
    write_golden(a.out, cfg, img, tok, merged, out, rimg, rtok, rout)
    print(f"wrote {a.out}: tokens {tuple(tok.shape)}, merged {tuple(merged.shape)}, out {tuple(out.shape)}; "
          f"resampled grid {rth}x{rtw}: tokens {tuple(rtok.shape)}, out {tuple(rout.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
