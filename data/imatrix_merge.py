#!/usr/bin/env python3
"""imatrix_merge.py -- add the fused gate_up entries an HF release needs to an imatrix.

DeepSeek-V4.1's release stores routed experts per expert and its shared expert as separate gate/up
weights, so the release's own imatrix (llama.cpp's collector output, run on the original checkpoint)
holds `<stem>.ffn_gate_exps` and `<stem>.ffn_up_exps`. Radiance's recipe fuses them
(`ffn_gate_up_exps.*`, `ffn_gate_up_shexp`), and the quantiser looks the importance up by the fused
name. This tool reads the imatrix, sums the two halves (the fused weight's input columns are the
gate's and the up's: llama.cpp sums them for fused tensors too), and writes a new imatrix with those
entries added. Everything else is carried through unchanged.

usage: python3 data/imatrix_merge.py IMATRIX IN OUT
"""
import sys
import numpy as np
import gguf


def load(path):
    r = gguf.GGUFReader(path)
    tensors = {}
    for t in r.tensors:
        tensors[t.name] = np.array(t.data, dtype=np.float32, copy=True) if t.tensor_type == 0 else None
    return r, tensors


def base_parts(name):
    if name.endswith(".in_sum2"):
        return name[:-8]
    if name.endswith(".counts"):
        return name[:-7]
    return None


def main() -> None:
    im, out = sys.argv[1], sys.argv[3]
    r, tensors = load(im)
    if any(v is None for v in tensors.values()):
        sys.exit("every imatrix tensor must be f32")
    pairs = {}
    for name in tensors:
        base = base_parts(name)
        if not base or not base.endswith(".weight"):
            continue
        stem_name = base[: -len(".weight")]
        for suf, key in ((".ffn_gate_exps", "gate"), (".ffn_up_exps", "up"),
                         (".ffn_gate_shexp", "gate_sh"), (".ffn_up_shexp", "up_sh")):
            if stem_name.endswith(suf):
                stem = stem_name[: -len(suf)]
                pairs.setdefault(stem, {})[key] = base
    added = {}
    for stem, p in pairs.items():
        if "gate" in p and "up" in p:
            g2, u2 = tensors[p["gate"] + ".in_sum2"], tensors[p["up"] + ".in_sum2"]
            gc, uc = tensors[p["gate"] + ".counts"], tensors[p["up"] + ".counts"]
            if g2.shape != u2.shape:
                sys.exit(f"{stem}: gate {g2.shape} != up {u2.shape}")
            added[f"{stem}.ffn_gate_up_exps.weight.in_sum2"] = g2 + u2
            added[f"{stem}.ffn_gate_up_exps.weight.counts"] = gc + uc
        if "gate_sh" in p and "up_sh" in p:
            g2, u2 = tensors[p["gate_sh"] + ".in_sum2"], tensors[p["up_sh"] + ".in_sum2"]
            gc, uc = tensors[p["gate_sh"] + ".counts"], tensors[p["up_sh"] + ".counts"]
            if g2.shape != u2.shape:
                sys.exit(f"{stem}: shexp gate {g2.shape} != up {u2.shape}")
            added[f"{stem}.ffn_gate_up_shexp.weight.in_sum2"] = g2 + u2
            added[f"{stem}.ffn_gate_up_shexp.weight.counts"] = gc + uc
    n_layers = len({k.split(".")[1] for k in pairs})
    w = gguf.GGUFWriter(out, "imatrix")
    w.add_int32("imatrix.version", 2)
    for name, v in tensors.items():
        w.add_tensor(name, v, raw_shape=v.shape, raw_dtype=gguf.GGMLQuantizationType.F32)
    for name, v in added.items():
        w.add_tensor(name, v, raw_shape=v.shape, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{out}: {len(tensors)} carried, {len(added)} fused gate_up entries for {n_layers} layers")


if __name__ == "__main__":
    main()
