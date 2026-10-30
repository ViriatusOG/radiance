"""Offline tiny reranker oracle; never run by CMake/CTest.

CPU torch, transformers==5.20.0 and Jinja2. Reuses the independently stored text/vision
weights and the pinned native reranker template; only the new tiny head is generated.
"""

import json
from pathlib import Path

import jinja2
import torch
import transformers
from transformers import (
    Ministral3Config,
    Mistral3Config,
    Mistral3Model,
    PixtralVisionConfig,
)

root = Path(__file__).parent / "data/golden"
text = json.loads((root / "nemotron35.json").read_text())
vision = json.loads((root / "nemotron35_vision.json").read_text())
mm = json.loads((root / "nemotron35_multimodal.json").read_text())
torch.set_num_threads(1)
tc = Ministral3Config(
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=128,
    vocab_size=320,
    rms_norm_eps=1e-5,
    is_causal=False,
    use_cache=False,
    rope_parameters={
        "rope_type": "yarn",
        "rope_theta": 1e6,
        "factor": 16.0,
        "original_max_position_embeddings": 16384,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "llama_4_scaling_beta": 0.1,
    },
)
vc = PixtralVisionConfig(
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=2,
    num_attention_heads=1,
    patch_size=14,
    image_size=1540,
    hidden_act="silu",
    rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
)
config = Mistral3Config(
    text_config=tc,
    vision_config=vc,
    spatial_merge_size=2,
    image_token_index=10,
    multimodal_projector_bias=False,
    projector_hidden_act="gelu",
    vision_feature_layer=-1,
    is_causal=False,
)
model = Mistral3Model(config).eval()
frequencies = {n: b.clone() for n, b in model.named_buffers()}
model.to(torch.bfloat16)
for n, b in frequencies.items():
    parent, attr = n.rsplit(".", 1)
    setattr(model.get_submodule(parent), attr, b)
model.set_attn_implementation("eager")
state = {
    n: torch.tensor(w["bf16"], dtype=torch.uint16)
    .view(torch.bfloat16)
    .reshape(w["shape"])
    for n, w in (text["weights"] | vision["weights"]).items()
}
state["language_model.embed_tokens.weight"] = state[
    "language_model.embed_tokens.weight"
].repeat(10, 1)
model.load_state_dict(state, strict=True)
torch.manual_seed(3508)
head = (torch.randn(1, 16) * 0.2).to(torch.bfloat16)


def pixels(index):
    image = vision["images"][index]
    h, w = image["size"]
    patches = torch.tensor(image["patches"], dtype=torch.uint16).view(torch.bfloat16)
    return (
        patches.reshape(h // 14, w // 14, 3, 14, 14)
        .permute(2, 0, 3, 1, 4)
        .reshape(3, h, w)
    )


def score(inputs, offsets=None):
    length = max(len(x["tokens"]) for x in inputs)
    ids = torch.full((len(inputs), length), 11, dtype=torch.long)
    mask = torch.zeros_like(ids)
    positions = torch.arange(length)[None].expand(len(inputs), -1).clone()
    images = []
    for i, x in enumerate(inputs):
        ids[i, : len(x["tokens"])] = torch.tensor(x["tokens"])
        mask[i, : len(x["tokens"])] = 1
        if offsets:
            positions[i] += offsets[i]
        images.extend(pixels(p["image"]) for p in x["placements"])
    kw = {}
    if images:
        sizes = torch.tensor([list(x.shape[1:]) for x in images])
        h, w = sizes.max(0).values.tolist()
        kw = {
            "pixel_values": torch.stack(
                [
                    torch.nn.functional.pad(x, (0, w - x.shape[2], 0, h - x.shape[1]))
                    for x in images
                ]
            ),
            "image_sizes": sizes,
        }
    with torch.inference_mode():
        hidden = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            **kw,
        ).last_hidden_state
        pooled = (
            hidden.masked_fill(~mask[..., None].bool(), 0).sum(1) / mask.sum(1)[:, None]
        )
        return (
            torch.nn.functional.linear(pooled.float(), head.float()).flatten().tolist()
        )


cases = []
for c in text["cases"]:
    inputs = [{"tokens": x, "placements": []} for x in c["inputs"]]
    cases.append(
        {
            **c,
            "scores": score(inputs, c["offsets"]),
            "head_only_scores": torch.nn.functional.linear(
                torch.tensor(c["pooled"]), head.float()
            )
            .flatten()
            .tolist(),
        }
    )
multimodal = []
for c in mm["cases"]:
    if c["name"] in ["image_only", "image_text", "text_and_image_batch"]:
        multimodal.append({**c, "scores": score(c["inputs"])})
env = jinja2.Environment()


def fail(s):
    raise ValueError(s)


env.globals["raise_exception"] = fail
template = env.from_string(
    (root.parent / "chat/nemotron35/rerank-template.jinja").read_text()
)
letter_ids = dict(zip("ABCDEF", [3, 4, 7, 8, 9, 6]))


def encode(s):
    return [5 if b == 32 else letter_ids.get(chr(b), 32 + b) for b in s.encode("utf-8")]


http = []
for name, q, doc, image in [
    ("text", "C", "A", None),
    ("controls", "[IMG] <A>", "CDF", None),
    ("image", "C", None, 1),
    ("image_text", "C", "CDF", 2),
    ("image_unicode_whitespace", "C", "CDF\u00a0\t\u3000", 2),
]:
    content = (
        doc
        if image is None
        else ([{"type": "text", "text": doc}] if doc else []) + [{"type": "image"}]
    )
    rendered = template.render(
        messages=[
            {"role": "query", "content": q},
            {"role": "document", "content": content},
        ]
    )
    tokens = []
    placements = []
    parts = rendered.split("[IMG]") if image is not None else [rendered]
    for i, part in enumerate(parts):
        if i:
            placements.append({"image": image, "tok": len(tokens)})
            tokens.extend(vision["images"][image]["expand"])
        tokens.extend(encode(part))
    x = {"tokens": tokens, "placements": placements}
    http.append(
        {
            "name": name,
            "query": q,
            "document": doc,
            "image": image,
            "rendered": rendered,
            "inputs": [x],
            "scores": score([x]),
        }
    )
# Independent component data explicitly exercises the two BF16 roundings, including pitches/tails.
h = torch.randn(10, 7).to(torch.bfloat16)
cu = [0, 3, 10]
means = torch.stack([(h[b:e].sum(0) / (e - b)) for b, e in zip(cu, cu[1:])])
result = {
    "provenance": {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "seed": 3508,
        "attention": "eager",
        "dtype": "bfloat16",
        "head_arithmetic": "float32",
        "template_revision": "f88f8d9104726747791283ab156d2fcdb29e90a7",
    },
    "head": {"shape": [1, 16], "bf16": head.flatten().view(torch.uint16).tolist()},
    "cases": cases,
    "multimodal": multimodal,
    "http": http,
    "pool": {
        "shape": [10, 7],
        "bf16": h.flatten().view(torch.uint16).tolist(),
        "cu": cu,
        "expected": means.flatten().view(torch.uint16).tolist(),
    },
}
path = root / "nemotron35_rerank.json"
path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
print(path, path.stat().st_size)
