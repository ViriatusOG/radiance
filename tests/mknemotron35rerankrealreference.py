"""Offline pinned-native reranker CPU oracle; never run by CMake or CTest.

Usage: python tests/mknemotron35rerankrealreference.py CHECKPOINT OUTPUT_JSON
Requires CPU torch 2.14.1, transformers 5.20.0 and Pillow 13. No network or remote code.
The built-in Mistral3Model loads the native backbone; the audited 8 KiB native score
head is read separately and applied exactly as NVIDIA's model.py specifies. Download
revision f88f8d9104726747791283ab156d2fcdb29e90a7. A local asset-manifest.json is
optional; otherwise standard Hugging Face metadata records the revision when available.
"""

import argparse
import base64
import hashlib
import importlib
import io
import json
from pathlib import Path

import PIL
import torch
import transformers
from PIL import Image
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer, PixtralProcessor
from transformers.models.pixtral.image_processing_pil_pixtral import (
    PixtralImageProcessorPil,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--sdpa-audit", action="store_true")
a = p.parse_args()
torch.set_num_threads(a.threads)
checkpoint = a.checkpoint.resolve()
metadata = (checkpoint / ".cache/huggingface/download").glob(
    "model-*.safetensors.metadata"
)
revisions = {path.read_text().splitlines()[0] for path in metadata}
if checkpoint.parent.name == "snapshots":
    revisions.add(checkpoint.name)
asset_manifest = checkpoint / "asset-manifest.json"
if asset_manifest.exists():
    assets = json.loads(asset_manifest.read_text())
    for name, entry in assets["files"].items():
        digest = hashlib.file_digest(
            (checkpoint / name).open("rb"), "sha256"
        ).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("Asset hash mismatch: " + name)
    revisions.add(assets["revision"])
if revisions and revisions != {"f88f8d9104726747791283ab156d2fcdb29e90a7"}:
    raise ValueError("Expected the pinned reranker revision, found: " + str(revisions))
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, local_files_only=True, trust_remote_code=False
)
processor = PixtralProcessor(
    tokenizer=tokenizer,
    image_processor=PixtralImageProcessorPil(
        patch_size=14, size={"longest_edge": 1120}
    ),
    patch_size=14,
    spatial_merge_size=2,
    chat_template=(checkpoint / "chat_template.jinja").read_text(),
)
index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
    "weight_map"
]
with safe_open(checkpoint / index["language_model.score.weight"], framework="pt") as f:
    head = f.get_tensor("language_model.score.weight").float()
assert head.shape == (1, 4096)
print("Loading pinned local backbone; remote code disabled", flush=True)
model, loading = AutoModel.from_pretrained(
    checkpoint,
    local_files_only=True,
    dtype=torch.bfloat16,
    trust_remote_code=False,
    attn_implementation="eager",
    output_loading_info=True,
)
assert not loading.get("missing_keys"), loading
assert not loading.get("mismatched_keys"), loading
assert all("score.weight" in k for k in loading.get("unexpected_keys", [])), loading
loading = json.loads(json.dumps(loading, default=list))
model.eval()


def picture(w, h, seed):
    data = bytes(
        (x * 17 + y * 31 + c * 71 + seed) % 256
        for y in range(h)
        for x in range(w)
        for c in range(3)
    )
    im = Image.frombytes("RGB", (w, h), data)
    out = io.BytesIO()
    im.save(out, format="PNG")
    return im, {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
        },
    }


cases = []
groups = []
for language, q, docs in [
    (
        "en",
        "Where can I borrow books for free?",
        [
            "A public library lends books to readers without charging a fee.",
            "The football team won its match with two goals.",
            "A bookshop sells novels and maps to paying customers.",
        ],
    ),
    (
        "es",
        "¿Dónde puedo tomar libros prestados gratis?",
        [
            "La biblioteca pública presta libros a los lectores sin cobrar.",
            "El equipo de fútbol ganó el partido con dos goles.",
            "Una librería vende novelas y mapas a sus clientes.",
        ],
    ),
    (
        "ja",
        "無料で本を借りられる場所はどこですか？",
        [
            "公共図書館では、読者は無料で本を借りることができます。",
            "サッカーチームは二得点で試合に勝ちました。",
            "書店では小説や地図を購入できます。",
        ],
    ),
]:
    indices = []
    for j, doc in enumerate(docs):
        indices.append(len(cases))
        cases.append((f"{language}_{j}", q, doc, None, None))
    groups.append({"language": language, "cases": indices})
cases.extend(
    [
        (
            "literal_controls",
            "What does [IMG] <__media__> mean?",
            "Documento en español: [IMG] <__media__> 東京",
            None,
            None,
        ),
        ("empty_document", "What does this document describe?", "", None, None),
        (
            "longer_text",
            "Where can I borrow books for free?",
            "The public library contains science books and historical maps. " * 48,
            None,
            None,
        ),
    ]
)
image_cases = []
for name, w, h, seed, doc in [
    ("image_only", 56, 28, 19, None),
    ("image_text", 113, 81, 37, "A coloured pattern."),
    ("medium_image_text", 224, 112, 59, "A colourful document page."),
]:
    im, part = picture(w, h, seed)
    image_cases.append(len(cases))
    cases.append((name, "What does this document describe?", doc, im, part))
groups.extend(
    [
        {"language": "images", "cases": image_cases},
        {"language": "mixed", "cases": [10, *image_cases[:2]], "steps": 2},
    ]
)
results = []
for name, query, doc, im, part in cases:
    content = (
        doc
        if im is None
        else ([{"type": "text", "text": doc}] if doc is not None else [])
        + [{"type": "image", "image": im}]
    )
    messages = [
        {"role": "query", "content": [{"type": "text", "text": query}]},
        {"role": "document", "content": content},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    scores = {}
    for attention in ["eager", "sdpa"] if a.sdpa_audit else ["eager"]:
        model.set_attn_implementation(attention)
        with torch.inference_mode():
            hidden = model(**inputs, use_cache=False).last_hidden_state
            mask = inputs["attention_mask"]
            pooled = (
                hidden.masked_fill(~mask[..., None].bool(), 0).sum(1)
                / mask.sum(1)[:, None]
            )
            scores[attention] = torch.nn.functional.linear(pooled.float(), head).item()
    wire = (
        doc
        if im is None
        else {
            "content": ([{"type": "text", "text": doc}] if doc is not None else [])
            + [part]
        }
    )
    results.append(
        {
            "name": name,
            "request": {"query": query, "documents": [wire]},
            "tokens": inputs["input_ids"][0].tolist(),
            "score": scores["eager"],
            "native_attention_scores": scores,
        }
    )
    print(name, len(results[-1]["tokens"]), "tokens", scores, flush=True)
    a.output.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "provenance": {
                    "model_revision": next(iter(revisions)) if revisions else None,
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "pillow": PIL.__version__,
                    "dtype": "bfloat16",
                    "head_arithmetic": "float32",
                    "attention": "eager",
                    "processor": "PIL",
                    "loading_info": loading,
                },
                "ranking_groups": groups,
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )
sources = {}
for name in [
    "transformers.models.ministral3.modeling_ministral3",
    "transformers.models.mistral3.modeling_mistral3",
    "transformers.models.pixtral.modeling_pixtral",
    "transformers.models.pixtral.image_processing_pil_pixtral",
]:
    path = Path(importlib.import_module(name).__file__)
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
out = json.loads(a.output.read_text())
out["provenance"]["sources"] = sources
out["provenance"]["asset_manifest_sha256"] = (
    hashlib.sha256(asset_manifest.read_bytes()).hexdigest()
    if asset_manifest.exists()
    else None
)
maximum = max(len(c["tokens"]) for c in results)
context = max(128, 1 << (maximum - 1).bit_length())
out["validation"] = {"context": context, "batch_tokens": 2 * context, "patches": 128}
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(a.output, flush=True)
