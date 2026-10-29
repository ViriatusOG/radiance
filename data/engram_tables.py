#!/usr/bin/env python3
"""engram_tables.py -- the Engram's derived tables of a DeepSeek-V4.1 release, for `rad-convert --add`.

The release ships the Engram's embedding tables but not the hash constants inference/engram.py
computes at load: the compressed token map (the tokenizer's normalisers over every token's decode),
the n-gram hash multipliers (numpy PCG64 streams) and the per-head bucket moduli (primes) with their
offsets into the joined table. This tool computes them in Radiance's own terms -- no llama.cpp
converter, no transformers: the token map is built from the release's tokenizer.json by reimplementing
HuggingFace's ByteLevel decode and the normaliser sequence, and the multipliers from numpy's PCG64.
It writes the four arrays as a small safetensors with the GGUF's element types (token map I32; the
rest U64 in the GGUF, which the core reads as I64):

  engram.token_map          I32 [vocab]
  engram.layer_multipliers  I64 [n_engram_layers * max_ngram]
  engram.head_vocab_sizes   I64 [n_engram_layers * (max_ngram - 1) * n_heads]
  engram.head_offsets       I64 [same]   (absolute row offsets into the joined table)

The deepseek41 architecture plugin, which this tree does not include, reads them under these names
when the checkpoint is the release. Reads config.json and tokenizer.json only (small).

usage: python3 data/engram_tables.py RELEASE_DIR OUT.safetensors
"""
import argparse
import json
import math
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np

SENTINEL = "\ue000"


def _is_prime(n: int) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


def engram_primes(n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int) -> list[list[list[int]]]:
    """EngramLayout.from_args: [layer][ngram order 2..max][head] bucket moduli, never reused across layers."""
    seen: set[int] = set()
    out: list[list[list[int]]] = []
    for _ in range(n_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cand = cur + 1
                while not _is_prime(cand) or cand in seen:
                    cand += 1
                seen.add(cand)
                sizes.append(cand)
                cur = cand
            per_ngram.append(sizes)
        out.append(per_ngram)
    return out


def engram_multipliers(layer_ids: list[int], max_ngram_size: int, compressed_vocab_size: int) -> list[list[int]]:
    """compute_hash_multipliers: odd, bounded so that token_id * multiplier cannot overflow int64."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append([int(v) * 2 + 1 for v in values])
    return rows


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-level alphabet: the printable bytes map to themselves, the rest to U+0100+."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_byte_dec = {v: k for k, v in _bytes_to_unicode().items()}


def _byte_level_decode(token: str) -> str:
    """ByteLevel: the token's characters are the byte alphabet; join the bytes, decode UTF-8."""
    raw = bytes(_byte_dec[c] for c in token)
    return raw.decode("utf-8", errors="replace")


def _normalize(text: str) -> str:
    """The converter's normaliser sequence, step for step: NFKC, NFD, strip accents, lowercase,
    collapse whitespace runs, trim, and keep a lone space as a sentinel."""
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.category(c).startswith("M"))
    text = "".join(c.lower() if c.isupper() else c for c in text)
    out, prev_space = [], False
    for c in text:
        if c in " \t\r\n":
            if not prev_space:
                out.append(" ")
            prev_space = True
        else:
            out.append(c)
            prev_space = False
    text = "".join(out)
    sentinel = False
    if text == " ":
        text, sentinel = SENTINEL, True
    stripped = text.strip()
    if sentinel:
        return " " if stripped == SENTINEL else stripped.replace(SENTINEL, " ")
    return stripped


def engram_compressed_token_map(release_dir: Path) -> tuple[list[int], int]:
    """token id -> compressed id (tokens that normalize alike collapse)."""
    tok = json.load(open(Path(release_dir) / "tokenizer.json"))
    vocab = tok["model"]["vocab"]                 # token string -> id
    added = tok.get("added_tokens") or []
    id_str: dict[int, str] = {i: s for s, i in vocab.items()}
    for a in added:
        id_str[a["id"]] = a["content"]
    added_ids = {a["id"] for a in added}
    key_to_new: dict[str, int] = {}
    lookup = [0] * (max(id_str) + 1)
    for token_id in range(max(id_str) + 1):
        token = id_str.get(token_id)
        if token is None:
            lookup[token_id] = key_to_new.setdefault(f"<missing {token_id}>", len(key_to_new))
            continue
        if token_id in added_ids:
            text = token
        else:
            text = _byte_level_decode(token)
        if "\ufffd" in text:
            key = token
        else:
            normalized = _normalize(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("out")
    a = ap.parse_args()
    cfg = json.load(open(Path(a.release) / "config.json"))
    hp = {**cfg, **cfg.get("text_config", {})}
    layer_ids = list(hp.get("engram_layer_ids") or [])
    if not layer_ids:
        sys.exit("the release has no Engram layers; nothing to write")
    max_ngram, n_heads = hp["engram_max_ngram_size"], hp["engram_n_heads"]
    primes = engram_primes(len(layer_ids), max_ngram, n_heads, hp["engram_vocab_size"])
    for li, layer_primes in enumerate(primes):
        if sum(p for per in layer_primes for p in per) != hp["engram_num_embeddings"][li]:
            sys.exit(f"engram layer {layer_ids[li]}: prime sum != num_embeddings")
    token_map, cvocab = engram_compressed_token_map(Path(a.release))
    if cvocab != hp["engram_compressed_vocab_size"] or len(token_map) != hp["vocab_size"]:
        sys.exit(f"compressed token map ({len(token_map)} -> {cvocab}) does not match the config")
    sizes = [p for layer in primes for per in layer for p in per]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    mults = [m for row in engram_multipliers(layer_ids, max_ngram, cvocab) for m in row]

    tensors = [("engram.token_map", "I32", "<i", token_map),
               ("engram.layer_multipliers", "I64", "<q", mults),
               ("engram.head_vocab_sizes", "I64", "<q", sizes),
               ("engram.head_offsets", "I64", "<q", offsets)]
    header, blobs, off = {}, [], 0
    for name, dt, fmt, vals in tensors:
        b = struct.pack(f"<{len(vals)}{fmt[1]}", *vals)
        header[name] = {"dtype": dt, "shape": [len(vals)], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"format": "pt", "source": "radiance data/engram_tables.py"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    print(f"{a.out}: token_map {len(token_map)} -> {cvocab}; {len(layer_ids)} Engram layers {layer_ids}; "
          f"{len(mults)} multipliers, {len(sizes)} heads, rows {sum(sizes)}")


if __name__ == "__main__":
    main()
