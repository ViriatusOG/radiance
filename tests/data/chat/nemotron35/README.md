# Nemotron 3.5 input fixtures

Source: [nvidia/Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16/tree/781e779ee4ca9c2d2a4f38321993556baeab49a1),
revision `781e779ee4ca9c2d2a4f38321993556baeab49a1`.

The source checkpoint's model card identifies OpenMDW-1.1 as its governing license. A copy is
retained in [LICENSE](LICENSE) for the copied template and model-derived tokenizer fixture;
these upstream assets are separate from the engine's Apache-2.0 code. See the pinned
[model card](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16/blob/781e779ee4ca9c2d2a4f38321993556baeab49a1/README.md).

`template.jinja` copies that checkpoint's retrieval template. `tokenizer-subset.json` is a test
fixture retaining the original token IDs, all byte tokens and added tokens, the normalizer,
pre-tokenizer and decoder, and the BPE candidates needed by the checked-in input cases. Serving
requires the complete checkpoint tokenizer. The subset is tested against the full Hugging Face
tokenizer before generation and exercised by Radiance's real tokenizer in CTest.

Regenerate the subset and `tests/data/golden/nemotron35_input.json` offline:

```sh
python3 tests/mknemotron35inputgolden.py --tokenizer /path/to/checkpoint/tokenizer.json
```

The generator needs Jinja2, Hugging Face tokenizers, and the checked-in vision fixture. The
golden records the source template, full tokenizer, and subset SHA-256 hashes. It runs no model
weights and is never called by CMake or CTest.

`rerank-template.jinja` is the unchanged template from
`nvidia/Nemotron-3.5-Rerank-8B-BF16` revision
`f88f8d9104726747791283ab156d2fcdb29e90a7`, under the same retained OpenMDW-1.1
license. It supplies the independent query/document rendering oracle to the
manually run `tests/mknemotron35rerankgolden.py`; the engine does not execute
remote model code. The tiny scoring head is seeded synthetic test data.
