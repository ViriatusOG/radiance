"""mkclefgolden.py CKPTDIR OUT.json -- the golden for tests/clef_decision_test.cpp: the release's own
head (joint_schema_model.py, beside the clef-flash weights, CLEF_DIR) over transformers' forward pass
of the tiny qwen35 the test writes, in float32, on the request in this file.

CKPTDIR is the tiny clef's directory as the test keeps it: run `RAD_E2E_KEEP=1
build/tests/clef_decision_test` and pass the directory it prints. The golden records the FNV-1a
hashes of model.safetensors and joint_head.safetensors (the test refuses a golden made from other
weights), the request, and each question's probabilities."""
import json, os, re, sys
from types import SimpleNamespace
import torch
from safetensors.torch import load_file

ckpt_dir, out = sys.argv[1], sys.argv[2]
# transformers routes Qwen3.5's delta-net and causal convolution to fla's Triton kernels wherever
# those packages import, which needs a card. The golden is the reference PyTorch path on the CPU,
# so the two packages are made unimportable before transformers looks for them.
sys.modules["fla"] = None
sys.modules["causal_conv1d"] = None
DEV = "cpu"
sys.path.insert(0, os.environ["CLEF_DIR"])
import joint_schema_model as jsm  # noqa: E402

REQUEST = {
    "state": {"text": "The parcel arrived two weeks late and the box was crushed, but the vase "
                      "inside was intact. I would like a partial refund for the delay.",
              "channel": "email"},
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
        "team": {"type": "choice", "instructions": "Which team should take this?",
                 "criteria": {"shipping": "Delivery delays and damage in transit",
                              "billing": "Charges and refunds",
                              "product": "Faults in the item itself"}},
        "tone": {"type": "score", "instructions": "How upset is the customer?",
                 "criteria": ["Calm", "Mildly annoyed", "Angry"]},
    },
}

SPECIAL = {"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258}


class ByteTokenizer:
    """The tiny tokenizer: one byte one id, the three specials as added tokens."""
    def __call__(self, text, add_special_tokens=False):
        ids = []
        for piece in re.split(r"(<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>)", text):
            if piece in SPECIAL:
                ids.append(SPECIAL[piece])
            else:
                ids.extend(piece.encode("utf-8"))
        return SimpleNamespace(input_ids=ids)   # encode_record reads .input_ids


def fnv(path):
    h = 0xcbf29ce484222325
    for b in open(path, "rb").read():
        h = ((h ^ b) * 0x100000001b3) & (2**64 - 1)
    return h


cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402
c = {k: v for k, v in cfg.items() if k not in ("architectures", "model_type")}
model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**c))
sd = {k.replace("model.language_model.", "model."): v.float()
      for k, v in load_file(os.path.join(ckpt_dir, "model.safetensors")).items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [k for k in missing if k != "lm_head.weight"]
if missing or unexpected:
    raise SystemExit(f"backbone: missing {missing}, unexpected {unexpected}")
model = model.float().eval().to(DEV)

head_cfg = json.load(open(os.path.join(ckpt_dir, "joint_head_config.json")))
head = jsm.JointSchemaHead(**head_cfg)
head.load_state_dict({k: v.float() for k, v in
                      load_file(os.path.join(ckpt_dir, "joint_head.safetensors")).items()},
                     strict=True)
head = head.float().eval().to(DEV)

rec = jsm.encode_record(ByteTokenizer(), REQUEST)
ids = torch.tensor([rec.input_ids], device=DEV)
with torch.no_grad():
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    logits = head(hidden, ids, torch.ones_like(ids), [rec], model.lm_head.weight)[0]
answers = {q.question_id: dict(zip(q.option_ids, ql.float().softmax(-1).cpu().tolist()))
           for q, ql in zip(rec.questions, logits)}
json.dump({"weights_fnv": "%016x" % fnv(os.path.join(ckpt_dir, "model.safetensors")),
           "head_fnv": "%016x" % fnv(os.path.join(ckpt_dir, "joint_head.safetensors")),
           "n_prompt_tokens": len(rec.input_ids),
           "request": REQUEST, "answers": answers},
          open(out, "w"), indent=1)
print(f"{out}: {len(rec.input_ids)} prompt tokens, {len(answers)} questions")
