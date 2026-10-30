"""mklayagolden.py CKPTDIR OUT.json -- the golden for tests/laya_decision_test.cpp: laya's own
Agent.predict (the package in LAYA_SRC, a checkout of github.com/NandhaKishorM/laya) on the tiny
Laya the test writes, on the CPU in float32, for the requests below.

CKPTDIR is the tiny Laya's directory as the test keeps it: run `RAD_E2E_KEEP=1
build/tests/laya_decision_test` and pass the directory it prints. The golden records the FNV-1a hash
of model.safetensors (the test refuses a golden made from other weights), the package's commit
(LAYA_COMMIT), each request, and the package's answers and usage for it.

The three requests: one whose state fits every question's sequence; one whose JSON-object state
is cut to the room each question leaves, with a 12-option choice (the 11+ temperature bucket,
whose fitted 0.1 the package clamps to 0.5); one whose state is a list of turns, which keeps its
end."""
import json
import os
import sys

ckpt_dir, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.environ["LAYA_SRC"])
from laya.agent import Agent  # noqa: E402

LONG = ("The parcel arrived two weeks late and the box was crushed, but the vase inside was "
        "intact. I would like a partial refund for the delay, and please tell the courier.")
REQUESTS = [
    {"state": "Charged twice for one order; refund one please.",
     "questions": {
         "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
         "team": {"type": "choice", "instructions": "Which team should take this?",
                  "criteria": {"billing": "Charges and refunds", "shipping": "Delivery",
                               "product": ""}},
         "tone": {"type": "score", "instructions": "How upset is the customer?",
                  "criteria": ["Calm", "Mildly annoyed", "Angry"]}}},
    {"state": {"message": LONG, "channel": "email"},
     "questions": {
         "topic": {"type": "choice", "instructions": "Topic?",
                   "criteria": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]},
         "spam": {"type": "noul", "instructions": {"ask": "Is this unsolicited?"},
                  "criteria": {"true": "unsolicited bulk mail"}},
         "team": {"type": "choice", "instructions": "Route it.",
                  "criteria": ["billing", "technical", "sales", "other"]}}},
    {"state": [{"role": "user", "content": LONG}, {"role": "agent", "content": "Sorry, checking."},
               {"role": "user", "content": "Any news on the refund?"}],
     "questions": {
         "waiting": {"type": "noul", "instructions": "Is the customer still waiting?"}}},
]


def fnv(path):
    h = 0xcbf29ce484222325
    for b in open(path, "rb").read():
        h = ((h ^ b) * 0x100000001b3) & (2**64 - 1)
    return h


agent = Agent(ckpt_dir, device="cpu")
results = [agent.predict(r["state"], r["questions"]) for r in REQUESTS]
golden = {
    "weights_fnv": "%016x" % fnv(os.path.join(ckpt_dir, "model.safetensors")),
    "laya_commit": os.environ.get("LAYA_COMMIT", ""),
    "requests": REQUESTS,
    "results": [{"answers": r["answers"], "usage": r["usage"]} for r in results],
}
with open(out, "w") as f:
    json.dump(golden, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(json.dumps([r["usage"] for r in results]))
