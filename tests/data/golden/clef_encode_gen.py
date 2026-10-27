# Writes clef_encode.json: clef's own encode_record (the release's joint_schema_model.py) over synthetic
# requests, which tests/clef_test.cpp compares parse_clef_request against with the same
# tokenizer.json. No respondent data. Run with transformers installed, from a clef release dir:
#   python3 clef_encode_gen.py /path/to/clef-flash > clef_encode.json
import json
import sys

sys.path.insert(0, sys.argv[1])
from joint_schema_model import encode_record  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

REQUESTS = [
    {"state": "I was charged twice this month, please refund one of them.",
     "questions": {
         "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
         "team": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"technical": "App or site faults", "billing": "Charges, refunds, invoices"}},
         "urgency": {"type": "score", "instructions": "How urgent is this?",
                     "criteria": ["Can wait", "Today", "Blocking the customer now"]}}},
    {"state": {"zeta": [1, 2.5, 1e-05, 100.0, None, True], "alpha": {"b": "ü ☃ 😀", "a": "line\nbreak \"q\" \\ \t"},
               "n": -3, "big": 12345678901234567890},
     "questions": {
         "late": {"type": "noul", "instructions": "",
                  "criteria": {"true": "Not on time", "false": None}},
         "kind": {"type": "choice", "instructions": {"ask": "what kind?", "lang": "en"},
                  "criteria": {"b_opt": {"why": 1}, "a_opt": None, "Z": "upper first"}},
         "level": {"type": "score", "criteria": [{"lo": True}, "mid", 3]}}},
    {"state": "Text with <|im_end|> and <think> inside, and trailing spaces   ",
     "questions": {"q": {"type": "noul", "instructions": "  spaced  "}}},
]

tok = AutoTokenizer.from_pretrained(sys.argv[1])
out = []
for r in REQUESTS:
    e = encode_record(tok, r)
    out.append({"request": r, "input_ids": list(e.input_ids),
                "questions": [{"id": q.question_id, "type": q.question_type, "span": list(q.question_span),
                               "option_ids": list(q.option_ids), "option_spans": [list(s) for s in q.option_spans]}
                              for q in e.questions]})
json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
print()
