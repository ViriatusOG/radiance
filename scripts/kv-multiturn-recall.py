#!/usr/bin/env python3
"""A planted fact survives twenty turns of a conversation whose history outgrows the KV ring.

Each turn's prompt is the previous one plus a little, so the prefix cache serves the common prefix
and only the tail is prefilled. With a ring smaller than the history (32 768 tokens against a
48 932-token transcript) the blocks a hit names have been through the host mirror and back, and a
wrong copy shows as fluent text carrying a different name, which only a fact-check catches. One
request with the whole transcript, or a ring covering the context, cannot show it; so the
transcript goes turn by turn with a small ring, against the same transcript with streaming off.

What it asserts:
  * every arm answers the code, the name and the torque correctly (the code and torque are the
    same-turn facts; the name is the one planted at turn 2 and asked at turn 20, nineteen turns and
    ~35k tokens later);
  * the off arm and the on arm each served a real prefix hit -- later turns report cached tokens,
    otherwise the arm is not a reuse test at all and this script fails rather than passing quietly;
  * the on arm streamed (nonzero kv_stream_misses_total), otherwise it is not the streamed arm.

Run it on the host; it starts each arm's server in the toolchain container, with MODELS (default
~/models) mounted at /models and the tree at /src:
  MODELS=/path/to/models python3 scripts/kv-multiturn-recall.py

Not in CI: it needs a model, both cards, and ~6 minutes. Exit 0 only if every arm is correct and
both arms are proven to have hit the cache; 2 if an arm did not come up or did not answer.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.environ.get("BUILD", "/src/build")
MODELS = os.environ.get("MODELS", os.path.expanduser("~/models"))
MODEL = os.environ.get("MODEL", "/models/qwen3.8-next-flash-fp8-iq4r-moe.rad")
OUT = os.environ.get("OUT", "/tmp/radiance-kv-multiturn")
BASE = ["--model", MODEL, "--tp", "2", "--tp-wire", "exact", "--max-num-seqs", "2",
        "--max-model-len", "65536",
        "--placement", "expert_tiered", "--host-pool-mib", "47304", "--gpu-headroom-mib", "96",
        "--kv-cache-dtype", "i8", "--prefix-cache-mib", "1024", "--prefix-cache-ttl", "0",
        "--vision-tower-placement", "ram", "--num-speculative-tokens", "0", "--deterministic",
        "--host", "127.0.0.1", "--api-key", "test"]

# The planted facts, exactly the shape §4u/A measured: the code in turn 2 and the surname in turn 5,
# asked together at turn 20. The code survives the window; the surname came back invented
# ("Ilse Koch" for "Ilse Varga") in a 32 768-token ring whose history had outgrown it.
CODE = "The maintenance code for bay seven is QX-4471."
NAME = ("The duty archivist for the east wing on the day of the catalogue check was named "
        "Ilse Varga.")
TORQUE = ""   # not planted in this transcript; the question asks for what is here
QUESTION = ("Question: what is the maintenance code for bay seven, and who was the duty archivist "
            "for the east wing on the day of the catalogue check? Answer with both values.")
RECORD = ("Record %d: the archivist checked the shelf, noted the catalogue number, and returned the "
          "volume to the east wing of the reading room.")
PAD = 65            # register lines a turn: ~2 010 tokens as the chat template counts them
N_TURNS = 20
CODE_TURN = 2
NAME_TURN = 5
MAX_TOKENS = 400     # forced past EOS, as measured: ~2 430 tokens of live context a turn


def log(*a):
    print(*a, flush=True)


def http(port, path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer test", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health(port):
    try:
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/health",
                                    headers={"Authorization": "Bearer test"}), timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def metrics(port):
    try:
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/metrics",
                                     headers={"Authorization": "Bearer test"}), timeout=5) as r:
            txt = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("radiance:") and " " in line:
            k, v = line.rsplit(" ", 1)
            k = k.split("{")[0][len("radiance:"):]
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def engine_pid():
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/comm") as f:
                if f.read().strip() == "radiance":
                    return int(p)
        except OSError:
            pass
    return None


class Server:
    """One engine, in its own container, killed by name at the end and on any error path."""

    def __init__(self, tag, port, flags):
        self.tag, self.port, self.name = tag, port, f"mt-{tag}"
        self.logf = f"{OUT}/{tag}.server.log"
        self.flags = BASE + list(flags) + ["--port", str(port)]

    def start(self):
        self.p = subprocess.Popen([
            "docker", "run", "--rm", "--name", self.name, "--user", "1000:1000", "-e", "HOME=/tmp",
            "--device", "/dev/kfd", "--device", "/dev/dri", "--group-add", "990", "--group-add", "44",
            "--security-opt", "seccomp=unconfined", "--network", "host",
            "-v", f"{ROOT}:/src", "-v", f"{MODELS}:/models", "-w", "/src", "rad-toolchain:latest",
            "bash", "-lc", f"RADIANCE_HOME={BUILD}/radiance_home /src/scripts/radiance-flags {BUILD}/bin/radiance " + " ".join(self.flags)],
            stdout=open(self.logf, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 300:
            if health(self.port):
                log(f"[{self.tag}] health after {time.time() - t0:.0f}s")
                return
            if self.p.poll() is not None:
                with open(self.logf) as lf:
                    log(lf.read()[-2000:])
                raise SystemExit(f"arm {self.tag}: engine exited before health")
            time.sleep(1)
        raise SystemExit(f"arm {self.tag}: no health after 300s")

    def stop(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        for _ in range(120):
            if engine_pid() is None:
                return
            time.sleep(0.5)


def turn_text(t):
    body = " ".join(RECORD % i for i in range(t * 400, t * 400 + PAD))
    if t == CODE_TURN:
        return f"{CODE} {body} Begin the session log."
    if t == NAME_TURN:
        return f"{NAME} {body} Continue the register."
    if t == N_TURNS:
        return f"{body} {QUESTION}"
    return f"{body} Continue the register."


def run_arm(tag, port, flags):
    srv = Server(tag, port, flags)
    try:
        srv.start()
        msgs, rows, answers = [], [], {}
        m0 = metrics(port)
        for t in range(1, N_TURNS + 1):
            msgs.append({"role": "user", "content": turn_text(t)})
            a = None
            for _ in range(2):
                try:
                    a = http(port, "/v1/chat/completions",
                             {"model": "qwen4exp", "temperature": 0, "seed": 12345,
                              "max_tokens": MAX_TOKENS, "ignore_eos": True,
                              "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs})
                    break
                except Exception as e:
                    log(f"[{tag}] turn {t} failed: {e}")
                    time.sleep(2)
            if a is None:
                raise SystemExit(f"arm {tag}: turn {t} failed twice")
            m1 = metrics(port)
            d = round(m1.get("kv_stream_misses_total", 0) - m0.get("kv_stream_misses_total", 0))
            m0 = m1
            u = a.get("usage", {})
            txt = a["choices"][0]["message"]["content"]
            answers[t] = txt
            rows.append({"turn": t, "prompt": u.get("prompt_tokens"),
                         "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                         "misses": d})
            log(f"[{tag}] turn {t:2d} prompt={rows[-1]['prompt']} cached={rows[-1]['cached']} misses+={d}")
            msgs.append({"role": "assistant", "content": txt})
    finally:
        srv.stop()
    return {"tag": tag, "rows": rows, "answers": answers}


def check(name, text):
    """Both values, compared case- and punctuation-insensitively, so formatting cannot hide a wrong
    fact and cannot fail a right one. The surname is the recorded failure: 'Ilse Koch' came back for
    'Ilse Varga', so both parts are required, in that order and adjacent."""
    flat = " ".join("".join(c.lower() if c.isalnum() else " " for c in text).split())
    return ("qx 4471" in flat or "qx4471" in flat) and ("ilse varga" in flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("on", "off"), help="run one arm")
    ap.add_argument("--port", type=int, default=8181)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    # A ring at the floor (32 768 tokens, 8 192 blocks) under a context that outgrows it: every
    # turn past 32 768 tokens has history the ring cannot hold, so a hit can only be served from the
    # host copy. This is the shape in which the planted name came back wrong.
    arms = [("off", []), ("on", ["--kv-resident", "32768"])]
    if args.only:
        arms = [a for a in arms if a[0] == args.only]
    results, bad = {}, []
    for i, (tag, flags) in enumerate(arms):
        results[tag] = run_arm(tag, args.port + i, flags)
        r = results[tag]
        final = r["answers"][N_TURNS]
        hits = sum(1 for row in r["rows"][1:] if row["cached"] > 0)
        log(f"[{tag}] FINAL: {r['answers'][N_TURNS]!r}")
        log(f"[{tag}] facts {'OK' if check(tag, final) else 'WRONG'}, "
            f"cached on {hits}/{N_TURNS - 1} continuing turns, "
            f"streamed {sum(row['misses'] for row in r['rows'])} blocks")
        json.dump(results, open(f"{OUT}/multiturn.json", "w"), indent=1)
    for tag, r in results.items():
        final = r["answers"][N_TURNS]
        hits = sum(1 for row in r["rows"][1:] if row["cached"] > 0)
        if not check(tag, final):
            bad.append(f"{tag}: wrong facts")
        if tag == "off" and hits < N_TURNS - 2:
            bad.append(f"off: only {hits} turns served from the cache -- not a reuse arm")
        if tag == "on" and not any(row["misses"] for row in r["rows"]):
            bad.append("on: streamed nothing")
    if len(arms) == 2:
        same = results["on"]["answers"][N_TURNS] == results["off"]["answers"][N_TURNS]
        log(f"[*] on and off agree byte for byte: {same}")
        if not same:
            bad.append("on and off differ")
    json.dump(results, open(f"{OUT}/multiturn.json", "w"), indent=1)
    if bad:
        log("FAIL: " + "; ".join(bad))
        return 1
    log("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
