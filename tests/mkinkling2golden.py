"""mkinkling2golden.py -- a float32 reference forward of Inkling 2 (`inkling2`, `inkling2_loop`).

    mkinkling2golden.py golden CKPTDIR OUT.h
        Also writes inkling2_logp_golden.h beside OUT.h: the full log-probabilities at every position.
        The golden of tests/engine_e2e_test.cpp: the tiny random checkpoint tests/tiny_model.h writes
        (CKPTDIR/inkling2, printed by `RAD_E2E_KEEP=1 build/tests/engine_e2e_test`) run over
        tiny_golden.h's prompt, the four most likely next tokens and their log-probabilities at every
        position, and the FNV-1a hash of its model.safetensors.

    mkinkling2golden.py real MODELDIR IDS [--top N] [--layers]
        The released checkpoint over the comma-separated token ids IDS: one JSON line a position with
        the top N ids and log-probabilities. `--layers` also prints the rms of every layer's output.
        This is how the reference itself is checked against llama.cpp (src/models/inkling2-loop.cpp).

transformers has no Inkling 2 class and vLLM has no loop, so this file is the reference: it follows
llama.cpp's graph (`llama_model_inkling2::graph`, `llama_model_inkling2_loop::graph`) op for op,
and vLLM's `vllm/models/inkling2` where the two describe the same thing (they agree on the
convolutions, the clips, the relative bias, AttnRes, SiTU and the routing). Nothing here is cached:
every call is one whole sequence from position 0, so the loop's per-pass KV and convolution
states, which only matter to a server, never appear. Run by hand, like the other mk*golden.py; the
build and the suite never call it.
"""
import json, math, os, re, sys

import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
F32 = torch.float32
EPS = 1e-6  # every norm of the model, AttnRes's and the controllers' included

# e2m1 as NVFP4 stores it: bit 3 the sign, then a two-bit exponent and one mantissa bit.
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=F32)


def rms(x, w=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if w is None else y * w


def l2clip(x, thr):
    """Each row scaled down to an L2 norm of at most thr (llama.cpp's clip, after its scale)."""
    n = x.norm(dim=-1, keepdim=True)
    return x * (thr / (n + 1e-6)).clamp(0.0, 1.0)


def mix(states, query):
    """Block attention residuals: a softmax over depth. states: list of [T, H]; query [Q, H].
    Scores read the rms-normed states, and so do the values (`attn_res_normalize_values`); the
    caller's norm applies its gain afterwards. -> [Q, T, H]."""
    s = rms(torch.stack(states))                                     # [N, T, H]
    p = torch.einsum("nth,qh->qtn", s, query) / math.sqrt(s.shape[-1])
    return torch.einsum("qtn,nth->qth", p.softmax(-1), s)


class Ckpt:
    def __init__(self, d):
        self.d = d
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.cfg = cfg
        self.t = cfg.get("text_config", cfg)
        self.files = {}
        idx = os.path.join(d, "model.safetensors.index.json")
        names = (json.load(open(idx))["weight_map"] if os.path.exists(idx)
                 else {k: "model.safetensors" for k in safe_open(os.path.join(d, "model.safetensors"), "pt").keys()})
        self.where = names
        q = os.path.join(d, "hf_quant_config.json")
        self.mxfp8 = set()
        if os.path.exists(q):
            for g in json.load(open(q))["model"]["parameter_groups"]:
                if g["compute_dtype"] == "mxfp8":
                    self.mxfp8.update(g["tensors"])
        self.prefix = "model.llm." if any(k.startswith("model.llm.") for k in names) else "model."

    def _f(self, name):
        fn = self.where[name]
        if fn not in self.files:
            self.files[fn] = safe_open(os.path.join(self.d, fn), "pt")
        return self.files[fn]

    def has(self, name):
        return self.prefix + name in self.where

    def get(self, name):
        n = self.prefix + name
        return self._f(n).get_tensor(n).to(F32)

    def raw_slice(self, name, i):
        n = self.prefix + name
        return self._f(n).get_slice(n)[i]

    def expert(self, layer, proj, e):
        """Routed expert e's weight [N, K] in f32: w13 (gate/up, interleaved by row) or w2.
        NVFP4: e2m1 codes, the even k in the low nibble, times an e4m3 scale per 16 of K, times
        amax[e] / outer_scale_denominator. MXFP8: e4m3 codes times 2^(ue8m0 - 127) per 32 of K.
        A bf16 expert reads as stored."""
        base = f"layers.{layer}.mlp.experts.{proj}_weight"
        w = self.raw_slice(base, e)
        if w.dtype == torch.uint8:
            sc = self.raw_slice(base + "_scale", e).to(F32)            # [N, K/16] e4m3
            lo, hi = E2M1[(w & 15).long()], E2M1[(w >> 4).long()]
            v = torch.stack([lo, hi], -1).reshape(w.shape[0], -1)      # [N, K]
            v = (v.view(v.shape[0], -1, 16) * sc[..., None]).reshape(v.shape)
            amax = self.get(base + "_amax")[e]
            den = self.get(base + "_outer_scale_denominator").reshape(-1)[0]
            return v * (amax / den)
        if w.dtype == torch.float8_e4m3fn:
            s = self.raw_slice(base + "_scale", e)
            s = torch.ldexp(torch.ones_like(s, dtype=F32), s.to(torch.int32) - 127)  # ue8m0
            v = w.to(F32)
            return (v.view(v.shape[0], -1, 32) * s[..., None]).reshape(v.shape)
        return w.to(F32)


class Inkling2:
    def __init__(self, ck):
        self.ck, t = ck, ck.t
        self.H = t["hidden_size"]
        self.L = t["num_hidden_layers"]
        self.local = set(t["local_layer_ids"])
        self.block = t["n_layers_per_attn_res_block"]
        self.kconv = t["sconv_kernel_size"]
        self.window = t["sliding_window_size"]
        self.extent = t["rel_extent"]
        self.d_rel = t["d_rel"]
        self.qk_clip, self.rel_clip = t["qk_l2_clip_threshold"], t["rel_l2_clip_threshold"]
        self.floor, self.alpha = t.get("log_scaling_n_floor"), t["log_scaling_alpha"]
        self.top_k, self.route_scale = t["num_experts_per_tok"], t["route_scale"]
        self.gate_cap, self.up_cap = t["mlp_gate_softcap"], t["mlp_up_softcap"]
        self.softcap = t.get("final_logit_softcapping") or 0.0
        self.n_vocab = t.get("unpadded_vocab_size") or t["vocab_size"]
        loop = t.get("loop_config")
        self.loop = (loop["core_start"], loop["core_end"], loop["num_passes"]) if loop else None
        if loop and loop.get("is_varloop"):
            raise SystemExit("inkling2: a variable loop (is_varloop) is not what this reference runs")
        if t.get("n_shared_experts", 0) or not t.get("use_value_norm_no_gamma", True) \
                or not t.get("attn_res_normalize_values", True):
            raise SystemExit("inkling2: a configuration this reference does not describe")
        self.w = {}
        if self.loop:
            # Core layer start + i mixes its q, k, v and mlp messages with
            # [q[2i], kv[2i], kv[2i+1], q[2i+1]] (llama.cpp's conversion; transformers' names are
            # cross_pass_messages.{queries,kv_queries}).
            q = ck.get("middle_cycle_cross_pass_queries")
            kv = ck.get("middle_cycle_cross_pass_kv_queries")
            s, e, _ = self.loop
            assert q.shape[0] == 2 * (e - s) and kv.shape == q.shape
            self.msg_q = {s + i: torch.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])
                          for i in range(e - s)}

    def W(self, name):
        if name not in self.w:
            self.w[name] = self.ck.get(name)
        return self.w[name]

    # ---------------------------------------------------------------- pieces

    def conv(self, x, prefix, dyn_in=None):
        """The short causal convolution with its residual: (x + conv(x) [+ dyn(x)]) / sqrt(n k + 1),
        n = 2 with input-dependent taps (the K and V convolutions of a full layer), 1 without.
        Tap `lag` multiplies x[t - lag]; a sequence starts on zeros. x: [T, C]."""
        k = self.kconv
        T, C = x.shape
        if self.ck.has(prefix + ".static_w"):
            taps = self.W(prefix + ".static_w")                         # [k, C], row = lag
        else:
            taps = self.W(prefix + ".weight").reshape(C, k).flip(1).T   # torch conv1d: last tap = now
        pad = torch.cat([x.new_zeros(k - 1, C), x])
        acc = torch.zeros_like(x)
        dyn = None
        if dyn_in is not None:
            wp = self.W(prefix + ".weight_proj")                        # [heads * k, H]
            heads = wp.shape[0] // k
            dyn = (dyn_in @ wp.T).view(T, heads, k)                     # row head * k + lag
        for lag in range(k):
            xs = pad[k - 1 - lag: k - 1 - lag + T]
            acc = acc + xs * taps[lag]
            if dyn is not None:
                acc = acc + (xs.view(T, heads, -1) * dyn[:, :, lag, None]).view(T, C)
        return (x + acc) / math.sqrt((2 if dyn is not None else 1) * k + 1)

    def control(self, x, msg, layer, which):
        """A loop pass's cross-pass message into a core layer's input:
        (1 - tanh(F g)) x + tanh(G g) msg, g = D [rms(x) | rms(msg)]."""
        p = f"layers.{layer}.controllers.{which}."
        g = torch.cat([rms(x), rms(msg)], -1) @ self.W(p + "down_weight").T
        gain = torch.tanh(g @ self.W(p + "gain_weight").T)
        forget = torch.tanh(g @ self.W(p + "forget_weight").T)
        return x - forget * x + gain * msg

    def attention(self, l, xq, xk, xv, tau):
        p = f"layers.{l}.attn."
        full = l not in self.local
        T = xq.shape[0]
        hd = 128 if "head_dim" not in self.ck.t else self.ck.t["head_dim"]
        if not full:
            hd = self.ck.t.get("swa_head_dim") or hd
        q = (xq @ self.W(p + "wq_du.weight").T).view(T, -1, hd)
        nh = q.shape[1]
        k = self.conv(xk @ self.W(p + "wk_dv.weight").T, p + "k_sconv", xk if full else None).view(T, -1, hd)
        v = self.conv(xv @ self.W(p + "wv_dv.weight").T, p + "v_sconv", xv if full else None).view(T, -1, hd)
        r = (xq @ self.W(p + "wr_du.weight").T).view(T, nh, self.d_rel)
        q = l2clip(q * self.W(p + "q_l2_clip.scale"), self.qk_clip)
        k = l2clip(k, self.qk_clip)
        v = rms(v)                                                     # no gain
        r = l2clip(r * self.W(p + "rel_l2_clip.scale"), self.rel_clip)
        if full and tau is not None:
            q, r = q * tau[:, None, None], r * tau[:, None, None]
        sink = xq @ self.W(p + "attn_sink_proj.weight").T              # [T, nh]
        gate = xq @ self.W(p + "gate_attn.weight").T                   # [T, nh * hd]

        # The relative bias: r . proj[:, d] at distance d = i - j below the extent, 0 beyond it
        # (llama.cpp's zero padded column, vLLM's masked_fill). The extent of a local layer is
        # its window, and both are 1024 in the release.
        extent = self.extent if full else self.window
        proj = self.W(p + "rel_logits_proj.proj")                      # [d_rel, extent]
        table = torch.einsum("thr,re->the", r, proj)                   # [T, nh, extent]
        i = torch.arange(T)[:, None]
        j = torch.arange(T)[None, :]
        dist = i - j
        bias = table.gather(2, dist.clamp(0, extent - 1).expand(nh, T, T).permute(1, 0, 2))
        bias = bias.masked_fill(((dist < 0) | (dist >= extent))[:, None, :], 0.0)  # [T, nh, T]
        visible = dist >= 0
        if not full:
            visible &= dist < self.window
        kk = k.repeat_interleave(nh // k.shape[1], 1)
        vv = v.repeat_interleave(nh // v.shape[1], 1)
        s = torch.einsum("thd,jhd->thj", q, kk) / math.sqrt(hd) + bias
        s = s.masked_fill(~visible[:, None, :], float("-inf"))
        # The sink is a logit of its own per (query, head): it joins the denominator and carries
        # no value. exp(s_j) / (exp(sink) + sum exp(s)).
        m = torch.maximum(s.amax(-1), sink)
        e = torch.exp(s - m[..., None])
        pr = e / (torch.exp(sink - m) + e.sum(-1))[..., None]
        o = torch.einsum("thj,jhd->thd", pr, vv).reshape(T, -1)
        o = o * torch.sigmoid(gate)
        o = o @ self.W(p + "wo_ud.weight").T
        return self.conv(o, f"layers.{l}.attn_sconv")

    def moe(self, l, x):
        p = f"layers.{l}.mlp."
        logits = x @ self.W(p + "gate.weight").T                       # f32 router
        pr = torch.sigmoid(logits)
        sel = (pr + self.W(p + "gate.bias")).topk(self.top_k, -1).indices
        wts = pr.gather(1, sel)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * self.route_scale
        out = torch.zeros_like(x)
        for e in sel.unique().tolist():
            rows, slot = (sel == e).nonzero(as_tuple=True)
            w13 = self.ck.expert(l, "w13", e)
            gate_w, up_w = w13[0::2], w13[1::2]                        # interleaved rows
            xe = x[rows]
            g, u = xe @ gate_w.T, xe @ up_w.T
            # SiTU: the sigmoid reads the raw gate, the softcaps cap gate and up.
            a = self.gate_cap * torch.tanh(g / self.gate_cap) if self.gate_cap > 0 else g
            a = a * torch.sigmoid(g)
            if self.up_cap > 0:
                u = self.up_cap * torch.tanh(u / self.up_cap)
            y = (a * u) @ self.ck.expert(l, "w2", e).T
            out.index_add_(0, rows, y * wts[rows, slot][:, None])
        return out

    # ---------------------------------------------------------------- the graph

    def layer(self, ip, cur, blocks, msgs=None, trace=None):
        full = ip not in self.local
        inp = cur
        if ip > 0:
            mixed = mix(blocks + [cur], self.W(f"attn_res.attn_queries.{ip - 1}").reshape(-1, self.H))
            ins = [mixed[j if full else 0] for j in range(3)]
        else:
            ins = [cur] * 3
        if msgs is not None:
            ins = [self.control(ins[j], msgs[j], ip, w) for j, w in enumerate(("attn", "k", "v"))]
        if full:
            ins = [rms(ins[j], self.W(f"layers.{ip}.{n}_input_norm.weight")) for j, n in enumerate("qkv")]
        else:
            ins = [rms(x, self.W(f"layers.{ip}.attn_norm.weight")) for x in ins]
        a = self.attention(ip, ins[0], ins[1], ins[2], self.tau)
        if ip % self.block == 0:
            blocks.append(inp)       # the first layer of a block closes the previous one
            cur = a
        else:
            cur = inp + a
        ffn_in = cur
        m = mix(blocks + [cur], self.W(f"attn_res.mlp_queries.{ip}").reshape(1, -1))[0]
        if msgs is not None:
            m = self.control(m, msgs[3], ip, "mlp")
        m = rms(m, self.W(f"layers.{ip}.mlp_norm.weight"))
        y = self.conv(self.moe(ip, m), f"layers.{ip}.mlp_sconv")
        cur = ffn_in + y
        if trace is not None:
            trace.append((ip, float(cur.pow(2).mean().sqrt())))
        return cur

    def forward(self, ids, trace=None, aux_at=()):
        """Logits [T, n_vocab] (softcapped, the padded vocabulary dropped). aux_at: the layer
        applications (0 .. 47 under the loop) whose input partial block to return as well, the
        DSpark draft's `target_layer_ids`; the application count itself is the final norm."""
        ids = torch.tensor(ids)
        pos = torch.arange(len(ids), dtype=F32)
        self.tau = None
        if self.floor is not None:
            self.tau = 1 + self.alpha * torch.log(torch.clamp((pos + 1) / self.floor, min=1.0))
        cur = rms(self._embed(ids), self.W("embed_norm.weight"))
        blocks, aux, n_app = [], {}, [0]

        def run(ip, cur, blocks, msgs=None):
            if n_app[0] in aux_at:
                aux[n_app[0]] = cur.clone()
            n_app[0] += 1
            return self.layer(ip, cur, blocks, msgs, trace)

        s, e, n = self.loop if self.loop else (self.L, self.L, 1)
        for ip in range(s):
            cur = run(ip, cur, blocks)
        prelude, pblocks = cur, list(blocks)
        for pss in range(n):
            bank = None
            if pss > 0:
                # The previous pass's bank: its blocks after the first core block, and its output.
                bank = blocks[len(pblocks) + 1:] + [cur]
                cur, blocks = prelude, list(pblocks)
            for ip in range(s, e):
                msgs = mix(bank, self.msg_q[ip]) if bank is not None else None
                cur = run(ip, cur, blocks, msgs)
        for ip in range(e, self.L):
            cur = run(ip, cur, blocks)
        h = rms(mix(blocks + [cur], self.W("attn_res.output_query").reshape(1, -1))[0], self.W("norm.weight"))
        if n_app[0] in aux_at:
            aux[n_app[0]] = h.clone()
        logits = h @ self.ck.get("unembed.weight").T
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits[:, :self.n_vocab], aux

    def _embed(self, ids):
        # One row at a time: the f32 table of the release is 1.85 GB.
        return torch.stack([self.ck.raw_slice("embed.weight", int(i)).to(F32) for i in ids])


def fnv(path):
    h = 0xcbf29ce484222325
    for b in open(path, "rb").read():
        h = ((h ^ b) * 0x100000001b3) & (2**64 - 1)
    return h


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "golden":
        # The golden of the first N bytes of tiny_golden.h's prompt (N = 24 unless given): the
        # full log-probability table at every position, over the vocabulary, as one header. The
        # tiny model is chaotic enough that a 130-token document drifts past any tolerance under
        # bf16; a short prompt keeps the bf16 reference and the f32 golden within a stated bound.
        ckpt_dir, out = sys.argv[2], sys.argv[3]
        n = int(sys.argv[4]) if len(sys.argv) >= 5 else 24
        here = os.path.dirname(os.path.abspath(__file__))
        prompt = json.loads(re.search(r"kTinyGoldPrompt\[\] = (\".*\");",
                                      open(os.path.join(here, "tiny_golden.h")).read()).group(1))
        ids = list(prompt.encode())[:n]
        d = os.path.join(ckpt_dir, "inkling2")
        lp = torch.log_softmax(Inkling2(Ckpt(d)).forward(ids)[0], -1)
        src = open(os.path.abspath(__file__), "rb").read()
        import hashlib
        L = ["/* inkling2_logp_golden.h -- GENERATED by mkinkling2golden.py (sha256",
             f" * {hashlib.sha256(src).hexdigest()}) under torch {torch.__version__}; do not edit. */",
             "#pragma once", "",
             f"/* the prompt: the first {len(ids)} bytes of tiny_golden.h's kTinyGoldPrompt */",
             f"enum {{ kInkling2GoldPositions = {len(ids)}, kInkling2GoldVocab = {lp.shape[1]} }};",
             f"static const char kInkling2GoldPrompt[] = {json.dumps(prompt[:len(ids)])};",
             f"static const float kInkling2LogP[kInkling2GoldPositions][kInkling2GoldVocab] = {{"]
        L += ["{" + ",".join(f"{float(x):.6g}" for x in row) + "}," for row in lp]
        L += ["};", ""]
        open(out, "w").write("\n".join(L))
        return
    if len(sys.argv) >= 4 and sys.argv[1] == "real":
        d, ids = sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
        top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
        trace = [] if "--layers" in sys.argv else None
        torch.set_num_threads(os.cpu_count())
        lp = torch.log_softmax(Inkling2(Ckpt(d)).forward(ids, trace)[0], -1)
        v, i = torch.topk(lp, top, -1)
        for p in range(len(ids)):
            print(json.dumps({"pos": p, "id": ids[p], "top": i[p].tolist(),
                              "logp": [round(float(x), 5) for x in v[p]]}))
        if trace is not None:
            for n, (ip, r) in enumerate(trace):
                print(json.dumps({"app": n, "layer": ip, "rms": round(r, 5)}))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
