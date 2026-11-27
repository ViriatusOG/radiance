#!/usr/bin/env python3
"""gg_mr_forms.py -- libr4d/r4d_ds41_gg_mr_forms.h, the measured table of multi-row ggml GEMM forms, from
r4d_selftest --perfmr.

    python3 tools/gg_mr_forms.py [--min-gain PCT] [--out FILE] BENCH.txt...

The multi-row form runs only where the table has an entry, and the stock kernel everywhere else, so an entry is
chosen against the STOCK kernel, not against the form's rule.

Reads every MRPERF line (r4d_selftest --perfmr: one (type, shape, M) with the stock rule "stock", the multi-row
form's rule "mr-rule" and every pinned multi-row form H<h>R<r>[c<mt>][w<wn>][-w64], each "name us [form]",
"OUTPUT DIFFERS" after a form whose bytes were not the stock rule's) in the files given; a queue log's lines work
too. One key is (ggml type, f32 out, N, K, M); a shape whose name ends in "head" is logits_gemm_ggml (f32 out).
The table gets a wave32 form for a key only when:

  - its bytes matched the stock kernel's in every run that measured the key;
  - it beats the stock kernel by at least --min-gain percent (default 3) in the mean of the runs' form/stock
    time ratios, and is not slower than the stock kernel in any run. The best such form by that mean is kept
    ("mr-rule" stands for its own resolved H and R with the rule's MT cap and WN, which mr_pick reproduces).

wave64 forms (-w64) are bench pins, not a runtime form, and are left out. Runs of consecutive measured M with the
same form become one M band; an M that was not measured gets no entry (the stock kernel)."""
import argparse
import re
import sys

TYPES = {"q8_0": "GG_Q8_0", "q5_K": "GG_Q5_K", "q6_K": "GG_Q6_K"}
HEAD = re.compile(r"MRPERF (\S+) (\S+) (\d+)x(\d+) M=(\d+)(?: \([\d.]+ MiB\))?([^:]*):(.*)$")
FORM = re.compile(r" (\S+) (-?[\d.]+)(?: \[[^\]]*\])?( OUTPUT DIFFERS)?;")
PIN = re.compile(r"^H(\d+)R(\d+)(?:c(\d+))?(?:w(\d+))?$")


def form_of(nm, N):
    if nm == "mr-rule":
        return (4, 2 if N >= 4096 else 1, 0, 0)
    p = PIN.match(nm)
    if not p:
        return None
    return (int(p.group(1)), int(p.group(2)), int(p.group(3) or 0), int(p.group(4) or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-gain", type=float, default=3.0)
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    runs, names, srcs = {}, {}, []
    for path in a.files:
        n = 0
        for line in open(path, errors="replace"):
            i = line.find("MRPERF ")
            if i < 0:
                continue
            h = HEAD.match(line[i:].rstrip("\n"))
            if not h:
                continue
            ty, shape, N, K, M = h.group(1), h.group(2), int(h.group(3)), int(h.group(4)), int(h.group(5))
            if ty not in TYPES:
                continue
            f32 = 1 if shape.endswith("head") else 0
            forms = {}
            for f in FORM.finditer(h.group(7)):
                us = float(f.group(2))
                if us > 0:
                    forms[f.group(1)] = (us, bool(f.group(3)))
            if "stock" not in forms:
                continue
            key = (ty, f32, N, K, M)
            runs.setdefault(key, []).append(forms)
            names.setdefault((ty, f32, N, K), set()).add(shape)
            n += 1
        srcs.append("%s (%d MRPERF)" % (path, n))
    best_of = {}
    for key, rs in runs.items():
        ty, f32, N, K, M = key
        best = None
        cands = {nm for forms in rs for nm in forms if nm == "mr-rule" or PIN.match(nm)}
        for nm in sorted(cands):
            fm = form_of(nm, N)
            ratios, ok = [], True
            for forms in rs:
                if nm not in forms:
                    ok = False
                    break
                us, differs = forms[nm]
                if differs:
                    ok = False
                    break
                ratios.append(us / forms["stock"][0])
            if not ok or not ratios or max(ratios) > 1.0:
                continue
            mean = sum(ratios) / len(ratios)
            if (1.0 - mean) * 100.0 < a.min_gain:
                continue
            if best is None or mean < best[1] - 1e-12:
                best = (fm, mean, len(ratios), nm)
        if best:
            stock_us = sum(f["stock"][0] for f in rs) / len(rs)
            best_of[key] = best + (stock_us,)
    rows = []
    groups = sorted({k[:4] for k in best_of}, key=lambda k: (list(TYPES).index(k[0]), k[1], k[2], k[3]))
    for g in groups:
        ms = sorted(k[4] for k in best_of if k[:4] == g)
        i = 0
        while i < len(ms):
            j = i
            fm = best_of[g + (ms[i],)][0]
            while j + 1 < len(ms) and ms[j + 1] == ms[j] + 1 and best_of[g + (ms[j + 1],)][0] == fm:
                j += 1
            ty, f32, N, K = g
            parts = []
            for m in ms[i:j + 1]:
                b = best_of[g + (m,)]
                parts.append("M%d %.1fus %+.0f%%" % (m, b[4], (b[1] - 1) * 100))
            note = "%s %s %s: %s" % (ty, "/".join(sorted(names[g])), b[3], ", ".join(parts))
            rows.append((TYPES[ty], f32, N, K, ms[i], ms[j]) + fm + (note,))
            i = j + 1
    out = sys.stdout if a.out == "-" else open(a.out, "w")
    out.write("/* r4d_ds41_gg_mr_forms.h -- GENERATED by tools/gg_mr_forms.py; edit the bench data, not this file.\n"
              " *\n"
              " * The multi-row form (r4d_ds41_ggml_mr.hip, mr_pick) a (ggml type, f32 out, N, K, M band) for\n"
              " * gemm_nt_ggml (f32 0) and logits_gemm_ggml (f32 1); every other key runs the stock kernel. A form here had\n"
              " * the stock kernel's bytes at its exact (type, shape, M) in every run and beat the stock kernel by >= %.1f %%\n"
              " * (mean form / stock time) without losing in any run. MT cap / WN 0 = the form's rule. Notes give the stock\n"
              " * time and the form's change a measured M.\n"
              " *\n"
              " * Sources:\n" % a.min_gain)
    for s in srcs:
        out.write(" *   %s\n" % s)
    out.write(" * %d entr%s. */\n" % (len(rows), "y" if len(rows) == 1 else "ies"))
    out.write("#define R4D_GG_MR_FORMS_N %d\n" % len(rows))
    out.write("static const MrTab kMrForms[] = {\n")
    out.write("    /* type   f32      N       K  M lo  M hi  H  R  MT cap WN */\n")
    for r in rows:
        out.write("    { %-8s %d, %6d, %6d, %3d, %3d, %d, %d, %3d, %d },   /* %s */\n" % ((r[0] + ",",) + r[1:10] + (r[10],)))
    if not rows:
        out.write("    { -1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },   /* none: the stock kernel throughout */\n")
    out.write("};\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
