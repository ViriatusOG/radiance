#!/usr/bin/env python3
"""dashboard_embed.py write|check -- the dashboard's page as the header the engine compiles in.

The page is built from dashboard/ by Node (npm run build -> dashboard/dist/index.html: one document
with its script, style and type inlined). The engine's build has no Node, so the built page is
committed as core/server/dashboard_page.h and this keeps the two together:

  write   turns dashboard/dist/index.html into the header, records a digest of the sources it was
          built from, and copies the build's list of bundled packages and their licences to
          dashboard/THIRD-PARTY-LICENSES.txt (`npm run embed` in dashboard/ builds and writes)
  check   exits 1 if dashboard/'s sources no longer have the recorded digest: someone changed the
          page's source and did not rebuild the header (the ctest `dashboard_page`)

The digest lives inside the header and only proves sources and claim agree; the served bytes are
hashed from outside by tests/ARTIFACTS (the ctest `artifact_sums`), so tampering with the header
alone fails. After a real rebuild, re-record both in the same commit:
  tests/artifact_sums.sh . --record core/server/dashboard_page.h
Name the artifacts. Reading the manifest into the command that overwrites it truncates it first, so
the substitution reads an empty file, sha256sum hashes its own empty stdin, and the manifest ends up
"e3b0c442…  -" -- the hash of nothing against a file named "-", which `sha256sum -c` answers with the
same empty stdin. That line checks nothing and passes forever (artifact_sums.sh refuses it).
"""
import hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH = os.path.join(ROOT, "dashboard")
HEADER = os.path.join(ROOT, "core", "server", "dashboard_page.h")
DELIM = "RADPAGE"
TOP = ["index.html", "package.json", "package-lock.json", "vite.config.ts", "components.json",
       "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json"]


def sources():
    out = [os.path.join(DASH, f) for f in TOP if os.path.exists(os.path.join(DASH, f))]
    for d, _, files in os.walk(os.path.join(DASH, "src")):
        out += [os.path.join(d, f) for f in files]
    return sorted(out)


def digest():
    h = hashlib.sha256()
    for p in sources():
        h.update(os.path.relpath(p, DASH).replace(os.sep, "/").encode() + b"\0")
        h.update(hashlib.sha256(open(p, "rb").read()).digest())
    return h.hexdigest()


# The interfaces of dashboard/src/lib/api.ts that describe a route's object, and the name the
# engine's test knows each object by.
READS = [("Stats", "stats"), ("RunningRequest", "requests"), ("DraftPosition", "draft_positions"),
         ("Experts", "experts"), ("ExpertTier", "tiers"), ("Sessions", "sessions"), ("Session", "session")]


def reads():
    """Every "object.field" the page reads: the members of those interfaces, nested ones aside."""
    src = open(os.path.join(DASH, "src", "lib", "api.ts"), encoding="utf-8").read()
    out = []
    for iface, obj in READS:
        at = src.index("export interface %s {" % iface)
        i = src.index("{", at) + 1
        depth, word, start = 0, "", True
        while depth >= 0:
            c = src[i]
            if c in "{([<":
                depth += 1
            elif c in "})]>":
                depth -= 1
            elif depth == 0:
                if c in ";\n":
                    word, start = "", True
                elif start and (c.isalnum() or c == "_"):
                    word += c
                elif c in "?:" and start and word:
                    out.append(obj + "." + word)
                    word, start = "", False
                elif not c.isspace():
                    start = False if not word else start
            i += 1
    return out


def recorded():
    with open(HEADER, encoding="utf-8") as f:
        for line in f:
            if "kPageSources" in line and '"' in line:
                return line.split('"')[1]
    return None


def write():
    page = open(os.path.join(DASH, "dist", "index.html"), encoding="utf-8").read()
    # The licences of what the page bundles, as the build listed them, go where the header goes.
    notices = open(os.path.join(DASH, "dist", "THIRD-PARTY-LICENSES.txt"), encoding="utf-8").read()
    with open(os.path.join(DASH, "THIRD-PARTY-LICENSES.txt"), "w", encoding="utf-8") as f:
        f.write(notices)
    if ")" + DELIM + '"' in page:
        sys.exit("the page contains the raw string's closing delimiter")
    # A string literal's length is bounded on some compilers; adjacent literals join without one.
    step = 12000
    parts = [page[i:i + step] for i in range(0, len(page), step)]
    body = "\n".join('R"%s(%s)%s"' % (DELIM, p, DELIM) for p in parts)
    with open(HEADER, "w", encoding="utf-8") as f:
        f.write("""/* core/server/dashboard_page.h -- the dashboard's page: one document with its script, style and
 * type inlined, so the engine serves it from memory with no asset directory and no fetch from
 * another origin (it runs on machines with no route to one).
 *
 * GENERATED by tools/dashboard_embed.py from the build of dashboard/; edit the sources there and
 * run `npm run embed` in that directory. The ctest `dashboard_page` fails while the sources and
 * this file disagree.
 */
#pragma once

/* A digest of dashboard/'s sources as of this build of the page. */
static const char* const kPageSources = "%s";

/* Every field the page reads of a route's object, as "object.field": the members of the
 * interfaces in dashboard/src/lib/api.ts, which the page's compiler holds it to. Ends with null. */
static const char* const kPageReads[] = {
%s
    nullptr
};

static const char* const kPage =
%s;
""" % (digest(), "\n".join('    "%s",' % r for r in reads()), body))
    print("wrote %s: %d bytes of page in %d pieces, sources %s" % (os.path.relpath(HEADER, ROOT), len(page.encode()), len(parts), digest()[:12]))


def check():
    have, want = recorded(), digest()
    if have != want:
        print("core/server/dashboard_page.h was built from other sources than dashboard/ holds now\n"
              "  recorded %s\n  sources  %s\nrun `npm run embed` in dashboard/ and commit the header" % (have, want))
        sys.exit(1)
    print("dashboard_page.h matches dashboard/ (%s)" % want[:12])


if __name__ == "__main__":
    {"write": write, "check": check}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
