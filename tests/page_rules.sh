#!/bin/sh
# page_rules.sh — the page rules that were each broken in production, stated where a build
# can check them. Every rule here came from a report of a tile that read wrong on a live lane.
#
# These read dashboard/src, not the shipped bytes, because the two tests beside this one close the
# rest of the ring: `dashboard_page` fails when the sources no longer match the digest baked into
# core/server/dashboard_page.h, and `artifact_sums` fails when that header differs from the hash in
# tests/ARTIFACTS. Source rule + those two digests means the served page obeys these too.
set -u
ROOT=${1:-$(dirname "$0")/..}
PAGE="$ROOT/dashboard/src/pages/performance.tsx"
BITS="$ROOT/dashboard/src/components/bits.tsx"
fails=0
bad() { fails=$((fails + 1)); echo "  FAIL $1"; }
good() { echo "  ok   $1"; }

[ -f "$PAGE" ] || { echo "no $PAGE"; exit 77; }

echo "== a tile's figure is the newest sample, never the newest one that had a value =="
# Series.last() scans back 300 samples (five minutes) for a non-null entry. The engine omits a
# column from a sample that measured nothing -- history.cpp drops prefillRate when zero prompt
# tokens were processed, prefill when no prefill-bearing step ran, step when no decode step ran,
# stepAny when no step ran -- so a tile built on last() keeps the previous figure on screen long
# after the work stopped. Rates may then read 0 (that is what happened); a duration reads a dash,
# because 0 ms is a step that cannot exist.
for col in step stepAny prefill prefillRate; do
  if grep -q "last(\"$col\")" "$PAGE"; then
    bad "\"$col\" is read with last(): the tile holds its stale figure while the column is null"
  else
    good "\"$col\" is read from the newest sample"
  fi
done
# The two draft tiles are the deliberate exception: acceptance and tokens-per-step are UNDEFINED,
# not zero, when a sample drafted nothing, and 0% acceptance reads as a drafter that broke. They
# freeze on purpose, so they are allowed to keep last().

echo "== a tile's detail line reads engine stats only =="
# A page-side window scan can produce a "reference" a poll cannot honestly be read against: one fast
# stretch sets the page-side peak and a page reload resets it, so a percent-against-best figure reads
# as a lie exactly when the lane got FASTER. The Step tile's detail is the engine's decode-step best
# and nothing else; a richer drift figure belongs in a history section, not a live tile.
grep -q 'of best' "$PAGE" \
  && bad "the page computes a percent-against-best drift figure" \
  || good "no page-side percent-against-best figure"

echo "== a tile's figure is a bare number; its unit is the tile's suffix =="
# A tile renders `<figure> <unit>`: the figure string, then the metric's `unit` as a separate span.
# So a figure that goes through a formatter which bakes the unit into the number prints it twice --
# the Step tile's fMs gave "25.34 ms" and the tile's suffix added a small ms after it. Formatters
# that append a unit belong on detail lines, which append nothing after them.
if grep -q 'id: "step"' "$PAGE"; then
  if sed -n '/id: "step"/,/lines:/p' "$PAGE" | grep -qE 'return \[fin\(v\) \? fMs\('; then
    bad "the Step tile's figure goes through a formatter that appends the unit the tile appends"
  else
    good "the Step tile's figure is a bare number, its unit suffix the only unit"
  fi
else
  bad "the Step tile is gone from the page"
fi

echo "== a rate's graph never joins across its gaps, and matches its own tile =="
# The prefill tile was on the other side of that bug: the graph drew prefillRate with uPlot's
# spanGaps switched on (the `join` prop), so it connected the two values around every idle gap and
# showed a flat ~3,500 tok/s prefill line across four minutes the lane spent decoding. Decode never
# does that: its line comes from a column history.cpp withholds unless steps_decode advanced. So a
# rate may join nothing -- `join` stays for sparse EVENTS (jobTime), where a gap is just a moment
# with no event -- and the tile and its graph must read the same column, or the pair cannot be read
# together (a tile at 0 beside a graph edge at 3,500 is two different figures, not a contradiction
# the reader can resolve).
if grep -qE '\{ id: "(prefill|decode|reqs|step)".*join: true' "$PAGE"; then
  bad "a rate tile passes join: true -- spanGaps draws throughput across samples that measured none"
else
  good "the rate graphs leave a gap where nothing was measured"
fi
if grep -qE 'id: "prefill".*col\("prefill"\).*one\("prefill"' "$PAGE"; then
  good "the Prefill tile and its graph read the same gated column"
else
  bad "the Prefill tile and its graph read different columns"
fi

echo "== a tile does not clip its own figure or its detail line =="
# `truncate` is overflow:hidden + nowrap. On the detail line it cut the drift clause off after the
# first figure ("best 11.45 ms · r…"); on the headline it shrank the figure before its unit sibling,
# so 12.91 ms came out as "12.91 ⋯". Both tile renderers had it, and fixing only components/bits.tsx
# changed nothing on the Performance page, which builds its own tile strip.
for f in "$PAGE" "$BITS"; do
  [ -f "$f" ] || continue
  # The tile's title keeps `truncate`: a label ellipsizing is fine. A figure and the detail line
  # under it are not -- those carry the numbers, and an ellipsis in one is a missing measurement.
  if grep -q 'className="truncate"{' "$f"; then
    bad "${f##*/} truncates the headline figure"
  elif grep -q 'min-h-4 truncate' "$f"; then
    bad "${f##*/} truncates its detail line"
  else
    good "${f##*/} lets its figure and detail line wrap"
  fi
done

echo
if [ "$fails" -ne 0 ]; then echo "page_rules: $fails check(s) failed"; exit 1; fi
echo "page_rules: all checks passed"
