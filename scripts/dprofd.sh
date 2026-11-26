#!/bin/sh
# Per-op device timing (--profile-ops) at depth and concurrency on the 27B dense model with its
# DFlash2 drafter: C requests of a ~16K-token prompt (PROMPT=<file>), so context-scaling ops
# (above all attn_paged) are timed at a realistic KV depth.
#
# The counters are cumulative and the run starts with prefill (~30 chunked steps of 4096 tokens),
# so one table mixes prefill and decode. The server dumps every EVERY steps
# (RADIANCE_PROFILE_EVERY) into the log; difference two dumps taken after prefill ended to get a
# decode-only table.
P=8129
WORK=$(mktemp -d "${TMPDIR:-/tmp}/dprofd.XXXXXX"); trap 'rm -rf "$WORK"' EXIT INT TERM
C="${C:-8}"
EVERY="${EVERY:-20}"
MT="${MT:-400}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
cd "$BUILD" || exit 1
pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done; sleep 3
: > "$HOME/dprofd.log"
GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 RADIANCE_PROFILE_EVERY="$EVERY" \
"$ROOT/scripts/radiance-flags" ./bin/radiance --model "${M:-$HOME/models/rad/qwen3.8-27b-iq8r-dflash.rad}" \
    --radiance-home radiance_home \
    --max-num-seqs 8 --max-model-len 24000 --checkpoint-slots 0 --no-prefix-cache \
    --tp 2 --tp-wire wht6 --tp-wire-min-kb 128 \
    --max-num-batched-tokens 4096 --num-speculative-tokens ${DEPTH:-auto} \
    --profile-ops --port $P >> "$HOME/dprofd.log" 2>&1 &
for i in $(seq 1 150); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] && break
  pgrep -f "bin/radianc[e]" >/dev/null || { echo "DIED"; grep -iE "^E " "$HOME/dprofd.log" | tail -5; exit 1; }
  sleep 5
done
PROMPT="${PROMPT:-$HOME/lpcode16k.txt}"
[ -s "$PROMPT" ] || { echo "no prompt file at $PROMPT"; exit 1; }
jq -Rs "{model:\"m\",prompt:.,max_tokens:$MT,temperature:0}" < "$PROMPT" > "$WORK/dpd.json"
PIDS=; i=0
while [ $i -lt $C ]; do
  curl -s -m 1800 http://localhost:$P/v1/completions -H "Content-Type: application/json" -d @"$WORK/dpd.json" -o "$WORK/dpd_$i.out" &
  PIDS="$PIDS $!"; i=$((i+1))
done
wait $PIDS
echo "C=$C completion_tokens $(jq -r '.usage.completion_tokens' "$WORK/dpd_0.out")"
sleep 2
pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done
grep -c "per-op device timing, rank 0" "$HOME/dprofd.log" | sed 's/^/dumps: /'
echo DPROFDDONE
