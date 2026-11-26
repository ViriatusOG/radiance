#!/bin/sh
# MiniCPM5-2B decode under rocprofv3 kernel tracing, mainly for the BLOCKS column: a 2B dense
# model at tp2 has small N per rank, so a decode GEMM may launch too few workgroups (2048/tile_n
# on a 64-CU card leaves most of it idle).
#   N=<decode tokens> OUT=<trace dir> scripts/mctrace.sh
P=8102
WORK=$(mktemp -d "${TMPDIR:-/tmp}/mctrace.XXXXXX"); trap 'rm -rf "$WORK"' EXIT INT TERM
M="${M:-$HOME/models/rad/minicpm5-2b-dspark.v2.rad}"
N="${N:-140}"
OUT="${OUT:-$HOME/traces/mc}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
cd "$BUILD" || exit 1
pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done; sleep 2
rm -rf "$OUT"; mkdir -p "$OUT"
env GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 $EXTRA_ENV \
  "$ROOT/libr4d/isa/trace.sh" "$OUT" -- \
  "$ROOT/scripts/radiance-flags" ./bin/radiance --model "$M" \
    --radiance-home radiance_home \
    --max-num-seqs 8 --max-model-len 16384 --tp 2 \
    --gpu-headroom-mib 160 --kv-cache-dtype i8 --max-num-batched-tokens 512 \
    $EXTRA --port $P > "$HOME/mctrace.log" 2>&1 &
for i in $(seq 1 90); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] && break
  pgrep -f "bin/radianc[e]" >/dev/null || { echo DIED; tail -5 "$HOME/mctrace.log"; exit 1; }
  sleep 2
done
curl -s -m 900 http://localhost:$P/v1/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"m\",\"prompt\":\"Write a long essay about the sea.\",\"max_tokens\":$N,\"temperature\":0,\"ignore_eos\":true}" -o "$WORK/mct.out"
echo "completion=$(jq -r .usage.completion_tokens "$WORK/mct.out" 2>/dev/null)"
pkill -f "bin/radianc[e]"
for i in $(seq 1 60); do pgrep -f "bin/radianc[e]" >/dev/null || break; sleep 1; done
for i in $(seq 1 120); do [ -s "$OUT/k_kernel_trace.csv" ] && break; sleep 2; done
for i in $(seq 1 60); do pgrep -f rocprofv[3] >/dev/null || break; sleep 2; done
echo TRACEDONE
