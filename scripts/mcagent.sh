#!/bin/sh
# What speculation is worth on an agent workload: /v1/chat/completions with tools, a constrained
# decode. Arms alternate across server restarts because n_spec is a server-level setting.
WORK=$(mktemp -d "${TMPDIR:-/tmp}/mcagent.XXXXXX"); trap 'rm -rf "$WORK"' EXIT INT TERM
T='[{"type":"function","function":{"name":"write","description":"write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"bash","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}},{"type":"function","function":{"name":"read","description":"read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]'
Q="Write a long, heavily commented C implementation of a chess board representation with FEN parsing, then explain every design decision in detail."
U="http://127.0.0.1:8111"
m() { curl -s -m 10 "$U/metrics" | grep -F "$1" | grep -v '^#' | awk '{print $NF}' | head -1; }
arm() { # $1 label  $2 extra
  P=8111 TP=2 MLEN=16384 SEQS=8 EXTRA="$2" sh "$(dirname "$0")/mcserve.sh" >/dev/null 2>&1 || { echo "$1 DIED"; return; }
  curl -s -m 300 -o /dev/null "$U/v1/completions" -H 'Content-Type: application/json' -d '{"model":"q","prompt":"warm","max_tokens":32,"temperature":0}'
  s0=$(m 'radiance:engine_steps_decode_total'); d0=$(m 'radiance:decode_tokens_total')
  t0=$(date +%s.%N)
  curl -s -m 600 -o "$WORK/a.out" "$U/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"tools\":$T,\"max_tokens\":900,\"temperature\":0}"
  t1=$(date +%s.%N)
  s1=$(m 'radiance:engine_steps_decode_total'); d1=$(m 'radiance:decode_tokens_total')
  n=$(jq -r '.usage.completion_tokens // 0' "$WORK/a.out")
  awk -v l="$1" -v t0="$t0" -v t1="$t1" -v n="$n" -v ds=$((s1-s0)) -v dd=$((d1-d0)) 'BEGIN{
    if (ds<=0) { printf "%-14s NO STEPS\n", l; exit }
    printf "%-14s %7.3f ms/step  %6.3f tok/step  %6.1f tok/s\n", l, 1000*(t1-t0)/ds, dd/ds, n/(t1-t0) }'
}
for pass in 1 2; do arm "spec-off" "--num-speculative-tokens 0"; arm "spec-auto" ""; done
pkill -f "bin/radianc[e]"
