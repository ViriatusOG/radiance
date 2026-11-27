#!/bin/bash
# docker/llvm-check.sh BUILD -- check the image's HIP compiler on the build's compile command for
# r4d_hc_bf16.hip, a file whose device code is not repeatable under ROCm's stock clang. The command
# is rerun four ways, outputs in a scratch directory:
#   1. pairing off, by ROCm's clang and by the image's: the device code must match byte for byte,
#      so the llvm stage's libLLVM is ROCm's apart from the patch;
#   2. pairing on, as the build ran it, twice by the image's clang: both must match the build's own
#      object, so pairing is repeatable.
# A mismatch fails the image build.
set -euo pipefail
B=$1
STOCK=/opt/rocm/lib/llvm/bin/clang++
IMAGE=/opt/rocm-repeatable/lib/llvm/bin/clang++
OBJCOPY=/opt/rocm/lib/llvm/bin/llvm-objcopy
O=libr4d/CMakeFiles/libr4d.dir/r4d_hc_bf16.hip.o
fail() { echo "llvm-check: $*" >&2; exit 1; }

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cd "$B"
# A build for one card (`docker/build.sh -t gfx1151`) compiles no libr4d -- that library is
# another family's -- so there is no compile command to rerun and nothing to check: the
# pairing this script verifies lives in device code this build does not contain.
cmds=$(ninja -t commands libr4d 2>/dev/null) || {
    echo "llvm-check: this build has no libr4d (its GPU targets are filtered); skipped"
    exit 0
}
line=$(grep -m1 -F -- "-o $O " <<< "$cmds") || fail "no compile command for $O"
eval "set -- $line"
args=("$@")
# The build compiles through ccache (Dockerfile); check the compiler behind it.
[ "${args[0]##*/}" != ccache ] || args=("${args[@]:1}")
[ "${args[0]}" = "$IMAGE" ] || fail "the build compiled with ${args[0]}, not $IMAGE"
for a in "${args[@]}"; do
    case "$a" in *amdgpu-enable-vopd*) fail "the build turned VOPD pairing off ($a)" ;; esac
done

# variant NAME COMPILER [FLAG]: rerun the build's command with COMPILER, object and depfile in the
# scratch directory and FLAG appended; dump the object's device code to NAME.fb.
variant() {
    local name=$1 cc=$2 extra=${3:-} i=1
    local v=("$cc")
    while [ $i -lt ${#args[@]} ]; do
        case "${args[$i]}" in
            -o)  v+=(-o "$T/$name.o");  i=$((i + 2)); continue ;;
            -MF) v+=(-MF "$T/$name.d"); i=$((i + 2)); continue ;;
            *)   v+=("${args[$i]}") ;;
        esac
        i=$((i + 1))
    done
    [ -z "$extra" ] || v+=("$extra")
    "${v[@]}" > "$T/$name.log" 2>&1 || { cat "$T/$name.log" >&2; return 1; }
    "$OBJCOPY" --dump-section=.hip_fatbin="$T/$name.fb" "$T/$name.o" /dev/null
}

OFF=-mllvm=-amdgpu-enable-vopd=false
pids=()
variant stock_off "$STOCK" "$OFF" & pids+=($!)
variant image_off "$IMAGE" "$OFF" & pids+=($!)
variant image_on1 "$IMAGE" & pids+=($!)
variant image_on2 "$IMAGE" & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || fail "a compile of $O failed"; done
"$OBJCOPY" --dump-section=.hip_fatbin="$T/build.fb" "$O" /dev/null

cmp -s "$T/stock_off.fb" "$T/image_off.fb" \
    || fail "with pairing off the image's clang and ROCm's emit different device code: the llvm stage did not rebuild ROCm's libLLVM"
cmp -s "$T/build.fb" "$T/image_on1.fb" && cmp -s "$T/build.fb" "$T/image_on2.fb" \
    || fail "with pairing on, three compiles of $O emitted different device code"
echo "llvm-check: r4d_hc_bf16.hip -- pairing off, ROCm's clang and the image's agree; pairing on, three compiles agree"
