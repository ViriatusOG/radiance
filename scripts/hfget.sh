#!/bin/sh
# hfget.sh -- fetch a Hugging Face repo with only curl and jq.
#
# The engine and its tooling do not depend on Python, so `huggingface-cli` is not assumed. The API
# returns a JSON file listing and every file has a stable resolve/ URL.
#
#   scripts/hfget.sh <repo-id> [dest-dir] [revision]
#
# Set HF_TOKEN for a gated or private repo. Re-running resumes: a file whose local size matches the
# listing is skipped and a short one is continued with `curl -C -`.
#
# A changed file is fetched again even at the same size. A complete run records each file's content
# key (the LFS sha256, or the size for a small file) in <dest>/.hfget-manifest, and a later run
# compares the listing against it. The manifest is written only when every file arrived, so its
# presence means the directory is a complete copy of some revision. An LFS file's key is verified:
# the download is hashed and compared against the oid the listing claims before the file is kept,
# so the hub cannot hand back one file for another and a resumed transfer cannot splice into a
# plausible size.
#
# A file whose listed name would land outside the destination -- absolute, carrying a `..`
# segment, holding control characters, or naming hfget's own manifest -- is not a repo listing:
# the whole listing is refused. Skipping such an entry silently would leave a directory that
# reads as complete and is not.
#
# Exit status: 0 complete; 3 nothing on disk was changed (the listing could not be read, or what is
# still to fetch does not fit), so a complete earlier copy is intact; anything else, a transfer
# failed and a file may be partial (re-run to resume). `radiance --model <repo-id>` serves the
# earlier copy on 3 and refuses on the rest.
#
# Duplicate weight formats are skipped: when the listing holds any .safetensors, other weight
# formats (pickle .bin, ONNX, GGUF, ...) are not fetched. A .pth is a duplicate only when it is the
# model's own weights again (`pytorch_model*.pth`, `consolidated*.pth`); any other is a part the
# safetensors do not hold (Fish Audio drama-3's codec.pth) and is fetched, as the checkpoint reader
# reads it (core/format/torch_zip.h).
#
# HFGET_JOBS concurrent transfers, default 4. These are long sequential reads: a few streams
# saturate the link and more only make the destination seek.
set -e

# ---- child mode: one file, named by "<size><tab><oid><tab><path>" in $1, everything else inherited.
# Size comes first because a non-LFS file has no oid and busybox `xargs -I` strips leading blanks,
# so a line starting with an empty oid field would arrive one tab short.
# sh cannot export a function, so each file is a re-entry of this script under `xargs -P`.
if [ -n "$HFGET_CHILD" ]; then
    size=${1%%	*}; rest=${1#*	}; oid=${rest%%	*}; path=${rest#*	}
    # The parent has vetted every name; this holds anyway, because $out is written straight
    # from it and an escape writes outside the directory the operator named.
    case "$path" in
        ""|/*|*/|.|..|../*|*/../*)
            echo "hfget: refusing a file named '$path', which does not land inside the destination" >&2
            exit 1;;
    esac
    key=${oid:-$size}                                  # LFS: sha256, non-LFS: size
    out=$HFGET_DEST/$path
    man=$HFGET_DEST/.hfget-manifest
    rec=""; [ -f "$man" ] && rec=$(p="$path" awk -F'\t' '$2==ENVIRON["p"]{print $1; exit}' "$man")
    if [ -f "$out" ] && [ "$(wc -c < "$out")" = "$size" ]; then
        if [ -z "$rec" ] || [ "$rec" = "$key" ]; then  # warm: size ok AND content-key matches (or unrecorded)
            echo "  = $path"
            printf '%s\t%s\n' "$key" "$path" >> "$HFGET_MAN_OUT"
            exit 0
        fi
        rm -f "$out"                                   # same size, new content: resume is size-based, so start fresh
        echo "  ~ $path (same size, new content)"
    elif [ -f "$out" ]; then
        echo "  + $path (resuming at $(wc -c < "$out") of $size)"
    else
        echo "  > $path"
    fi
    mkdir -p "$(dirname "$out")"
    if [ -n "$HF_TOKEN" ]; then
        curl -sSL --fail --retry 5 --retry-delay 5 --retry-all-errors -C - \
             -H "Authorization: Bearer $HF_TOKEN" \
             -o "$out" "https://huggingface.co/$HFGET_REPO/resolve/$HFGET_REV/$path"
    else
        curl -sSL --fail --retry 5 --retry-delay 5 --retry-all-errors -C - \
             -o "$out" "https://huggingface.co/$HFGET_REPO/resolve/$HFGET_REV/$path"
    fi
    got=$(wc -c < "$out")
    [ "$got" = "$size" ] || { echo "hfget: $path is $got bytes, listing says $size" >&2; exit 1; }
    # An LFS name carries the sha256 the hub claims. Believe it only after computing it: the
    # size check alone would take a swapped file at the same length, and a spliced resume at a
    # resumed length. A mismatch deletes the file -- a fresh transfer is the only repair -- and
    # nothing reaches the manifest.
    if [ -n "$oid" ]; then
        got=$(sha256sum "$out" | awk '{print $1}')
        [ "$got" = "${oid#sha256:}" ] || {
            echo "hfget: $path hashes to $got, the listing claims ${oid#sha256:}" >&2
            rm -f "$out"
            exit 1; }
    fi
    printf '%s\t%s\n' "$key" "$path" >> "$HFGET_MAN_OUT"
    exit 0
fi

# ---- parent ------------------------------------------------------------------------------------
repo=$1
[ -n "$repo" ] || { echo "usage: hfget.sh <repo-id> [dest-dir] [revision]" >&2; exit 2; }
dest=${2:-$HOME/models/$repo}
rev=${3:-main}
jobs=${HFGET_JOBS:-4}
command -v jq >/dev/null || { echo "hfget: jq is required" >&2; exit 2; }
command -v curl >/dev/null || { echo "hfget: curl is required" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "hfget: sha256sum is required to verify LFS files" >&2; exit 2; }

echo "hfget: listing $repo@$rev"
# The listing is paginated (fifty entries per page, next page in a `Link: ...; rel="next"`
# header). Reading only the first page would silently yield an incomplete checkpoint whose files
# all pass the size check, so follow the cursor until there is no next link.
api="https://huggingface.co/api/models/$repo/tree/$rev?recursive=true&expand=true"
listing='[]'
page=0
while [ -n "$api" ]; do
    hdr=$(mktemp "${TMPDIR:-/tmp}/hfhdr.XXXXXX")
    # Report a failed request here rather than letting set -e exit silently with curl's status;
    # exit 3, since nothing has been written yet.
    st=0
    if [ -n "$HF_TOKEN" ]; then
        body=$(curl -sSL --fail-with-body -D "$hdr" -H "Authorization: Bearer $HF_TOKEN" "$api") \
            || st=$?
    else
        body=$(curl -sSL --fail-with-body -D "$hdr" "$api") || st=$?
    fi
    [ "$(printf '%s' "$body" | jq -r 'type' 2>/dev/null)" = "array" ] || {
        rm -f "$hdr"
        echo "hfget: listing $repo@$rev failed (curl exit $st)" >&2
        [ -n "$body" ] && {
            echo "hfget: $(printf '%s' "$body" | jq -r '.error // .message // .' 2>/dev/null)" >&2
            echo "hfget: a gated or private repo needs HF_TOKEN in the environment" >&2; }
        exit 3; }
    listing=$(printf '%s\n%s' "$listing" "$body" | jq -s 'add')
    api=$(tr -d '\r' < "$hdr" | sed -n 's/^[Ll]ink: *<\([^>]*\)>; *rel="next".*/\1/p' | head -1)
    rm -f "$hdr"
    page=$((page + 1))
    [ "$page" -gt 200 ] && { echo "hfget: listing did not terminate after 200 pages" >&2; exit 1; }
done
echo "hfget: $page page(s) of listing"
# A file name that does not land inside the destination is not a listing oddity: an absolute
# path, a `..` segment, a control character (a tab splices one entry into two fields of the
# transfer list), or a name shadowing hfget's own manifest, which the next run would read back
# as its record of verified content. Name the entry and stop; skipping silently would leave a
# directory that reads as complete and is not.
bad=$(printf '%s' "$listing" | jq -r '[.[] | select(.type == "file")
       | select((.path | test("(^/|(^|/)\\.\\.(/|$))"))
             or (.path | test("[[:cntrl:]]"))
             or (.path | test("(^|/)[.]hfget-manifest")))] | map(.path) | join(", ")')
[ -n "$bad" ] && { echo "hfget: the listing names '$bad', which does not land inside the destination" >&2; exit 1; }
have_st=$(printf '%s' "$listing" | jq '[.[] | select(.path | endswith(".safetensors"))] | length')
sel='.[] | select(.type == "file")'
[ "$have_st" -gt 0 ] && sel="$sel"' | select((.path | test("\\.(bin|msgpack|h5|onnx|onnx_data|gguf)$") or test("(^|/)(pytorch_model|consolidated)[^/]*\\.pth$")) | not)'

list=$(mktemp "${TMPDIR:-/tmp}/hfget.XXXXXX")
trap 'rm -f "$list" "$dest/.hfget-manifest.tmp"' EXIT INT TERM
printf '%s' "$listing" | jq -r "$sel"' | "\(.size)\t\(.lfs.oid // "")\t\(.path)"' > "$list"

n=$(wc -l < "$list")
[ "$n" -gt 0 ] || { echo "hfget: the listing has no files" >&2; exit 3; }
need=$(awk -F'\t' '{s += $1} END {printf "%.1f", s / 1073741824}' "$list")
mkdir -p "$dest"
: > "$dest/.hfget-manifest.tmp"
# Count what is still to fetch, not the repo's size, so a mostly complete copy larger than the free
# space does not read as a download that does not fit. A file counts from its local length, or in
# full when the manifest says its content changed.
todo=0
# Split by expansion, as the child does: a tab is IFS whitespace, so `read` would merge a non-LFS
# file's empty key into the next field.
while IFS= read -r line; do
    size=${line%%	*}; rest=${line#*	}; oid=${rest%%	*}; path=${rest#*	}
    cur=0
    if [ -f "$dest/$path" ]; then cur=$(wc -c < "$dest/$path"); fi
    if [ "$cur" = "$size" ] && [ -f "$dest/.hfget-manifest" ]; then
        rec=$(p="$path" awk -F'\t' '$2==ENVIRON["p"]{print $1; exit}' "$dest/.hfget-manifest")
        if [ -n "$rec" ] && [ "$rec" != "${oid:-$size}" ]; then cur=0; fi
    fi
    if [ "$cur" -lt "$size" ]; then todo=$((todo + size - cur)); fi
done < "$list"
# Free space in KiB from POSIX `df -Pk` (1024-byte blocks); the container image's busybox df lacks
# GNU's `-BG --output=avail`.
free=$(df -Pk "$dest" | awk 'NR == 2 { print $4 }')
echo "hfget: $n file(s), $need GiB -> $dest"
[ "$have_st" -gt 0 ] && echo "hfget: safetensors present, pickle weight formats skipped"
echo "hfget: $(awk -v t="$todo" 'BEGIN { printf "%.1f", t / 1073741824 }') GiB to fetch," \
     "$(awk -v f="$free" 'BEGIN { printf "%.1f", f / 1048576 }') GiB free on the destination filesystem"
awk -v t="$todo" -v f="$free" 'BEGIN { exit !(t > f * 1024) }' \
    && { echo "hfget: THE DOWNLOAD DOES NOT FIT" >&2; exit 3; }

export HFGET_CHILD=1 HFGET_DEST="$dest" HFGET_REPO="$repo" HFGET_REV="$rev" HFGET_MAN_OUT="$dest/.hfget-manifest.tmp"
self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
tr '\n' '\0' < "$list" | xargs -0 -P "$jobs" -I{} sh "$self" "{}" \
    || { echo "hfget: incomplete -- re-run to resume" >&2; exit 1; }
mv "$HFGET_MAN_OUT" "$dest/.hfget-manifest"
echo "hfget: complete -- $(du -sh "$dest" | cut -f1) in $dest"
