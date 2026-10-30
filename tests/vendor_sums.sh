#!/bin/sh
# vendor_sums.sh -- checks the vendor tree, byte for byte, against vendor/SHA256SUMS.
#
# vendor/UPSTREAM records what every lifted file is, from which commit, and what we changed. This
# script is the mechanical half of that record: every file under vendor/ -- the lifts, our patch
# series, the record itself -- is hashed, and anything that differs from its recorded hash or
# arrived without being recorded is an unrecorded change to a lifted dependency, which is exactly
# what the record exists to catch. The build does not consult it; a test does (ctest:
# vendor_sums), so it runs wherever the suite runs.
#
#   tests/vendor_sums.sh <repo-root>
#
# Regenerate the manifest in the same commit that changes vendor/:
#   find vendor -type f ! -name SHA256SUMS | sort | xargs sha256sum > vendor/SHA256SUMS
set -e
root=${1:?usage: vendor_sums.sh <repo-root>}
cd "$root"
command -v sha256sum >/dev/null || { echo "vendor_sums: sha256sum is required" >&2; exit 1; }
[ -f vendor/SHA256SUMS ] || { echo "vendor_sums: vendor/SHA256SUMS is missing" >&2; exit 1; }
sha256sum -c --status vendor/SHA256SUMS || {
    echo "vendor_sums: a vendor file differs from its recorded hash;" \
         "run sha256sum -c vendor/SHA256SUMS to see which" >&2
    exit 1; }
# The other direction: sha256sum -c never notices a file the manifest does not name -- an added
# header sliding into a lift is silent without this half.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM
find vendor -type f ! -name SHA256SUMS | sort > "$tmp/have"
awk '{print $2}' vendor/SHA256SUMS | sort > "$tmp/want"
diff "$tmp/want" "$tmp/have" || {
    echo "vendor_sums: the vendor tree and the manifest name different files;" \
         "regenerate the manifest in the commit that changes vendor/" >&2
    exit 1; }
echo "vendor_sums: $(wc -l < "$tmp/have") files, every hash matches"
