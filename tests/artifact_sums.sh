#!/bin/sh
# artifact_sums.sh -- checks committed built artifacts against tests/ARTIFACTS.
#
# Some committed files are not sources but a build of sources: core/server/dashboard_page.h is a
# Node build of dashboard/, committed because the engine's build has no Node. The digest recorded
# INSIDE such an artifact proves only that the sources and the artifact's own claim agree --
# tampering with the artifact together with its embedded digest reads as an honest build. This
# manifest hashes the artifact itself from outside, the mechanical half that vendor/SHA256SUMS is
# to vendor/UPSTREAM: the dashboard_page test keeps sources and header in step, this one checks
# the served bytes. (A tamper that updates the artifact and this manifest in one breath remains,
# exactly as with a vendor file -- the manifest is against drift and lone tampering, not against
# a commit that rewrites the whole record.)
#
#   tests/artifact_sums.sh <repo-root>                     check, as ctest runs it
#   tests/artifact_sums.sh <repo-root> --record <file>...  re-record, in the commit that rebuilt
#
# Re-record names the artifacts explicitly and writes through a temporary file. Do NOT re-record by
# reading the manifest into the command that overwrites it: the shell truncates tests/ARTIFACTS for
# the redirect BEFORE the command substitution reads it, so awk sees an empty file, sha256sum is
# left with no file argument, and it hashes its own empty stdin. That writes the line
# "e3b0c442…  -" -- the hash of nothing, against a file named "-". And `sha256sum -c` resolves that
# name as stdin too, so it re-reads the same empty input, matches, and the check passes whatever the
# artifact contains. A manifest destroyed that way makes this test unable to fail: it reported
# "every hash matches" across a tampered dashboard_page.h. The checks below are what close it.
set -e
root=${1:?usage: artifact_sums.sh <repo-root> [--record <artifact>...]}
cd "$root"
command -v sha256sum >/dev/null || { echo "artifact_sums: sha256sum is required" >&2; exit 1; }
[ -f tests/ARTIFACTS ] || { echo "artifact_sums: tests/ARTIFACTS is missing" >&2; exit 1; }

# sha256sum of an empty input: the value a self-truncating re-record leaves behind.
EMPTY=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

if [ "${2:-}" = "--record" ]; then
    [ $# -ge 3 ] || { echo "artifact_sums: --record needs the artifact paths, e.g. --record core/server/dashboard_page.h" >&2; exit 1; }
    shift 2
    for f in "$@"; do
        [ -f "$f" ] || { echo "artifact_sums: refuses to record $f: no such file" >&2; exit 1; }
    done
    sha256sum "$@" > tests/ARTIFACTS.new
    mv tests/ARTIFACTS.new tests/ARTIFACTS
    echo "artifact_sums: recorded $# artifact(s) in tests/ARTIFACTS"
    exit 0
fi

# A line can satisfy `sha256sum -c` and still pin nothing: it may name a file that is not in the
# tree, or "-" (stdin), or carry the hash of empty input. Each of those reads as a green test while
# the artifact is unchecked, so each is a failure here.
bad=0
while read -r sum name; do
    [ -n "$sum" ] || continue
    if [ "$name" = "-" ] || [ -z "$name" ]; then
        echo "artifact_sums: tests/ARTIFACTS names '-' (sha256sum's stdin marker): re-record it" >&2
        bad=1
    elif [ ! -f "$name" ]; then
        echo "artifact_sums: tests/ARTIFACTS names a file that is not there: $name" >&2
        bad=1
    fi
    if [ "$sum" = "$EMPTY" ]; then
        echo "artifact_sums: tests/ARTIFACTS records the hash of empty input for '${name:-<none>}': re-record it" >&2
        bad=1
    fi
done < tests/ARTIFACTS
[ "$bad" = 0 ] || exit 1

sha256sum -c --status tests/ARTIFACTS || {
    echo "artifact_sums: a committed artifact differs from its recorded hash;" \
         "run sha256sum -c tests/ARTIFACTS to see which" >&2
    exit 1; }
echo "artifact_sums: $(wc -l < tests/ARTIFACTS) artifact(s), every hash matches"
