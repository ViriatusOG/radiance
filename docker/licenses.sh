#!/bin/sh
# docker/licenses.sh ROOT FILE... -- record the licenses of all third-party software in the
# runtime image.
#
# FILE... are the build-image paths of every file copied into ROOT from outside the radiance
# install: the shared libraries, comgr, busybox, curl, jq and the CA bundle's directory. Each is
# traced to its installing package, or to the FFmpeg stage's /opt/ffmpeg. Package license files go
# to ROOT/usr/share/doc/<package>/, the texts Debian's copyright files refer to go to
# ROOT/usr/share/common-licenses/, and ROOT/usr/share/doc/THIRD-PARTY lists each component with its
# version, license files and source. A file no package owns, or a package with no license file,
# fails the build, so no software ships with an unrecorded license.
set -eu
R=$1; shift
D=$R/usr/share/doc
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
mkdir -p "$D"

owner() {   # owner PATH -- the package that installed PATH, under any name usrmerge gives it
  for p in "$1" "$(readlink -f "$1")"; do
    for q in "$p" "/usr$p" "${p#/usr}"; do
      o=$(dpkg-query -S "$q" 2>/dev/null | grep -v "^diversion " | head -1) || continue
      [ -n "$o" ] && { echo "${o%%:*}"; return 0; }
    done
  done
  return 1
}

licfiles() {   # licfiles PACKAGE -- the license files PACKAGE installed that are present
  { dpkg-query -L "$1" | grep -iE '/(copyright|licen[cs]e[^/]*|copying[^/]*|notice[^/]*)$' || true
    echo "/usr/share/doc/$1/copyright"; } |
  while read -r f; do [ -f "$f" ] && readlink -f "$f"; done | sort -u
}

ffmpeg=
for f in "$@"; do
  case "$f" in
    /opt/ffmpeg/*) ffmpeg=1 ;;
    *) p=$(owner "$f") || { echo "licenses.sh: no package owns $f" >&2; exit 1; }
       echo "$p $(basename "$f")" >> "$W/owned" ;;
  esac
done

rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)
{
  cat <<EOF
Third-party software in this image
==================================

/opt/radiance is radiance, under the Apache License 2.0; its license and notices are in
/opt/radiance/share/doc/radiance/. Everything else in the image is the software listed below, each
under its own license. A package's license files are in /usr/share/doc/<package>/, and the license
texts those files refer to are in /usr/share/common-licenses/.

The Ubuntu and ROCm packages are the distributions' binaries, copied unmodified. The source of an
Ubuntu package is its source package at the version listed: \`apt-get source NAME=VERSION\`, or
the Launchpad page given. The ROCm libraries are from AMD's ROCm $rocm; their source is at
https://github.com/ROCm.
EOF
  if [ -n "$ffmpeg" ]; then
    cat <<EOF

FFmpeg is built from the release tarball named below, unmodified, with the configure options in
/usr/share/doc/ffmpeg/CONFIGURATION. Built without --enable-gpl and --enable-nonfree, it is under
the GNU LGPL 2.1 or later. Its libraries are shared objects in /usr/lib/x86_64-linux-gnu and can
be replaced.

ffmpeg
  source:  $(cat /opt/ffmpeg/share/doc/ffmpeg/SOURCE)
  files:   $(cd /opt/ffmpeg/lib && ls *.so.* | grep -E '\.so\.[0-9]+$' | tr '\n' ' ' | sed 's/ $//')
  license: /usr/share/doc/ffmpeg/LICENSE.md, /usr/share/doc/ffmpeg/COPYING.LGPLv2.1
EOF
  fi
  for p in $(cut -d' ' -f1 "$W/owned" | sort -u); do
    files=$(grep "^$p " "$W/owned" | cut -d' ' -f2 | sort -u | tr '\n' ' ' | sed 's/ $//')
    lic=$(licfiles "$p")
    [ -n "$lic" ] || { echo "licenses.sh: package $p ships no license file" >&2; exit 1; }
    mkdir -p "$D/$p"
    dest=""
    for l in $lic; do
      t="$D/$p/$(basename "$l")"
      if [ -e "$t" ] && ! cmp -s "$l" "$t"; then
        t="$D/$p/$(basename "$(dirname "$l")")-$(basename "$l")"
      fi
      cp "$l" "$t"
      dest="$dest${dest:+, }${t#"$R"}"
    done
    ver=$(dpkg-query -W -f='${Version}' "$p")
    src=$(dpkg-query -W -f='${source:Package} ${source:Version}' "$p")
    echo
    echo "$p $ver"
    case "$(dpkg-query -L "$p")" in
      */opt/rocm*) echo "  source:  ROCm $rocm, https://github.com/ROCm" ;;
      *) echo "  source:  $src (Ubuntu), https://launchpad.net/ubuntu/+source/${src% *}/${src#* }" ;;
    esac
    echo "  files:   $files"
    echo "  license: $dest"
  done
} > "$W/THIRD-PARTY"
mv "$W/THIRD-PARTY" "$D/THIRD-PARTY"

if [ -n "$ffmpeg" ]; then
  mkdir -p "$D/ffmpeg"
  cp /opt/ffmpeg/share/doc/ffmpeg/* "$D/ffmpeg/"
fi
mkdir -p "$R/usr/share"
cp -a /usr/share/common-licenses "$R/usr/share/"
