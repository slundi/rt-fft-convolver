#!/usr/bin/env bash
# Build a Debian package that installs the CLAP + VST3 bundles into the standard
# system plugin paths.  Requires the bundles to already exist under
# `target/bundled` (run `just bundle-all` first) and `dpkg-deb` on PATH.
#
# LV2 note: NIH-plug exports CLAP + VST3 only.  An LV2 build would need a
# CLAP→LV2 bridge; when one exists its `.lv2` bundle goes in /usr/lib/lv2.
set -euo pipefail

version="${1:?usage: build-deb.sh VERSION}"
root="$(cd "$(dirname "$0")/.." && pwd)"
bundled="$root/target/bundled"
dist="$root/dist"

command -v dpkg-deb >/dev/null || {
    echo "build-deb: dpkg-deb not found (install dpkg)" >&2
    exit 1
}
[ -d "$bundled" ] || {
    echo "build-deb: no bundles in $bundled — run 'just bundle-all' first" >&2
    exit 1
}

arch="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
pkgroot="$(mktemp -d)"
trap 'rm -rf "$pkgroot"' EXIT

install -d "$pkgroot/usr/lib/clap" "$pkgroot/usr/lib/vst3" "$pkgroot/DEBIAN"
for clap in "$bundled"/*.clap; do cp -a "$clap" "$pkgroot/usr/lib/clap/"; done
for vst3 in "$bundled"/*.vst3; do cp -a "$vst3" "$pkgroot/usr/lib/vst3/"; done

cat >"$pkgroot/DEBIAN/control" <<EOF
Package: rt-ir-plugins
Version: ${version}
Section: sound
Priority: optional
Architecture: ${arch}
Maintainer: Sébastien Lundi <slundi@stamus-networks.com>
Depends: libc6, libgl1, libx11-6
Description: rt-ir-cab & rt-ir-reverb convolution plugins (CLAP + VST3)
 Zero-latency guitar/bass cabinet simulator and a true-stereo convolution
 reverb, built on the rt-fft-convolver engine.
EOF

mkdir -p "$dist"
out="$dist/rt-ir-plugins_${version}_${arch}.deb"
dpkg-deb --build --root-owner-group "$pkgroot" "$out"
echo "wrote $out"
