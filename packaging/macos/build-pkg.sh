#!/usr/bin/env bash
# Build a macOS installer .pkg that drops the CLAP + VST3 bundles into the
# standard user plugin folders.  Run on macOS with the bundles already built
# (`just bundle-all`).
#
# Signing + notarisation (fill in your identities):
#   productsign --sign "Developer ID Installer: NAME (TEAMID)" unsigned.pkg signed.pkg
#   xcrun notarytool submit signed.pkg --apple-id … --team-id … --password … --wait
#   xcrun stapler staple signed.pkg
set -euo pipefail

version="${1:?usage: build-pkg.sh VERSION}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
bundled="$root/target/bundled"
dist="$root/dist"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

install -d "$staging/Library/Audio/Plug-Ins/CLAP" "$staging/Library/Audio/Plug-Ins/VST3"
cp -R "$bundled"/*.clap "$staging/Library/Audio/Plug-Ins/CLAP/"
cp -R "$bundled"/*.vst3 "$staging/Library/Audio/Plug-Ins/VST3/"

mkdir -p "$dist"
pkgbuild \
    --identifier "org.codeberg.slundi.rt-ir-plugins" \
    --version "$version" \
    --install-location "/" \
    --root "$staging" \
    "$dist/rt-ir-plugins-${version}.pkg"
echo "wrote $dist/rt-ir-plugins-${version}.pkg (unsigned — see header for signing)"
