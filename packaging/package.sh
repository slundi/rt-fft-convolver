#!/usr/bin/env bash
# Stage per-platform release archives into dist/ from the bundles in
# target/bundled.  Produces a zip of the CLAP + VST3 bundles for the host
# platform, a .deb on Linux, and a SHA256SUMS over dist/.
#
# Bundle first: `just bundle-all` (or the release CI matrix per OS).
set -euo pipefail

version="${1:?usage: package.sh VERSION}"
root="$(cd "$(dirname "$0")/.." && pwd)"
bundled="$root/target/bundled"
dist="$root/dist"

[ -d "$bundled" ] || {
    echo "package: no bundles in $bundled — run 'just bundle-all' first" >&2
    exit 1
}
mkdir -p "$dist"

os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
Linux) plat="linux-${arch}" ;;
Darwin) plat="macos-${arch}" ;;
MINGW* | MSYS* | CYGWIN*) plat="windows-${arch}" ;;
*) plat="${os}-${arch}" ;;
esac

# `.tar.gz` on Unix (tar is always present), `.zip` on Windows runners.
case "$os" in
Linux | Darwin)
    archive="$dist/rt-ir-plugins-${version}-${plat}.tar.gz"
    rm -f "$archive"
    (cd "$bundled" && tar czf "$archive" ./*.clap ./*.vst3)
    ;;
*)
    # Windows runners ship bsdtar (`tar.exe`), which infers zip from `-a`.
    archive="$dist/rt-ir-plugins-${version}-${plat}.zip"
    rm -f "$archive"
    (cd "$bundled" && tar -a -c -f "$archive" ./*.clap ./*.vst3)
    ;;
esac
echo "wrote $archive"

# Native OS installer where we can build one.
if [ "$os" = "Linux" ] && command -v dpkg-deb >/dev/null; then
    "$root/packaging/build-deb.sh" "$version"
fi

"$root/packaging/checksums.sh" "$dist"
