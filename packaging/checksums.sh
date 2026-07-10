#!/usr/bin/env bash
# SHA-256 checksums for every artifact in a directory (default: the CLAP/VST3
# bundles).  `.vst3` bundles are directories, so their files are hashed
# individually.  Writes a `SHA256SUMS` file in that directory.
set -euo pipefail

dir="${1:-target/bundled}"
[ -d "$dir" ] || {
    echo "checksums: no such directory: $dir" >&2
    exit 1
}

cd "$dir"
: >SHA256SUMS
shopt -s nullglob
for entry in *.clap *.vst3 *.zip *.tar.gz *.deb *.pkg *.exe; do
    if [ -d "$entry" ]; then
        find "$entry" -type f -print0 | sort -z | xargs -0 sha256sum >>SHA256SUMS
    else
        sha256sum "$entry" >>SHA256SUMS
    fi
done
cat SHA256SUMS
