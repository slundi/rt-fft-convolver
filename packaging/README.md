# Packaging

Bundling and installer scripts for `rt-ir-cab` and `rt-ir-reverb`.

## Bundles (all platforms)

NIH-plug's bundler produces the plugin bundles into `target/bundled/`:

```sh
just bundle-all          # CLAP + VST3, with the egui editor (needs X11/OpenGL)
just bundle-headless     # CLAP + VST3, no editor (no system libraries)
```

- **CLAP**: `rt-ir-cab.clap`, `rt-ir-reverb.clap` (single files).
- **VST3**: `rt-ir-cab.vst3/`, `rt-ir-reverb.vst3/` (bundle directories).

> **LV2:** NIH-plug exports CLAP and VST3 only.  An LV2 build needs a CLAP→LV2
> bridge; the installers reserve `/usr/lib/lv2` (Linux) for when one is added.

## Per-platform archives

```sh
just package 0.1.0       # zip of the bundles for the host platform → dist/
                         # (+ a .deb on Linux, + SHA256SUMS)
```

## Installers

| Platform | Tool | Script | Installs to |
|----------|------|--------|-------------|
| Linux    | `dpkg-deb` | [`build-deb.sh`](build-deb.sh) | `/usr/lib/{clap,vst3}` |
| macOS    | `pkgbuild` | [`macos/build-pkg.sh`](macos/build-pkg.sh) | `/Library/Audio/Plug-Ins/{CLAP,VST3}` |
| Windows  | Inno Setup | [`windows/rt-ir-plugins.iss`](windows/rt-ir-plugins.iss) | `Common Files\{CLAP,VST3}` |

Each installer takes a version argument / define and reads from `target/bundled`.
macOS signing + notarisation and Windows code-signing are documented inline in
their scripts — fill in your Developer ID / certificate before shipping.

## Release

CI (`.github/workflows/release.yaml`) builds the bundles on each OS runner,
packages them, and attaches the archives + `SHA256SUMS` to the tag's release.
Locally, `just release vX.Y.Z` publishes the host-platform archive to Codeberg
(needs `CODEBERG_TOKEN`).
