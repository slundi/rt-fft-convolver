---
name: package-plugin
description: Work through Phase 5 (Packaging & release) for a plugin — CLAP/VST3 bundling via NIH-plug's xtask, per-platform archives, Debian/macOS/Windows installers, preset banks, and the Codeberg release flow. Use for any roadmap "Phase 5: Packaging & release" section or when asked to bundle/package/release a plugin.
---

# Phase 5: packaging & release

All of this is already wired through the `justfile` and `packaging/` scripts —
use them, don't hand-roll. NIH-plug exports **CLAP + VST3 only**; LV2 needs a
CLAP→LV2 bridge (not yet added — installers reserve `/usr/lib/lv2` for later).

## 0. Register the plugin
Before bundling, the plugin must be known to the tooling:
- root `Cargo.toml`: in `[workspace] members` **and** `default-members`.
- `justfile`: add it to `plugins :=` and to `lint-gui` / `ci-gui` if it ships a
  `gui` editor (otherwise `bundle-all` / CI skip it).

## 1. Bundle (CLAP + VST3 → `target/bundled/`)
```sh
just bundle rt-<name>        # one plugin, with the egui editor (needs X11/GL)
just bundle-all              # every plugin in `plugins`, with editor
just bundle-headless         # every plugin, no editor (no system libs) → --features nih
just bundle-cross rt-<name> aarch64-unknown-linux-gnu
```
Under the hood: `cargo xtask bundle <p> --release --features gui` (see
`xtask/src/main.rs` → `nih_plug_xtask`).

## 2. Archives, checksums, installers
```sh
just checksums               # SHA256SUMS of every bundle
just package <version>       # stage per-platform archive (+ .deb on Linux) → dist/
just deb <version>           # Debian package from target/bundled
```
Scripts: `packaging/package.sh`, `packaging/build-deb.sh`,
`packaging/checksums.sh`, `packaging/macos/build-pkg.sh` (signed `.pkg`),
`packaging/windows/*.iss` (Inno Setup `.exe`). Read `packaging/README.md`.

## 3. CI matrix (Phase 5 target)
Linux x86_64 + aarch64, macOS universal, Windows x86_64. Local mirrors:
`just ci` (core, no system libs), `just ci-gui` (where X11/GL exist),
`just ci-local` (full pipeline through `nix develop`). The GUI bundle needs the
GL/X11 libs — on a bare machine, reuse the `just check-gui` devshell approach.

## 4. Presets
Each roadmap's Phase 5 lists a named preset bank (e.g. limiter: mastering /
broadcast / streaming). Ship them as the plugin's factory presets; keep the
names and target values from the roadmap.

## 5. Docs
Add a user manual under `docs/manual-<name>.md` (see `docs/manual-cab.md`,
`docs/manual-reverb.md`) — mode-selection guide, control reference, and any
routing diagram the roadmap calls for.

## 6. Release (Codeberg)
```sh
just changelog               # git-cliff → CHANGELOG.md (commit before tagging)
just tag v1.2.3              # annotated tag → triggers the release workflow
just release v1.2.3          # package + publish assets via the Forgejo API (needs CODEBERG_TOKEN)
```
Commits must be conventional (git-cliff builds the changelog from them).
End release/PR bodies with the standard Co-Authored-By / Generated-with lines.
