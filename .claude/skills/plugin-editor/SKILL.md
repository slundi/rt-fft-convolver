---
name: plugin-editor
description: Implement Phase 3 (UI) for an `rt-<name>` plugin — the pure, unit-tested `ui.rs` helpers plus the `nih-plug-egui` editor gated behind the `gui` feature, with lock-free telemetry from the audio thread. Use for any roadmap "Phase 3: UI" section, or when asked to add/edit a plugin editor. Includes the exact Nix command to compile/clippy the `gui` feature.
---

# Phase 3: the plugin editor

**Framework decision (fixed): `nih-plug-egui`** — every plugin in this
workspace uses it (`rt-pitch`, `rt-limiter`, `rt-vox`). Do not reach for iced.

Split the work the way the siblings do: pure logic in `ui.rs` (tested), egui
drawing in `plugin.rs` behind `#[cfg(feature = "gui")]`.

## 1. `rt-<name>/src/ui.rs` — pure, egui-free, tested
Put every value/format/normalise helper here so it unit-tests **without** the
plugin framework. Re-export shared bits and add plugin-specific ones:
```rust
pub use rt_ir_plugin_core::{MidiActivity, MidiStatus, PadState, pad_state};
// e.g. hz_to_note_name, format_hz, confidence_fraction, spectral_heights,
//      format_interval/level/pan/detune, latency_ms/format_latency_ms
```
Test the boundaries (A4→"A4", silence→zeros, floor/peak of a dB mapping, clamps).
See `rt-vox/src/ui.rs` for the reference set.

## 2. Telemetry: audio thread → editor (lock-free)
Publish read-only state through **packed atomics** — never lock on the audio
thread. Pattern (`rt-limiter`, `rt-vox`):
- scalars: `Arc<AtomicU32>` holding `f32::to_bits(..)` (pitch Hz, confidence,
  latency, sample rate) or a small code (MIDI-activity LED).
- arrays (spectra): `Arc<Vec<AtomicU32>>`, one packed `f32` per bin.
- Gate anything expensive on `self.params.editor_state.is_open()` so, e.g., a
  display-only FFT probe runs **only while the editor is open** (see
  `rt_vox_core::SpectralProbe` + `VoxProcessor::probe_spectrum`).

## 3. `plugin.rs` editor (behind `gui`)
- `#[cfg(feature="gui")] use nih_plug_egui::{EguiState, create_egui_editor, egui, widgets};`
- Add `#[cfg(feature="gui")] #[persist="editor-state"] editor_state: Arc<EguiState>`
  to the `#[derive(Params)]` struct; init with `EguiState::from_size(w, h)`.
- Clone the telemetry `Arc`s into the `create_egui_editor` closure; read with
  `f32::from_bits(atomic.load(Relaxed))`.
- Controls are `widgets::ParamSlider::for_param(&params.x, setter)`.
- Keyed-off-`Mode` plugins: `match params.mode.value()` to show the right panel.
- `ctx.request_repaint()` each frame so meters animate without automation.

## 4. Feature hygiene (so non-gui builds stay clean)
- gui-only fields/consts/imports get `#[cfg(feature="gui")]`.
- Always-present telemetry atomics must be **written** in `process()` (a
  `.store()` counts as a use) or clippy's dead-code lint fires in the `nih`
  (no-gui) build. Anything read *only* in the editor closure → gate it.

## 5. Verify the `gui` feature (the non-obvious part)
The `gui` build links GL/X11/xcb/wayland. This machine/devshell may lack them,
so use the wrapper recipe (added to the justfile):
```sh
just check-gui rt-<name>      # nix devshell + cargo clippy --features gui
```
Equivalent raw command if `just` is unavailable:
```sh
nix develop --impure --expr 'let pkgs = (builtins.getFlake "nixpkgs").legacyPackages.${builtins.currentSystem}; in pkgs.mkShell { nativeBuildInputs = [ pkgs.pkg-config ]; buildInputs = [ pkgs.libGL pkgs.xorg.libX11 pkgs.xorg.libxcb pkgs.xorg.libXcursor pkgs.xorg.libXrandr pkgs.xorg.libXi pkgs.libxkbcommon pkgs.wayland ]; }' \
  --command cargo clippy -p rt-<name> --features gui -- -D warnings
```
Missing `xcb.pc` / `x11-xcb` / `gl.pc` errors mean a lib is missing from
`buildInputs` — `libxcb` and `libglvnd`(→`libGL`) are the ones usually forgotten.

## 6. Finish
- `cargo test -p rt-<name>` (ui.rs tests run in the default build).
- `cargo clippy` clean on **default**, **nih**, and **gui** (via `just check-gui`).
- Tick the roadmap Phase 3 checkboxes; suggest a `feat(rt-<name>): ... Phase 3 editor` commit.
