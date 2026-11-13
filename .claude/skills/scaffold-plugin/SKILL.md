---
name: scaffold-plugin
description: Scaffold a new `rt-<name>-core` engine crate + `rt-<name>` nih-plug plugin pair with this workspace's standard layout, Cargo/workspace wiring, feature flags, and test skeleton. Use when starting Phase 1/2 of any `docs/Roadmap-##-*.md`, or whenever the user asks to create a new plugin / DSP crate in this repo.
---

# Scaffold a new `*-core` + plugin pair

Every plugin in this workspace follows one shape: a **framework-agnostic
`rt-<name>-core`** DSP crate and a thin **`rt-<name>`** `nih-plug` wrapper.
Copy the most recent sibling (`rt-vox` / `rt-vox-core`, or `rt-limiter`) rather
than inventing structure. Only the DSP differs — the plumbing is identical.

## 0. Read first
- The relevant `docs/Roadmap-##-<name>.md` (Phases 0–5 and the control list).
- The closest existing pair (`rt-vox` for FFT/voice work, `rt-limiter` for
  sample-domain dynamics, `rt-drive` for oversampled). Mirror it.
- `../../CLAUDE.md` → **Plugin conventions** section (the always-on rules).

## 1. Core crate `rt-<name>-core`
```
rt-<name>-core/
├── Cargo.toml          # copy rt-vox-core/Cargo.toml; deps: realfft (if FFT),
│                       #   rt-fft-convolver (Ring/FftHandler/DenormalGuard),
│                       #   rt-pitch-core (if reusing shifters/formant/pitch)
└── src/
    ├── lib.rs          # module docs + `pub use` re-exports of the public API
    ├── <engine>.rs     # the DSP; process_block() is allocation-free, no locks
    └── test_support.rs # #[cfg(test)] helpers (make_sine/sawtooth, rms, dominant_frequency_hz)
```
Rules: allocate in `new()` (document "**Allocates** — off-RT only"); wrap FFT
paths in `DenormalGuard::new()`; every public type gets unit tests with known
IRs (unit pulse, sine, silence→silence).

## 2. Plugin crate `rt-<name>`
```
rt-<name>/
├── Cargo.toml          # crate-type = ["lib","cdylib"]; features:
│                       #   default=[], nih=["dep:nih_plug"], gui=["nih","dep:nih_plug_egui"]
└── src/
    ├── lib.rs          # pub mod params/engine/ui; `#[cfg(feature="nih")] mod plugin;`
    ├── params.rs       # framework-agnostic `*Params` (Copy) + ranges +
    │                   #   clamped() + structural_changed()
    ├── engine.rs       # `*Controller` (off-RT builder, ir_handoff publisher) +
    │                   #   `*Processor` (audio thread, polls handoff, RT-safe)
    ├── ui.rs           # pure egui-free helpers + tests (see plugin-editor skill)
    └── plugin.rs       # nih-plug wrapper (nih feature); editor behind gui
```
- `params.rs` is the single source of truth; `plugin.rs` mirrors it with
  `#[derive(Enum)]`/`#[derive(Params)]` types and a `snapshot()` converter.
- **Threading:** anything that reallocates on a parameter change goes through
  the `rt_ir_plugin_core::ir_handoff` (Controller publishes, Processor polls).
  `structural_changed()` decides what forces a rebuild; everything else is
  applied in place each block. Keep engines the audio thread mutates (e.g. a
  harmoniser with MIDI note state) **resident** in the Processor, not behind the
  handoff, so a rebuild never drops that state.

## 3. Workspace wiring (root `Cargo.toml`)
Add **both** crates to `[workspace] members` **and** `default-members`
(keep `xtask` last in `members` only). Verify: `cargo metadata --no-deps`.

## 4. justfile
The `plugins :=` variable and `lint-gui` / `ci-gui` recipes currently list only
`rt-ir-cab rt-ir-reverb` — **add the new plugin** if it ships a `gui` editor so
CI covers it.

## 5. Tests (the mandatory triad — see the rt-safety-audit skill)
`rt-<name>/tests/`: `integration.rs`, `rt_safety.rs`, `rt_alloc.rs`.

## 6. Finish
1. `cargo test -p rt-<name>-core -p rt-<name>`
2. `cargo fmt` && `cargo clippy -p rt-<name>-core -p rt-<name> --all-targets`
   && `cargo clippy -p rt-<name> --features nih`
3. `cargo machete rt-<name> rt-<name>-core`
4. Tick the roadmap Phase 1/2 checkboxes with file links.
5. Suggest a conventional commit (`feat(rt-<name>): ...`), Co-Authored-By line.
