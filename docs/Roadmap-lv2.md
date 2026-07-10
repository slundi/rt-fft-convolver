# Roadmap: LV2 Plugins (`rt-ir-cab` & `rt-ir-reverb`)

This roadmap describes two audio plugins built on top of the
`rt-fft-convolver` crate.  Both plugins live in a separate workspace
and depend on this crate as a library.

- **`rt-ir-cab`** — Guitar / bass cabinet simulator using zero-latency
  hybrid convolution.  Optimised for short IRs (< 500 ms).
- **`rt-ir-reverb`** — Convolution reverb using partitioned FFT
  convolution.  Optimised for long true-stereo IRs (0.5 – 10 s).

Both plugins target **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug), so a single
Rust codebase produces every format.

## Design principles

- **8 IR slots per plugin**, all pre-loaded at startup or when the
  user assigns / changes a file.  No lazy-load-on-CC: the first
  press must never dropout on stage.
- **Slot selection sources**: automation parameter, MIDI Program
  Change, MIDI CC (all live simultaneously).
- **RT-safe swap**: file I/O + resample + partitioning happen on an
  LV2 worker (or `nih-plug` background task).  The audio thread only
  ever swaps a prepared IR pointer under a lock-free handoff.
- **Crossfade between any two slots** (not just A→B).  Duration is a
  user-controllable parameter.
- **Empty slots pass audio through dry** — never panic, never
  silence when the guitarist hits an unassigned patch button.

## Prerequisites in `rt-fft-convolver`

These roadmap items must land in the core library before the plugins
can ship a stable release:

- [x] **HybridConvolver** (Phase 5 of the core roadmap):
  - Needed by `rt-ir-cab` for true 0-sample latency.
  - Wraps `DirectConvolver` + delayed `UniformPartitionEngine` so
    head + tail don't double-count taps.
- [x] **Cross-fading IR switch** (Phase 4 of the core roadmap):
  - Both plugins depend on it.  Design the API as
    `(from_slot, to_slot, duration_samples)` from day one — not a
    fixed A↔B toggle — so N-slot switching works with the same
    primitive.

## Phase 0: Shared foundation (`rt-ir-plugin-core`)

Internal crate reused by both plugins.  Not published to crates.io.

> **Status (done):** the crate is deliberately **framework-agnostic** — it has
> no `nih-plug` dependency.  It provides the reusable, RT-safe primitives and
> unit-tests them in isolation; the `nih-plug` wiring (background-task loading,
> `Plugin` impls) is a thin layer added in the plugin crates (Phase 1/2).

- [x] **Workspace layout**:
  - `rt-ir-plugin-core` — shared logic.
  - `rt-ir-cab` — cab plugin (placeholder until Phase 1).
  - `rt-ir-reverb` — reverb plugin (placeholder until Phase 2).
- [x] **`IrSlotBank<T, const N: usize = 8>`**:
  - Owns `N` `Option<T>` slots (`T` = a built convolver or `PreparedIr`).
  - `PreparedIr::load(path, rate)` resamples off-RT; the plugin's `nih-plug`
    background task calls it and `IrSlotBank::load(index, …)`.
  - `request_activation(index)` records intent; `take_pending()` performs the
    RT-safe swap on the next block.  Empty slots stay activatable (dry).
- [x] **Lock-free RT handoff**:
  - `triple_buffer`-backed `IrPublisher` / `IrConsumer` between worker and
    audio thread.  Wait-free, allocation-free read; the retired convolver is
    dropped on the producer, never on the audio thread.
- [x] **MIDI routing**:
  - `program_to_slot` (Program Change → slot index, out-of-range ignored).
  - `cc_to_slot` (assignable CC → slot index, `value / 16`, clamped).
  - `MidiRouter` unifies both; host-UI parameter changes route through the
    same `IrSlotBank::request_activation()` path.
- [x] **Parametric slot crossfader**:
  - `SlotCrossfader<T: MonoConvolver>` with a **per-fade** duration
    (`ms_to_samples` maps the `crossfade_ms` param, 5 – 200 ms).
  - Equal-power blend via a rotation recurrence (no per-sample `sin`/`cos`,
    no table).  Runs two convolvers in parallel during the transition and
    hands the retired one back for off-RT disposal.

## Phase 1: `rt-ir-cab` plugin

Guitar / bass cabinet simulator.

- [ ] **I/O**:
  - Mono in → stereo out (default), stereo in → stereo out.
  - Uses `HybridConvolver` per output channel.
- [ ] **Slot controls** (× 8):
  - `IR File N` (LV2 patch:message, VST3/CLAP file path property).
  - `Slot Enabled N` (bool) — allow disabling a slot without
    unloading it.
- [ ] **Global controls**:
  - `Current Slot` (int 0–7, automatable).
  - `Crossfade Time` (5–200 ms).
  - `Input Gain` (−24 … +24 dB).
  - `Output Gain` (−24 … +24 dB).
  - `Mix` (0–100 %, defaults to 100 %).
  - `Low Cut` (20–500 Hz, 12 dB/oct, pre-convolution).
  - `High Cut` (2–20 kHz, 12 dB/oct, post-convolution).
  - `IR Normalize` (bool, uses the `Mixer` loudness computation).
  - `Bypass` (LV2 standard).
- [ ] **MIDI learn**:
  - Right-click `Current Slot` → "MIDI learn" to bind a PC or CC.
- [ ] **CPU meter**:
  - Report partition count × block size as a read-only property.
- [ ] **Presets**:
  - Ship a handful of neutral tone-shaping presets (bright, dark,
    scooped) that only touch the EQ — not the IR itself.

## Phase 2: `rt-ir-reverb` plugin

Convolution reverb with true-stereo support.

- [ ] **I/O**:
  - Stereo in → stereo out.
  - Uses `TrueStereoConvolver` when the IR has 4 channels,
    `StereoConvolver` for 2-channel IRs.
- [ ] **Slot controls** (× 8):
  - `IR File N` (patch:message).
  - `Slot Enabled N` (bool).
- [ ] **Global controls**:
  - `Current Slot` (int 0–7, automatable).
  - `Crossfade Time` (5–500 ms — longer than cab, tails need it).
  - `Pre-Delay` (0–500 ms).
  - `Dry` (−∞ … +6 dB, separate fader).
  - `Wet` (−∞ … +6 dB, separate fader).
  - `Stereo Width` (0–200 %, wet only).
  - `Low Cut` (20–500 Hz, wet only).
  - `High Cut` (2–20 kHz, wet only).
  - `IR Reverse` (bool).
  - `IR Length` (1–100 %, truncates the tail for CPU control).
  - `Bypass`.
- [ ] **MIDI learn**: same as cab.
- [ ] **CPU meter**: same as cab.
- [ ] **Presets**:
  - Small room / large room / hall / plate / spring — assuming a
    small bundled IR pack (CC-licensed) is shipped with the plugin.

## Phase 3: UI

- [ ] **`nih-plug-egui`** (or `nih-plug-iced`) frontend, shared
  between both plugins where possible.
- [ ] **Slot rack**: 8 pads, drag-and-drop file loading.  Active
  slot highlighted; empty slots visibly greyed out.
- [ ] **IR waveform preview** per slot (pre-computed off-thread).
- [ ] **MIDI activity indicator** on the currently mapped CC/PC.
- [ ] **Global bypass with LED** — matches host bypass state.

## Phase 4: Testing

- [ ] **Unit tests** in `rt-ir-plugin-core`:
  - Slot bank loads, swaps, and releases without leaks.
  - Crossfader produces bit-identical output when
    `from_slot == to_slot`.
- [ ] **Integration tests**:
  - Load a known IR, feed a unit impulse, compare output against
    the raw IR (both plugins).
  - MIDI Program Change on channel 1 switches slots.
- [ ] **Host validation**:
  - `lv2lint` + `lv2ls` for LV2 conformance.
  - `clap-validator` for CLAP.
  - `pluginval` for VST3.
- [ ] **RT-safety audit**:
  - `no_std` allocator hook in tests to catch any allocation in
    `process()`.
  - Denormal guard active in every callback (via
    `DenormalGuard` from the core crate).

## Phase 5: Packaging & release

- [ ] **CI**: cross-compile for Linux (x86_64, aarch64), macOS
  (universal), Windows (x86_64) via GitHub Actions / Codeberg CI.
- [ ] **Bundling**:
  - `.lv2` bundle with manifest.
  - `.clap` single-file.
  - `.vst3` bundle.
- [ ] **Installers**:
  - Debian `.deb` (LV2 + CLAP + VST3 in standard paths).
  - macOS `.pkg` signed and notarised.
  - Windows `.exe` (Inno Setup).
- [ ] **Documentation**:
  - Per-plugin manual under `docs/`.
  - Screenshots + a short video for each plugin.
- [ ] **Release channels**:
  - Codeberg release tags with checksums.
  - Optional: submit to KVR / linux-audio user announcements.

## Out of scope (future work)

- Per-slot EQ (would 8× the parameter count — do it globally instead).
- IR editor / trimmer built into the plugin (users have DAWs).
- Non-linear cab modelling (this crate is linear-only by design).
- Snapshots / patch morphing beyond the 8-slot crossfade.
