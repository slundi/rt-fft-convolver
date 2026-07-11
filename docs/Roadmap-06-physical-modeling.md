# Roadmap: `rt-physical` (Physical / Waveguide Modelling)

Physical-modelling synthesis / effect plugin.  Covers plucked and
bowed string models, tube-amp state-space stages, and modal
resonators.  This is a broader domain than a single plugin —
[decision] whether the deliverable is one plugin with modes or
multiple plugins sharing a core crate.

**Reference implementations**: Pianoteq (piano physical model),
AAS Chromaphone / String Studio, Line 6 Variax, Roland V-Guitar /
V-Drums, Logic Sculpture, Elektron Analog Rytm (analog + models).

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **RT-safe integrator loops** — the trapezoidal / semi-implicit
  integrators used for waveguide and state-space models must be
  allocation-free and free of NaN / Inf sinks.
- **Denormal safety** — waveguides with light damping decay through
  the denormal range.  FTZ/DAZ must be set.
- **Sample-accurate MIDI note handling** — physical models are
  usually voiced synthesisers as much as effects.
- **Excitation as first-class input** — for models used as effects,
  the audio input is the excitation (pick, bow, mallet).  For synth
  use, MIDI notes drive an internal excitation generator.
- **Multiple model families** shipped, but each is complex enough
  that a single default per family is the initial scope.

## Prerequisites in `rt-fft-convolver`

- [ ] `DenormalGuard`.
- [ ] Nothing else strictly required; this plugin is largely
  independent of the convolution engine.

## Phase 0: Model-family decisions

- [ ] **Which families to include in v1** — [decision]:
  - Karplus-Strong extended (plucked string).
  - Digital waveguide (with body / bridge filtering).
  - Modal (bank of biquad resonators).
  - State-variable / state-space tube (nonlinear circuit).
- [ ] **Excitation source** — [decision]:
  - Audio input (effect use).
  - MIDI-triggered internal excitation (synth use).
  - Both (voice-per-note internally).
- [ ] **Polyphony**:
  - [decision] max voice count (typical: 8 – 32).
  - Voice stealing strategy (oldest / quietest).

## Phase 1: `rt-physical-core` (engine crate)

- [ ] **Workspace layout**:
  - `crates/rt-physical-core` — model primitives + voice manager.
  - `crates/rt-physical` — plugin.
- [ ] **Delay-line primitives**:
  - Fractional-delay (Lagrange or all-pass) for pitch-continuous
    waveguides.
  - Ring-buffer implementation, all-in-place.
- [ ] **Filters**:
  - Bridge / loop filter (1-pole low-pass).
  - Body filter (small biquad bank or IR-based — [decision] IR
    would reuse `rt-fft-convolver`).
- [ ] **Nonlinear stages** (for the tube family):
  - Reuse the shaper primitives from `rt-drive-core`
    ([Roadmap-02-oversampled-drive.md]) — potentially extract into
    a shared `rt-nonlinear-core` crate.
- [ ] **Voice manager**:
  - Fixed-capacity `[Voice; MAX_VOICES]` allocated at construction.
  - Note-on / note-off / voice-steal without allocations.
- [ ] **Modal bank**:
  - Array of biquads (resonator per mode).
  - Excitation impulse → sum of decaying sinusoids.

## Phase 2: `rt-physical` plugin

- [ ] **I/O**:
  - Audio in → audio out (effect use).
  - MIDI in → audio out (synth use).
  - Stereo out via body / bridge stereoisation.
- [ ] **Model selector**:
  - Dropdown: plucked, bowed, struck, tube-stage, modal.
- [ ] **Plucked-string controls**:
  - `Pluck Position` (0–100 %).
  - `Pick Hardness` (0–100 %).
  - `Damping` (0–100 %).
  - `Body Type` (dropdown: none, small, dreadnought, cello, custom
    IR).
  - `Detune` (cents).
- [ ] **Bowed-string controls**:
  - `Bow Force`, `Bow Velocity`, `Bow Position`.
- [ ] **Modal controls**:
  - `Modes` (1–32).
  - `Frequency Ratios` preset (metal, wood, glass, custom).
  - `Decay Slope` (bright modes decay faster / slower).
- [ ] **Global controls**:
  - `Gain`.
  - `Mix` (effect mode only).
  - `Polyphony` (voice limit).
  - `Bypass`.
- [ ] **MIDI**:
  - Full note routing with velocity → excitation energy mapping.
  - Aftertouch → damping (optional).
  - CC on the model's most-used parameter.
- [ ] **CPU meter** (per-voice cost × active voices).

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Model-specific parameter layout** — dropdown switches the
  visible controls.
- [ ] **Voice-activity meter**: active vs. maximum polyphony.
- [ ] **String / mode animation** for visual feedback (optional).

## Phase 4: Testing

- [ ] **Unit tests** (`rt-physical-core`):
  - Karplus-Strong: pitch of a note-on matches the requested
    frequency within ±5 cents.
  - Modal bank: single-mode response is a decaying sinusoid at the
    correct frequency.
  - Voice stealing: no state leak between reused voices.
  - Nonlinear tube stage: DC-blocker cancels DC step > 60 dB.
- [ ] **RT-safety audit**:
  - Zero allocations in `process()` including on note-on.
  - Denormal guard active.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets**: acoustic guitar (small / dread), classical
  guitar (nylon), harp, marimba, glockenspiel, bowed violin,
  metal-plate mode, custom tube-preamp voice.
- [ ] **Documentation**: manual with a model / control cross-reference.

## Out of scope (future work)

- Full physical piano model (Pianoteq scope — many years of R&D).
- Multiphysics coupling (string ↔ body feedback) — deferred.
- GPU rendering — not helpful for RT audio.
- Neural physical modelling (learn a model from a real instrument)
  — see `rt-neural-amp` for the neural path instead.
