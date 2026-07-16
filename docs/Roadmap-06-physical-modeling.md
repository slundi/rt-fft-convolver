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

## Prerequisites in `rt-fft-convolver` ✅

- `DenormalGuard` — **satisfied**. Fully implemented in
  `src/utils/denormals.rs` and re-exported from the crate root
  (`src/lib.rs:34`): RAII FTZ+DAZ guard (x86/x86_64 MXCSR,
  AArch64 FPCR, no-op elsewhere), restores the exact prior FPU
  state on drop, RT-safe (one register read+write), test-covered.
  This is the load-bearing prerequisite: waveguide loops and modal
  resonators with light damping decay *through* the denormal range
  after note-off, so the render callback must be wrapped in a
  `DenormalGuard` to avoid the classic tail-CPU spike.
- **Optional reuse (not a blocker)** — the body / bridge filter of
  the plucked and waveguide families can be an impulse response
  convolved with the existing partitioned engine
  (`src/engine/partition.rs`), letting a captured guitar-body IR
  stand in for a hand-tuned biquad body model. Wired opportunistically
  in Phase 1, not required for v1.
- Nothing else in the convolution engine is required; the model
  families are self-contained DSP.

## Phase 0: Model-family decisions ✅

### Deliverable shape — **decision: one plugin, model dropdown, shared `rt-physical-core`**

Resolves the top-of-file [decision]. Ship a *single* `rt-physical`
plugin whose model selector switches families, backed by one
`rt-physical-core` crate (the Phase 1 layout). All families share the
same fixed-capacity voice manager, fractional-delay primitive and
loop/body filters — only the per-voice DSP kernel differs — so a
shared core is where the reuse lives. A single instance also lets a
guitarist A/B "plucked" against "modal" without reloading.

- *Rejected*: **one plugin per family**. Multiplies packaging, CI,
  and UI surface for no DSP benefit, and blocks in-session A/B.

### v1 families — **decision: Karplus-Strong extended + Modal bank** (waveguide as a KS tier; bowed & tube deferred)

Two engines cover the useful ground for a guitar-oriented suite:

1. **Karplus-Strong extended (plucked string)** — cheapest convincing
   string; directly on-brand for a guitar library.
2. **Modal bank (struck / metallic)** — same *excitation → resonator*
   shape but a bank of biquads instead of a delay loop; covers
   marimba, glockenspiel, plate, bells.

The listed **digital waveguide "with body/bridge filtering"** is *not*
a separate engine — a KS-extended string already **is** a simplified
waveguide. It becomes the "waveguide tier" by swapping the fixed loop
filter for a tunable bridge filter plus an optional body IR (see
prerequisite reuse). Low marginal cost, so it rides on the KS kernel.

- *Deferred*: **bowed string** (needs a nonlinear bow-string friction
  scattering junction — MSW-style — which is fiddly to keep
  RT-stable) and **state-space tube** (a nonlinear circuit, a
  different discipline; belongs with the shared nonlinear-shaper
  extraction from `rt-drive-core`, not the voiced-instrument core).

**Example — KS-extended, A2 = 110 Hz at 44.1 kHz.** The loop delay
that sets pitch is `L = fs / f0 = 44100 / 110 ≈ 400.909` samples. The
integer part (400) is the ring-buffer read offset; the fractional
`0.909` is realised with a first-order all-pass
`A(z) = (a + z⁻¹)/(1 + a·z⁻¹)`. **The common bug:** the one-pole loop
filter itself contributes phase delay `Dφ`, so the tuned delay must be
`L − Dφ − 1`, not `L`. Skip that compensation and every note plays
sharp — worse in the high register where `Dφ` is a larger fraction of
`L`. Decay is set by the loop-filter gain `g < 1`; for a target
`T60`, `g = 10^(−3·L / (T60·fs))`.

**Example — Modal, one glockenspiel mode at 2000 Hz, T60 = 1.5 s.**
Each mode is a two-pole resonator
`y[n] = 2R·cosθ·y[n−1] − R²·y[n−2] + x[n]` with pole angle
`θ = 2π·f/fs = 2π·2000/44100 ≈ 0.2850 rad` and radius
`R = 10^(−3 / (T60·fs)) = 10^(−3 / (1.5·44100)) ≈ 0.99990`. An
excitation impulse produces a decaying sinusoid at `f` — sum a handful
of these (inharmonic ratios for metal, harmonic-ish for wood) to build
a struck tone.

### Excitation source — **decision: both, voice-per-note internally**

Every voice owns an internal excitation generator; the *source* of its
trigger is what differs:

- **MIDI note-on (synth use)** → synthesise a short excitation burst
  (filtered noise for a pluck, an impulse/mallet shape for modal),
  windowed by the `Pick Hardness` control and scaled by velocity.
- **Audio input (effect use)** → the incoming audio *is* the
  excitation, injected at the pluck-position tap; MIDI still selects
  pitch/gates the voice.

Doing both is nearly free once the injection point is a first-class
input, and it is what makes the same plugin work as a synth *and* as a
"play a chord through the resonator" effect.

**Example — velocity → energy.** A MIDI note-on at velocity `v`
(0–127) injects a noise burst of length `~2–5 ms` whose peak is
`gain·(v/127)`, low-pass-filtered with cutoff rising with
`Pick Hardness` (soft pick = duller excitation). The burst is written
into the delay line at the fractional tap
`round(pluck_position · L)`, which sets the comb-filter timbre of the
pluck.

### Polyphony — **decision: 16 voices, steal quietest-then-oldest**

Fixed `[Voice; 16]` allocated at construction. **16** is the guitar
sweet spot — 6 strings plus ring-over/overlap headroom for fast
passages and let-ring chords — while staying well under the per-voice
CPU ceiling of the modal bank. Not user-growable at runtime (would
allocate); the `Polyphony` control instead *caps* how many of the 16
may sound.

Voice-stealing is **quietest-first, oldest as tie-break**: on a note-on
with all voices active, steal the voice with the lowest running
envelope RMS (least audible to drop), breaking ties by age. Cross-fade
the steal over `~2 ms` so the reused voice doesn't click.

- *Rejected*: **8** (too tight for six-string let-ring) and **32**
  (doubles worst-case modal CPU for headroom a guitar rarely uses).
  **Pure oldest-note** stealing is simpler but audibly clips a
  still-ringing loud note in favour of a decayed one — RMS-first
  avoids that.

**Example.** 16 voices sounding, note-on arrives. Scan the 16
envelope followers, find index `k` with `min(rms)`; if two are within
1 dB, pick the older. Ramp voice `k`'s output gain to 0 over ~88
samples (2 ms @ 44.1 kHz), reset its delay line / resonator state,
then start the new note — no allocation, no lock, no click.

## Phase 1: `rt-physical-core` (engine crate)

- [ ] **Workspace layout**:
  - `rt-physical-core` — model primitives + voice manager.
  - `rt-physical` — plugin.
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
