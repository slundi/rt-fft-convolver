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

## Phase 1: `rt-physical-core` (engine crate) ✅

- [x] **Workspace layout**:
  - `rt-physical-core` — model primitives + voice manager. **Done**
    (added to the workspace `members` / `default-members`).
  - `rt-physical` — plugin. *Deferred to Phase 2* (this phase is the
    framework-agnostic core only, mirroring `rt-fdn-core`).
- [x] **Delay-line primitives**:
  - Fractional delay via a first-order all-pass tuning filter
    (`allpass.rs`, `TuningAllpass`) — chosen over Lagrange for exact
    unity magnitude, as in the Phase 0 example.
  - Integer ring buffer, power-of-two, all-in-place, denormal-flushed
    (`delay.rs`, `DelayLine`).
- [x] **Filters**:
  - Bridge / loop filter (`filter.rs`, `LoopFilter`) — a one-zero
    damping filter with a broadband decay gain and a closed-form
    phase delay folded into the tuning budget.
  - Body filter as a peaking biquad (`biquad.rs`, `Biquad::peaking`).
    IR-based body via `rt-fft-convolver` remains the optional upgrade
    noted in the prerequisites.
- [ ] **Nonlinear stages** (for the tube family):
  - *Deferred* — the tube family is out of the Phase 0 v1 scope; the
    shared `rt-nonlinear-core` extraction happens when it lands.
- [x] **Voice manager**:
  - Fixed-capacity `[Voice; N]` allocated at construction
    (`manager.rs`, `VoiceManager<N>`; `MAX_VOICES = 16`,
    `PhysicalVoices` alias).
  - Note-on / note-off / quietest-then-oldest voice-steal, all
    allocation-free, with a ~2 ms anti-click fade on (re)trigger.
- [x] **Modal bank**:
  - Array of biquad resonators (`modal.rs`, `ModalBank`, up to
    `MAX_MODES = 32`) with metal / wood / glass `ModePreset`s.
  - Excitation impulse → sum of decaying sinusoids, with a
    `decay_slope` making bright modes decay faster.

Excitation (`excitation.rs`, `Excitation`) covers the Phase 0 "both
sources" decision: a velocity/hardness-shaped, pluck-positioned noise
burst for MIDI/synth use; the plugin feeds audio in for effect use.
53 unit tests (pitch within ±5 cents, single-mode decaying sinusoid,
no state leak across reused voices, quietest-first stealing, finite
output on sustained tails); `cargo fmt` + `clippy` clean.

## Phase 2: `rt-physical` plugin ✅

Framework-agnostic DSP (`params.rs`, `engine.rs`) with the `nih-plug`
wrapper (`plugin.rs`) behind the opt-in **`nih`** feature — the same
split as `rt-fdn-reverb`, so `cargo test` / `clippy` stay offline.

- [x] **I/O**:
  - MIDI in → audio out (synth use) — sample-accurate note routing.
  - Audio in → audio out (effect use) via the `Mix` dry/wet blend;
    the plugin declares both a stereo-input and a no-input layout.
    *(Per-voice audio-signal excitation — audio driving the string
    loop directly — is a future enhancement; Phase 2 blends the input
    at the bus.)*
  - Stereo out via body stereoisation (two body filters spread ±3 %).
- [x] **Model selector** — dropdown over the two v1 families
  (`Plucked`, `Modal`).  Bowed / struck / tube stay deferred per the
  Phase 0 scope.
- [x] **Plucked-string controls**: `Pluck Position`, `Pick Hardness`,
  `Damping`, `Body Type` (none / small / dreadnought / cello — custom
  IR deferred), `Detune` (± one semitone in cents).
- [ ] **Bowed-string controls** — *deferred* with the bowed family.
- [x] **Modal controls**: `Modes` (1–32), `Frequency Ratios` preset
  (metal / wood / glass), `Decay Slope`.
- [x] **Global controls**: `Gain`, `Mix`, `Polyphony`, `Bypass`, plus
  a shared `Decay` (sustain / T60) needed by both families.
- [x] **MIDI**:
  - Full note on/off routing with velocity → excitation energy.
  - Aftertouch (channel pressure) → `Damping`.
  - CC 74 (Brightness) → `Damping` (the model's most-used control),
    with host-move arbitration.
- [x] **CPU meter** — render-time / block-budget proxy stored for the
  Phase 3 editor, plus `active_voice_count()` on the engine.

18 plugin unit tests (per-family note-on, bypass, mix, gain scaling,
body stereo decorrelation, model-switch silencing, polyphony cap,
reset, detune, finite output under load) + 1 new core detune test;
`nih` feature compiles for CLAP/VST3; `cargo fmt` + `clippy` clean on
both the default and `nih` builds.

## Phase 3: UI ✅

Behind the opt-in **`gui`** feature (`gui = ["nih", "dep:nih_plug_egui"]`),
so the default and `nih` builds stay offline / X11-free — the same split
`rt-fdn-reverb` uses.  The framework-agnostic display maths live in
`src/ui.rs` (unit-testable without the plugin runtime); the editor closure
in `src/plugin.rs` only turns those numbers into widgets.

- [x] **`nih-plug-egui`** — **decision: egui** (over `nih-plug-iced`).
  Every other plugin editor in the workspace (IR cab / reverb, drive,
  spectral, FDN reverb) is egui; matching keeps one immediate-mode idiom,
  one set of X11/OpenGL CI deps, and lets the shared LED / meter drawing
  style carry over.  `iced`'s retained model buys nothing here.
- [x] **Model-specific parameter layout** — a `Plucked` / `Modal` selector
  (buttons bound to the `model` param) switches which control block the
  grid shows; shared `Decay` and the global `Gain` / `Mix` / `Polyphony`
  are always visible.
- [x] **Voice-activity meter** — a row of pips, `active / polyphony` lit,
  fed by `active_voice_count()` stored per block into an atomic; the cap is
  the live `Polyphony` value.
- [x] **String / mode animation** — realised as a **partial-spectrum**
  visualiser: bars laid out along the frequency-ratio axis show the comb-
  filtered harmonics of a pluck (`|sin(n·π·pluck_position)|` with a
  damping/brightness tilt) or the preset's inharmonic modal partials.  Plus
  a note-activity LED in the header.

`ui.rs` carries 6 unit tests (centre-pluck nulls even harmonics, damping
darkens the highs, modal spectrum follows the preset ratios, mode-count cap,
finite/in-range amplitudes, dB endpoints); `cargo fmt` + `clippy` clean on
the default and `nih` builds (the `gui` build links X11/OpenGL and is
exercised in CI, as with the other egui plugins).

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
