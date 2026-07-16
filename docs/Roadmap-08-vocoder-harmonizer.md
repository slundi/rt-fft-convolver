# Roadmap: `rt-vox` (Vocoder / Harmoniser / Formant Shifter)

Voice-processing plugin combining vocoder (carrier + modulator),
harmoniser (single input → chord output), and formant shifter.
**Decision (Phase 0): one plugin, three modes** sharing a single
`rt-vox-core` engine crate — see Phase 0 below.

**Reference implementations**: TC-Helicon VoiceLive, Boss VE-20 /
VE-500, DigiTech Vocalist, Antares Harmony Engine, Waves OVox,
iZotope VocalSynth.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **Pitch detection is monophonic** — voice-focused; not polyphonic
  input.
- **Latency ≤ 15 ms** for live use.  Higher-quality "studio" mode
  optional at longer latency.
- **RT-safe** — pitch tracker, formant analyser, and resynthesiser
  all allocation-free on the audio thread.
- **MIDI chord input** for the harmoniser — chord notes come from a
  MIDI keyboard, guitar controller, or automation.
- **Formant control is first-class** — vocoder and harmoniser both
  gain from correct formant handling.  Reuse the pitch-shifter
  formant path from `rt-pitch-core` if formant support is added
  there ([Roadmap-03-pitch-shift.md]).

## Prerequisites in `rt-fft-convolver` ✅

- `DenormalGuard` — present and tested
  ([src/utils/denormals.rs](../src/utils/denormals.rs)); RAII FTZ/DAZ
  guard, re-exported by `rt-pitch-core`.
- `dsp/fft_handler.rs` — present and tested
  ([src/dsp/fft_handler.rs](../src/dsp/fft_handler.rs)); allocation-free
  real→complex / complex→real FFT, reused for FFT-vocoder band grouping
  and formant estimation.
- **Bonus prerequisite met**: `rt-pitch-core` is already built
  (Roadmap-03 Phases 0–1 complete) and exports exactly the primitives
  this plugin needs — see the reuse decisions below.

## Phase 0: Algorithm decisions ✅

The dominant Phase 0 finding is that **`rt-pitch-core` already ships
RT-safe, tested implementations** of pitch tracking, formant estimation,
and pitch shifting. Every decision below therefore favours *reuse* over a
second parallel implementation, keeping this plugin inside the "≤ 15 ms,
allocation-free, no second tracker to maintain" design principles.

- **Plugin packaging** — **decision: one plugin, three modes**
  (`Vocoder` / `Harmoniser` / `Formant Shifter`) over a shared
  `rt-vox-core` crate, matching every other `*-core` + plugin pair in
  the workspace.
  - *Options considered.* (a) One plugin with a mode selector — the
    three modes share the pitch tracker, formant analyser, FFT infra,
    and the global `Mix`/`Output`/`Bypass` controls Phase 2 already
    lists. (b) Three separate plugins over a shared core — triples
    bundling, CI, installers, and preset banks for no user benefit.
  - *Examples in the wild.* iZotope VocalSynth and Waves OVox are single
    plugins with switchable modules; that is the (a) model.

- **Pitch detection** — **decision: reuse
  `rt_pitch_core::PitchTracker`** (autocorrelation + parabolic
  interpolation) — the "simpler baseline" option, already RT-safe.
  - *Options considered.* YIN is marginally more robust on octave errors
    but is a *second* tracker to build and test; CREPE-style neural adds
    heavy CPU and a model-weights dependency, breaking the RT /
    allocation-free principle. The existing tracker already emits
    `PitchEstimate { frequency_hz, period_samples, confidence }` and has
    a polyphony guard (`is_polyphonic`) — everything the harmoniser and
    formant tracker need.
  - *Example.* A 440 Hz sine frame → `PitchEstimate { frequency_hz ≈
    440, confidence ≈ 1.0 }`; a C-major chord → `is_polyphonic() ==
    true`, so PSOLA-style modes auto-bypass (Phase 4 acceptance test).

- **Pitch-shift back-end** — **decision: reuse `rt-pitch-core`** via
  its off-RT-swappable `Shifter` trait (`process_block`,
  `latency_samples`, `reset`); no voice-specific re-implementation.
  - *Options considered.* An independent voice-tuned shifter duplicates
    ~2 000 lines of already-tested PV/WSOLA/PSOLA code. The trait already
    exposes `PhaseVocoder` (default, handles chords), `Wsola`
    (low-latency mono), and `Psola` (mono voice) so the harmoniser can
    pick per-voice.
  - *Example.* A harmony voice a major third up:
    `let ratio = semitones_to_ratio(4.0); // ≈ 1.26`, fed to a
    `Box<dyn Shifter>` instance; the ±12-semitone Phase 2 range fits
    inside `MIN_RATIO..=MAX_RATIO` (0.25..=4.0 = ±24 semitones).

- **Formant analysis** — **decision: reuse
  `rt_pitch_core::FormantEstimator`, all three `FormantMethod`s
  selectable, default `Lpc` for the voice path.**
  - *Options considered.* LPC (Levinson-Durbin all-pole) is the textbook
    voice formant model and the natural fit for the vocoder's
    envelope-transfer stage — chosen as default here even though
    `rt-pitch-core`'s own default is `Cepstral` (cheapest, reuses the PV
    spectrum). `TrueEnvelope` is offered for highest quality (iteration
    count is capped for RT bounding). All three already exist and are
    allocation-free after construction.
  - *Example.* At 48 kHz the voice LPC order ≈ `sample_rate / 1000 + 2`
    ≈ 50; `estimate(magnitude, &mut envelope)` yields the smooth
    spectral envelope, and `preserve(orig_env, shifted_env, &mut mag)`
    holds formants still when only pitch is shifted.

- **Vocoder topology** — **decision: FFT-based magnitude-envelope
  transfer, with the `Bands` control implemented as log-spaced FFT-bin
  grouping** (a true SVF channel bank is deferred as a later "analog"
  voicing, mirroring Roadmap-03's deferred-hybrid approach).
  - *Options considered.* (a) Analog channel bank — 16–32 state-variable
    band-pass filters with per-band envelope-follower cross-modulation;
    distinctive resonant character but a new filter bank to tune and
    test, and it does not reuse the FFT infra the prerequisite calls
    out. (b) FFT-based — reuses `FftHandler` + the `rt-pitch-core` STFT
    framing, gives arbitrary per-bin resolution, integrates directly with
    the formant path, and is cheaper to get correct for a 1.0. Grouping
    bins into N log-spaced bands preserves the familiar `Bands`
    (8/16/24/32) control and much of the analog voicing.
  - *Example.* Per STFT frame: take the **modulator** magnitude
    spectrum, sum bin energies into 16 log-spaced bands, smooth each with
    an attack/release envelope follower, then scale the **carrier**'s
    bins in each band by that envelope and IFFT. Silent modulator →
    silent output; unity-envelope modulator → carrier passes through
    (Phase 4 pass-through test).

- **Harmoniser voicing** — **decision: all three modes selectable
  via a `Snap Mode` control (Free / Scale / MIDI), default MIDI** per the
  "MIDI chord input" design principle.
  - *Options considered.* Free = fixed per-voice semitone offsets
    (simplest, no musical context). Scale = snap each computed harmony
    note to the nearest degree of `Key` + `Scale` (diatonic thirds,
    etc.). MIDI = harmony pitches taken directly from held MIDI notes —
    the most expressive and the stated primary use case. All three share
    the same per-voice `Shifter`; only the interval source differs.
  - *Example.* Input note C4: **Free** `+7` → G4; **Scale** (C major,
    "third above") → E4; **MIDI** holding E4 + G4 → two voices tracking
    E4 and G4. Each resolves to a ratio via `semitones_to_ratio`.

## Phase 1: `rt-vox-core` (engine crate)

- [ ] **Workspace layout**:
  - `rt-vox-core` — engine.
  - `rt-vox` — plugin.
- [ ] **Pitch tracker** (YIN or chosen):
  - Frame size, hop size, confidence output.
  - Off-RT-safe: uses only fixed-size scratch buffers.
- [ ] **Formant analyser**:
  - LPC coefficients per frame (chosen order, typically 12–18).
  - Envelope reconstruction from coefficients.
- [ ] **Vocoder engine**:
  - Channel bank of state-variable band-pass filters (or FFT bins).
  - Envelope follower per band on the modulator.
  - Amplitude modulation of the carrier by the modulator envelope.
- [ ] **Harmoniser engine**:
  - N voices × pitch-shifter.
  - Voice manager with MIDI-driven interval assignment.
- [ ] **Formant shifter**:
  - Independent path from pitch shift; user can shift formants
    without shifting pitch, and vice versa.

## Phase 2: `rt-vox` plugin

- [ ] **I/O**:
  - Mono in (voice) + sidechain in (carrier, for vocoder mode) →
    stereo out.
  - Stereo in → stereo out for the harmoniser / formant modes.
- [ ] **Mode selector**:
  - `Vocoder` / `Harmoniser` / `Formant Shifter`.
- [ ] **Vocoder controls**:
  - `Bands` (8 / 16 / 24 / 32).
  - `Attack`, `Release` (envelope followers).
  - `Formant Shift` (± 1 octave).
  - `Sibilance Handling` (bool + amount) — passes highs from the
    modulator directly.
- [ ] **Harmoniser controls**:
  - `Voice N Interval` (× 4 voices, e.g. −12 … +12 semitones).
  - `Voice N Level`.
  - `Voice N Pan`.
  - `Voice N Formant Shift`.
  - `Voice N Detune`.
  - `Snap Mode` (free / scale / MIDI).
  - `Key` and `Scale` (if scale mode).
- [ ] **Formant Shifter controls**:
  - `Formant Shift` (± 1 octave).
  - `Pitch Shift` (± 12 semitones, if desired).
- [ ] **Global controls**:
  - `Mix` (0–100 %).
  - `Output Gain`.
  - `Bypass`.
- [ ] **MIDI**:
  - Note in → chord voices for the harmoniser.
  - CC on `Formant Shift` and per-voice `Interval`.
- [ ] **Latency reporting** per mode.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Pitch tracking indicator**: current detected pitch (Hz +
  note name) and confidence bar.
- [ ] **Formant display**: current spectral envelope overlaid on
  the input spectrum.
- [ ] **Voice grid** for the harmoniser: 4 rows × interval / level /
  formant / detune columns.
- [ ] **MIDI activity indicator**.

## Phase 4: Testing

- [ ] **Unit tests** (`rt-vox-core`):
  - Pitch tracker: 440 Hz sine → 440 Hz ± 2 Hz output.
  - Vocoder pass-through: matches direct carrier when the modulator
    is silent (or at unity envelope).
  - Harmoniser: unison voice at 0 semitones is bit-identical (or
    within noise) to the dry.
  - Formant shifter: pitch is preserved when only formant is
    shifted (± 5 cents drift maximum).
- [ ] **Integration tests**:
  - MIDI chord notes trigger the harmoniser voices sample-accurately.
- [ ] **RT-safety audit**:
  - Zero allocations in `process()`, including on MIDI note-on.
  - Denormal guard active.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets**: robot vocoder, choir vocoder, third-above
  harmoniser, octave-up harmoniser, doubler (5-cent detune × 2),
  gender-swap formant.
- [ ] **Documentation**: manual with a mode-selection guide and
  MIDI routing diagram.

## Out of scope (future work)

- Polyphonic pitch correction — different problem (per-voice
  tracking); if pursued, a separate plugin.
- Talkbox emulation (physical excitation model) — see
  `rt-physical` ([Roadmap-06-physical-modeling.md]).
- Speech synthesis / TTS — not a real-time-audio concern.
- Convolution-based formant transfer — deferred; would reuse
  `rt-fft-convolver`.
