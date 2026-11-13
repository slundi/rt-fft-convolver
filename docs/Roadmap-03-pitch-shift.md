# Roadmap: `rt-pitch` (Real-Time Pitch Shift / Time Stretch)

Real-time pitch-shifting plugin capable of interval shifts (musical
harmoniser), continuous shifts (whammy-style bends), and optional
formant-preserving mode.

**Reference implementations**: DigiTech Whammy, Eventide PitchFactor
/ H9, Boss PS-6, Little Alter Boy, zplane élastique, Waves
SoundShifter, MAutoPitch.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **Latency ≤ 15 ms** in the "live" mode (guitar / vocal
  monitoring).  Longer FFT modes available for offline / studio.
- **Latency is user-selectable** via a `Quality` control that maps
  to FFT / grain size.  Reported to the host as
  `nih_plug::latency_samples()` so the DAW can compensate on other
  tracks.
- **No allocations, no locks** on the audio thread.  All buffers,
  FFT plans, and windows sized at construction / preset change.
- **Formant preservation** — [decision] whether to ship it in the
  first release; adds significant DSP complexity.
- **8 pitch presets / intervals** with click-free crossfade on
  change (see `CrossfadingConvolver` design principle).

## Prerequisites in `rt-fft-convolver` ✅

- Reuse `dsp/fft_handler.rs` (`realfft`-backed real-to-complex
  wrapper).  Same abstraction, different consumer.
- `DenormalGuard` — spectral processing produces denormals when
  bins are near-silent.
- Ring-buffer utilities — **decision: split (Option C)**.  Core
  half **done**: generic, RT-safe `Ring<T: Copy + Default>` FIFO
  implemented in `utils/ring.rs` and re-exported as
  `rt_fft_convolver::Ring` — reused by delay / pre-delay / lookahead
  effects across the `rt-*` family.  The STFT overlap-add framer
  (`OverlapAdd` / `StftFramer`) remains private to `rt-pitch`
  (Phase 1) until a second consumer justifies promoting it (its shape
  depends on the Phase 0
  algorithm choice).  The `DirectConvolver` doubled-buffer stays as-is
  — it is a SIMD sliding window, not a FIFO, so it is not refactored
  onto `Ring<T>`.

## Phase 0: Algorithm decisions ✅

- **Primary algorithm** — **decision: all three selectable** behind an
  off-RT `Shifter` trait (`Box<dyn Shifter>` swapped at preset/param
  change; RT path unaffected).  Each mode reports its own
  `latency_samples()` so the host recompensates on switch.
  - **Phase Vocoder** ("Smooth") — STFT + Puckette peak-lock;
    polyphonic/chords, arbitrary ratios.  Default mode.
  - **WSOLA** ("Tight/Mono") — time-domain, low latency, punchy
    transients, best on monophonic leads.
  - **PSOLA / SOLA-FS** ("Mono Voice") — pitch-synchronous OLA for
    monophonic voice.
    - ⚠️ Requires a monophonic pitch tracker and **degrades on
      polyphonic input**.  Constraint: label the mode "Mono Voice" and
      auto-warn / bypass when polyphony is detected.
  - *(Hybrid PV+time-domain routing deferred to a post-1.0 upgrade — the
    `Shifter` trait leaves room for it.)*
- **Formant-preservation strategy** — **decision: user on/off toggle
  + selectable method**.  Exposes a `Formant Method` control:
  - **Cepstral liftering** — default; reuses the PV spectrum, cheapest.
  - **LPC-based** spectral envelope resampling — voice-tuned.
  - **True Envelope** estimation — highest quality; iteration count
    capped for RT bounding.
  - ⚠️ Cost: three RT-bounded envelope estimators to implement + test.
    Formant methods apply only to spectral (PV) mode; document behaviour
    under WSOLA/PSOLA (PSOLA is already partially formant-preserving).
- **FFT sizes to expose** — **settled**: at 4× overlap the
  algorithmic latency ≈ one frame.
  - 512 (~10.7 ms @ 48 kHz) — "Live" mode.
  - 1024 (~21.3 ms) — "Standard" mode.
  - 2048 (~42.7 ms) — "Studio" mode.
- **Overlap factor** — **decision: 4× fixed** (hop = frame/4,
  satisfies Hann COLA).  Keeps buffer sizes and reported latency
  deterministic for host compensation and the RT-safety audit; may
  become an "Advanced" control post-1.0.

## Phase 1: `rt-pitch-core` (engine crate) ✅

- **Workspace layout**:
  - `rt-pitch-core` — algorithm. **Done** (`rt-pitch` plugin is Phase 2).
- **`Shifter` trait** (dispatch boundary for the three algorithms):
  - `latency_samples(&self) -> usize`, `reset(&mut self)`,
    `process_block(&mut self, in: &[f32], out: &mut [f32])`.
  - Selection swapped off-RT (`Box<dyn Shifter>`); each impl owns its
    own buffers and reports its own latency.
- **PV shifter** (STFT via `realfft`, Hann default; Blackman-Harris
  optional) — `phase_vocoder.rs`, bin-remap (`smbPitchShift`):
  - Puckette peak-locking (toggle, default on).
  - Transient detection (spectral flux → phase reset on strong onsets).
- **WSOLA shifter** (time-domain, resample ∘ similarity-search
  time-stretch) — `wsola.rs`.
- **PSOLA shifter** + **monophonic pitch tracker** — `psola.rs`,
  `pitch.rs` (octave-error-corrected autocorrelation); polyphony
  detector → warn / bypass.
- **Shifter core (shared)** — `shifter.rs`:
  - `set_ratio(f32)` (smoothed); lock-free `SharedRatio` atomic cell for
    the GUI→RT parameter; semitone/ratio helpers.
- **Formant preservation** (spectral/PV mode; three estimators) —
  `formant.rs`:
  - Cepstral liftering (default), LPC (Levinson-Durbin), True Envelope
    (capped iterations).  Whiten on analysis, re-impose the original
    envelope on synthesis; user-selectable via `set_formant_preservation`.
- **Preset bank** — `preset.rs`:
  - 8 slots with click-free **equal-power** crossfade between ratios.

*(All 44 `rt-pitch-core` unit tests pass; `cargo clippy -D warnings`
clean; RT-safety: `DenormalGuard` on each `process_block`, buffers sized
at construction, output FIFOs on the shared `Ring<T>`.)*

## Phase 2: `rt-pitch` plugin ✅

- **I/O**:
  - Mono in → mono out or stereo in → stereo out.
  - Dual voice mode: independent shift per channel (harmoniser use) —
    `dual_voice` routes `voice2_preset` to the right channel.
- **Preset controls** (× 8):
  - `Interval N` (−24 … +24 semitones + cents fine tune, `0.01 st` step).
  - `Preset Enabled N`.
- **Global controls**:
  - `Current Preset` (int 0–7).
  - `Crossfade Time` (5–200 ms).
  - `Algorithm` (Smooth / Tight-Mono / Mono-Voice — selects the
    `Shifter`; "Mono-Voice" = PSOLA, warns on polyphony).
  - `Quality` (Live / Standard / Studio — sets FFT size).
  - `Mix` (0–100 %, wet/dry).
  - `Formant Preservation` (bool) — spectral (PV) mode only.
  - `Formant Method` (Cepstral / LPC / True Envelope) — active when
    formant preservation is on.
  - `Dry Delay Compensation` (bool) — align dry with wet after
    reported latency.
  - `Bypass`.
- **MIDI**:
  - Program Change → preset index (with a host-knob-move override so
    automation reclaims control after the user acts).
  - CC on `Current Preset` and `Interval` — every continuous control is
    an automatable parameter hosts bind to any CC (MIDI-learn).
  - Optional: MIDI note in → chase note as target pitch — *deferred*
    (post-Phase-3 leftover; the shared-ratio plumbing leaves room).
- **Latency reporting** to the host per selected `Quality`.

**Architecture** — mirrors the `rt-drive` split: a framework-agnostic
`engine` (off-RT `PitchController` builds the `Box<dyn Shifter>` voices;
audio-thread `PitchProcessor` polls the lock-free handoff and renders) plus a
thin `nih` wrapper (`plugin.rs`, behind the `nih` feature).  Click-free preset
changes use **two shifter voices per channel** blended by the core
`PresetBank` equal-power crossfade — both voices always render so the incoming
one is never faded in from its silent warm-up.  Structural changes (algorithm /
quality / formant) rebuild off-RT; everything else is applied in place.

*(All `rt-pitch` unit tests pass — 14 engine/params + the full 462-test
workspace; `cargo clippy` clean on default and `--features nih`; the CLAP /
VST3 wrapper builds.  The `nih-plug-egui` editor is Phase 3.)*

## Phase 3: UI ✅

-  **`nih-plug-egui`** — **decision: egui** (matches the sibling
  `rt-drive` editor; behind the `gui` feature so `cargo test` / `clippy`
  stay offline).
-  **Preset rack**: 8 pads (2 × 4) with the interval displayed as
  semitones + cents (`format_interval`); a pad click sets `Current
  Preset`, and the pad colour reflects active / enabled / disabled via
  the shared `pad_state`.
-  **Pitch trajectory display**: real-time, display-smoothed line of
  the current output pitch (`PitchTrajectory` ring, fed the selected
  interval published from the audio thread) — useful for
  expression-pedal / whammy automation.
-  **Latency badge** showing current reported latency in ms
  (`latency_ms` / `format_latency_ms`, from the audio-thread-published
  sample count + host sample rate).
-  **MIDI activity indicator**: header LED driven by the shared
  `MidiActivity` LED state (green = recognised Program Change, red =
  unrecognised), pulsed from the audio thread via a relaxed atomic.

**Architecture** — the pure editor logic (interval labels, latency
conversion, the trajectory ring) lives in a framework-agnostic
`ui.rs` and is unit-tested without `egui`; the `nih-plug-egui` drawing
is a thin `gui`-gated layer in `plugin.rs` fed by four relaxed atomics
(current pitch, latency samples, sample rate, MIDI pulse) so the audio
thread never blocks on the editor.

*(All `rt-pitch` tests pass — engine/params + 5 new `ui` tests, full
workspace green; `cargo clippy` clean on default, `--features nih`, and
`--features gui`; the egui editor compiles against `nih-plug-egui`.)*

## Phase 4: Testing ✅ (host validation deferred to Phase 5)

- [x] **Unit tests** (`rt-pitch-core`):
  - Ratio = 1.0 reconstructs the input within the noise floor
    (`unison_reconstructs_within_noise_floor`: correlation > 0.98,
    scale-corrected residual < −20 dB — STFT overlap-add is not
    bit-identical, but Hann COLA at 4× is high-fidelity).
  - Ratio = 2.0 (octave up) on a 440 Hz sine → dominant 880 Hz peak
    (`octave_up_doubles_pitch`).
  - Ratio changes crossfade cleanly — the equal-power crossfade holds
    RMS steady with no dropout/spike (`preset_change_has_no_click` in
    the `rt-pitch` integration suite, where the two-voice blend lives).
- [x] **Integration tests** (`rt-pitch/tests/integration.rs`):
  - Reference frequencies from a **manual DFT** (a self-contained
    Goertzel probe — no external `librosa` vectors): octave up/down and
    fifth land at the predicted Hz end-to-end through the engine.
  - MIDI PC / CC → preset routing (`program_to_slot` / `cc_to_slot`
    drive the engine to the routed preset's pitch; out-of-range PC is
    ignored; the CC sweep spans all eight slots).
  - Latency warm-up matches the reported value
    (`output_is_silent_through_the_reported_latency`).
- [x] **RT-safety audit**:
  - Allocator hook: zero allocations in `process()`
    (`rt-pitch/tests/rt_alloc.rs` — a counting `#[global_allocator]`
    proves the busiest stereo / dual-voice / formant render allocates
    nothing in steady state).
  - Denormal guard active — silence flushes to true zero
    (`silence_flushes_to_true_zero`).
  - Latency matches the reported value exactly, including across a
    quality rebuild (`reported_latency_matches_build`).
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval` —
  *deferred to Phase 5*: needs the built `.lv2` / `.clap` / `.vst3`
  bundles and the external validators (the `gui` build already requires
  system X11/GL libraries absent from the plain dev shell).

*(12 new tests — 1 core fidelity + 8 engine integration + 2 RT-safety +
1 zero-allocation; full workspace green at 480 passing; `cargo clippy`
clean on default, `--features nih`, `--features gui`, and `--all-targets`.)*

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets**: octave up, octave down, fifth, fourth, minor
  third, major third, 5-cent detune (chorus effect), 20-cent detune.
- [ ] **Documentation**: manual with a latency / quality trade-off
  table.

## Out of scope (future work)

- Polyphonic pitch correction (iZotope Nectar-style) — needs
  per-voice tracking; separate plugin.
- Auto-tune / snap-to-scale — [decision] whether to add as a
  Phase 6 to this plugin or a separate `rt-tune`.
- Time-stretching without pitch shift — related algorithm but
  different use case (offline / clip-based); host handles this via
  warping.
- Multi-voice harmony from a single input (Antares Harmony Engine
  style) — needs chord input; separate plugin.
