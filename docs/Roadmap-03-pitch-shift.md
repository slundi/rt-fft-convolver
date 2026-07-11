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

## Prerequisites in `rt-fft-convolver`

- [ ] Reuse `dsp/fft_handler.rs` (`realfft`-backed real-to-complex
  wrapper).  Same abstraction, different consumer.
- [ ] `DenormalGuard` — spectral processing produces denormals when
  bins are near-silent.
- [ ] Ring-buffer utilities — [decision] whether to extract a shared
  `Ring<T>` into the core crate or keep it private to `rt-pitch`.

## Phase 0: Algorithm decisions

- [ ] **Primary algorithm** — [decision]:
  - Phase Vocoder (STFT + phase-lock).
  - WSOLA (time-domain, cheaper, lower quality on complex signals).
  - SOLA-FS or PSOLA (better for monophonic voice).
  - Hybrid (phase vocoder for polyphonic, WSOLA for percussive).
- [ ] **Formant-preservation strategy** — [decision]:
  - LPC-based spectral envelope resampling.
  - Cepstral liftering.
  - True Envelope estimation.
  - None (formants shift with pitch — cheaper, "chipmunk" artefact).
- [ ] **FFT sizes to expose**:
  - 512 (~11 ms @ 48 kHz) — "Live" mode.
  - 1024 (~21 ms) — "Standard" mode.
  - 2048 (~43 ms) — "Studio" mode.
- [ ] **Overlap factor**: 4× fixed, or exposed as control — [decision].

## Phase 1: `rt-pitch-core` (engine crate)

- [ ] **Workspace layout**:
  - `crates/rt-pitch-core` — algorithm.
  - `crates/rt-pitch` — plugin.
- [ ] **Analysis / synthesis frame**:
  - Windowed STFT via `realfft`.
  - Hann or Blackman-Harris window (Hann default).
- [ ] **Phase-lock heuristics** (if phase vocoder chosen):
  - Puckette peak-locking.
  - Transient detection (bypass phase update on strong onsets).
- [ ] **Shifter core**:
  - `set_ratio(f32)` — off-RT; the RT path reads an atomic-updated
    ratio.
  - `process_block(&mut self, in: &[f32], out: &mut [f32])`.
- [ ] **Formant preservation** (if enabled):
  - Envelope estimation on the input frame.
  - Envelope re-application on the shifted frame.
- [ ] **Preset bank**:
  - 8 slots with click-free crossfade between ratios.

## Phase 2: `rt-pitch` plugin

- [ ] **I/O**:
  - Mono in → mono out or stereo in → stereo out.
  - Dual voice mode: independent shift per channel (harmoniser use).
- [ ] **Preset controls** (× 8):
  - `Interval N` (−24 … +24 semitones + cents fine tune).
  - `Preset Enabled N`.
- [ ] **Global controls**:
  - `Current Preset` (int 0–7).
  - `Crossfade Time` (5–200 ms).
  - `Quality` (Live / Standard / Studio — sets FFT size).
  - `Mix` (0–100 %, wet/dry).
  - `Formant Preservation` (bool) — only if the algorithm supports
    it.
  - `Dry Delay Compensation` (bool) — align dry with wet after
    reported latency.
  - `Bypass`.
- [ ] **MIDI**:
  - Program Change → preset index.
  - CC on `Current Preset` and `Interval` (for whammy-pedal-style
    continuous bends).
  - Optional: MIDI note in → chase note as target pitch.
- [ ] **Latency reporting** to the host per selected `Quality`.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Preset rack**: 8 pads with the interval displayed as
  semitones + cents.
- [ ] **Pitch trajectory display**: real-time line showing the
  current output pitch (useful for expression-pedal / whammy).
- [ ] **Latency badge** showing current reported latency in ms.
- [ ] **MIDI activity indicator**.

## Phase 4: Testing

- [ ] **Unit tests** (`rt-pitch-core`):
  - Ratio = 1.0 produces bit-identical output (or within noise
    floor, given windowing).
  - Ratio = 2.0 (octave up) applied to a 440 Hz sine produces a
    dominant 880 Hz peak.
  - Ratio changes crossfade cleanly (no discontinuity > −60 dBFS in
    the wet signal at the transition).
- [ ] **Integration tests**:
  - Reference test vectors from Python (librosa or manual STFT).
  - MIDI PC / CC → preset routing.
- [ ] **RT-safety audit**:
  - Allocator hook: zero allocations in `process()`.
  - Denormal guard active.
  - Latency matches the reported value exactly.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

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
