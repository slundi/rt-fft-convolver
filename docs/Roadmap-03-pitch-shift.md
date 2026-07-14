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

## Phase 1: `rt-pitch-core` (engine crate)

- [ ] **Workspace layout**:
  - `rt-pitch-core` — algorithm.
  - `rt-pitch` — plugin.
- [ ] **`Shifter` trait** (dispatch boundary for the three algorithms):
  - `latency_samples(&self) -> usize`, `reset(&mut self)`,
    `process_block(&mut self, in: &[f32], out: &mut [f32])`.
  - Selection swapped off-RT (`Box<dyn Shifter>`); each impl owns its
    own buffers and reports its own latency.
- [ ] **PV shifter** (STFT via `realfft`, Hann default; Blackman-Harris
  optional):
  - Puckette peak-locking.
  - Transient detection (bypass phase update on strong onsets).
- [ ] **WSOLA shifter** (time-domain, correlation search window).
- [ ] **PSOLA shifter** + **monophonic pitch tracker** (epoch detection);
  polyphony detector → warn / bypass.
- [ ] **Shifter core (shared)**:
  - `set_ratio(f32)` — off-RT; the RT path reads an atomic-updated
    ratio.
- [ ] **Formant preservation** (spectral/PV mode; three estimators):
  - Cepstral liftering (default), LPC, True Envelope (capped iterations).
  - Envelope estimation on the input frame, re-application on the
    shifted frame.
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
