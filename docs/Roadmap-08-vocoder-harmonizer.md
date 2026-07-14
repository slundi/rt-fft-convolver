# Roadmap: `rt-vox` (Vocoder / Harmoniser / Formant Shifter)

Voice-processing plugin combining vocoder (carrier + modulator),
harmoniser (single input → chord output), and formant shifter.
[decision] whether these are one plugin with modes or three
separate plugins sharing a core crate.

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

## Prerequisites in `rt-fft-convolver`

- [ ] `DenormalGuard`.
- [ ] `dsp/fft_handler.rs` — reused for the vocoder channel-bank
  filter and formant estimation.

## Phase 0: Algorithm decisions

- [ ] **Pitch detection** — [decision]:
  - YIN (time-domain, low latency, robust for voice).
  - CREPE-style small neural (higher CPU, higher accuracy).
  - Autocorrelation with parabolic interpolation (simpler baseline).
- [ ] **Pitch-shift back-end** — [decision]:
  - Reuse `rt-pitch-core` (must be built first).
  - Independent implementation optimised for voice.
- [ ] **Formant analysis** — [decision]:
  - LPC (fast, standard).
  - True Envelope.
  - Cepstral liftering.
- [ ] **Vocoder topology** — [decision]:
  - Analog-style channel bank (16 – 32 band-pass filters,
    envelope-follower cross-modulation).
  - FFT-based (magnitude-envelope transfer per bin).
- [ ] **Harmoniser voicing** — [decision]:
  - Free intervals (independent per voice).
  - Scale-aware snap-to-scale (needs a key/scale input).
  - MIDI-driven chord (harmony notes from MIDI).

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
