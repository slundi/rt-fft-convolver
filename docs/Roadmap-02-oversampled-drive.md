# Roadmap: `rt-drive` (Oversampled Nonlinear Drive Plugin)

Real-time distortion / overdrive / saturation plugin built around
oversampled waveshaping.  A single reusable engine can host many
"pedal" or "amp stage" flavours by swapping the shaper curve and
tone-stack parameters.

**Reference implementations**: Waves GTR3, IK AmpliTube stages,
Klon Centaur / TS808 / ProCo Rat / Big Muff emulations, U-he Satin,
Softube Tape, Neural DSP drive stages.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **Aliasing floor below −80 dBFS** at 48 kHz for the loudest input,
  worst-case shaper.  Achieved via oversampling + FIR anti-aliasing.
- **Oversampling factor is a user setting** — 2× / 4× / 8× / 16× —
  with 4× as the default for the "quality" mode and 2× for the
  "efficiency" mode.
- **Zero-latency shapers only** — no lookahead filters, no
  linear-phase FIRs on the audio path.  Group-delay from the
  oversampler polyphase filters is compensated implicitly (same on
  the up- and down-leg).
- **Multiple shaper flavours** shipped as a data-driven curve set:
  soft-clip, hard-clip, asymmetric diode, tube-triode, tape.
- **Tone stack modelled per pedal** — simple 3-band Baxandall or
  Fender/Marshall stack topology (state-variable filter), not IR-based.
- **Denormal-safe** — the shaper output can decay very slowly on
  release; FTZ/DAZ must be set for the whole callback.

## Prerequisites in `rt-fft-convolver`

- [ ] `DenormalGuard` (already available in `utils/denormals.rs`).
- [ ] `mixer.rs` gain-staging patterns — reuse the dB-to-linear
  utilities and dry/wet crossfade.

## Phase 0: DSP primitives (`rt-drive-core`)

Internal crate, reusable by other nonlinear-effect plugins.

- [ ] **Workspace layout**:
  - `crates/rt-drive-core` — resampling + shapers + tone stack.
  - `crates/rt-drive` — plugin.
- [ ] **Polyphase FIR resampler**:
  - Half-band FIR chained per octave up to 16×.
  - Kaiser-window design; stopband > 96 dB, transition < 0.05 × fs.
  - Fixed coefficient tables at compile time (`const fn` or
    generated at build).
- [ ] **Shaper curves** — [decision] on primary set:
  - Soft: `tanh`, `x / sqrt(1 + x²)`, cubic soft-clip.
  - Hard: symmetric and asymmetric clip.
  - Diode: exponential model (single/double stage).
  - Tube: Koren-style triode approximation.
  - Tape: Chowning / SATURN-style bias-shifted tanh.
- [ ] **Tone stack**:
  - Baxandall (2-band, low/high) as default.
  - Optional: state-variable topology mirroring Fender / Marshall
    tone stacks (bass, mid, treble).
- [ ] **DC-blocker** post-shaper (1-pole HPF ~ 20 Hz).
- [ ] **SIMD kernels**:
  - Vectorised shaper evaluation (batch of 4 / 8 samples).
  - Vectorised polyphase FIR (SIMD-across-taps).

## Phase 1: Engine integration

- [ ] **`DriveStage` struct**:
  - Owns: upsampler, shaper (fn pointer or enum dispatch), downsampler,
    DC-blocker.
  - `process_block(&mut self, in: &[f32], out: &mut [f32])` — RT-safe.
- [ ] **`ShaperBank`**:
  - Enum of shipped shapers with parameters (drive, asymmetry, bias).
  - Data-driven so new shapers can be added without touching the
    plugin code.
- [ ] **Latency reporting**:
  - Report 0 samples (FIR delays cancel between up- and down-leg).
  - Verify with a unit-impulse round-trip.

## Phase 2: `rt-drive` plugin

- [ ] **I/O**:
  - Mono in → mono out; stereo in → stereo out (independent per
    channel).
- [ ] **Preset controls**:
  - `Model` (dropdown, 8+ shipped shapers × tone stacks).
  - `Drive` (0–100 %).
  - `Level` (−24 … +24 dB, post-shaper).
  - `Tone` (Baxandall) — `Bass`, `Mid` (if 3-band), `Treble`.
  - `Mix` (0–100 %).
- [ ] **Quality controls**:
  - `Oversampling` (1× / 2× / 4× / 8× / 16×).
  - `Bias` (−1 … +1, injects DC pre-shaper for asymmetric flavour).
  - `Bypass`.
- [ ] **MIDI**:
  - CC on `Drive` and `Level` at minimum.
  - Program Change → preset index.
- [ ] **CPU meter** as read-only property.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Shaper transfer curve visualiser**: real-time x-y plot of
  the current shaper + drive setting.
- [ ] **Input / output VU meters**.
- [ ] **Oversampling badge** showing current factor and estimated CPU.
- [ ] **Preset A/B compare** with copy-to-clipboard.

## Phase 4: Testing

- [ ] **Unit tests** (`rt-drive-core`):
  - Resampler round-trip: `downsample(upsample(x)) ≈ x` within
    −80 dB at the highest oversampling factor.
  - Each shaper: monotonicity check (or documented non-monotonic
    intent).
  - DC-blocker attenuates a DC step by > 60 dB after 500 ms.
- [ ] **Aliasing floor test**:
  - Feed a 5 kHz sine at full scale through each shaper, measure the
    spectrum in a headroom-aware bin selection, assert
    aliases < −80 dBFS at 4×.
- [ ] **RT-safety audit**:
  - Allocator hook: zero allocations in `process()`.
  - Bench each shaper at each oversampling factor with `criterion`.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Preset pack**: ~20 pedal / amp-stage emulations covering
  clean-boost → high-gain distortion → tape saturation.  Ships
  in-plugin (not as external files).
- [ ] **Documentation**: manual under `docs/`, per-shaper transfer
  curve reference.

## Out of scope (future work)

- IR-based cab / speaker simulation — pair with `rt-ir-cab`.
- Nonlinear circuit modelling (Wave Digital Filters, state-space) —
  higher fidelity but ~5× the CPU; separate plugin if ever done.
- Feedback loops (fuzz-face-style bias interaction between drive and
  input) — deferred; requires nonlinear equation solving.
- Amp modelling with tonal memory (bias, sag) — see `rt-neural-amp`
  instead.
