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

- [x] `DenormalGuard` (available in `utils/denormals.rs`) — RAII FTZ/DAZ
  guard, RT-safe (one MXCSR/FPCR read+write on entry/exit, no alloc/lock).
  Wrap the whole render callback with it.  A software `flush_to_zero`
  fallback is also available.
- [x] `mixer.rs` dry/wet crossfade — equal-power (cos/sin) blend with
  `ir_gain` folded into `wet_gain` so `process()` is two mul/sample.
  Reusable directly.
- [ ] **dB↔linear helpers** — NOT present in the crate today (the mixer
  works purely in linear gains; there is no `db_to_linear`/`linear_to_db`
  anywhere in `src/`).  Must be written for `Level` (−24…+24 dB) and
  `Bias`.  [decision] where they live — recommended: a shared
  `rt-fft-convolver::utils::gain` module so `rt-neural-amp` and the other
  roadmapped plugins reuse one implementation, rather than duplicating in
  `rt-drive-core`.

## Phase 0: DSP primitives (`rt-drive-core`)

Internal crate, reusable by other nonlinear-effect plugins.

- [ ] **Workspace layout**:
  - `rt-drive-core` — resampling + shapers + tone stack.
  - `rt-drive` — plugin.
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

## Pedal model format (Phase 0/1)

A "pedal" is **not** a plugin, binary, or DSP module — it is a
**data-driven preset descriptor** that parameterises the one shared
`DriveStage` engine.  A pedal carries *no code*, only coefficients and
enum selectors; this is what makes "add a new pedal without touching the
plugin code" (Phase 1, `ShaperBank`) actually true.

A pedal is a point in the *(input gain × shaper × bias × tone stack ×
output gain)* space:

```rust
/// One "pedal" = coefficients + enum selectors over the shared engine.
/// Purely data — no DSP lives here.
pub struct PedalModel {
    pub name: &'static str,          // display name (see Sourcing & legal)
    pub input_gain_db: f32,          // pre-shaper drive trim
    pub shaper: Shaper,              // SoftTanh | AsymDiode { .. } | Tube { .. } | …
    pub bias: f32,                   // DC injected pre-shaper (asymmetry)
    pub tone: ToneStack,             // Baxandall { .. } | Svf3Band { bass, mid, treble }
    pub output_gain_db: f32,         // post-shaper make-up
    pub default_oversampling: u8,    // 2 / 4 / 8 / 16
}
```

- [ ] **Authoring format**: human-editable `RON` or `TOML` curve set
  under `pedals/`, deserialised via `serde`.
- [ ] **Compile-time bake** (default, matches Phase 5 "ships in-plugin,
  not external files"): a `build.rs` reads `pedals/*.toml` and codegens a
  `static [PedalModel]` table.  Zero runtime cost, allocation-free on the
  render thread.
- [ ] **Runtime-loadable** (deferred): same struct via `serde`, loaded
  off the audio thread into a double-buffered slot for user-authored
  pedals.
- [ ] **Note**: `input_gain_db` / `output_gain_db` depend on the dB↔linear
  helper listed in Prerequisites — resolve that decision first.

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
  in-plugin (not as external files).  Each preset is a `PedalModel`
  (see "Pedal model format").
- [ ] **Documentation**: manual under `docs/`, per-shaper transfer
  curve reference.

### Sourcing & legal (preset pack)

We model *transfer functions + tone stacks*, not firmware, so the source
material is circuit-level:

- **Circuit analysis / schematics** (public, hobbyist):
  - **electrosmash.com** — best single resource; full mathematical
    analysis of the Boss DS-1, Ibanez TS808/TS9, ProCo Rat, Big Muff Pi,
    MXR Distortion+, including clipping-stage transfer functions and
    tone-stack RC networks.  Nearly direct → `Shaper` + `ToneStack` params.
  - **freestompboxes.org**, **diystompboxes.com** — schematics + measured
    behaviour.
  - **Klon Centaur** — fully reverse-engineered; schematic + germanium
    diode clipping stage public.
- **Deriving the curve**:
  - **SPICE** (ngspice / LTspice) — build the schematic, DC sweep →
    static transfer curve, AC sweep → tone-stack response; fit to a
    `Shaper` enum (e.g. asymmetric diode → exponential model).
  - **Direct measurement** — slow full-scale sine/ramp through a real
    pedal via an interface; good for validation, messier than SPICE for
    primary sourcing (nonlinearity + the pedal's own filtering).
  - **Academic** — Yeh/Abel/Smith (CCRMA) diode-clipper & tube papers,
    DAFx proceedings — parametric models directly.
- **⚠️ Trademark**: "Tube Screamer", "Klon", "RAT", "DS-1", "Big Muff"
  are trademarks.  The circuits/topologies are fair to model, but ship
  presets under descriptive / punning names (industry norm: TS → *Green
  808*, Klon → *Gold Centaur*, Rat → *Rodent*, DS-1 → *Orange Dist*).
  Keep any "modelled after…" reference in the manual, not on the preset
  name.

## Out of scope (future work)

- IR-based cab / speaker simulation — pair with `rt-ir-cab`.
- Nonlinear circuit modelling (Wave Digital Filters, state-space) —
  higher fidelity but ~5× the CPU; separate plugin if ever done.
- Feedback loops (fuzz-face-style bias interaction between drive and
  input) — deferred; requires nonlinear equation solving.
- Amp modelling with tonal memory (bias, sag) — see `rt-neural-amp`
  instead.
