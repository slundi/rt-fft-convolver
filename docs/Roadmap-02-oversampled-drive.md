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

## Prerequisites in `rt-fft-convolver` ✅

- `DenormalGuard` (available in `utils/denormals.rs`) — RAII FTZ/DAZ
  guard, RT-safe (one MXCSR/FPCR read+write on entry/exit, no alloc/lock).
  Wrap the whole render callback with it.  A software `flush_to_zero`
  fallback is also available.
- `mixer.rs` dry/wet crossfade — equal-power (cos/sin) blend with
  `ir_gain` folded into `wet_gain` so `process()` is two mul/sample.
  Reusable directly.
- **dB↔linear helpers** — NOT present in the crate today (the mixer
  works purely in linear gains; there is no `db_to_linear`/`linear_to_db`
  anywhere in `src/`).  Must be written for `Level` (−24…+24 dB) and
  `Bias`.  [decision] where they live — recommended: a shared
  `rt-fft-convolver::utils::gain` module so `rt-neural-amp` and the other
  roadmapped plugins reuse one implementation, rather than duplicating in
  `rt-drive-core`.

## Phase 0: DSP primitives (`rt-drive-core`) ✅

Internal crate, reusable by other nonlinear-effect plugins.

- **Workspace layout**:
  - `rt-drive-core` — resampling + shapers + tone stack + DC blocker +
    SIMD kernels.  Added to the workspace; framework-agnostic (no
    plugin-format dep), mirrors the `rt-neural-core` layout.
  - `rt-drive` — plugin.  **Deferred to Phase 2** (needs `nih-plug`; an
    empty shell would not build).
- **Polyphase FIR resampler** (`resampler.rs`):
  - Half-band FIR chained per octave up to 16× (`Oversampler` cascade of
    2× `Upsampler2x` / `Downsampler2x` stages).
  - Kaiser-window design; default stopband 96 dB, transition 0.05 × fs.
  - **Deviation**: coefficients are computed once in `Oversampler::new`
    (off the audio thread), *not* `const fn` — stable Rust can't evaluate
    the Bessel-I₀ Kaiser window in const context.  Tables are then fixed
    for the object's lifetime; render path is allocation-free.  A
    `build.rs` can emit const tables later if wanted.
- **Shaper curves** — [decision] resolved, full set shipped as the
  `Shaper` enum (`shaper.rs`), all monotonic + origin-passing + bounded:
  - Soft: `Tanh`, `Reciprocal` (`x/√(1+x²)`), `CubicSoft`.
  - Hard: `HardClip { threshold, asymmetry }` (symmetric + asymmetric).
  - Diode: `Diode { drive }` (exponential clipper).
  - Tube: `Tube` — simplified asymmetric triode-style curve (*not* a full
    Koren plate-current model; that stays circuit-modelling / out of scope).
  - Tape: `Tape { bias }` (bias-shifted tanh).
- **Tone stack** (`tone.rs`), RBJ-cookbook biquads:
  - `Baxandall` (2-band low/high shelf) as default.
  - `ThreeBand` (low-shelf + peaking mid + high-shelf) for a
    Fender/Marshall-style bass/mid/treble control.
- **DC-blocker** post-shaper — `DcBlocker`, 1-pole HPF (default 20 Hz).
- **SIMD kernels** (`kernels.rs`, `pulp` — same choice as
  `rt-neural-core`):
  - Vectorised soft-clip block (`soft_clip_block`).
  - Vectorised polyphase FIR dot (`fir_dot`, SIMD-across-taps), backs the
    resampler.
  - Note: transcendental shapers (`tanh`/exp families) stay scalar for now
    (LLVM auto-vectorises the rational ones); a SIMD `tanh` approximation
    is a follow-up if profiling warrants it.

## Phase 1: Engine integration

- [x] **`DriveStage` struct** (`stage.rs`):
  - Owns `Oversampler`, `Shaper` (enum dispatch), `DcBlocker`, `ToneStack`,
    plus pre-shaper gain+bias and post-shaper make-up gain.
  - `process_block(&mut self, input, output)` — RT-safe (no alloc/lock/
    panic; scratch pre-sized to `max_block × factor`).  Chain:
    `↑ → ×gain → +bias → shaper → ↓ → ×level → DC-block → tone`, with the
    nonlinearity inside the oversampled region.
  - Built from a `PedalModel` via `from_model` / `with_design`; gains use
    the shared `rt_fft_convolver::db_to_linear`.
- [x] **`ShaperBank`** — realised as the data-driven `PedalModel` + the
  `const PEDALS` bank (`pedal.rs`).  The `Shaper` enum already carries the
  per-curve params (drive, asymmetry, bias); a pedal is pure data over the
  shared engine, so new pedals need no engine change.
- [x] **Latency reporting** (`DriveStage::latency_samples`):
  - **Correction to the original note**: the linear-phase half-band cascade
    is **not** 0 samples.  The up- and down-leg group delays *add* (they do
    not cancel — "same on both legs" only means the filter is symmetric),
    giving a non-zero integer latency reported for host compensation.
    factor = 1 reports 0.  IIR DC-block/tone add phase, not reportable
    latency.
  - Measured once at construction via a unit-impulse round trip; a unit
    test asserts the reported value matches the impulse peak (±1).

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

- [x] **`PedalModel` struct + shipped bank** implemented in `pedal.rs`
  (5 presets: Clean Boost, Green Overdrive, Rodent Distortion, Tube
  Warmth, Tape Saturator).  Tone carried as a data-only `ToneConfig`
  (Bypass / Baxandall / ThreeBand) built into a live `ToneStack` at
  construction.
- [x] **Authoring format** — [decision] **TOML via `toml-span`** (not
  `serde`/RON).  Pedals live in `pedals/*.toml`; `toml-span`'s byte-offset
  spans give IDE-style `PedalLoadError`s on hand-edited files.  Manual
  `ValueInner`-walking parser with range validation; `[shaper]`/`[tone]`
  tables tagged by a `kind` string.
- [x] **Ships in-plugin**: the shipped `pedals/*.toml` are
  `include_str!`-embedded and parsed once at start-up by `shipped_pedals()`
  — no external files at run time, allocation-free render path.  (Chosen
  over a `build.rs` const bake so TOML stays the single source of truth
  and the same parser serves user pedals.)
- [ ] **Runtime-loadable user pedals**: `PedalModel::from_toml_str` already
  parses arbitrary documents; the remaining piece — discovering a user
  pedal directory (XDG) and a double-buffered hot-swap slot — is
  **deferred to Phase 2** (plugin).
- [x] **Note**: `input_gain_db` / `output_gain_db` use the shared
  `rt_fft_convolver::db_to_linear` helper (Prerequisites decision resolved
  — it lives in the root crate; `rt-drive-core` now depends on it).

## Phase 2: `rt-drive` plugin

Implemented as the `rt-drive` crate: framework-agnostic `params` + `engine`
(unit-tested, offline), with the `nih-plug` wrapper behind the opt-in `nih`
feature (`cargo build -p rt-drive --features nih`) — same convention as
`rt-neural-amp`.  The engine uses the lock-free `rt-ir-plugin-core` handoff:
the control thread rebuilds per-channel `DriveStage`s on structural changes
(model / oversampling), the audio thread applies continuous controls in place.

- [x] **I/O**: mono → mono and stereo → stereo, one independent `DriveStage`
  per channel (two `AudioIOLayout`s).
- [x] **Preset controls**:
  - `Model` — `IntParam` over the runtime TOML bank, index → pedal name via
    a value-to-string formatter (runtime-sized, so not a compile-time enum).
  - `Drive` (0–100 %) → ±24 dB input-gain offset on the pedal (50 % = as
    authored).
  - `Level` (−24 … +24 dB) → offset on the pedal's output gain.
  - `Tone` — `Bass` / `Mid` / `Treble` (−12…+12 dB), a plugin-owned 3-band
    stack (independent of the pedal's authored voicing; 0 dB = flat).
  - `Mix` (0–100 %) — dry/wet.
- [x] **Quality controls**:
  - `Oversampling` (`Preset` / 1× / 2× / 4× / 8× / 16×; `Preset` = pedal
    default).
  - `Bias` (−1 … +1) → offset on the pedal's pre-shaper DC bias.
  - `Bypass`.
- [x] **MIDI**:
  - Program Change → `Model` index (via `program_to_slot`).
  - CC on `Drive` / `Level`: all continuous controls are automatable
    `nih-plug` params, bound to any CC via host-side MIDI learn (the
    idiomatic path — `nih-plug` has no in-`process` setter for continuous
    params).
- [x] **CPU meter**: last-block render-time / wall-clock-budget, packed into
  an `AtomicU32` for the (Phase-3) editor to read.
- [ ] **Deferred to Phase 3/leftover**: `nih-plug-egui` editor; user-pedal
  directory discovery + hot-swap; `DenormalGuard` is already applied per
  callback.

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
