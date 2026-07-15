# Roadmap: `rt-fdn-reverb` (Algorithmic FDN Reverb)

Algorithmic reverb plugin based on a Feedback Delay Network (FDN).
Complements `rt-ir-reverb` (convolution-based, from [Roadmap-lv2.md])
by providing modulated, tweakable spaces that a static IR cannot
produce cheaply.

**Reference implementations**: Valhalla VintageVerb / Room / Delay,
Strymon BigSky / El Capistan, Eventide Blackhole, Lexicon PCM
series, TC Electronic Hall of Fame, FabFilter Pro-R.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **True stereo** — 4-channel FDN with cross-coupled matrix mixing.
- **Modulated delays** for chorus-y motion in the tail (essential
  for the "lush" character of classic algorithmic reverbs).
- **Zero-latency dry** — the wet signal accumulates over time so
  there is no pre-delay in the dry path.
- **RT-safe modulation** — LFO / random-walk phases updated
  per-sample, delay lines sized at construction.
- **Parameter smoothing** on all continuous controls to avoid
  zipper noise from automation.
- **8 preset slots** — same UX as the other plugins in this suite.

## Prerequisites in `rt-fft-convolver` ✅

- `DenormalGuard` — **satisfied**. Reuse
  `src/utils/denormals.rs` verbatim: RAII FTZ+DAZ guard (x86/x86_64
  MXCSR, AArch64 FPCR, no-op elsewhere), restores exact prior FPU
  state on drop, RT-safe (one register read+write), fully tested.
  Exactly the >5 s feedback-tail case it was designed for — wrap the
  whole render callback.
- Parameter-smoothing utility — **decision: build a new one,
  local to `rt-fdn-core`**. There is nothing to extract: `mixer.rs`
  computes equal-power gains *set-then-apply* (no per-sample ramp)
  and `crossfade.rs` blends per-sample but only for IR hot-swap.
  Neither is a reusable per-sample smoother. A small one-pole /
  linear ramp (~40 lines) lives in the new engine crate, matching
  the existing pattern where core crates own their DSP (the `nih`
  plugin layer is feature-gated). Promote to shared `src/utils/`
  only if a second plugin needs it. This is a **Phase 1 build item**,
  not a Phase 0 blocker.

## Phase 0: Algorithm decisions ✅

### FDN size — **decision: 8×8**

- The industry-standard sweet spot (Lexicon / Valhalla / FabFilter
  Pro-R territory): dense enough for convincing hall / plate tails,
  ~half the per-sample cost of 16×16, and a **power of two** — which
  is what unlocks the fast Hadamard matrix below.
- *Rejected*: **4×4** (sparse, flutters on long decays — kept only as
  a possible "small room" variant later) and **16×16** (Blackhole-
  style density, but the extra 8 delay lines + matrix cost isn't
  justified for a guitar-oriented 1.0).

### Feedback matrix — **decision: Hadamard (fast Walsh–Hadamard)**

- With `N = 8` (power of two) the FWHT is `O(N log N)` using **only
  adds and subtracts** — no per-sample multiplies — and is exactly
  orthonormal, so it is maximally lossless and maximally diffusive.
- Implemented as an in-place butterfly over the 8-element state
  vector, then scaled by `1/√N` to preserve energy.
- *Rejected*: **Householder** (real, any size, but slightly less
  diffuse — retained as the fallback if a future size becomes
  non-power-of-two) and **random orthogonal** (`O(N²)` dense matmul
  per sample, more CPU for marginal audible gain).

### Delay-line lengths — **decision: coprime prime set**

- Eight **mutually prime** lengths around a target mean, scaled at
  construction by `Size` and the actual sample rate:
  `{743, 977, 1123, 1319, 1493, 1657, 1811, 1997}` samples @ 48 kHz
  (≈ 15 – 42 ms).
- Coprime lengths maximise modal density and push the delay-echo
  pattern's common period out to their product, avoiding audible
  flutter and metallic ringing.
- Recompute (round to nearest prime / integer) when `Size` or `fs`
  changes; sizes are in samples so ms scales with `fs`.

### Damping — **decision: per-path 1-pole low-pass (given)**

- One 1-pole low-pass inside **each** of the 8 feedback paths, giving
  frequency-dependent decay (highs die faster than lows — the natural
  air-absorption behaviour). Cutoff driven by the `Damping` control.
- Not contentious; this is the standard FDN damping topology.

### Modulation source — **decision: per-delay LFOs, decorrelated phases**

- One sine LFO per delay line, each seeded with a **decorrelated
  starting phase** (e.g. evenly spread over 2π), producing the classic
  "lush" chorus-y tail motion of Strymon / Valhalla algorithmic verbs.
- Cost is `N` phase increments + `N` fractional delay reads per sample
  — cheap and RT-safe (delay buffers sized at construction, fractional
  read via linear/all-pass interpolation).
- *Rejected*: **shared random-walk driver** (cheaper and more organic,
  but flatter motion and can sound diffuse-mushy on sustained chords).

### Early reflections — **decision: FDN-only for 1.0**

- The 8×8 Hadamard diffusion already produces credible early energy;
  a dedicated pre-FDN tapped-delay ER section would roughly double the
  delay-line accounting for modest audible gain.
- Deferred, not cancelled: a short ER tap bank is a natural refinement
  for the "room" / "plate" spaces in a later revision.

## Phase 1: `rt-fdn-core` (engine crate) ✅

- **Workspace layout**:
  - `rt-fdn-core` — reverb engine (this phase).
  - `rt-fdn-reverb` — plugin (deferred to Phase 2).
  - Framework-agnostic; depends only on the root crate for
    `flush_to_zero` / `DenormalGuard` (mirrors `rt-spectral-core`).
- **Parameter-smoothing utility** (the prerequisite): one-pole
  exponential [`smoother.rs`] with a stall-snap so `f32` glides
  terminate cleanly.  Used by the engine for size and mod-depth.
- **Delay-line primitives**:
  - `ModulatedDelay` [`delay.rs`] — fractional, linear-interpolated,
    per-line LFO ([`lfo.rs`]), power-of-two circular buffer sized at
    construction (the "cache-aligned buffers" item), explicit
    read()/write() split for the matrix feedback step.
  - Schroeder all-pass diffuser + chain [`allpass.rs`].
- **FDN engine** [`fdn.rs`]:
  - Compile-time size `Fdn<const N: usize>` (presets use `N = 8`).
  - Lossless fast Walsh–Hadamard mixing matrix [`matrix.rs`].
  - Per-branch feedback gains derived from T60 (uniform decay).
  - `process_sample_stereo(&mut self, f32, f32) -> (f32, f32)` with
    decorrelated stereo tap / injection sign patterns.
- **Tone controls inside the loop** [`filter.rs`]:
  - Per-branch one-pole high-cut (damping) + low-cut (high-pass).
- **Preset bank** [`preset.rs`]:
  - 8 `Space` presets (room, hall, plate, spring, chamber, cave,
    shimmer, non-linear) over the Phase 0 coprime delay set.
- **Tests**: 48 passing (per-module unit + engine integration:
  stability over 60 s at max decay, decay ordering, stereo
  decorrelation, silent-in/silent-out, reset, all presets finite,
  live size sweep, 96 kHz).  `cargo clippy -D warnings` clean.

## Phase 2: `rt-fdn-reverb` plugin

- [ ] **I/O**:
  - Stereo in → stereo out.
- [ ] **Preset controls** (× 8):
  - `Space N` (dropdown: room, hall, plate, spring, chamber, cave,
    shimmer, non-linear).
  - `Preset Enabled N`.
- [ ] **Global controls**:
  - `Current Preset` (int 0–7).
  - `Crossfade Time` (5–500 ms — [note] tails need long fades).
  - `Pre-Delay` (0–500 ms).
  - `Decay` (0.1 – 20 s, T60).
  - `Size` (0–100 %, scales all delay lengths).
  - `Damping` (0–100 %, low-pass strength).
  - `Diffusion` (0–100 %, all-pass feedback amount).
  - `Modulation Depth` (0–100 %).
  - `Modulation Rate` (0.1 – 5 Hz).
  - `Low Cut` (20–500 Hz, wet only).
  - `High Cut` (2–20 kHz, wet only).
  - `Stereo Width` (0–200 %).
  - `Dry` (−∞ … +6 dB).
  - `Wet` (−∞ … +6 dB).
  - `Freeze` (bool — feedback → 1.0, no new input into FDN).
  - `Bypass`.
- [ ] **MIDI**:
  - PC → preset.
  - CC on `Decay`, `Size`, `Freeze` (great for live use).
- [ ] **CPU meter** as read-only property.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Preset rack**: 8 pads with space name.
- [ ] **Impulse-response visualiser**: run a virtual impulse through
  the current settings and display the resulting decay envelope.
  Updated on parameter change.
- [ ] **Freeze indicator** with an obvious LED.
- [ ] **MIDI activity indicator**.

## Phase 4: Testing

- [ ] **Unit tests** (`rt-fdn-core`):
  - Stability: no divergence over 60 s at maximum decay.
  - T60 measurement matches the requested decay within ±10 %.
  - Freeze mode: RMS is constant to within 0.1 dB over 10 s after
    engaging.
  - Modulation reset does not click.
- [ ] **RT-safety audit**:
  - Zero allocations in `process()`.
  - Denormal guard active in every callback.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets** (×8 default): small room, chamber, medium hall,
  large hall, plate, spring, shimmer (with pitch-up feedback tap),
  non-linear / reverse.
- [ ] **Documentation**: manual with a T60 vs. size chart per
  preset.

## Out of scope (future work)

- Convolution-based reverb — that is `rt-ir-reverb`
  (see [Roadmap-lv2.md]).
- Hybrid FDN + convolution (early reflections from IR, late tail
  from FDN) — deferred; interesting but doubles the CPU accounting.
- Diffuse-field / Ambisonics rendering — separate plugin domain.
- Sidechained ducking of the wet signal — [decision] whether to
  add later as a global feature.
