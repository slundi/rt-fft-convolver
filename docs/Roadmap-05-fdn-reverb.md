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

## Prerequisites in `rt-fft-convolver`

- [ ] `DenormalGuard` — feedback loops with tails > 5 s reach the
  denormal regime quickly.
- [ ] Parameter-smoothing utility — [decision] whether to extract
  from `mixer.rs` or keep local.

## Phase 0: Algorithm decisions

- [ ] **FDN size** — [decision]:
  - 4×4 (cheap, "small room" character).
  - 8×8 (standard, best diffusion / CPU tradeoff).
  - 16×16 (dense, "lush" tail, higher CPU).
- [ ] **Feedback matrix** — [decision]:
  - Householder (fast, real).
  - Hadamard (fast, requires size = power of two).
  - Random orthogonal (higher diffusion, slower).
- [ ] **Delay-line lengths**: mutually prime around a target mean
  length; [decision] on exact values (typically 700 – 4500 samples
  @ 48 kHz).
- [ ] **Damping**: 1-pole low-pass inside each feedback path.
- [ ] **Modulation source**: [decision] per-delay LFOs vs. a shared
  random-walk driver.
- [ ] **Early reflections**:
  - [decision] whether to include a short pre-FDN tapped-delay
    section for early reflections, or leave that to the FDN itself.

## Phase 1: `rt-fdn-core` (engine crate)

- [ ] **Workspace layout**:
  - `rt-fdn-core` — reverb engine.
  - `rt-fdn-reverb` — plugin.
- [ ] **Delay-line primitives**:
  - `ModulatedDelay { buf: Vec<f32>, mod_lfo, mod_depth }`.
  - All-pass diffuser (Schroeder-style) chain.
  - Cache-aligned buffers.
- [ ] **FDN engine**:
  - Compile-time-parameterised size (`FDN<N: usize>`).
  - Feedback matrix as a `const fn` or generated at construction.
  - `process_sample_stereo(&mut self, in_l: f32, in_r: f32) ->
    (f32, f32)`.
- [ ] **Tone controls inside the loop**:
  - Low-cut and high-cut per feedback path.
- [ ] **Preset bank**:
  - 8 preset structs (delay lengths, matrix, damping, mod depth).

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
