# Roadmap: `rt-multiband` (Multi-Band Compressor / Limiter / Expander)

Multi-band dynamics processor with configurable crossover and
per-band compression, limiting, or expansion.

**Reference implementations**: FabFilter Pro-MB, Waves C4 / C6,
iZotope Ozone Dynamics, TDR Nova, Sonnox Oxford Dynamics.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **Band count** — [decision]:
  - Fixed at 4 (simplest UI).
  - Fixed at 6.
  - User-configurable 2–6.
- **Crossover topology** — [decision]:
  - Linkwitz-Riley 4th-order (24 dB/oct, phase-coherent when summed
    within a band).
  - Linear-phase FIR (higher CPU + latency, ideal for mastering).
  - Both, user-selectable.
- **Per-band mode** — each band independently compressor, expander,
  gate, or bypass.
- **RT-safe** — all state per-band and per-channel, allocation-free
  on the audio thread.
- **Sidechain input** — per-band sidechain filter, plus external
  sidechain — [decision] whether external sidechain is v1 or v2.
- **Zero-latency mode** with the IIR crossover; higher-latency
  linear-phase mode reported accurately to the host.

## Prerequisites in `rt-fft-convolver`

- [ ] `DenormalGuard`.
- [ ] For the linear-phase crossover option: reuse
  `rt-fft-convolver` as an FIR convolution engine (this is the
  primary architectural link to this library).
- [ ] Parameter-smoothing utility — [decision] whether to extract
  from `mixer.rs` or keep local.

## Phase 0: DSP decisions

- [ ] **Detector topology** — [decision]:
  - Peak.
  - RMS.
  - Peak / RMS blend.
  - Program-dependent (auto).
- [ ] **Knee** — [decision]:
  - Hard knee only.
  - Soft knee (user-configurable width).
  - Both.
- [ ] **Envelope follower** — [decision]:
  - Analog-modelled (attack/release with capacitor charge curve).
  - Linear.
  - RC + look-ahead.
- [ ] **Look-ahead** — [decision]:
  - Off (zero-latency).
  - Fixed (e.g. 3 ms).
  - User-configurable.
- [ ] **Stereo linking** — [decision]:
  - Linked (max detector across L/R).
  - Independent.
  - M/S with per-band selectable.

## Phase 1: `rt-multiband-core` (engine crate)

- [ ] **Workspace layout**:
  - `rt-multiband-core` — engine.
  - `rt-multiband` — plugin.
- [ ] **Crossover primitives**:
  - Linkwitz-Riley 4th-order biquad pairs.
  - Optional linear-phase FIR via `rt-fft-convolver`.
- [ ] **Per-band processor**:
  - `BandProcessor { detector, gain_computer, envelope_follower,
    makeup_gain }`.
  - `process_sample(&mut self, in: f32) -> f32` — RT-safe.
- [ ] **Sidechain routing**:
  - Internal (band content).
  - External (aux input) — if in scope.
  - Sidechain filter (HPF + shelf).
- [ ] **Band-sum reconstruction**:
  - Linkwitz-Riley sums to unity when bands are bypassed
    (verified with a bypass test).

## Phase 2: `rt-multiband` plugin

- [ ] **I/O**:
  - Stereo in → stereo out.
  - Sidechain in (if scope).
- [ ] **Global controls**:
  - `Band Count` (2–6, if configurable).
  - `Crossover Type` (IIR / linear-phase).
  - `Input Gain`.
  - `Output Gain`.
  - `Mix` (parallel / dry-wet).
  - `Bypass`.
- [ ] **Per-band controls** (× N):
  - `Crossover Frequency N` (Hz) — sets the upper edge of band N.
  - `Mode` (compressor / expander / gate / bypass).
  - `Threshold`.
  - `Ratio`.
  - `Attack`.
  - `Release`.
  - `Knee`.
  - `Makeup Gain`.
  - `Solo` (audition only this band).
  - `Mute` (silence this band).
- [ ] **Sidechain controls** (per band):
  - `SC Source` (self / external).
  - `SC HPF` (Hz).
  - `SC Listen` (solo the sidechain signal).
- [ ] **MIDI**:
  - CC on `Threshold` and `Makeup` per band (bank-switching may be
    needed for LV2 CC limits).
- [ ] **Latency reporting** for the linear-phase mode.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Spectrum display** with draggable crossover lines.
- [ ] **Per-band gain-reduction meters**.
- [ ] **Per-band transfer-curve display**: threshold, ratio, knee.
- [ ] **Solo / Mute buttons per band**.
- [ ] **Global meters**: input, output.

## Phase 4: Testing

- [ ] **Unit tests** (`rt-multiband-core`):
  - Crossover reconstruction: bypassed bands sum to bit-identical
    (IIR) or delayed-identical (linear-phase).
  - Compressor: 20 dB above threshold at ratio 2:1 with hard knee
    produces −10 dB gain reduction (steady state).
  - Expander: below threshold at ratio 2:1 produces the correct
    downward expansion.
  - Attack / release: gain reduction reaches target within ± 10 %
    of the specified time constant.
- [ ] **Sidechain tests**:
  - External sidechain path routes correctly (if in scope).
- [ ] **RT-safety audit**:
  - Zero allocations in `process()`.
  - Denormal guard active.
  - Reported latency matches actual delay in linear-phase mode.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets**: mastering-glue, drum-bus, vocal-de-esser (single
  narrow band at 5–8 kHz in compress mode), broadcast-loudness,
  bass-tightener (low-band only).
- [ ] **Documentation**: manual with per-preset intent and per-band
  parameter reference.

## Out of scope (future work)

- Multi-band saturation / drive — combine `rt-multiband` with
  `rt-drive` at the band level, or a separate dedicated plugin.
- Dynamic EQ (parametric bands with dynamic gain) — related but a
  different plugin class.
- Adaptive band splitting (auto-choose crossover points) — deferred.
- Multi-band brickwall limiting — see
  [Roadmap-07-true-peak-limiter.md] and consider a
  chained-plugin workflow instead.
