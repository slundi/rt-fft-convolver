# Roadmap: `rt-adaptive` (Adaptive Filters — Feedback Suppression / Hum / Echo)

Adaptive-filter plugin covering feedback suppression (live PA
howlback), acoustic echo cancellation (AEC), and adaptive hum
removal.

**Reference implementations**: Waves X-FDBK, Sabine FBX, ART
Feedback Eliminator, Shure DFR, Behringer Feedback Destroyer,
Speex / WebRTC AEC.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **Very low CPU** — target well under 1 % per instance.  Adaptive
  filters are typically simple loops with SIMD-friendly kernels.
- **RT-safe** — no allocations, no locks on the audio thread.
  Coefficient updates happen in-place.
- **Modes are independent** — feedback suppression, hum removal,
  and AEC share the LMS / RLS core but have very different UX.
  [decision] whether to ship as one plugin with modes or three
  separate plugins.
- **Automatic and manual detection** — auto for typical live use,
  manual notch-programming as a fallback for stubborn feedback.
- **No latency in the direct signal path** — adaptive filters here
  are notches / cancellers, not lookahead processors.

## Prerequisites in `rt-fft-convolver`

- [ ] `DenormalGuard`.
- [ ] Optional reuse of `dsp/fft_handler.rs` for frequency-domain
  adaptive filters (block-LMS).

## Phase 0: Algorithm decisions

- [ ] **Adaptive-filter algorithm** — [decision] per mode:
  - Time-domain LMS (simplest, cheapest).
  - Normalised LMS (NLMS, more stable across gain).
  - Recursive Least Squares (RLS, faster convergence, higher CPU).
  - Frequency-domain block LMS (partitioned FFT — reuses
    `rt-fft-convolver`).
- [ ] **Feedback detector** — [decision]:
  - Continuous FFT peak-tracker + Q-narrow notch placement.
  - Autocorrelation-based howlback detection.
- [ ] **Hum removal** — [decision]:
  - Static comb-notch (fundamental + harmonics).
  - Adaptive: track the fundamental frequency drift (mains
    frequency can drift ± 0.5 Hz).
- [ ] **AEC reference input**:
  - Required if the plugin is used in an AEC scenario (broadcast,
    conferencing).  [decision] whether v1 supports AEC or only
    feedback/hum.

## Phase 1: `rt-adaptive-core` (engine crate)

- [ ] **Workspace layout**:
  - `crates/rt-adaptive-core` — engine.
  - `crates/rt-adaptive` — plugin.
- [ ] **LMS / NLMS / RLS** primitives:
  - `AdaptiveFilter<const N: usize>` — fixed-length tap vector.
  - `update(&mut self, input: f32, error: f32)` — per-sample update.
- [ ] **Feedback detector**:
  - Rolling FFT (small size, e.g. 512) for peak detection.
  - Peak-hold + threshold logic to distinguish howlback from tone.
- [ ] **Notch bank**:
  - Fixed capacity (e.g. 24 notches).
  - Biquad per notch, RT-safely allocated / deallocated (bool
    enable per slot).
- [ ] **Hum tracker**:
  - Fundamental estimation via zero-crossing rate or autocorrelation.
  - Comb-notch coefficient update.
- [ ] **Optional block-LMS** using `rt-fft-convolver` for AEC.

## Phase 2: `rt-adaptive` plugin

- [ ] **I/O**:
  - Stereo in → stereo out.
  - Sidechain in for AEC (reference / far-end signal), if scope.
- [ ] **Mode selector**:
  - `Feedback Suppressor` / `Hum Remover` / `Echo Canceller`.
- [ ] **Feedback controls**:
  - `Sensitivity` (0–100 %).
  - `Max Notches` (1–24).
  - `Notch Q` (0.5 – 50).
  - `Notch Depth` (−48 … 0 dB).
  - `Auto / Manual` (in Manual, user places notches).
  - `Freeze` — stop new notch allocation, keep existing.
  - `Reset` — clear all notches.
- [ ] **Hum controls**:
  - `Fundamental` (50 / 60 Hz + custom).
  - `Harmonics` (1–10).
  - `Adaptive` (bool) — track mains drift.
  - `Depth` (dB).
- [ ] **AEC controls** (if scope):
  - `Filter Length` (samples / ms).
  - `Convergence Speed` (μ for NLMS).
  - `Double-Talk Detector Threshold`.
- [ ] **Global controls**:
  - `Output Gain`.
  - `Bypass`.
- [ ] **MIDI**:
  - CC on `Freeze` and `Reset` (footswitch-friendly).
- [ ] **CPU meter** as read-only property.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Spectrum display** with active notches highlighted.
- [ ] **Notch inventory**: list of active notches with frequency,
  Q, depth, age.
- [ ] **Feedback-detection LED** — flashes when a new notch is
  placed.
- [ ] **Hum-fundamental tracker** display (if adaptive hum is on).

## Phase 4: Testing

- [ ] **Unit tests** (`rt-adaptive-core`):
  - LMS convergence: input = white noise + tone, filter should
    converge to null the tone within N samples (documented bound).
  - Notch bank: adding a 1 kHz notch at Q=10 attenuates a 1 kHz
    sine by > 20 dB.
  - Hum tracker: 60 Hz + harmonics input has fundamental identified
    within ± 0.5 Hz.
- [ ] **Integration tests**:
  - Loopback simulation for feedback: a synthetic delay-line
    "acoustic path" is fed back into the input, plugin should
    prevent divergence.
- [ ] **RT-safety audit**:
  - Zero allocations in `process()`, including on notch
    allocation.
  - Denormal guard active.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets**: live-PA feedback (sensitive), broadcast-hum
  (60 Hz), stage-hum (50 Hz), lecture-mic AEC (if AEC in scope).
- [ ] **Documentation**: manual with a live-sound engineer's
  quick-start guide.

## Out of scope (future work)

- Deep-learning-based noise suppression (RNNoise, Krisp) — that
  is closer to spectral processing; see
  [Roadmap-04-spectral-processing.md].
- Blind source separation (BSS) — different problem class.
- Full digital signal processor (DSP) automixer — separate plugin
  domain.
