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
  and AEC share the NLMS / notch core but have very different UX.
  **Decision (Phase 0): one plugin, three modes** over a shared
  `rt-adaptive-core` crate — see Phase 0 below.
- **Automatic and manual detection** — auto for typical live use,
  manual notch-programming as a fallback for stubborn feedback.
- **No latency in the direct signal path** — adaptive filters here
  are notches / cancellers, not lookahead processors.

## Prerequisites in `rt-fft-convolver` ✅

- `DenormalGuard` — present and tested
  ([src/utils/denormals.rs](../src/utils/denormals.rs)); RAII FTZ/DAZ
  guard, re-exported at the crate root. Wraps the near-silent NLMS /
  notch tails so a quiet room never drives coefficients into denormal
  arithmetic.
- `FftHandler` — present and tested
  ([src/dsp/fft_handler.rs](../src/dsp/fft_handler.rs)); allocation-free
  real→complex / complex→real FFT, reused for the feedback detector's
  rolling analysis FFT and (later) the optional frequency-domain
  block-LMS AEC path.
- `Ring` and dB helpers (`db_to_linear` / `linear_to_db`) — present and
  exported; used for the detector's sliding analysis window and for the
  notch-depth / sensitivity controls.
- **Bonus prerequisites met in sibling `*-core` crates** — the two
  hardest DSP pieces already exist, tested and RT-safe, so this plugin
  reuses rather than re-implements (see Phase 0):
  - [`rt_spectral_core::Biquad`](../rt-spectral-core/src/biquad.rs) — an
    RBJ notch in transposed direct-form II with a state-preserving
    `set_notch`; its own unit test proves > 20 dB rejection at centre,
    exactly this roadmap's Phase 4 bar.
  - [`rt_spectral_core::HumNotchBank`](../rt-spectral-core/src/hum.rs) —
    a fundamental-plus-harmonics comb of those notches.
  - [`rt_pitch_core::PitchTracker`](../rt-pitch-core/src/pitch.rs) —
    normalised-autocorrelation fundamental estimation with a confidence
    score, the natural drift tracker for adaptive hum.
  - Core→core path dependencies are already the workspace norm
    (`rt-vox-core` depends on both `rt-fft-convolver` and
    `rt-pitch-core`), so `rt-adaptive-core` may depend on
    `rt-fft-convolver`, `rt-pitch-core`, and `rt-spectral-core` without
    any plugin-framework leak.

## Phase 0: Algorithm decisions ✅

The dominant Phase 0 finding mirrors Roadmap-08: the two hardest pieces
of DSP this plugin needs — a state-preserving RBJ **notch** and an
autocorrelation **fundamental tracker** — already ship, tested and
RT-safe, in sibling `*-core` crates. Every decision below therefore
favours *reuse* over a second parallel implementation, keeping the
plugin inside its "well under 1 % CPU, allocation-free, no latency in the
direct path" design principles. **v1 ships all three modes**, with a
time-domain **NLMS** echo canceller (the frequency-domain block-LMS path
is deferred to a later version).

- **Plugin packaging** — **decision: one plugin, three modes**
  (`Feedback Suppressor` / `Hum Remover` / `Echo Canceller`) over a
  shared `rt-adaptive-core` crate, matching every other `*-core` +
  plugin pair in the workspace.
  - *Options considered.* (a) One plugin with a mode selector — the
    modes share the notch primitive, the rolling analysis FFT, and the
    global `Output Gain` / `Bypass` / MIDI `Freeze`+`Reset` controls
    Phase 2 already lists. (b) Three separate plugins over a shared core
    — triples bundling, CI, installers, and preset banks for no user
    benefit. Waves X-FDBK, Sabine FBX, and Shure DFR are each single
    boxes with switchable behaviour, i.e. the (a) model.

- **Adaptive-filter algorithm** — **decision: per mode, the cheapest
  approach that meets the mode's job** — the modes do *not* all run a
  per-sample adaptive filter:
  - *Feedback suppression* — **detection-driven notch placement**, no
    per-sample adaptive filter. The rolling FFT detector (below) locates
    howling partials and programmes `rt_spectral_core::Biquad` notches;
    the direct path is a static notch cascade, so it adds zero latency
    and near-zero CPU between detections.
  - *Hum removal* — **adaptive comb of biquad notches** (below); again
    no LMS, just periodic coefficient retuning from a tracked
    fundamental.
  - *AEC* — **normalised LMS (NLMS)**, time-domain, a fixed-length tap
    vector `AdaptiveFilter<const N: usize>` with a per-sample
    `update(input, error)`.
  - *Options considered (AEC core).* Plain **LMS** is the simplest but
    its step size must be retuned whenever the far-end gain changes;
    **NLMS** normalises by input energy and stays stable across levels
    for one extra divide per block — the right default for a live tool.
    **RLS** converges fastest but is `O(N²)` per sample, which blows the
    < 1 % CPU budget for any useful tap count. **Frequency-domain block
    LMS** (partitioned FFT, reusing `rt-fft-convolver`) is the best fit
    for long echo tails but adds block latency to the path — **deferred**
    to v1.1 as an optional long-filter mode.
  - *Example.* At 48 kHz an `N = 512`-tap NLMS covers ≈ 10.7 ms of echo
    delay; per sample `y = wᵀx`, `e = d − y`,
    `w += μ · e · x / (‖x‖² + ε)`, with `μ` exposed as
    `Convergence Speed` and adaptation frozen by the double-talk
    detector.

- **Feedback detector** — **decision: continuous rolling FFT
  peak-tracker + peak-hold / threshold logic + Q-narrow notch
  placement**, reusing `rt_fft_convolver::FftHandler` over a 512-point
  sliding window.
  - *Options considered.* (a) FFT peak-tracker — a howling partial shows
    as a spectral peak that *keeps growing and holds* frame-to-frame,
    which the peak-hold logic distinguishes from a loud-but-decaying
    musical tone; this is the classic Sabine FBX / Shure DFR approach and
    reuses the FFT infra the prerequisite already calls out.
    (b) Autocorrelation howlback detection — cheaper but conflates a
    sustained musical note with feedback and gives no direct frequency to
    place the notch at. Chose (a).
  - *Example.* A 3.2 kHz partial that rises for ≥ N consecutive frames
    and exceeds the `Sensitivity` threshold → allocate the next free
    `Biquad` notch at 3.2 kHz, `Q` and depth from the controls; the
    feedback-detection LED flashes (Phase 3). `Freeze` stops new
    allocation; `Reset` clears the bank.

- **Hum removal** — **decision: adaptive comb-notch** — reuse
  `rt_spectral_core::HumNotchBank` (fundamental + harmonics) as the
  static comb, and when `Adaptive` is on **retune it from a tracked
  fundamental** via `rt_pitch_core::PitchTracker`; `Adaptive` off leaves
  the comb pinned at the selected 50 / 60 Hz.
  - *Options considered.* (a) Static comb only — simplest, but mains
    frequency drifts ± 0.5 Hz and a fixed notch then leaks the
    fundamental. (b) Adaptive — the autocorrelation tracker already
    emits a confidence-scored fundamental, so a slow drift of the comb's
    base frequency (rate-limited to avoid chasing music) tracks mains
    wander while keeping notches narrow. (a) is the `Adaptive = false`
    sub-case of (b), so both ship from one path.
  - *Example.* Selected 60 Hz, `Harmonics = 5` → notches at 60/120/180/
    240/300 Hz; with `Adaptive` on and a measured 59.94 Hz fundamental,
    the whole comb slides to 59.94/119.88/… so the third harmonic stays
    inside its narrow notch.

- **AEC reference input / scope** — **decision: AEC is in scope for
  v1** (per the user's Phase 0 call). The plugin takes a **sidechain
  reference (far-end) input**; the NLMS filter models the echo path from
  reference to the near-end mic and subtracts it. A **double-talk
  detector** freezes adaptation while the near-end talker is active so
  the canceller does not train on local speech and diverge.
  - *Options considered.* Ship only feedback/hum (smaller v1) vs. include
    NLMS AEC vs. include FFT block-LMS AEC. Chose NLMS: it covers
    broadcast / conferencing / lecture-mic echo at low CPU and no path
    latency; the heavier block-LMS long-filter path stays deferred (see
    the algorithm decision above).
  - *Example.* Far-end program on the sidechain, near-end lavalier on the
    main input: with no near-end speech the detector lets NLMS converge
    and the leaked far-end echo drops into the noise floor; when the
    near-end talker starts, the detector freezes `w` and the direct path
    passes their voice untouched.

## Phase 1: `rt-adaptive-core` (engine crate) ✅

Engine crate built, unit-tested (19 tests), clippy- and machete-clean, and
wired into the workspace `members` + `default-members`.  Every RLS option was
dropped per the Phase 0 decision.

- **Workspace layout**:
  - [`rt-adaptive-core`](../rt-adaptive-core) — engine (created; added to the
    workspace).  `rt-adaptive` — plugin (deferred to Phase 2).
- **LMS / NLMS** primitive
  ([`AdaptiveFilter<const N>`](../rt-adaptive-core/src/adaptive.rs)):
  - Fixed-length inline `[f32; N]` tap vector; `AdaptiveFilter::nlms(mu)` /
    `::lms(mu)` constructors (RLS dropped — Phase 0).
  - The per-sample `update(input, error)` from the roadmap is realised as two
    O(N) steps so the caller owns the error: `predict(input) -> y` (shift the
    sample in, return `wᵀ·x`) then `adapt(error)` (`w += μ·e·x / (‖x‖²+ε)`).
    Running `‖x‖²` is maintained in O(1).
  - Convergence proven by a system-identification test (identifies a known
    echo path from white noise, residual → ~0).
- **Feedback detector**
  ([`FeedbackDetector`](../rt-adaptive-core/src/detector.rs)):
  - Rolling Hann-windowed FFT (default 512) over a `Ring`, `DenormalGuard`-wrapped.
  - Per-bin peak-hold + peak-to-average threshold (sensitivity-mapped) with a
    persistence count distinguishes a sustained howl from broadband music;
    reports the confirmed frequency (parabolically interpolated).
- **Notch bank** ([`NotchBank`](../rt-adaptive-core/src/notch.rs)):
  - Fixed [`MAX_NOTCHES`](../rt-adaptive-core/src/notch.rs) = 24 slots, one
    reused `rt_spectral_core::Biquad` notch each, `enabled` bool per slot.
  - `allocate`/`release`/`reset` are RT-safe coefficient math; a `max_active`
    cap (`Max Notches`), near-duplicate suppression, and per-slot age/freq/Q
    inventory for the Phase 3 UI.  Meets the > 20 dB Phase 4 rejection bar.
- **Hum tracker** ([`HumTracker`](../rt-adaptive-core/src/hum.rs)):
  - Fundamental estimation via reused `rt_pitch_core::PitchTracker`
    (autocorrelation), confidence-gated and clamped to `nominal ± max_drift`.
  - Drives a reused `rt_spectral_core::HumNotchBank` comb; `Adaptive` off pins
    it to the nominal 50 / 60 Hz.
- **Block-LMS** — intentionally **deferred to v1.1** per the Phase 0 algorithm
  decision (the FFT long-filter AEC path adds latency); the time-domain NLMS
  core above covers v1.

## Phase 2: `rt-adaptive` plugin ✅

Plugin crate built on the framework-agnostic
[`params`](../rt-adaptive/src/params.rs) +
[`engine`](../rt-adaptive/src/engine.rs) split, with the `nih-plug` wrapper
([`plugin`](../rt-adaptive/src/plugin.rs)) behind the `nih` feature.  Controls
map to `AdaptiveParams` (Copy) with `clamped()` + `structural_changed()`; the
[`AdaptiveController`](../rt-adaptive/src/engine.rs) rebuilds the echo bank off
the audio thread on a filter-length change, everything else applied in place.
The Phase-3 egui editor is deferred (no `gui` feature yet).

- **I/O**:
  - Stereo in → stereo out, and mono → mono, each with a **mono sidechain**
    (the AEC far-end reference), per the Phase 0 "AEC in scope" decision.
- **Mode selector**: `Feedback Suppressor` / `Hum Remover` / `Echo Canceller`
  (one plugin, three resident engines — a mode switch never drops notches or a
  converged echo path).
- **Feedback controls**: `Sensitivity` (0–100 %), `Max Notches` (1–24),
  `Notch Q` (0.5–50), `Auto`, `Freeze`, `Reset` (rising-edge clears the bank).
  - *`Notch Depth` deferred* — the reused `rt_spectral_core::Biquad` is a
    full-depth RBJ notch, which already meets the > 20 dB Phase 4 bar; a
    variable-depth notch biquad is a v1.1 refinement.  *Manual* placement UI is
    Phase 3 (Auto off currently just disables auto-allocation).
- **Hum controls**: `Fundamental` (40–70 Hz, covers 50 / 60 + custom),
  `Harmonics` (1–10), `Adaptive` (track mains drift), `Hum Q` (the narrowness
  the reused `HumNotchBank` exposes, in place of a dB `Depth`).
- **AEC controls**: `Filter Length` (256 / 512 / 1024 / 2048 taps — the
  structural control), `Convergence Speed` (μ for NLMS), `Double-Talk
  Threshold` (dB; per-channel block-energy detector freezes adaptation when the
  near-end dominates).
- **Global controls**: `Output Gain` (−24…24 dB), `Bypass` (bit-transparent).
- **MIDI**: `Freeze` / `Reset` are automatable `BoolParam`s bound to a
  footswitch CC via host MIDI learn (the idiomatic `nih-plug` path; no plugin
  MIDI handling needed).
- **CPU meter** — the processor exposes read-only telemetry getters
  (`active_notches`, `tracked_hum_hz`, `feedback_events`) for the Phase 3 editor;
  a true CPU-percentage meter is deferred to that phase.
- **Tests**: the mandatory triad ships —
  [`integration.rs`](../rt-adaptive/tests/integration.rs) (incl. the feedback
  loopback / anti-divergence test), [`rt_safety.rs`](../rt-adaptive/tests/rt_safety.rs),
  [`rt_alloc.rs`](../rt-adaptive/tests/rt_alloc.rs) (zero-alloc including on notch
  placement).

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
