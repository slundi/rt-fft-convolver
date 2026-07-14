# Roadmap: `rt-loudness-meter` (EBU R128 / LUFS Loudness Meter)

Loudness / true-peak metering plugin compliant with EBU R128,
ITU-R BS.1770-4, and streaming-platform delivery specs.

**Reference implementations**: Youlean Loudness Meter, iZotope
Insight, Waves WLM Plus, TC Clarity M, NUGEN VisLM.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **Analysis-only** — the plugin does not modify the audio signal.
  Pure metering / measurement.
- **Zero latency** in the audio path.  Meter values update on their
  own cadence.
- **Reference-compliant** — passes the BS.1770-4 test vectors
  exactly.  This is the primary quality gate.
- **Multiple channel configurations** — mono, stereo, 5.1, 7.1
  (LUFS meters need surround support for broadcast use).
  [decision] whether v1 supports beyond stereo.
- **RT-safe** — filter states are per-channel biquads, integration
  buffers are ring-buffers.
- **History logging** off-RT for offline review of a session.

## Prerequisites in `rt-fft-convolver`

- [ ] `DenormalGuard`.
- [ ] Polyphase resampler primitives (from `rt-drive-core` if
  extracted into a shared crate) for true-peak measurement at 4×.

## Phase 0: Compliance decisions

- [ ] **Standards to comply with** — [decision]:
  - EBU R128 (broadcast, −23 LUFS).
  - ITU-R BS.1770-4 (measurement algorithm — mandatory).
  - AES71 (broadcast).
  - Streaming targets (Spotify −14, YouTube −14, Apple Music −16,
    Tidal −14, Amazon Music −14).
- [ ] **Metering windows to expose** — [decision]:
  - Momentary (400 ms).
  - Short-term (3 s).
  - Integrated (session).
  - Range (LRA).
  - True peak (dBTP).
  - Peak (dBFS).
- [ ] **Channel configurations** — [decision]:
  - Mono, stereo (v1).
  - 5.1, 7.1 (v2).

## Phase 1: `rt-loudness-core` (engine crate)

- [ ] **Workspace layout**:
  - `rt-loudness-core` — measurement engine.
  - `rt-loudness-meter` — plugin.
- [ ] **K-weighting filter** (BS.1770-4):
  - Two biquad stages (pre-filter + RLB high-pass).
  - Per-channel state.
- [ ] **Mean-square integrator**:
  - 400 ms window for momentary.
  - 3 s window for short-term.
  - Gated running mean for integrated (relative and absolute gates
    per BS.1770-4).
- [ ] **True-peak detector**:
  - 4× oversampling, peak-hold over a rolling window.
- [ ] **LRA (loudness range)**:
  - Histogram of short-term values, 10th–95th percentile spread.
- [ ] **Result-broadcast channel**:
  - Off-RT consumer reads latest values via atomics for UI /
    logging.

## Phase 2: `rt-loudness-meter` plugin

- [ ] **I/O**:
  - Stereo in → stereo out (pass-through).
  - Multichannel — [decision].
- [ ] **Meter controls**:
  - `Target` (dropdown: EBU R128 / Spotify / YouTube / Apple Music /
    custom).
  - `Custom Target` (LUFS + dBTP if custom).
  - `Reset` (clears integrated + LRA + history).
  - `Start / Stop` (integration control — some workflows start
    integration on transport play).
  - `Bypass` (irrelevant — the plugin doesn't modify audio, but
    LV2 requires it).
- [ ] **Logging controls**:
  - `Log to File` (bool + path).  Off-RT worker writes CSV /
    JSON.
- [ ] **MIDI**:
  - CC on `Reset` and `Start / Stop` (transport integration).
- [ ] **Reporting**:
  - All meter values exposed as read-only host parameters so a
    DAW can log or automate against them.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Large numeric readouts**: momentary, short-term,
  integrated LUFS, LRA, true peak, peak, RMS.
- [ ] **Meter graph**: last 60 s of short-term LUFS scrolling.
- [ ] **Histogram**: LRA distribution.
- [ ] **Target-compliance LED**: green / amber / red versus the
  chosen target.
- [ ] **Playhead-integration**: reset / hold on transport if the
  host provides transport state.

## Phase 4: Testing

- [ ] **Unit tests** (`rt-loudness-core`):
  - BS.1770-4 reference test vectors — the plugin **must** pass
    all of them within tolerance.  Non-negotiable.
  - K-weighting frequency response matches the standard curve.
  - True-peak of a 5 kHz full-scale sine at 44.1 kHz is measured
    correctly to within 0.1 dBTP.
  - Integrated LUFS gating: silence excluded from the integrated
    result per the absolute-gate rule.
- [ ] **Multichannel tests** (if in scope):
  - 5.1 channel weighting matches BS.1770 (LFE excluded, side
    channels weighted 1.41).
- [ ] **RT-safety audit**:
  - Zero allocations in `process()`.
  - Denormal guard active.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets**: EBU R128, Netflix, Spotify, YouTube, Apple Music,
  Tidal, Amazon Music, podcast (−16 LUFS), broadcast (−23 LUFS).
- [ ] **Documentation**: manual with a delivery-spec cheatsheet.

## Out of scope (future work)

- Auto-gain to hit a target LUFS — that is limiter / maximiser
  territory ([Roadmap-07-true-peak-limiter.md]).
- Spectral / vectorscope / correlation meter — [decision] whether
  to add later.  These are common companion features but not part
  of BS.1770.
- Dolby dialogue-intelligence metering — proprietary, out of scope.
- ATSC A/85 (US broadcast) — same math family as R128; add later
  if requested.
