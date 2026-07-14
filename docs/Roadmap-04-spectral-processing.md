# Roadmap: `rt-spectral` (Spectral Processing Plugin Suite)

Real-time spectral (FFT-based) processing engine covering
denoising, spectral gating, spectral compression, transient
handling, and click / hum removal.  Ships as a single plugin with
mode selection, or as multiple plugins sharing a core crate —
[decision].

**Reference implementations**: iZotope RX, Waves NS1 /
Clarity Vx, FabFilter Pro-DS, Cedar DNS, Waves W43.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **Latency exposed to the host** — one FFT frame minus one hop
  (typically 21 – 85 ms).  Not a "live" plugin by default; targeted
  at mixing / mastering / dialogue cleanup.
- **All processing per FFT bin, per frame** — no allocations, no
  locks on the audio thread.
- **Noise profile capture is off-RT** — the audio thread only reads
  the current profile via atomic pointer swap.
- **Overlap-add reconstruction** with a hop / window pair that
  guarantees COLA (Constant Overlap-Add) — typically 75 % overlap
  with Hann.
- **Modes ship as separate operators** on the same spectrum:
  denoise → gate → compress → click-remove.  User can enable /
  bypass each independently.

## Prerequisites in `rt-fft-convolver`

- [ ] Reuse `dsp/fft_handler.rs` (`realfft`) — same real-to-complex
  wrapper.
- [ ] `DenormalGuard` — spectral magnitudes below the noise floor
  are exactly the case denormals appear in.
- [ ] Ring-buffer utilities (see [Roadmap-03-pitch-shift.md]) —
  shared code candidate.

## Phase 0: Algorithm decisions

- [ ] **Denoise algorithm** — [decision]:
  - Spectral subtraction (classic, artefact-prone).
  - Wiener filtering (better SNR floor).
  - Minima-Controlled Recursive Averaging (MCRA) noise estimation
    + Wiener.
  - Deep-learning based (small on-device model) — see also
    `rt-neural-amp` inference stack.
- [ ] **Transient handling** — [decision]:
  - Onset-detected bypass of denoise/compress on transients.
  - No special handling (simplest).
- [ ] **FFT size(s)**:
  - 2048 @ 48 kHz (~43 ms) default.
  - 4096 optional (~85 ms, better frequency resolution).
- [ ] **Window / overlap**:
  - Hann + 75 % overlap default.  [decision] whether to expose.
- [ ] **Click / hum detection**:
  - Median-filter-based click detection.
  - Notch-filter bank for hum (50/60 Hz + harmonics).

## Phase 1: `rt-spectral-core` (engine crate)

- [ ] **Workspace layout**:
  - `rt-spectral-core` — algorithms.
  - `rt-spectral` — plugin(s).
- [ ] **STFT front-end**:
  - Windowed real FFT via `realfft`.
  - COLA-verified hop/window pair.
- [ ] **Bin operators** (chain-of-responsibility style):
  - `NoiseProfile { magnitudes: Vec<f32> }` — captured off-RT.
  - `denoise(&mut Bin, &NoiseProfile, params)`.
  - `spectral_gate(&mut Bin, threshold_db, ratio)`.
  - `spectral_compress(&mut Bin, threshold_db, ratio, knee)`.
  - `notch_hum(&mut Bin, fundamental_hz, harmonics)`.
  - `click_repair(&mut Bin, prev: &Bin, threshold)`.
- [ ] **Noise-profile capture API** (off-RT):
  - `learn_profile(&[f32]) -> NoiseProfile`.
  - Median magnitude across N frames, per-bin.
- [ ] **Profile hot-swap**:
  - `arc-swap` between capture thread and audio thread.

## Phase 2: `rt-spectral` plugin

- [ ] **I/O**:
  - Stereo in → stereo out.
  - Per-channel STFT (mid/side option — [decision]).
- [ ] **Mode controls** (one enable-flag each):
  - `Denoise` (bool).
  - `Spectral Gate` (bool).
  - `Spectral Compressor` (bool).
  - `Hum Removal` (bool).
  - `Click Removal` (bool).
- [ ] **Denoise controls**:
  - `Learn` (momentary button) — captures 500 – 2000 ms of noise
    profile from the current input.
  - `Reduction` (0–24 dB).
  - `Sensitivity` (−30 … +30 dB above the profile).
- [ ] **Gate controls**: `Threshold`, `Range`, `Attack`, `Release`.
- [ ] **Compressor controls**: `Threshold`, `Ratio`, `Knee`,
  `Makeup`.
- [ ] **Hum controls**: `Fundamental` (50 / 60 Hz + custom),
  `Harmonics` (1–10), `Notch Q`.
- [ ] **Click controls**: `Sensitivity`, `Max Length` (samples).
- [ ] **Global controls**:
  - `Mix` (0–100 %).
  - `Output Gain`.
  - `Bypass`.
- [ ] **Latency reporting** — one frame minus one hop.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Real-time spectrogram** (last N seconds) — highest-value UI
  feature for this plugin class.
- [ ] **Noise-profile display** overlaid on the spectrum.
- [ ] **Per-mode enable / bypass indicators**.
- [ ] **Metering**: input, output, gain reduction (per operator).

## Phase 4: Testing

- [ ] **Unit tests** (`rt-spectral-core`):
  - STFT round-trip is bit-identical (or within window overlap
    error) when all operators are pass-through.
  - Denoise on white noise + tone reduces noise floor without
    affecting the tone's magnitude by more than 0.5 dB.
  - Hum notch attenuates a pure 60 Hz sine by > 40 dB.
  - Click repair detects and interpolates a single-sample impulse.
- [ ] **Integration tests**:
  - Reference frames captured from a NumPy STFT implementation.
  - Profile hot-swap under load (no glitches).
- [ ] **RT-safety audit**:
  - Zero allocations in `process()`.
  - Latency matches reported value exactly.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets**: dialogue cleanup, room-tone reduction, guitar
  amp hiss, vinyl click removal starter preset.
- [ ] **Documentation**: manual with a workflow guide for the
  learn-then-process pattern.

## Out of scope (future work)

- ML-based source separation (Demucs, Spleeter) — orders of
  magnitude more CPU, not a real-time target.
- Spectral morphing / cross-synthesis — separate plugin.
- Convolution-based deconvolution / de-reverb — see
  [Ideas.md](Ideas.md) §1 for the convolution engine path.
- Loudness / LUFS metering — separate plugin
  (`rt-loudness-meter`, roadmap 10).
