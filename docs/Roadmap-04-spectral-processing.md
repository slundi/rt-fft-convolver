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

## Prerequisites in `rt-fft-convolver` ✅

- Reuse `dsp/fft_handler.rs` (`realfft`) — same real-to-complex
  wrapper.
- `DenormalGuard` — spectral magnitudes below the noise floor
  are exactly the case denormals appear in.
- Ring-buffer utilities (see [Roadmap-03-pitch-shift.md]) —
  shared code candidate.

## Phase 0: Algorithm decisions ✅

### Denoise algorithm — **decision: Wiener with decision-directed SNR, learn-seeded; MCRA optional**

- **Core**: per-bin **Wiener gain** driven by the **decision-directed
  (Ephraim–Malah) a-priori SNR** estimate.  The DD estimator is what
  keeps Wiener free of *musical noise* — it smooths the a-priori SNR
  across frames, so isolated random bin spikes don't turn into
  warbling tones.
  - Gain: `G(k) = ξ(k) / (1 + ξ(k))`, with a-priori SNR
    `ξ(k) = α·(|Y_{prev}|²·G_prev²)/N(k) + (1−α)·max(γ(k)−1, 0)`,
    a-posteriori SNR `γ(k) = |Y(k)|² / N(k)`, smoothing `α ≈ 0.98`.
  - `N(k)` is the per-bin noise power from the learned profile.
- **Noise floor / oversubtraction control** exposed as `Sensitivity`
  and `Reduction`: gain is floored at `10^(−Reduction/20)` so residual
  is a controllable bed of attenuated noise, not a hole.
- **Learn-seeded** (RX / Waves W43 workflow): the `Learn` button
  captures a static per-bin profile off-RT (median magnitude across
  frames → power).  The RT thread only reads it via `arc-swap`.
- **MCRA** (Minima-Controlled Recursive Averaging) is kept as an
  **optional continuous-tracking mode** for the no-learn / "set and
  forget" path (NS1 / Cedar DNS style).  It adds per-bin minimum
  tracking state but no new RT allocations.
- *Rejected*: plain **spectral subtraction** (musical-noise prone — it
  survives only as the degenerate `α=0`, oversubtraction>1 case) and
  **deep-learning** denoise (defer to `rt-neural-amp` stack; out of
  scope, see §Out of scope).

### Transient handling — **decision: onset-detected soft bypass**

- Compute **spectral flux** `SF = Σ_k max(|Y(k)|−|Y_{prev}(k)|, 0)`
  once per frame (essentially free — magnitudes already computed).
- On an onset (flux exceeds an adaptive median threshold), **crossfade
  the denoise/compress depth toward unity** for that frame so pick
  attacks and drum transients aren't smeared.  Gate/hum/click are
  unaffected.
- High value for a guitar-oriented suite (pick attack preservation);
  the cost is one scalar reduction per frame.

### FFT size(s) — **settled: 1024 / 2048 / 4096, default 2048**

- Reported latency = **one full frame `N`** (Phase 1 implementation
  decision, see below):

  | N | frame @48 kHz | latency (`N`) | use |
  |---|---|---|---|
  | 1024 | 21.3 ms | **21.3 ms** | "Live" |
  | 2048 | 42.7 ms | **42.7 ms** | "Studio" (default) |
  | 4096 | 85.3 ms | **85.3 ms** | "HiRes" (best freq. resolution) |

  - ⚠️ **Revised during Phase 1** (was tentatively `N − hop = 3N/4`).
    The `N − hop` group delay only holds when the host block size is
    `>= hop`.  With a *streaming* overlap-add framer and per-block pull,
    a block-size-**independent** latency provably requires a full frame:
    prime the analyzer with `N − hop` zeros *and* the synthesis FIFO
    with one `hop` (`N − hop + hop = N`).  Since DAWs use blocks as small
    as 16–64 samples (well below a 512-sample hop), reporting `N` is the
    only value that stays exact under automation and buffer-size
    changes.  This matches the design-principle note's original
    "21–85 ms" (= `N`) figures.  Recompute on size / sample-rate change
    (sizes are in samples, so ms scales with `fs`).  The click despiker
    adds a further fixed `(window−1)/2` (2 samples at the default
    5-tap), always imposed so toggling it never shifts PDC.

### Window / overlap — **decision: Hann analysis+synthesis, 75 % overlap, fixed**

- Both analysis and synthesis windowed with **Hann** (effective `Hann²`
  weighting), **hop = N/4**.  `Hann²` at 75 % overlap satisfies COLA;
  normalise by the constant OLA gain so an all-pass chain is unity.
- **Not exposed** (like the pitch plugin's fixed 4× overlap): keeps
  reported latency and buffer sizes deterministic for host
  compensation and the RT-safety audit.  A 50 % "low-CPU" advanced
  option is a post-1.0 candidate.
- Reuse the existing Hann/COLA helper (shared crate, see prerequisites).

### Click / hum removal — **decision: time-domain stages, not per-bin operators**

This revises the Phase 1 operator list.  A click is a *broadband,
flat-spectrum* event and hum is a *few pure lines* — both are clumsy
and lossy to treat per-STFT-bin.  Handle them in the time domain,
flanking the STFT core:

- **Hum**: cascaded **biquad notch bank** at the fundamental (50 / 60 Hz
  or custom) + up to N harmonics, `RBJ` notch with adjustable `Q`.
  Surgical, sample-accurate, no added latency.  Reuse the existing
  `biquad.rs`.
- **Click**: **median-filter detector** (order 5–9) on the time-domain
  signal; flag samples whose deviation from the median exceeds
  `Sensitivity`, then **interpolate** the flagged run (cubic / short AR
  model) up to `Max Length` samples.  A pre-STFT stage operating on the
  input ring.
- ⚠️ **Phase 1 impact**: drop `notch_hum` / `click_repair` from the
  per-bin `Bin` operator chain; add a `TimeDomainPre` (click) and
  `TimeDomainPost` (hum) stage instead.  The per-bin chain remains
  denoise → gate → compress.

## Cost model (CPU / memory / complexity)

Baseline: **48 kHz, stereo, N = 2048, hop = 512**, `K = N/2+1 = 1025`
bins, frame rate `fs/hop = 93.75 frames/s` per channel.

### Per-operator complexity (per frame, per channel)

| Stage | Complexity | Notes |
|---|---|---|
| STFT fwd + inv | `O(N log N)` | ~110 k flops/frame @ N=2048 |
| Window + OLA | `O(N)` | 2 mul-add passes |
| Denoise (Wiener + DD SNR) | `O(K)` | ~20 flops/bin + 1 `arc-swap` load/frame |
| Spectral gate | `O(K)` | ~8 flops/bin |
| Spectral compress | `O(K)` | ~15 flops/bin (log/knee) |
| Transient (spectral flux) | `O(K)` | 1 scalar/frame |
| Hum notch (time domain) | `O(M · hop)` | M biquads, M = harmonics |
| Click (time domain) | `O(w · hop)` | median window `w` |

### CPU estimate

FFT dominates and is cheap: fwd+inv ≈ 110 k flops/frame → `×93.75 ×2 ch
≈ 21 Mflops/s`.  All bin operators worst-case (~60 flops/bin) add
`≈ 12 Mflops/s`.  Time-domain notch/click add a few Mflops/s.  Total
arithmetic is **well under 100 Mflops/s** against multi-Gflop cores.

Realistic wall-clock (including windowing memory traffic, `arc-swap`
loads, denormal handling — hence `DenormalGuard`) per **stereo
instance**:

| N | est. CPU (1 core, modern x86-64) |
|---|---|
| 1024 | ~1.5 – 3 % (2× frame rate, higher overhead/sample) |
| 2048 | **~1 – 2 %** |
| 4096 | ~0.8 – 1.5 % (half the frame rate) |

Larger N is *cheaper per second* (fewer frames) but costs latency and
frequency smearing — the trade-off, not CPU, drives the size choice.

### Memory (RT-resident, per stereo instance @ N=2048)

| Buffer | Size |
|---|---|
| Input ring + OLA accumulator (×2 ch) | 4 × 2048 f32 = 32 KB |
| Spectrum + scratch (`FftHandler`) | ~24 KB |
| Hann window (shared) | 8 KB |
| Noise profile, double-buffered (`arc-swap`) | 2 × 1025 f32 = 8 KB |
| Biquad notch state, click median window | < 1 KB |
| **Total RT footprint** | **~70 – 80 KB** |

Off-RT / UI only (not on the audio thread): the Phase 3 spectrogram
history is the memory hog — 5 s × 93.75 frames/s × 1025 bins × 4 B ≈
**1.9 MB per channel**.  Down-sample bins/time for the display buffer;
never size the RT path to it.

### Implementation complexity ranking (effort, not runtime)

1. **STFT front-end + COLA** — low (reuse `FftHandler` + window helper).
2. **Gate / compress** — low (per-bin static curves).
3. **Hum notch** — low (reuse `biquad.rs`).
4. **Wiener + DD SNR denoise** — medium (per-bin SNR state, tuning).
5. **Click detect + interpolate** — medium (false-positive tuning vs.
   guitar transients — coordinate with the transient detector).
6. **MCRA continuous noise tracking** — medium-high (minimum-statistics
   state machine); optional, can land after 1.0.

## Phase 1: `rt-spectral-core` (engine crate) ✅

Implemented as the `rt-spectral-core` workspace crate (framework-agnostic,
no plugin-format dependency).  43 unit tests + integration tests; `cargo
fmt`/`clippy` clean.

- [x] **Workspace layout**:
  - `rt-spectral-core` — algorithms (this phase).
  - `rt-spectral` — plugin(s) (Phase 2).
- [x] **STFT front-end** (`window.rs`, `stft.rs`):
  - Windowed real FFT via `realfft` (reuses `FftHandler`).
  - COLA-verified Hann², 75 % overlap; numerically-measured OLA norm.
  - Streaming `Analyzer`/`Synthesizer` on `Ring<f32>`, primed so latency
    is one frame, block-size independent.
- [x] **Operators** (concrete per-op structs, explicit ordered pipeline —
  cleaner than a `&mut Bin` trait; each carries its own per-bin state):
  - `NoiseProfile { power: Vec<f32> }` — per-bin **power**, captured
    off-RT (denoiser works in the power domain).
  - `Denoiser::process` — Wiener + decision-directed SNR, transient depth.
  - `SpectralGate::process` — per-bin expander, attack/release envelope.
  - `SpectralCompressor::process` — per-bin static soft-knee curve.
  - `HumNotchBank::process_block` — **time-domain** biquad notch bank
    (Phase 0 revision; `biquad.rs`, `hum.rs`).
  - `ClickRepair::process_block` — **time-domain** median despiker
    (Phase 0 revision; `click.rs`).
  - `TransientDetector::observe` — spectral-flux onset soft-bypass.
- [x] **Noise-profile capture API** (off-RT, `profile.rs`):
  - `learn_profile(&[f32], fft_size, hop) -> NoiseProfile`.
  - Median magnitude across frames, per-bin → power.
- [x] **Profile hot-swap**:
  - `triple_buffer` SPSC channel (`profile_channel` → `ProfileSender` /
    `ProfileReceiver`); wait-free, allocation-free read on the audio
    thread.  Chosen over `arc-swap` — already in the workspace lockfile
    and fits the one-writer/one-reader topology exactly.

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
