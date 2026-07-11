# Roadmap: `rt-limiter` (True-Peak Limiter / Maximizer)

Look-ahead brickwall limiter with true-peak (inter-sample-peak)
detection.  Targets both mastering (transparent limiting) and
loudness-maximiser use.

**Reference implementations**: FabFilter Pro-L 2, Waves L2 / L3-16,
iZotope Ozone Maximizer, Sonnox Oxford Limiter, JLM TrueLim.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **True-peak detection** via oversampled peak lookup.  ITU-R
  BS.1770-4 recommends 4× oversampling as sufficient; go with 4× as
  the default, expose 2× / 4× / 8× / 16×.
- **Look-ahead** — [decision] whether fixed (e.g. 5 ms) or
  user-configurable (0.1 – 20 ms).  Reported to the host as
  latency.
- **Multiple release curves** — [decision] on the shipped set:
  linear, exponential, program-dependent (auto-release).
- **RT-safe** — all buffers sized at construction; only atomic
  parameter reads on the audio thread.
- **Loudness-aware makeup** — [decision] whether to include an
  integrated LUFS meter and auto-gain feature.  (Overlaps with
  `rt-loudness-meter` from [Roadmap-10-loudness-metering.md].)

## Prerequisites in `rt-fft-convolver`

- [ ] `DenormalGuard`.
- [ ] Polyphase resampler primitives from `rt-drive-core`
  ([Roadmap-02-oversampled-drive.md]) — reuse rather than duplicate.
  [decision] whether to promote these into a shared
  `rt-nonlinear-core` / `rt-oversample-core` crate.

## Phase 0: Algorithm decisions

- [ ] **Look-ahead length** — [decision]:
  - Fixed (typically 5 ms).
  - User-configurable (0.1 – 20 ms).
- [ ] **Gain-computer topology** — [decision]:
  - Envelope follower + feed-forward gain reduction.
  - Feedback (Waves L1-style, chainable).
  - Hybrid with adaptive attack.
- [ ] **Release curve set** — [decision]:
  - Linear, exponential, "auto" (program-dependent), "punchy",
    "smooth".
- [ ] **Channel linkage**:
  - Stereo-linked (default).
  - M/S optional.
  - Fully independent (4-channel mode) — [decision].
- [ ] **Dithering** — [decision] whether to include a bit-depth
  reducer + noise shaper at output (mastering feature).

## Phase 1: `rt-limiter-core` (engine crate)

- [ ] **Workspace layout**:
  - `crates/rt-limiter-core` — engine.
  - `crates/rt-limiter` — plugin.
- [ ] **Oversampler**:
  - Reuse `rt-drive-core`'s polyphase FIR chain.
- [ ] **True-peak detector**:
  - Oversampled peak-hold over a rolling look-ahead window.
  - Ring-buffer implementation.
- [ ] **Gain-computer**:
  - Threshold, ceiling, attack (implicit = look-ahead length),
    release.
  - Smoothed gain reduction curve to avoid distortion at the
    knee.
- [ ] **Delay line** for the dry signal (matches look-ahead
  length).
- [ ] **Output ceiling clipper** as a safety net (should never
  engage if the limiter is doing its job).
- [ ] **Dither / noise-shaper** (if included in scope).

## Phase 2: `rt-limiter` plugin

- [ ] **I/O**:
  - Stereo in → stereo out.
  - Optional 4-channel mode — [decision].
- [ ] **Core controls**:
  - `Threshold` (−60 … 0 dB).  Signal above this is limited down.
  - `Ceiling` (−12 … 0 dB).  Output true-peak limit.
  - `Release` (1 – 2000 ms + "auto").
  - `Style` (dropdown: transparent / punch / smooth /
    aggressive).
  - `Look-ahead` (0.1 – 20 ms, if user-configurable).
- [ ] **Advanced controls**:
  - `Oversampling` (1× / 2× / 4× / 8× / 16×).
  - `Channel Link` (stereo linked / M/S / independent).
  - `Auto Release` (bool).
  - `Dither` (off / 16-bit / 20-bit / 24-bit, with shaper choice).
  - `Bypass`.
- [ ] **Loudness display** — [decision]:
  - True-peak (dBTP), integrated LUFS, short-term LUFS, gain
    reduction max.
- [ ] **MIDI**:
  - Not strictly needed for a mastering plugin; CC on `Threshold`
    optional.
- [ ] **CPU meter** as read-only property.

## Phase 3: UI

- [ ] **`nih-plug-egui`** or **`nih-plug-iced`** — [decision].
- [ ] **Gain-reduction meter** with peak-hold.
- [ ] **True-peak in / out meters** (dBTP-calibrated).
- [ ] **LUFS meter** (short-term + integrated) — if included in
  scope.
- [ ] **Threshold / ceiling knobs with numeric readout**.
- [ ] **History display**: last N seconds of gain reduction as a
  scrolling graph.

## Phase 4: Testing

- [ ] **Unit tests** (`rt-limiter-core`):
  - True-peak of the output never exceeds the ceiling for any
    input at any tested oversampling factor (test with a
    high-frequency signal designed to alias).
  - Sub-threshold input is bit-identical to the delayed dry signal
    (bypass equivalence).
  - Attack: gain reduction reaches target within look-ahead
    samples ± 1.
  - Release: gain reduction returns to 0 dB within the specified
    time (± 10 %).
- [ ] **Loudness compliance**:
  - Reference BS.1770-4 test vectors, if LUFS is included.
- [ ] **RT-safety audit**:
  - Zero allocations in `process()`.
  - Denormal guard active.
  - Reported latency matches actual sample-accurate delay.
- [ ] **Host validation**: `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Presets**: mastering-transparent, mastering-loud, drum-bus,
  vocal-bus, broadcast (−1 dBTP, −23 LUFS if LUFS included), streaming
  (−1 dBTP, −14 LUFS).
- [ ] **Documentation**: manual with a "how to set the ceiling for
  streaming platforms" table.

## Out of scope (future work)

- Multi-band limiting — that is `rt-multiband-compressor`
  (see [Roadmap-11-multiband-compressor.md]).
- Clipper-plus-limiter hybrid (e.g. Kazrog KClip topology) —
  deferred.
- Sidechain input for external ducking — [decision] whether to
  add later.
