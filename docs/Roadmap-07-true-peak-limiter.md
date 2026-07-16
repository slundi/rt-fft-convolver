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

## Prerequisites in `rt-fft-convolver` ✅

- `DenormalGuard` — **satisfied**. Fully implemented in
  `src/utils/denormals.rs` and re-exported from the crate root
  (`src/lib.rs:34`, `rt_fft_convolver::{DenormalGuard, flush_to_zero}`):
  RAII FTZ+DAZ guard (x86/x86_64 MXCSR, AArch64 FPCR, no-op
  elsewhere), restores the prior FPU state on drop, RT-safe (one
  register read+write), test-covered. Load-bearing here because the
  gain-reduction envelope and dry-signal delay line ring *down toward
  zero* on quiet passages; without FTZ the release tail would spike
  CPU on "silence".
- Polyphase resampler — **satisfied**. `rt-drive-core`'s
  `Oversampler` (`rt-drive-core/src/lib.rs:58`) is a Kaiser-windowed
  half-band polyphase up/down-sampler for factors {1, 2, 4, 8, 16},
  ≥96 dB default stop-band attenuation, RT-safe `upsample`/`downsample`
  with all scratch pre-sized at construction. This is exactly the
  true-peak oversampling primitive the limiter needs — reused, not
  duplicated.
- **Shared-crate promotion — decision: do NOT promote; depend on
  `rt-drive-core` directly.** `rt-limiter-core` takes a path
  dependency on `rt-drive-core` and uses its `Oversampler`. A
  dedicated `rt-oversample-core` extraction was considered and
  *rejected for now*: it buys nothing at runtime and the cost of
  pulling the (small, leaf) drive engine transitively is acceptable.
  Revisit only if a third consumer needs oversampling without the
  drive engine, or if `rt-drive-core`'s dependency surface grows.

## Phase 0: Algorithm decisions ✅

### Look-ahead length — **decision: user-configurable, 0.1 – 20 ms**

Follows FabFilter Pro-L 2. The dry-signal delay line and the
peak-hold look-ahead ring are both sized for the **20 ms maximum** at
construction (RT-safe: no reallocation), and the runtime control only
selects a shorter read offset. Reported host latency = current
look-ahead + oversampler group delay. Short settings (0.1–1 ms) read
as punchy/aggressive; long settings (10–20 ms) are transparent on
dense material.

**Example — 20 ms max at 96 kHz.** Delay-line capacity =
`ceil(0.020 · 96000)` = 1920 samples per channel, allocated once. A
5 ms runtime setting reads the tap at `round(0.005 · 96000)` = 480
samples back. Changing the control never touches the allocation.

- *Rejected*: **fixed 5 ms** — simpler, but the runtime cost of the
  configurable version is zero once the buffer is sized for the max,
  so there is no reason to give up the flexibility.

### Gain-computer topology — **decision: feed-forward envelope follower**

The detector reads the oversampled true-peak over the look-ahead
window, the gain computer derives the required reduction, and that
gain is applied to the delayed dry signal. This is the only topology
that lets Phase 4's hard guarantee hold cleanly — *output true-peak
never exceeds the ceiling* — because the reduction is computed from a
peak the audio has **not yet reached**.

**Example — instantaneous target gain.** For a detected true-peak
`p` (linear) and ceiling `c` (linear), the target gain is
`g_target = min(1, c / p)`. With `c = 10^(−1/20) ≈ 0.8913` (−1 dBTP)
and a detected peak `p = 1.5` (+3.52 dBFS true-peak),
`g_target ≈ 0.594` (−4.52 dB reduction). The look-ahead lets that
gain be reached *by the sample the peak arrives*, so the peak is
caught rather than clipped. Attack is therefore implicit: it equals
the look-ahead length.

- *Rejected*: **feedback (Waves L1-style)** — derives reduction from
  the output, cannot use look-ahead, and makes a hard true-peak
  ceiling difficult to guarantee.
- *Rejected*: **hybrid adaptive-attack** — deferred; the
  program-dependent behaviour we want is delivered through the
  release curve ("auto") instead, without a second topology.

### Release curve set — **decision: linear + exponential + "auto" as engine curves; "punchy"/"smooth" as layered styles**

Three curves live in the engine; the two voiced presets are *styles*
that parameterise them (avoids duplicating curve code, per the Phase 2
`Style` control):

- **Linear** — constant dB/s recovery; predictable, can pump.
- **Exponential** — one-pole RC recovery; the natural default.
- **Auto (program-dependent)** — dual time-constant (fast + slow)
  blended by transient density; the "transparent" mode.
- **Punchy** = exponential with a faster initial release;
  **Smooth** = exponential/auto with a longer time constant.

**Example — exponential release coefficient.** For a release time
`t_r` (seconds to ~63 % recovery) at sample rate `fs`, the one-pole
coefficient is `α = exp(−1 / (t_r · fs))`. At `t_r = 100 ms`,
`fs = 48 kHz`: `α = exp(−1 / 4800) ≈ 0.999792`. Gain recovers as
`g[n] = α · g[n−1] + (1 − α) · g_target`.

### Channel linkage — **decision: stereo-linked (default) + M/S; defer fully-independent 4-channel**

Stereo-linked drives both channels from a single (max-of-channels)
gain-reduction value, preserving the stereo image — the mastering
default. M/S applies the same linked computer to a mid/side matrix
for width control. Fully-independent per-channel reduction is
*deferred*: it risks audible image shift on stereo material and
complicates plugin I/O (revisit with the out-of-scope surround work).

- *Rejected for v1*: **fully independent (4-channel)** — see the
  I/O `[decision]` in Phase 2; not shipped in v1.

### Dithering — **decision: include a bit-depth reducer + noise shaper, feature-gated**

A true-peak limiter is typically the *last* stage before export, so
dither belongs here. `rt-limiter-core` ships a bit-depth reducer
(16/20/24-bit) plus a selectable noise shaper, behind a Cargo
**feature flag** so the crate stays lean for callers that don't need
it. Applied strictly *after* the ceiling stage, at the output word
length.

- *Rejected*: **exclude entirely** — would force a downstream tool
  for the final export step the plugin is otherwise positioned to
  own.

## Phase 1: `rt-limiter-core` (engine crate)

- [ ] **Workspace layout**:
  - `rt-limiter-core` — engine.
  - `rt-limiter` — plugin.
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
