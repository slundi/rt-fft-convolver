# Roadmap: `rt-multiband` (Multi-Band Compressor / Limiter / Expander)

Multi-band dynamics processor with configurable crossover and
per-band compression, limiting, or expansion.

**Reference implementations**: FabFilter Pro-MB, Waves C4 / C6,
iZotope Ozone Dynamics, TDR Nova, Sonnox Oxford Dynamics.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **Band count** — **Decision (Phase 0): fixed at 4 in v1**
  (3 crossovers); configurable 2–6 deferred to v2 — see Phase 0.
- **Crossover topology** — **Decision (Phase 0): Linkwitz-Riley
  4th-order IIR in v1** (24 dB/oct, phase-coherent when summed within a
  band, zero latency); linear-phase FIR via `rt-fft-convolver` deferred
  to v2 — see Phase 0.
- **Per-band mode** — each band independently compressor, expander,
  gate, or bypass.
- **RT-safe** — all state per-band and per-channel, allocation-free
  on the audio thread.
- **Sidechain input** — per-band internal sidechain filter (HPF +
  shelf). **Decision (Phase 0): external (aux-input) sidechain deferred
  to v2** — see Phase 0.
- **Zero-latency mode** with the IIR crossover; the higher-latency
  linear-phase mode (v2) will be reported accurately to the host.

## Prerequisites in `rt-fft-convolver` ✅

- `DenormalGuard` — present and tested
  ([src/utils/denormals.rs](../src/utils/denormals.rs)); RAII FTZ/DAZ
  guard, re-exported at the crate root. Wraps each band's envelope
  follower and gain computer so a silent band (or a silent passage in
  any band) never drives the one-pole detector states or the crossover
  biquad states into denormal arithmetic.
- FIR convolution engine for the linear-phase crossover — **present.**
  The partitioned convolver — the heavy, hard-to-get-right part — already
  exists and is tested (`UniformPartitionEngine`, `HybridConvolver`,
  `OfflineConvolver`, all re-exported at the crate root). The v1 crossover
  is LR4 IIR (zero-latency, see Phase 0), so this link is **not on the v1
  critical path**; it is the drop-in engine for the v2 linear-phase mode
  (design windowed-sinc band-split FIRs, convolve, report latency).
- Parameter-smoothing utility — **decision: keep local.** There is
  nothing clean to extract: [mixer.rs](../src/engine/mixer.rs) has no
  general smoother (only an equal-power dry/wet crossfade folded into two
  gains at set-time), and `rt_limiter_core::GainComputer` is a
  look-ahead-limiter-specific reduction smoother, not a generic ramp.
  `rt-multiband-core` gets its own small one-pole `ParamSmoother` for
  click-free crossover-frequency / threshold / makeup changes; nih-plug's
  own `Smoother` still handles wrapper-level param smoothing in Phase 2.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Local one-pole `ParamSmoother` in core** (chosen) | 2 f32/param (current + target) | 1 mul-add/param/sample | Low — ~15 lines, no cross-crate coupling |
    | Extract a shared smoother from `mixer.rs` | same | same | Med — mixer has no smoother to extract; would mean *adding* one there and re-exporting for one consumer |
    | Depend on `rt_limiter_core::GainComputer` | its full release-curve state | its per-sample cost | Med — wrong shape (feed-forward limiter reduction, not a generic parameter ramp) |

## Phase 0: DSP decisions ✅

The organising Phase 0 finding is that **v1 is deliberately the
zero-latency, IIR, fixed-4-band core** — every heavier option
(linear-phase FIR crossover, external sidechain, look-ahead, >4 /
configurable bands, M/S) is a clean v2 add-on that does not change the
v1 signal-flow contract. That keeps the v1 audio path allocation-free
and zero-latency, honours the "reuse over re-implementation" rule (the
FIR path drops onto the existing convolver when v2 wants it), and lets
the per-band dynamics engine be exhaustively unit-tested against the
Phase 4 targets before any latency or aux-bus plumbing is layered on.

Cost figures below are per instance at 48 kHz, stereo, 4 bands; "block"
means the host's processing block (≤ 512 samples assumed for sizing).

### Scope / topology decisions

- **Band count** — **decision: fixed at 4 in v1**; user-configurable
  2–6 deferred to v2.
  - *Rationale.* 4 bands (3 crossovers) is the classic multiband layout
    (Waves C4) and covers every Phase 5 preset — mastering-glue,
    drum-bus, vocal-de-esser (one active band, others bypassed),
    broadcast-loudness, bass-tightener. A fixed count keeps the state
    arrays, UI, and crossover tree static, so nothing reallocates on a
    parameter change.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Fixed 4** (chosen) | 4× band state + 3 LR4 crossovers/ch (a few hundred bytes) | 3 crossovers + 4 band processors/sample | Low — static arrays, static UI |
    | Fixed 6 | ~1.5× the above (5 crossovers, 6 bands) | ~1.5× | Low, but a heavier fixed UI for marginal benefit at guitar/bus scope |
    | Configurable 2–6 | sized for 6, N active | N-dependent | Med–High — off-thread crossover-tree rebuild on count change + UI reflow |

- **Crossover topology** — **decision: Linkwitz-Riley 4th-order IIR
  (24 dB/oct) in v1**; linear-phase FIR (via `rt-fft-convolver`)
  deferred to v2.
  - *Rationale.* LR4 is a cascade of two identical Butterworth 2nd-order
    biquads per split; the summed low+high output of one split is
    all-pass (magnitude-flat), which is what makes bypassed bands
    reconstruct the input — the Phase 4 bit-identical bypass test. It is
    zero-latency, matching the design principle. Linear-phase FIR is
    ideal for mastering but adds latency, CPU, FIR band-split design, and
    delayed-reconstruction tests; since the convolver it would reuse
    already exists, deferring it costs nothing architecturally.
  - *Reconstruction note (>2 bands).* Summing three LR4 splits naïvely
    leaves a phase error between non-adjacent bands. The standard fix is
    an **all-pass correction** (or processing splits as a serial tree)
    so the bypassed-band sum stays flat — see Phase 1 "Band-sum
    reconstruction". This is the one genuinely subtle part of the IIR
    path and gets its own test.
  - *Options considered.*

    | Option | Memory | CPU load | Latency | Complexity |
    |---|---|---|---|---|
    | **LR4 IIR, v1** (chosen) | 2 biquads/crossover/ch | low (a handful of mul-adds/sample/split) | 0 | Med — all-pass reconstruction across 3 splits |
    | Linear-phase FIR now | FFT partitions/band (KB–MB per IR length) | high (FFT/block) | FIR half-length | High — band-split design + latency reporting + delayed-recon tests |
    | Both, selectable, v1 | union of the two | mode-dependent | mode-dependent | High — ships both paths and the switch in v1 |

- **Sidechain** — **decision: internal (band-content) sidechain with a
  per-band SC HPF + shelf in v1**; external aux-input sidechain deferred
  to v2.
  - *Rationale.* An internal SC filter covers the common cases
    (de-essing, bass-ducking, tilting the detector without touching the
    audio path) at the cost of one biquad per band. External sidechain
    needs an nih-plug aux input bus and a per-host support matrix
    (`clap-validator` / `pluginval` aux-port coverage), which is a
    packaging/validation concern better isolated to its own version.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Internal SC filter, v1** (chosen) | 1 biquad/band/ch | 1 biquad/band/sample | Low |
    | External aux input, v1 | + aux input buffers | low (routing only) | Med–High — aux-bus plumbing + host validation matrix |

- **Look-ahead** — **decision: off (zero-latency) in v1**;
  user-configurable look-ahead deferred to v2.
  - *Rationale.* Keeps the v1 promise of a zero-latency path and avoids a
    full-signal delay line + host latency reporting. When v2 adds it,
    `rt_limiter_core::DelayLine` is the ready-made, tested delay
    primitive to reuse.
  - *Options considered.*

    | Option | Memory | CPU load | Latency | Complexity |
    |---|---|---|---|---|
    | **Off, v1** (chosen) | 0 | 0 | 0 | Low |
    | Fixed (e.g. 3 ms) | ~fs·3 ms f32/ch (~576 B/ch) | low | 3 ms (reported) | Med — delay line + latency reporting |
    | User-configurable | max-lookahead buffer/ch | low | param-dependent (reported) | Med |

### Per-band dynamics decisions

- **Detector topology** — **decision: Peak / RMS blend with a per-band
  blend control** (0 = pure RMS, 1 = pure peak); program-dependent
  (auto) deferred.
  - *Rationale.* Low bands want RMS (musical, level-riding, ignores
    transients); high bands and the de-esser band want peak (transient
    control). A single per-band blend knob subsumes both pure modes and
    is what the reference designs (FabFilter Pro-MB) expose. RMS costs
    one square + one one-pole mean-square state per band.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Peak / RMS blend** (chosen) | 1 mean-square state/band/ch (+ peak state) | square + one-pole + blend/sample | Low–Med |
    | Peak only | peak state only | cheapest | Low, but no musical low-band riding |
    | RMS only | mean-square state | one-pole/sample | Low, but poor transient control up top |
    | Program-dependent (auto) | + adaptation state | + adaptation logic | Med–High — deferred; blend covers the need |

- **Knee** — **decision: soft knee with user-configurable width;
  width = 0 gives an exact hard knee** (so "both" fall out of one gain
  computer).
  - *Rationale.* A single quadratic-interpolation soft-knee gain
    computer degenerates to a hard knee at width 0, so there is no reason
    to ship two code paths. The extra cost is a branch and a quadratic
    only in the ± width/2 region around threshold.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Soft knee, width param (0 = hard)** (chosen) | none extra | branch + quadratic near threshold | Low |
    | Hard knee only | none | cheapest | Low, but no soft-knee musical option |
    | Separate hard + soft paths | none | same | Med — two computers to keep in sync for no gain |

- **Envelope follower** — **decision: analog-modelled branching one-pole**
  with separate attack/release capacitor-charge coefficients
  `coeff = exp(−1 / (t · fs))`, computed off-RT on any time-constant
  change.
  - *Rationale.* The RC one-pole is the standard, cheap, well-understood
    follower and directly matches the Phase 4 attack/release
    time-constant test (±10 % of the specified τ). "Linear" ramps are
    less musical; "RC + look-ahead" is subsumed by the (deferred)
    look-ahead decision.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Analog-modelled RC one-pole** (chosen) | 1 f32 state/band/ch; 2 coeffs precomputed | 1 branch + 1 one-pole/sample | Low |
    | Linear attack/release | 1 state | 1 add/clamp/sample | Low, but less natural, harder to hit τ test |
    | RC + look-ahead | + delay line | + delay | Med — depends on the deferred look-ahead |

- **Stereo linking** — **decision: linked (max detector across L/R) as
  the default, with a per-band Independent toggle**; M/S-selectable
  linking deferred to v2.
  - *Rationale.* Linking the detector (drive both channels' gain from
    `max(|L|, |R|)`) prevents the stereo-image shift that independent
    per-channel gain reduction causes — the right default. Independent is
    a free toggle (skip the `max`). M/S linking needs an encode/decode
    matrix wrapped around the whole band processor and is a v2 feature.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Linked + per-band Independent toggle** (chosen) | shared detector/band | 1 `max`/sample when linked | Low |
    | Linked only | shared detector | 1 `max`/sample | Low, but no independent option |
    | M/S per-band selectable | + M/S matrix state | + encode/decode/sample | Med — deferred to v2 |

## Phase 1: `rt-multiband-core` (engine crate) ✅

- **Workspace layout**:
  - `rt-multiband-core` — engine
    ([Cargo.toml](../rt-multiband-core/Cargo.toml), wired into the root
    `members` **and** `default-members`).
  - `rt-multiband` — plugin (Phase 2, not yet scaffolded).
- **Crossover primitives**
  ([crossover.rs](../rt-multiband-core/src/crossover.rs)):
  - Linkwitz-Riley 4th-order biquad pairs (two cascaded Butterworth
    sections per split) built on a local RBJ
    [`Biquad`](../rt-multiband-core/src/biquad.rs) (LP/HP/allpass/shelf).
  - Linear-phase FIR via `rt-fft-convolver` — deferred to v2 (Phase 0).
- **Per-band processor**
  ([band.rs](../rt-multiband-core/src/band.rs)):
  - `BandProcessor` = [`SidechainFilter`](../rt-multiband-core/src/sidechain.rs)
    → [`Detector`](../rt-multiband-core/src/detector.rs) (peak/RMS blend)
    → [`EnvelopeFollower`](../rt-multiband-core/src/envelope.rs)
    (analog attack/release)
    → [`GainComputer`](../rt-multiband-core/src/gain_computer.rs)
    (soft-knee compressor/expander/gate) → makeup gain.
  - `process(&mut self, left, right) -> (f32, f32)` — RT-safe, stereo with
    `StereoLink` linked / independent (the sample-domain equivalent of the
    roadmap's `process_sample`, done stereo so detector linking is real).
- **Sidechain routing**
  ([sidechain.rs](../rt-multiband-core/src/sidechain.rs)):
  - Internal (band content) — the detector-path signal.
  - External (aux input) — deferred to v2 (Phase 0).
  - Sidechain filter (HPF + high shelf), per band, per channel.
- **Band-sum reconstruction**
  ([engine.rs](../rt-multiband-core/src/engine.rs)):
  - Linkwitz-Riley bands sum to a magnitude-flat (all-pass) copy of the
    input when bypassed — verified by
    `all_bands_bypassed_reconstruct_the_input` (< 0.1 dB error) plus a
    per-probe-tone flatness test in `crossover.rs`.
  - Parameter smoothing is the local
    [`ParamSmoother`](../rt-multiband-core/src/smoother.rs) (Phase 0
    prerequisite decision).

**Tests.** 46 unit tests across the nine modules (crossover reconstruction,
compressor −10 dB @ 2:1 hard-knee acceptance, expander downward-expansion,
attack/release ±10 % τ, detector blend, solo/mute isolation, silence→silence).
Plugin `tests/` triad lands with the Phase 2 `rt-multiband` wrapper.

## Phase 2: `rt-multiband` plugin ✅

The plugin ([Cargo.toml](../rt-multiband/Cargo.toml)) follows the family shape
minus the controller: because [`MultibandEngine`](../rt-multiband-core/src/engine.rs)
allocates **nothing** after construction and every control is an in-place
setter, the engine lives resident in the audio-thread
[`MultibandProcessor`](../rt-multiband/src/engine.rs) (no `ir_handoff` seam in
v1). [`params.rs`](../rt-multiband/src/params.rs) is the framework-agnostic
`Copy` model; [`plugin.rs`](../rt-multiband/src/plugin.rs) is the `nih` wrapper
(nested per-band param groups + a `snapshot()` converter);
[`ui.rs`](../rt-multiband/src/ui.rs) holds egui-free meter helpers for Phase 3.

- **I/O** — stereo → stereo and mono → mono
  ([plugin.rs](../rt-multiband/src/plugin.rs) `AUDIO_IO_LAYOUTS`). External
  (aux) sidechain in deferred to v2 (Phase 0), so no aux buses.
- **Global controls**
  ([params.rs](../rt-multiband/src/params.rs)):
  - `Band Count` — fixed 4 in v1 (Phase 0), so not a control.
  - `Crossover Type` — IIR only in v1 (Phase 0), so not a control.
  - `Input Gain`, `Output Gain`, `Mix`, `Bypass` — plus `Link`
    (linked/independent) and three `Crossover N` edges.
- **Per-band controls** (× 4) — `Crossover Frequency` (global edges),
  `Mode`, `Threshold`, `Ratio`, `Attack`, `Release`, `Knee`, `Makeup`,
  `Solo`, `Mute`, plus `Detector Blend` (peak/RMS). Declared as four nested
  `#[nested(array, group = "Bands")]` param groups.
- **Sidechain controls** (per band)
  ([sidechain](../rt-multiband-core/src/sidechain.rs)):
  - `SC Source` — self only in v1 (external deferred, Phase 0).
  - `SC HPF` (Hz) — per band.
  - `SC Listen` — deferred to the Phase 3 editor.
- **MIDI** — per-band `Threshold` (CC 20–23) and `Makeup` (CC 24–27) via
  `MidiConfig::MidiCCs`; a CC override is reclaimed when the host moves the
  knob ([`decode_cc`](../rt-multiband/src/params.rs), `apply_midi` in
  [plugin.rs](../rt-multiband/src/plugin.rs)).
- **Latency reporting** — v1 is zero-latency (IIR crossover);
  `set_latency_samples(0)`. The v2 linear-phase mode will report its FIR delay
  through the same [`latency_samples`](../rt-multiband/src/engine.rs) hook.

**Tests.** The plugin `tests/` triad —
[`integration.rs`](../rt-multiband/tests/integration.rs) (bypass transparency,
bypassed-band reconstruction, per-band compression, solo/mute, independent
linking, live crossover retune), [`rt_safety.rs`](../rt-multiband/tests/rt_safety.rs)
(zero latency, finite/bounded, silent-tail decay, edge-case blocks) and
[`rt_alloc.rs`](../rt-multiband/tests/rt_alloc.rs) (allocation-free steady
state + per-block parameter changes + mono) — plus params/engine/ui unit
tests: **33 tests** across the crate.

## Phase 3: UI ✅

The editor is split the family way: pure, egui-free logic in
[`ui.rs`](../rt-multiband/src/ui.rs) (unit-tested — transfer curve, log-frequency
axis, spectrum heights, dB/GR/Hz formatters) and the `nih-plug-egui` drawing in
[`plugin.rs`](../rt-multiband/src/plugin.rs) behind `#[cfg(feature = "gui")]`.
The audio thread publishes read-only telemetry through packed atomics (per-band
GR, in/out peak, a log-spaced spectrum array); the display-only
[`SpectrumProbe`](../rt-multiband/src/spectrum.rs) FFT runs **only while the
editor is open** (`editor_state.is_open()` gate).

- **`nih-plug-egui`** — **decision: egui**, matching every other plugin in
  the workspace (`rt-limiter`, `rt-vox`, `rt-pitch`); `nih-plug-iced` not used.
- **Spectrum display** with **draggable crossover lines** — log-frequency
  magnitude bars from the [`SpectrumProbe`](../rt-multiband/src/spectrum.rs),
  with three crossover lines the user drags to set `Crossover 1/2/3`
  (`begin/set/end_set_parameter`, nearest-line grab).
- **Per-band gain-reduction meters** — a vertical GR bar per band, fed by
  the `band_gr_db` telemetry atomics.
- **Per-band transfer-curve display** (threshold / ratio / knee / mode) —
  [`ui::transfer_output_db`](../rt-multiband/src/ui.rs) sampled across the
  threshold range, drawn against the 1:1 diagonal.
- **Solo / Mute buttons per band** — `ParamSlider`s on the per-band
  `solo`/`mute` bool params.
- **Global meters**: input, output — horizontal dBFS bars from the
  `input_dbfs` / `output_dbfs` telemetry atomics.

**Verified.** `just check-gui rt-multiband` (Nix devshell with GL/X11) —
`cargo clippy --features gui -- -D warnings` clean. `ui.rs` / `spectrum.rs`
add 12 unit tests (log-axis round-trip, spectral-height floor/peak, transfer
curve = compressor acceptance, spectrum tone-lands-in-its-bin); **45 tests**
across the crate.

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
