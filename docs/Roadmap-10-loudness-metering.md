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
  **Decision (Phase 0): mono + stereo in v1; 5.1 / 7.1 deferred to v2**
  — see Phase 0 below.
- **RT-safe** — filter states are per-channel biquads, integration
  buffers are ring-buffers.
- **History logging** off-RT for offline review of a session.

## Prerequisites in `rt-fft-convolver` ✅

- `DenormalGuard` — present and tested
  ([src/utils/denormals.rs](../src/utils/denormals.rs)); RAII FTZ/DAZ
  guard, re-exported at the crate root
  ([src/lib.rs](../src/lib.rs)). Wraps the K-weighting filter and
  mean-square integrator so a long silent passage (a paused session,
  a fade-out tail) never drives the biquad states or the running-sum
  accumulators into denormal arithmetic.
- Polyphase resampler for 4× true-peak — **present, and one layer
  better than the prerequisite asked for.** `rt-drive-core` already
  exposes the Kaiser half-band polyphase
  [`Oversampler`](../rt-drive-core/src/resampler.rs) (the "shared
  crate" the note anticipated), **and**
  [`rt_limiter_core::TruePeakDetector`](../rt-limiter-core/src/detector.rs)
  already wraps it into a tested BS.1770-style inter-sample-peak
  envelope (`detect_block` → max `|x|` over the oversampled
  sub-samples). The meter reuses that primitive directly rather than
  re-implementing 4× true-peak — see the true-peak decision in Phase 0.
- `db_to_linear` / `linear_to_db` dB helpers — present and exported;
  used for the dBFS / dBTP readouts and the target-compliance
  thresholds.

## Phase 0: Compliance decisions ✅

The organising Phase 0 finding is that **every listed "standard" shares
one measurement algorithm — BS.1770-4 — and differs only in its target
numbers.** R128, the streaming specs, and the broadcast specs are not
distinct DSP; they are distinct *presets* (a target LUFS + a dBTP
ceiling) over a single gated-K-weighted-loudness engine. That collapses
the standards decision into "implement BS.1770-4 once, ship the rest as
Phase 5 presets," and keeps v1 inside its analysis-only, zero-latency,
reference-compliant design principles. The second finding is that the
two heaviest DSP pieces — 4× true-peak oversampling and the K-weighting
biquads — already exist tested in sibling `*-core` crates, so this
plugin reuses rather than re-implements.

Cost figures below are per instance at 48 kHz; "block" means the
host's processing block (≤ 512 samples assumed for sizing).

- **Standards to comply with** — **decision: implement ITU-R BS.1770-4
  as the sole measurement engine; ship EBU R128, the streaming targets,
  and the broadcast targets as presets (Phase 5) over it.**
  - *Rationale.* BS.1770-4 is mandatory and non-negotiable (it is the
    Phase 4 quality gate). R128 (−23 LUFS, −1 dBTP), Spotify/YouTube/
    Tidal/Amazon (−14), Apple Music (−16), podcast (−16), broadcast
    (−23) all *measure* loudness identically; they only change the
    target line the compliance LED compares against. AES71 is a
    broadcast-interconnect/metadata context, not a separate loudness
    maths — treated as a target/label, not a second algorithm.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **One BS.1770-4 engine + preset targets** (chosen) | 1× engine; presets are ~2 floats each (negligible) | 1× engine cost | Low — one gated-loudness path, exhaustively test-vector-verified once |
    | Per-standard measurement paths | N× integrator/histogram state | N× (redundant) | High — N code paths to keep in sync with the test vectors; no upside |

  - *Example.* Switching `Target` from EBU R128 to Spotify changes the
    compliance thresholds from (−23 LUFS, −1 dBTP) to (−14 LUFS,
    −1 dBTP); the momentary/short-term/integrated/LRA/true-peak numbers
    themselves are byte-for-byte identical — only the LED colour logic
    moves.

- **True-peak measurement** — **decision: reuse
  `rt_limiter_core::TruePeakDetector` (4× oversampling) per channel**
  rather than re-implement the polyphase path, per the workspace
  "reuse over re-implementation" rule.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Reuse `TruePeakDetector` @ 4×** (chosen) | scratch `max_block·4` f32 + half-band FIR state per channel (~8 KB/ch @ 512) | dominant per-sample cost of the whole meter, but a fraction of a limiter (no gain stage); est. well under 1 % | Low — tested primitive; `detect_block` gives the per-sample envelope, we peak-hold its max |
    | Depend on `rt_drive_core::Oversampler` directly, write the max-abs loop here | same scratch | same | Low-med — ~10 lines duplicated from the limiter; no benefit (metering ignores the latency-alignment that `TruePeakDetector` adds) |
    | 2× oversampling | half the scratch | ~half the true-peak CPU | Low, but BS.1770-4 specifies ≥ 4× — fails the 0.1 dBTP Phase 4 test |

  - *Note.* Metering is analysis-only, so the detector's reported group
    delay is irrelevant here (we want the max, not sample-aligned gain
    reduction) — a further reason the tested primitive drops straight in.

- **Metering windows to expose** — **decision: ship all listed windows
  — Momentary (400 ms), Short-term (3 s), Integrated (gated session),
  LRA, True peak (dBTP), Sample peak (dBFS) — plus RMS** (the UI in
  Phase 3 already lists RMS). They are cheap because they share the
  per-100 ms-block mean-square rather than each buffering raw samples.
  - *Implementation that makes them cheap.* K-weighted samples are
    accumulated into a running mean-square over **100 ms gating blocks**
    (BS.1770's grid). Momentary = mean of the last **4** block energies;
    short-term = last **30**. Integrated and LRA read a **histogram of
    block/short-term loudness** (0.1 LU bins, ≈ 700 bins spanning −70…0
    LUFS) so the relative gate and the 10th–95th-percentile spread are
    O(1) to update and need no unbounded log on the RT thread.
  - *Options considered.*

    | Window | Memory (per channel/meter) | CPU load | Complexity |
    |---|---|---|---|
    | Momentary / Short-term via **100 ms block sums** (chosen) | ring of 30 f64 block energies (~240 B) | 1 mul-add/sample + one divide/100 ms | Low |
    | Momentary / Short-term via **raw sample rings** (rejected) | 400 ms → ~19 k f32/ch (~77 KB); 3 s → ~144 k f32/ch (~576 KB) | same | Low, but ~100× the memory for no accuracy gain |
    | Integrated + LRA via **histogram** (chosen) | ~700 bins ×4 B ≈ 2.8 KB, fixed | O(1) per block | Med — gate logic (abs −70 LUFS, rel −10 LU) |
    | Integrated via **full block log** (rejected) | unbounded, grows with session | O(1) append, O(n) recompute | RT-unsafe (grows / reallocates) |

  - *Example.* At 48 kHz a 100 ms block is 4800 K-weighted squared
    samples summed into one f64; the integrated meter's histogram bin
    for that block's loudness is incremented; nothing per-sample is
    retained, so a 3-hour session costs the same RAM as a 3-second one.

- **Channel configurations** — **decision: mono + stereo in v1; 5.1 /
  7.1 deferred to v2.** BS.1770 channel weighting (C/L/R = 1.0, LFE
  excluded, Ls/Rs = +1.5 dB ≈ 1.41 linear) is designed in from the
  start so the v1 summation is the surround formula with the surround
  weights set to the stereo subset — v2 is added channels, not a
  rewrite.
  - *Options considered.*

    | Option | Memory | CPU load | Complexity |
    |---|---|---|---|
    | **Mono + stereo v1** (chosen) | 1–2× per-channel state (biquads + true-peak scratch) | 1–2× true-peak (the multiplier) | Low — covers music/streaming/podcast, the bulk of users |
    | 5.1 / 7.1 in v1 | 6–8× per-channel state (~48–64 KB true-peak scratch) | 6–8× true-peak cost | Med-high — host **channel-layout negotiation** (nih-plug `AudioIOLayout`), surround bus plumbing, and multichannel test vectors; broadcast-only audience |

  - *Rationale.* True-peak oversampling is the per-channel CPU
    multiplier, so surround multiplies the meter's only non-trivial
    cost 6–8×; deferring it keeps v1 light and ships the 90 % case
    sooner. The weighted-sum is written surround-shaped now to make v2
    additive. (This resolves the same `[decision]` in the Phase 2 I/O
    section.)

- **UI framework** — **decision: `nih-plug-egui`**, matching every other
  plugin in the workspace (`rt-limiter`, `rt-adaptive`, … all use
  `nih_plug_egui`) so the shared [`plugin-editor`](../.claude/skills)
  procedure and the lock-free telemetry pattern apply unchanged.
  (Resolves the `[decision]` in Phase 3.) `nih-plug-iced` was the
  alternative but would be the only iced editor in the repo — extra
  build surface and no reuse.

- **Spectral / vectorscope / correlation meter** — **decision: out of
  scope for v1** (kept in the roadmap's out-of-scope list). They are not
  part of BS.1770 and would add an FFT/analysis path unrelated to the
  loudness gate; revisit only on request.

## Phase 1: `rt-loudness-core` (engine crate) ✅

Engine crate built, unit- and integration-tested (33 tests), clippy- and
machete-clean, and wired into the workspace `members` + `default-members`.  The
full-scale-stereo-sine ≈ 0 LUFS BS.1770 calibration check passes end-to-end.

- **Workspace layout**:
  - [`rt-loudness-core`](../rt-loudness-core) — measurement engine (created).
    `rt-loudness-meter` — plugin (deferred to Phase 2).
- **K-weighting filter** (BS.1770-4)
  ([kweight.rs](../rt-loudness-core/src/kweight.rs)):
  - Two biquad stages (shelving pre-filter + RLB high-pass), transposed
    direct-form II, per-channel [`KWeightingFilter`].
  - Sample-rate-independent (De Man / `libebur128` bilinear design); a unit
    test asserts the 48 kHz coefficients match the standard's table.
- **Mean-square integrator**
  ([integrator.rs](../rt-loudness-core/src/integrator.rs),
  [lra.rs](../rt-loudness-core/src/lra.rs)):
  - 400 ms momentary + 3 s short-term over a fixed 100 ms sub-block ring (no
    raw-sample buffering — O(1) memory in session length).
  - Gated running mean for integrated via a fixed [`LoudnessHistogram`]
    (absolute −70 LUFS + relative −10 LU gates, BS.1770-4).
- **True-peak detector**
  ([truepeak.rs](../rt-loudness-core/src/truepeak.rs)):
  - 4× oversampling with peak-hold — **reuses**
    [`rt_limiter_core::TruePeakDetector`] (Phase 0 decision), one per channel,
    plus a sample-peak (dBFS) hold alongside.
- **LRA (loudness range)**
  ([lra.rs](../rt-loudness-core/src/lra.rs)):
  - Histogram of short-term values, gated (−20 LU) 10th–95th percentile spread
    per EBU Tech 3342, sharing the [`LoudnessHistogram`] machinery.
- **Result-broadcast channel**
  ([broadcast.rs](../rt-loudness-core/src/broadcast.rs)):
  - [`MeterBroadcast`] — seven `f32`-packed `AtomicU32`s; the audio thread
    `store`s [`MeterValues`] wait-free, the off-RT UI / logger `load`s the
    latest set.  [`LoudnessMeter::publish`] wires the whole chain to it.

## Phase 2: `rt-loudness-meter` plugin ✅

Plugin crate built ([rt-loudness-meter](../rt-loudness-meter)), wired into the
workspace, unit- and triad-tested (23 tests), clippy-clean on both the default
and `--features nih` builds, and machete-clean.  Analysis-only: audio passes
through bit-for-bit (a `rt_safety` test asserts it).  There is **no**
`*Controller` / `ir_handoff` — nothing reallocates on a parameter change, so a
resident [`MeterProcessor`](../rt-loudness-meter/src/engine.rs) is built once in
`initialize`.

- [x] **I/O** ([plugin.rs](../rt-loudness-meter/src/plugin.rs)):
  - Stereo in → stereo out and mono in → mono out (pass-through).
  - Multichannel — **v2** (5.1 / 7.1); see Phase 0 channel-config
    decision.
- [x] **Meter controls**
  ([params.rs](../rt-loudness-meter/src/params.rs)):
  - `Target` (dropdown: EBU R128 / Streaming −14 / Apple·Podcast −16 /
    Custom) — the presets collapse Spotify/YouTube/Tidal/Amazon (all −14)
    into one entry per the Phase-0 "one engine, preset targets" decision;
    [`Target::thresholds`] resolves the (LUFS, dBTP) pair.
  - `Custom Target` (`custom_target_lufs` + `custom_target_dbtp`, used when
    `Target = Custom`).
  - `Reset` (rising-edge clears integrated + LRA + peak holds).
  - `Start / Stop` (`running`) — freezes integrated + LRA while momentary /
    short-term / peaks stay live (new [`LoudnessMeter::set_integrating`]).
  - `Bypass` (formal no-op — audio is never modified — present for LV2).
- [x] **Logging controls**
  ([engine.rs](../rt-loudness-meter/src/engine.rs)):
  - `Log` bool → the off-RT [`MeterLogger`](../rt-loudness-meter/src/engine.rs)
    worker polls the lock-free broadcast into a CSV / JSON file; started /
    stopped via the host background executor, never on the audio thread.  The
    file-*path* picker lands with the Phase 3 editor (a `#[persist]` path field
    is already in place); until a path is set, `Log` is inert.
- [x] **MIDI**:
  - `Reset` and `Start / Stop` are automatable `BoolParam`s a host binds to a
    footswitch / transport CC via MIDI learn (the idiomatic `nih-plug` path),
    so `MIDI_INPUT` stays `None` — matching the sibling `rt-limiter` /
    `rt-adaptive` decision.
- [~] **Reporting**:
  - Meter values are published every block through the lock-free
    [`MeterBroadcast`] (the editor and logger read it).  `nih-plug` has no
    first-class *output-parameter* concept, so exposing the readings as
    DAW-automatable host parameters is not supported by the framework —
    deferred / N-A; the broadcast is the reporting channel.

## Phase 3: UI ✅

Editor built behind the `gui` feature, with the pure drawing/format logic in
[ui.rs](../rt-loudness-meter/src/ui.rs) (13 unit tests) and the `egui` drawing
in [plugin.rs](../rt-loudness-meter/src/plugin.rs).  The editor reads the
lock-free [`MeterBroadcast`] each frame — no extra telemetry atomics were
needed, since the broadcast already carries every reading.  `gui` clippy is
clean under `just check-gui rt-loudness-meter`.

- [x] **`nih-plug-egui`** — the workspace framework (Phase 0); persisted
  `#[persist = "editor-state"] editor_state: Arc<EguiState>`.
- [x] **Large numeric readouts** — momentary, short-term, integrated LUFS, LRA,
  true peak (dBTP), sample peak (dBFS), RMS, via [`ui::format_lufs`] /
  [`format_lu`] / [`format_dbtp`] / [`format_dbfs`] (silence → `−∞`).
- [x] **Meter graph** — scrolling short-term LUFS over
  [`LoudnessHistory`](../rt-loudness-meter/src/ui.rs) (600 samples ≈ 60 s at the
  100 ms short-term cadence; the editor pushes at 10 Hz regardless of repaint
  rate), with the target line drawn across it.
- [x] **Histogram** — the recent short-term distribution via
  [`ui::histogram_bins`] (drawn from the same history ring rather than plumbing
  the core's gated histogram through telemetry).
- [x] **Target-compliance LED** — green / amber / red from
  [`ui::compliance`]: within ±0.5 LU of target → green, ±2 LU → amber, else red;
  the true-peak ceiling is a hard limit (over → red regardless of loudness).
- [x] **Playhead-integration** — [`ui::effective_running`] holds integrated /
  LRA while the host transport is stopped; a host that reports no position
  (`pos_samples() == None`, e.g. a live monitor) leaves the manual `Start/Stop`
  in charge.  (Auto-*reset* on transport rewind is a later refinement; manual
  `Reset` covers it for now.)

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
- Spectral / vectorscope / correlation meter — **out of scope for v1**
  (Phase 0); common companion features but not part of BS.1770.  Add
  later on request.
- Dolby dialogue-intelligence metering — proprietary, out of scope.
- ATSC A/85 (US broadcast) — same math family as R128; add later
  if requested.
