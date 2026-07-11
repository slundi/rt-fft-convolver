# Roadmap: `rt-neural-amp` (Neural Amp Modeling Plugin)

Real-time neural amp / pedal modelling plugin.  Runs a pre-trained
inference graph over each audio sample to emulate a specific piece of
gear (amp head, preamp, distortion pedal).

**Target model formats to support** — [decision]:
- NAM (`.nam` JSON, Steven Atkinson).
- Proteus / GuitarML (`.json` from Aida/GuitarML training).
- ONNX subset (for internal / third-party models).

**Reference implementations**: NAM, Proteus (GuitarML), Neural DSP
Nano Cortex, Tonex, Line 6 Helix Native, Kemper Profiler.

Plugin format target: **LV2 + CLAP + VST3** via
[`nih-plug`](https://github.com/robbert-vdh/nih-plug).

## Design principles

- **RT-safe inference**: no allocations, no locks, no dynamic
  reshaping on the audio thread.  All buffers sized at model load.
- **Sample-by-sample latency = 0**: no lookahead, no framing.  Some
  architectures (WaveNet) buffer internally — this is state, not
  lookahead.
- **Model hot-swap** off the audio thread, with click-free crossfade
  (like `CrossfadingConvolver`).
- **8 model slots**, same UX as `rt-ir-cab` / `rt-ir-reverb`
  (see [Roadmap-lv2.md](Roadmap-lv2.md)).  MIDI PC + CC + parameter
  automation all switch slots.
- **SIMD kernels** for dense matmul, conv1d, and activation
  functions.  Portable path first (`std::simd` or auto-vec), AVX2/NEON
  path second.
- **This plugin does NOT include cab simulation** — it pairs
  naturally with `rt-ir-cab` in the user's signal chain.

## Prerequisites in `rt-fft-convolver`

None strictly required.  Optional reuse:
- [ ] `DenormalGuard` from `utils/denormals.rs` — every inference tick
  must run with FTZ/DAZ set, since ReLU + saturation activations can
  produce denormals when the signal decays.
- [ ] `CrossfadingConvolver` design pattern — reuse for model swap
  (`from_model → to_model` linear crossfade over N samples).

## Phase 0: Algorithm & format decisions

- [ ] **Decide primary model architecture(s)** — [decision]:
  - LSTM (small, ~1 % CPU, NAM/Proteus).
  - WaveNet-style dilated conv1d (larger, higher fidelity).
  - Temporal Convolutional Network (TCN).
  - Multiple, with a runtime dispatcher.
- [ ] **Decide inference backend** — [decision]:
  - Hand-rolled Rust + `std::simd`.
  - [`candle`](https://github.com/huggingface/candle) with CPU
    backend.
  - [`tract`](https://github.com/sonos/tract) for ONNX.
  - `ndarray` + `matrixmultiply`.
- [ ] **Decide precision** — [decision]:
  - f32 only (safest).
  - f16 / bf16 for smaller CPU footprint on ARM.
- [ ] **Model file loader** (off-RT):
  - Parse chosen format(s) to an internal `PreparedModel` struct
    with pre-transposed weight matrices and layer descriptors.

## Phase 1: `rt-neural-core` (engine crate)

Internal crate, not published to crates.io initially.

- [ ] **Workspace layout**:
  - `crates/rt-neural-core` — inference engine.
  - `crates/rt-neural-amp` — plugin.
- [ ] **Layer primitives**, all `#[inline]` and allocation-free:
  - Dense (matmul + bias).
  - Conv1D (causal, dilated).
  - LSTM cell.
  - Activations: tanh, sigmoid, ReLU, GELU (LUT-based fast paths).
- [ ] **`PreparedModel`**:
  - Owns weight tensors in cache-friendly layout.
  - `process_sample(&mut self, x: f32) -> f32` — the RT entry point.
  - `reset()` — zeroes internal state (LSTM h/c, conv1d ring buffers).
- [ ] **SIMD kernels**:
  - Portable via `std::simd` first.
  - Feature-gated AVX2 / NEON overrides.
  - Benchmarked against a scalar baseline.
- [ ] **Slot bank** analogous to `IrSlotBank`:
  - 8 `Option<PreparedModel>` slots.
  - Off-RT load via `nih-plug` background task.
  - Lock-free swap via `arc-swap` or `triple_buffer`.

## Phase 2: `rt-neural-amp` plugin

- [ ] **I/O**:
  - Mono in → mono out (models are inherently mono).
  - Second instance for stereo (host places two on L/R).
- [ ] **Slot controls** (× 8):
  - `Model File N` (patch:message / file property).
  - `Slot Enabled N` (bool).
- [ ] **Global controls**:
  - `Current Slot` (int 0–7, automatable).
  - `Crossfade Time` (5–200 ms).
  - `Input Gain` (−24 … +24 dB) — critical: neural models are
    trained at a specific input level.
  - `Output Gain` (−24 … +24 dB).
  - `Mix` (0–100 %).
  - `Model Level Trim` (−12 … +12 dB, per-slot, non-automatable).
  - `Bypass`.
- [ ] **MIDI learn**: right-click → learn PC / CC on `Current Slot`.
- [ ] **CPU meter**: read-only property reporting inference time as
  a % of the audio block deadline.

## Phase 3: UI

- [ ] **`nih-plug-egui`** (or `nih-plug-iced`) — [decision, should
  match choice made for `rt-ir-cab`].
- [ ] **Slot rack**: 8 pads, drag-and-drop model loading.
- [ ] **Input meter** with a "sweet spot" indicator (models are
  input-level sensitive).
- [ ] **Model metadata display**: architecture, parameter count,
  training sample rate, source gear name.
- [ ] **MIDI activity indicator**.

## Phase 4: Testing

- [ ] **Unit tests** (`rt-neural-core`):
  - Layer primitives match a reference NumPy / PyTorch computation
    to within 1e-6.
  - Slot bank loads, swaps, releases without leaks.
  - Model hot-swap is bit-identical to a static load when
    `crossfade_ms == 0` and slots hold the same model.
- [ ] **Integration tests**:
  - Load a known NAM model, run a unit impulse, compare against a
    Python reference output (bit or near-bit exact per the
    reference implementation's tolerance).
- [ ] **RT-safety audit**:
  - Allocator hook in tests to assert zero allocations in
    `process()`.
  - Denormal guard active in every callback.
- [ ] **Host validation**:
  - `lv2lint`, `clap-validator`, `pluginval`.

## Phase 5: Packaging & release

- [ ] **CI**: Linux (x86_64 + aarch64), macOS universal, Windows
  x86_64.  Same matrix as `rt-ir-cab`.
- [ ] **Bundling**: `.lv2`, `.clap`, `.vst3`.
- [ ] **Installers**: Debian `.deb`, macOS signed `.pkg`, Windows
  `.exe`.
- [ ] **Sample models** — [decision]:
  - Ship a small pack of CC-licensed community models (ToneHunt has
    a permissive subset), or ship none and link to sources.
- [ ] **Documentation**: manual under `docs/`, input-level guide,
  known-good model index.

## Out of scope (future work)

- Cab simulation — use `rt-ir-cab` after this plugin.
- Training / capture workflow — out of scope for the plugin;
  reference NAM Trainer or Proteus Trainer.
- Non-guitar sources (bass has different training data; may work
  by chance but not a supported use case).
- GPU inference — audio latency requirements make CPU the right
  choice for real-time.
