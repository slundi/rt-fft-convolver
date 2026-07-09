# Roadmap: Rust Real-Time Convolution Library

This roadmap outlines the development of a zero-latency, high-performance convolution engine for guitar effects.

## Phase 1: Core Mathematical Foundation (dsp/ & utils/)
- [x] **FFT Abstraction (`fft_handler.rs`)**:
  - Integrate `rustfft`.
  - Create a wrapper to handle forward and inverse FFTs for real-to-complex signals.
- [x] **Direct Form Convolution (`direct.rs`)**:
  - Implement a simple time-domain convolution for the first block (head) of the Impulse Response.
  - Goal: Achieve 0-sample latency for the initial attack.
- [x] **Anti-Denormal Protection (`denormals.rs`)**:
  - Implement FTZ (Flush To Zero) logic to prevent CPU spikes during silent passages.

## Phase 2: Partitioned Engine (engine/)
- [x] **Uniform Partitioning (`partition.rs`)**:
  - Split long IRs into equal blocks of size `N`.
  - Implement the "Overlap-Save" or "Overlap-Add" algorithm.
- [x] **Frequency Domain Delay Line**:
  - Create a buffer to store past frequency blocks and multiply them with IR partitions.
- [x] **Stereo & True Stereo Support**:
  - Parallel processing for Left/Right channels.
  - Support for 4-channel convolution (L->L, L->R, R->L, R->R).

## Phase 3: Audio Professional Features (utils/ & lib.rs)
- [x] **Resampling Engine (`resampler.rs`)**:
  - Automatically resample IR files (WAV) to match the host's sample rate (44.1k, 48k, 96k).
- [x] **Normalization & Gain Staging (`mixer.rs`)**:
  - Auto-calculate IR loudness to prevent clipping.
  - Implement a "Dry/Wet" mix control.

## Phase 4: Optimization & Studio Mode
- [x] **SIMD Optimization**:
  - Use ~~`packed_simd`~~ `simsimd` or auto-vectorization for complex multiplications.
- [x] **Offline High-Fidelity Mode**:
  - Create a non-partitioned, high-precision processing path for studio rendering.
- [x] **Cross-fading IR Switch**:
  - Allow switching between cabinets (Cab A to Cab B) without audio clicks.

## Phase 5: Follow-up performance / robustness
- [x] **Hybrid zero-latency engine**:
  - `HybridConvolver` type that wraps `DirectConvolver` + a delayed
    `UniformPartitionEngine` so `head + tail` composition doesn't
    double-count taps.
- [ ] **`realfft` for the render FFT**:
  - Replace `rustfft`'s complex-in-complex-out plan with a real-to-complex
    plan.  Cuts inner-loop memory and roughly halves the FFT / IFFT cost.
    Requires updating `FftHandler`, `UniformPartitionEngine`, and the
    `FreqDomainDelayLine` spectrum length (`N + 1` instead of `2N`).
- [ ] **Split-real/imag IR storage for the complex-MAC**:
  - Pre-compute IR partitions as separate re/im vectors so the SIMD kernel
    can use plain vector loads instead of interleaved shuffle patterns.
    Needs a criterion-based benchmark harness first to validate the gain.
- [ ] **Serial denormal-guard tests**:
  - `x86_ftz_daz_bits_are_set_and_restored` mutates thread-global MXCSR and
    can race with `nested_guards_restore_state` under `cargo test` default
    parallelism.  Gate FTZ-mutating tests behind a `serial_test` lock (or
    a hand-rolled `std::sync::Mutex`) so they run one at a time.
- [ ] **Resampler output-buffer reuse**:
  - `resample()` calls `vec![0.0; process_all_needed_output_len]` then
    `truncate()`s the tail.  On multi-channel loads this is one allocation
    per channel; switch to `Vec::with_capacity` + `resize()` + `truncate()`
    to skip the double-init cost.  IR loading is non-RT so this is minor,
    but matters for large multi-channel IRs.
