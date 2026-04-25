# Roadmap: Rust Real-Time Convolution Library

This roadmap outlines the development of a zero-latency, high-performance convolution engine for guitar effects.

## Phase 1: Core Mathematical Foundation (dsp/ & utils/)
- [ ] **FFT Abstraction (`fft_handler.rs`)**:
  - Integrate `rustfft`.
  - Create a wrapper to handle forward and inverse FFTs for real-to-complex signals.
- [ ] **Direct Form Convolution (`direct.rs`)**:
  - Implement a simple time-domain convolution for the first block (head) of the Impulse Response.
  - Goal: Achieve 0-sample latency for the initial attack.
- [ ] **Anti-Denormal Protection (`denormals.rs`)**:
  - Implement FTZ (Flush To Zero) logic to prevent CPU spikes during silent passages.

## Phase 2: Partitioned Engine (engine/)
- [ ] **Uniform Partitioning (`partition.rs`)**:
  - Split long IRs into equal blocks of size `N`.
  - Implement the "Overlap-Save" or "Overlap-Add" algorithm.
- [ ] **Frequency Domain Delay Line**:
  - Create a buffer to store past frequency blocks and multiply them with IR partitions.
- [ ] **Stereo & True Stereo Support**:
  - Parallel processing for Left/Right channels.
  - Support for 4-channel convolution (L->L, L->R, R->L, R->R).

## Phase 3: Audio Professional Features (utils/ & lib.rs)
- [ ] **Resampling Engine (`resampler.rs`)**:
  - Automatically resample IR files (WAV) to match the host's sample rate (44.1k, 48k, 96k).
- [ ] **Normalization & Gain Staging (`mixer.rs`)**:
  - Auto-calculate IR loudness to prevent clipping.
  - Implement a "Dry/Wet" mix control.

## Phase 4: Optimization & Studio Mode
- [ ] **SIMD Optimization**:
  - Use `packed_simd` or auto-vectorization for complex multiplications.
- [ ] **Offline High-Fidelity Mode**:
  - Create a non-partitioned, high-precision processing path for studio rendering.
- [ ] **Cross-fading IR Switch**:
  - Allow switching between cabinets (Cab A to Cab B) without audio clicks.
