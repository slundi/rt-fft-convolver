---
name: rt-safety-audit
description: Add or run the real-time-safety test triad for a plugin engine — zero-allocation audit, denormal/silence-to-zero + latency checks, and end-to-end integration tests. Use when writing Phase 4 tests, when adding tests for a new `*Processor`, or when the user asks to verify RT-safety / no-allocation / denormal handling.
---

# RT-safety audit (the `tests/` triad)

Every plugin ships three test binaries under `rt-<name>/tests/`. They exercise
the public `*Controller` / `*Processor` end-to-end. Copy `rt-vox/tests/*` or
`rt-limiter/tests/*` and adapt the assertions.

## `rt_alloc.rs` — zero allocations in `process()`
Installs a **counting global allocator** and asserts a steady-state render
allocates nothing — including on MIDI note-on (the Roadmap Phase-4 wording).
It lives in its own binary so the global allocator is isolated.
```rust
#[global_allocator] static ALLOCATOR: CountingAllocator = CountingAllocator;
// alloc/realloc bump an AtomicUsize; dealloc doesn't.
```
Method: build the **busiest** config off the measured path, pre-allocate the
audio buffers, warm up ~12 blocks, then assert `ALLOC_COUNT` is unchanged across
~128 steady-state blocks (fire a `note_on`/`note_off` mid-loop). See
`rt-vox/tests/rt_alloc.rs`.

## `rt_safety.rs` — denormals, latency, robustness
- **Silence → true zero:** feed silence for ~200 blocks; every output sample
  `abs() < 1e-12`. Proves the `DenormalGuard` flushed near-silent FFT tails
  (do this per mode for multi-mode plugins).
- **Reported latency matches the build:** `processor.reported_latency()` ==
  `processor.latency_samples(mode)` after a render.
- **Varying block sizes:** render `[max, 64, 1, 200, max]`-length blocks without
  panicking; output stays finite.
- **Bypass is bit-transparent:** output equals input exactly.

## `integration.rs` — behaviour end-to-end
Use a **Goertzel** power probe (`power_at(signal, hz)`) or zero-crossing count —
no external reference vectors. Cover the roadmap's Phase-4 acceptance items, e.g.
- pitch tracker: 440 Hz sine → 440 Hz ± tolerance;
- vocoder: silent modulator → silence; carrier gated by the modulator envelope;
- harmoniser: MIDI note → the right interval; note-off silences the voice;
- formant shift: pitch preserved (fundamental dominates neighbours).

## Engine-invariant reminders these tests defend
- `process_block` / `process`: no `alloc`, no `lock`, no `panic`; all storage
  reserved in `new()`.
- Wrap FFT/near-silent paths in `rt_fft_convolver::DenormalGuard::new()`.
- MIDI note handling mutates only fixed-size arrays (no `Vec` growth).

## Run
```sh
cargo test -p rt-<name>          # all three binaries
cargo test -p rt-<name> --test rt_alloc   # the allocation audit alone
```
Also part of host validation (Phase 4/5): `lv2lint`, `clap-validator`,
`pluginval` against the bundles from the package-plugin skill.
