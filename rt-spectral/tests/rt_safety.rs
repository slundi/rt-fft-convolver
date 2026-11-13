//! RT-safety audit for `SpectralProcessor::process` (Roadmap-04 Phase 4:
//! "Zero allocations in `process()`").
//!
//! This test binary installs a **counting global allocator** and asserts that a
//! steady-state render — including the per-block telemetry publish and the
//! learn-capture recorder path — performs no heap allocation.  It lives in its
//! own file so the counting allocator (and the single-threaded measurement
//! window it depends on) is isolated from the other test binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_spectral::spectral_core::{
    ClickParams, CompressParams, DenoiseParams, GateParams, HumParams, SpectralParams,
    TransientParams,
};
use rt_spectral::spectral_engine;

/// Global allocation counter, incremented on every `alloc` / `realloc`.
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A `System`-backed allocator that tallies allocations for the audit.
struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const SR: u32 = 48_000;
const FFT: usize = 2048;
const BLOCK: usize = 256;

/// Every operator enabled — the busiest configuration, so every per-bin and
/// time-domain stage plus the telemetry publish is exercised on the measured
/// path.
fn all_on() -> SpectralParams {
    SpectralParams {
        click: ClickParams {
            enabled: true,
            sensitivity: 4.0,
            max_len: 8,
        },
        denoise: DenoiseParams {
            enabled: true,
            reduction_db: 18.0,
            sensitivity_db: 3.0,
        },
        gate: GateParams {
            enabled: true,
            threshold_db: -50.0,
            range_db: 24.0,
            attack_ms: 5.0,
            release_ms: 100.0,
        },
        compress: CompressParams {
            enabled: true,
            threshold_db: -24.0,
            ratio: 3.0,
            knee_db: 6.0,
            makeup_db: 3.0,
        },
        hum: HumParams {
            enabled: true,
            fundamental_hz: 50.0,
            harmonics: 3,
            q: 30.0,
        },
        transient: TransientParams {
            enabled: true,
            sensitivity: 1.5,
        },
        mix: 0.8,
        output_gain_db: 2.0,
        bypass: false,
    }
}

/// Fill `buffer` in place with a 220 Hz sine (no allocation).
fn fill_tone(buffer: &mut [f32], block_index: usize) {
    for (n, sample) in buffer.iter_mut().enumerate() {
        let t = (block_index * BLOCK + n) as f32 / SR as f32;
        *sample = 0.4 * (std::f32::consts::TAU * 220.0 * t).sin();
    }
}

/// Measure allocations while rendering `blocks` blocks in place, starting at
/// `start_block`.  Buffers are reused (allocated by the caller off the measured
/// path); the returned count is allocations attributable to `process`.
fn count_allocs(
    processor: &mut rt_spectral::SpectralProcessor,
    params: &SpectralParams,
    left: &mut [f32],
    right: &mut [f32],
    start_block: usize,
    blocks: usize,
) -> usize {
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for offset in 0..blocks {
        fill_tone(left, start_block + offset);
        fill_tone(right, start_block + offset);
        let mut channels: [&mut [f32]; 2] = [left, right];
        processor.process(&mut channels, params);
    }
    ALLOC_COUNT.load(Ordering::Relaxed) - before
}

/// Steady-state render + the Learn capture path are both allocation-free on the
/// audio thread.
///
/// A **single** test owns the whole measured region: the counting allocator is
/// process-wide across threads, so two allocator tests running concurrently
/// would cross-contaminate.  Keeping one sequential test (as `rt-pitch` does)
/// makes the measurement window unambiguous.
#[test]
fn process_is_allocation_free_on_the_audio_thread() {
    // Keep the telemetry endpoint alive for the whole test so nothing is
    // dropped (deallocated) mid-measurement.
    let (_controller, mut processor, _telemetry) = spectral_engine(SR, FFT, BLOCK, 2);
    let all_on = all_on();

    // Pre-allocate the audio buffers off the measured path and reuse them.
    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];

    // Warm up past the pipeline's one-time lazy allocations (FFT internals, the
    // first-enabled-operator configuration).
    for block_index in 0..64 {
        fill_tone(&mut left, block_index);
        fill_tone(&mut right, block_index);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &all_on);
    }

    // Steady state, busiest configuration: 128 blocks must allocate nothing.
    let steady = count_allocs(&mut processor, &all_on, &mut left, &mut right, 64, 128);
    assert_eq!(
        steady, 0,
        "process() allocated {steady} time(s) in steady state"
    );

    // The Learn capture path (arm + in-place record + completion flag) is
    // likewise allocation-free — only the off-RT controller learn allocates,
    // and the audio thread never calls it.
    processor.begin_learn(BLOCK * 32);
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for offset in 0..64 {
        fill_tone(&mut left, 192 + offset);
        fill_tone(&mut right, 192 + offset);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &all_on);
        let _ = processor.take_pending_learn();
    }
    let during_learn = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        during_learn, 0,
        "Learn capture allocated {during_learn} time(s) on the audio thread"
    );
}
