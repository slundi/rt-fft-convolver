//! Zero-allocation audit for `PitchProcessor::process` (Roadmap-03 Phase 4:
//! "Allocator hook: zero allocations in `process()`").
//!
//! This test binary installs a **counting global allocator** and asserts that
//! a steady-state render performs no heap allocation.  It lives in its own file
//! so the counting allocator — and the single-threaded measurement window it
//! depends on — is isolated from the other test binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_pitch::{PitchController, PitchParams};

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

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK: usize = 256;

#[test]
fn process_does_not_allocate_in_steady_state() {
    // Stereo, dual-voice, formant on — the busiest configuration, so every
    // per-channel buffer and the formant estimator are exercised.  All of it
    // is built here, off the measured path.
    let params = PitchParams {
        current_preset: 1,
        voice2_preset: 3,
        dual_voice: true,
        mix: 0.7,
        formant_preservation: true,
        dry_delay_compensation: true,
        ..Default::default()
    };
    let (_controller, mut processor) = PitchController::new(BLOCK, SAMPLE_RATE, 2, &params);

    // Pre-allocate the audio buffers outside the measured region and reuse them
    // in place (so buffer creation never counts against the audit).
    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];

    // Warm up: settle the shifters and force any first-call lazy work.
    for block_index in 0..8 {
        fill_tone(&mut left, block_index);
        fill_tone(&mut right, block_index);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &params);
    }

    // Measure: 128 steady-state blocks must allocate nothing.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block_index in 8..(8 + 128) {
        fill_tone(&mut left, block_index);
        fill_tone(&mut right, block_index);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &params);
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);

    assert_eq!(
        after,
        before,
        "process() allocated {} time(s) in steady state",
        after - before
    );
}

/// Fill `buffer` in place with a 220 Hz sine (no allocation).
fn fill_tone(buffer: &mut [f32], block_index: usize) {
    for (n, sample) in buffer.iter_mut().enumerate() {
        let t = (block_index * BLOCK + n) as f32 / SAMPLE_RATE;
        *sample = 0.5 * (std::f32::consts::TAU * 220.0 * t).sin();
    }
}
