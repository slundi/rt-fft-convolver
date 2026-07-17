//! Zero-allocation audit for `AdaptiveProcessor::process` (Roadmap-09 Phase 4:
//! "Zero allocations in `process()`, including on notch allocation").
//!
//! This test binary installs a **counting global allocator** and asserts that a
//! steady-state render — including the moment the feedback detector places a new
//! notch — performs no heap allocation.  It lives in its own file so the
//! counting allocator is isolated from the other test binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_adaptive::params::Mode;
use rt_adaptive::{AdaptiveController, AdaptiveParams};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

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
fn process_and_notch_allocation_do_not_allocate() {
    // Feedback mode fed a strong howl so a notch is placed inside the measured
    // region — notch allocation must itself be allocation-free.
    let params = AdaptiveParams {
        mode: Mode::FeedbackSuppressor,
        sensitivity: 0.8,
        max_notches: 24,
        ..Default::default()
    };
    let (_controller, mut processor) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 2, &params);

    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];

    // Warm up outside the measured region (fill the detector ring, force any
    // first-call lazy work).
    for block_index in 0..8 {
        fill_howl(&mut left, block_index);
        fill_howl(&mut right, block_index);
        let mut main: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut main, None, &params);
    }

    // Measure: steady-state blocks (including the frames that place notches)
    // must allocate nothing.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block_index in 8..(8 + 128) {
        fill_howl(&mut left, block_index);
        fill_howl(&mut right, block_index);
        let mut main: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut main, None, &params);
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);

    assert!(
        processor.active_notches() >= 1,
        "test did not exercise a notch allocation"
    );
    assert_eq!(
        after,
        before,
        "process() allocated {} time(s) in steady state",
        after - before
    );
}

/// Fill `buffer` with a strong 2 kHz howl (no allocation).
fn fill_howl(buffer: &mut [f32], block_index: usize) {
    for (n, sample) in buffer.iter_mut().enumerate() {
        let t = (block_index * BLOCK + n) as f32 / SAMPLE_RATE;
        *sample = (std::f32::consts::TAU * 2_000.0 * t).sin();
    }
}
