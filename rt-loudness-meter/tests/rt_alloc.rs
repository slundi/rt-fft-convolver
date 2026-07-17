//! Zero-allocation audit for `MeterProcessor::process` (Roadmap-10 Phase 4:
//! "Zero allocations in `process()`").
//!
//! Installs a **counting global allocator** and asserts that a steady-state
//! render — feeding the K-weighting filters, the gated integrator (including
//! the 100 ms boundaries that increment the histograms), the true-peak
//! detectors, and the lock-free broadcast publish — allocates nothing.  It
//! lives in its own binary so the counting allocator is isolated from the other
//! test binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_loudness_core::MeterBroadcast;
use rt_loudness_meter::{LoudnessParams, MeterProcessor};

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

const SAMPLE_RATE: u32 = 48_000;
const BLOCK: usize = 256;
/// A light true-peak half-band design keeps construction cheap.
const LIGHT: (f64, f64) = (80.0, 0.1);

#[test]
fn process_does_not_allocate_in_steady_state() {
    let broadcast = Arc::new(MeterBroadcast::new());
    let mut processor =
        MeterProcessor::with_true_peak_design(2, SAMPLE_RATE, BLOCK, broadcast, LIGHT.0, LIGHT.1);
    let params = LoudnessParams::default();

    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];

    // Warm up outside the measured region: fill the sub-block ring, cross a few
    // 100 ms boundaries so histogram bins are first-touched, force lazy work.
    for block_index in 0..64 {
        fill_tone(&mut left, block_index);
        fill_tone(&mut right, block_index);
        processor.process(&[&left, &right], &params);
    }

    // Measure: steady-state blocks (including the frames that roll a 100 ms
    // sub-block and bump the gated histograms) must allocate nothing.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block_index in 64..(64 + 256) {
        fill_tone(&mut left, block_index);
        fill_tone(&mut right, block_index);
        processor.process(&[&left, &right], &params);
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);

    assert!(
        processor.values().integrated_lufs.is_finite(),
        "test did not accumulate a measurement"
    );
    assert_eq!(
        after,
        before,
        "process() allocated {} time(s) in steady state",
        after - before
    );
}

/// Fill `buffer` with a 1 kHz tone at −20 dBFS (no allocation).
fn fill_tone(buffer: &mut [f32], block_index: usize) {
    let amplitude = 0.1f32; // ≈ −20 dBFS
    for (offset, sample) in buffer.iter_mut().enumerate() {
        let time = (block_index * BLOCK + offset) as f32 / SAMPLE_RATE as f32;
        *sample = amplitude * (std::f32::consts::TAU * 1_000.0 * time).sin();
    }
}
