//! Phase-4 RT-safety audit: the render path must not allocate.
//!
//! A counting global allocator wraps the system allocator; when *armed*, every
//! `alloc`/`realloc` bumps a counter.  We build and warm up a [`DriveStage`]
//! (allocation is expected there — it is off-thread setup), then arm the
//! counter and drive many blocks through `process_block` plus the live setters.
//! Any allocation in that window fails the test.
//!
//! Note: nothing inside the armed window may allocate — in particular no
//! `println!`/`format!` — or it would be (correctly) counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rt_drive_core::{DriveStage, PedalModel, Shaper, ToneConfig};

/// Allocations counted while [`ARMED`] is set.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
/// When true, the allocator counts allocations.
static ARMED: AtomicBool = AtomicBool::new(false);

struct CountingAllocator;

// SAFETY: this simply forwards every call to the system allocator, optionally
// incrementing a counter first; it never changes the returned pointers or
// layouts, so it upholds every `GlobalAlloc` invariant the `System` allocator
// does.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// `DriveStage::process_block` (and the live setters) must not allocate.
#[test]
fn process_block_is_allocation_free() {
    // A pedal that exercises every stage: oversampler, an asymmetric shaper,
    // DC blocker and a live 3-band tone stack.
    let pedal = PedalModel {
        name: "RtSafety".to_string(),
        input_gain_db: 12.0,
        shaper: Shaper::HardClip {
            threshold: 0.7,
            asymmetry: 0.2,
        },
        bias: 0.1,
        tone: ToneConfig::ThreeBand {
            bass_db: 2.0,
            mid_db: -1.0,
            treble_db: 3.0,
        },
        output_gain_db: -6.0,
        default_oversampling: 4,
    };
    // A light resampler design keeps the test quick; allocation behaviour is
    // identical to the default design.
    let block = 256;
    let mut stage = DriveStage::with_design(&pedal, 48_000.0, block, 80.0, 0.1);

    let input: Vec<f32> = (0..block)
        .map(|n| 0.5 * (std::f32::consts::TAU * 0.01 * n as f32).sin())
        .collect();
    let mut output = vec![0.0f32; block];

    // Warm up (off-thread setup / any first-call laziness) before arming.
    for _ in 0..8 {
        stage.process_block(&input, &mut output);
    }

    ARMED.store(true, Ordering::Relaxed);
    for index in 0..128 {
        // Continuous controls, applied every block as the plugin would.
        stage.set_input_gain_db(6.0 + index as f32 * 0.01);
        stage.set_output_gain_db(-3.0);
        stage.set_bias(0.05);
        // Tone redesign (occasional in practice) must also be alloc-free.
        stage.set_tone(ToneConfig::ThreeBand {
            bass_db: 1.0,
            mid_db: 0.0,
            treble_db: 2.0,
        });
        stage.process_block(&input, &mut output);
    }
    ARMED.store(false, Ordering::Relaxed);

    let allocations = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(
        allocations, 0,
        "render path allocated {allocations} times (must be zero)"
    );
}
