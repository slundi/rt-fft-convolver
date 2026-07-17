//! Zero-allocation audit for `VoxProcessor::process` (Roadmap-08 Phase 4:
//! "Zero allocations in `process()`, including on MIDI note-on").
//!
//! This test binary installs a **counting global allocator** and asserts that a
//! steady-state render — including a MIDI note-on mid-stream — performs no heap
//! allocation.  It lives in its own file so the counting allocator is isolated
//! from the other test binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_vox::params::{Mode, VoiceParams};
use rt_vox::{VoxController, VoxParams};
use rt_vox_core::SnapMode;

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
fn process_and_note_on_do_not_allocate() {
    // The busiest configuration: harmoniser mode with all four voices enabled,
    // so every per-voice phase vocoder + formant shifter is exercised.
    let mut params = VoxParams {
        mode: Mode::Harmoniser,
        snap_mode: SnapMode::Midi,
        mix: 0.7,
        dry_delay_compensation: true,
        ..Default::default()
    };
    params.voices = [VoiceParams {
        interval: 0.0,
        level: 1.0,
        pan: 0.0,
        formant_shift: 3.0,
        detune_cents: 5.0,
        enabled: true,
    }; rt_vox::params::NUM_VOICES];

    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 2, &params);

    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];

    // Warm up outside the measured region: settle the shifters and force any
    // first-call lazy work.
    for block_index in 0..12 {
        fill_tone(&mut left, block_index);
        fill_tone(&mut right, block_index);
        let mut main: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut main, None, &params);
    }

    // Measure: steady-state blocks (plus MIDI note-ons) must allocate nothing.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block_index in 12..(12 + 128) {
        // Note-on / note-off on the audio thread must be allocation-free.
        if block_index == 20 {
            processor.note_on(64);
        }
        if block_index == 40 {
            processor.note_off(64);
        }
        fill_tone(&mut left, block_index);
        fill_tone(&mut right, block_index);
        let mut main: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut main, None, &params);
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
