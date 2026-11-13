//! RT-safety audit for `PhysicalEngine` (Roadmap-06 Phase 4: "Zero allocations
//! in `process()` including on note-on").
//!
//! Installs a **counting global allocator** and asserts that steady-state
//! rendering — including per-block `apply_params`, sample-accurate `note_on` /
//! `note_off` (voice allocation *and* stealing) and a model switch — performs no
//! heap allocation on the audio-thread path.  It lives in its own test binary so
//! the process-wide counting allocator is isolated from the other test
//! binaries.
//!
//! The companion RT-safety guarantee — a `DenormalGuard` wrapping every callback
//! — is satisfied by the plugin's `process` (which opens one at the top of the
//! callback) and by the engine's flush-to-zero writes, checked end-to-end by
//! `rt-physical-core`'s `decaying_tail_flushes_to_exact_zero`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_physical::physical_core::Model;
use rt_physical::{InstrumentParams, PhysicalEngine};

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

const SR: f32 = 48_000.0;
const BLOCK: usize = 256;

/// Fill a stereo block with a 220 Hz sine (effect-mode excitation input).
fn fill_tone(left: &mut [f32], right: &mut [f32], block_index: usize) {
    for sample in 0..left.len() {
        let time = (block_index * BLOCK + sample) as f32 / SR;
        let value = 0.4 * (std::f32::consts::TAU * 220.0 * time).sin();
        left[sample] = value;
        right[sample] = -value;
    }
}

/// A steady-state render — including per-block `apply_params`, note on/off
/// (allocation *and* stealing) and a model switch — is allocation-free on the
/// audio thread.
///
/// A single test owns the whole measured region: the counting allocator is
/// process-wide, so concurrent allocator tests would cross-contaminate.
#[test]
fn process_is_allocation_free_on_the_audio_thread() {
    let mut params = InstrumentParams::default();
    let mut engine = PhysicalEngine::new(SR, &params);

    // Pre-allocate the audio buffers off the measured path and reuse them.
    let mut in_l = vec![0.0f32; BLOCK];
    let mut in_r = vec![0.0f32; BLOCK];
    let mut out_l = vec![0.0f32; BLOCK];
    let mut out_r = vec![0.0f32; BLOCK];

    // Warm up past any one-time lazy setup.
    for block in 0..16 {
        fill_tone(&mut in_l, &mut in_r, block);
        engine.apply_params(&params);
        engine.process(&in_l, &in_r, &mut out_l, &mut out_r);
    }

    // 1. Steady state with a running note: no allocation.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    engine.note_on(60, 0.9);
    for block in 16..80 {
        fill_tone(&mut in_l, &mut in_r, block);
        engine.apply_params(&params);
        engine.process(&in_l, &in_r, &mut out_l, &mut out_r);
    }
    engine.note_off(60);
    let steady = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(steady, 0, "steady-state render allocated {steady} time(s)");

    // 2. A flurry of note-ons — more than the 16 voices, forcing voice
    //    stealing — must not allocate.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block in 80..160 {
        fill_tone(&mut in_l, &mut in_r, block);
        // Two fresh notes per block quickly exhausts polyphony and steals.
        engine.note_on(40 + (block % 48) as u8, 0.8);
        engine.note_on(52 + (block % 24) as u8, 0.7);
        engine.process(&in_l, &in_r, &mut out_l, &mut out_r);
    }
    let notes = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        notes, 0,
        "note-on / voice stealing allocated {notes} time(s)"
    );

    // 3. Switching model family (silences the outgoing bank) must not allocate.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    params.model = Model::Modal;
    for block in 160..224 {
        fill_tone(&mut in_l, &mut in_r, block);
        engine.apply_params(&params);
        engine.note_on(64, 0.8);
        engine.process(&in_l, &in_r, &mut out_l, &mut out_r);
    }
    let switch = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(switch, 0, "model switch allocated {switch} time(s)");
}
