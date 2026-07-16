//! RT-safety audit for `ReverbEngine` (Roadmap-05 Phase 4: "Zero allocations
//! in `process()`").
//!
//! Installs a **counting global allocator** and asserts that steady-state
//! rendering — including per-block `apply_params`, a live preset crossfade and a
//! freeze toggle — performs no heap allocation on the audio-thread path.  It
//! lives in its own test binary so the process-wide counting allocator is
//! isolated from the other test binaries.
//!
//! The companion RT-safety guarantee — a `DenormalGuard` wrapping every
//! callback — is satisfied by the plugin's `process` (which opens one at the top
//! of the callback) and by the engine's flush-to-zero feedback writes, checked
//! in `rt-fdn-core`'s `tail_flushes_to_exact_zero`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_fdn_reverb::{ReverbEngine, ReverbParams};

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
const BLOCK: usize = 256;

/// Fill `buffer` in place with a 220 Hz sine (no allocation).
fn fill_tone(left: &mut [f32], right: &mut [f32], block_index: usize) {
    for n in 0..left.len() {
        let t = (block_index * BLOCK + n) as f32 / SR as f32;
        let value = 0.4 * (std::f32::consts::TAU * 220.0 * t).sin();
        left[n] = value;
        right[n] = -value;
    }
}

/// A steady-state render — including per-block `apply_params`, a preset
/// crossfade and a freeze toggle — is allocation-free on the audio thread.
///
/// A single test owns the whole measured region: the counting allocator is
/// process-wide, so concurrent allocator tests would cross-contaminate.
#[test]
fn process_is_allocation_free_on_the_audio_thread() {
    let mut params = ReverbParams::default();
    let mut engine = ReverbEngine::new(SR, &params);

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

    // 1. Steady state: 128 blocks, no allocation.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block in 16..144 {
        fill_tone(&mut in_l, &mut in_r, block);
        engine.apply_params(&params);
        engine.process(&in_l, &in_r, &mut out_l, &mut out_r);
    }
    let steady = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(steady, 0, "steady-state render allocated {steady} time(s)");

    // 2. A preset crossfade (switch active slot) must not allocate.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    params.current_preset = 5;
    for block in 144..208 {
        fill_tone(&mut in_l, &mut in_r, block);
        engine.apply_params(&params);
        engine.process(&in_l, &in_r, &mut out_l, &mut out_r);
    }
    let crossfade = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        crossfade, 0,
        "preset crossfade allocated {crossfade} time(s)"
    );

    // 3. A freeze toggle must not allocate.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    params.freeze = true;
    for block in 208..272 {
        fill_tone(&mut in_l, &mut in_r, block);
        engine.apply_params(&params);
        engine.process(&in_l, &in_r, &mut out_l, &mut out_r);
    }
    params.freeze = false;
    engine.apply_params(&params);
    engine.process(&in_l, &in_r, &mut out_l, &mut out_r);
    let freeze = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(freeze, 0, "freeze toggle allocated {freeze} time(s)");
}
