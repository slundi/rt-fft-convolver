//! RT-safety audit for [`LimiterProcessor`] (Roadmap-07 Phase 4: "Zero
//! allocations in `process()`").
//!
//! Installs a **counting global allocator** and asserts that steady-state
//! rendering — including per-block continuous-parameter application, a
//! look-ahead change and a channel-link change (both of which reset engine
//! state but must not reallocate, since every buffer is sized for the 20 ms
//! maximum look-ahead at construction) — performs no heap allocation on the
//! audio-thread path.  It lives in its own test binary so the process-wide
//! counting allocator is isolated from the other test binaries.
//!
//! Structural (oversampling) changes *do* reallocate, but that happens on the
//! control thread inside `LimiterController::update` — never in
//! `LimiterProcessor::process` — so it is deliberately outside the measured
//! region here.
//!
//! The companion `DenormalGuard` guarantee (FTZ/DAZ around the callback) is
//! covered end-to-end by `rt-limiter-core`'s `denormal_guard_flushes_the_tail`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_limiter::{LimiterController, LimiterParams, LinkChoice};

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

/// Fill a stereo block with a hot 220 Hz sine so the limiter is actively
/// reducing gain (the reduction path is what we care about).
fn fill_tone(left: &mut [f32], right: &mut [f32], block_index: usize) {
    for sample in 0..left.len() {
        let time = (block_index * BLOCK + sample) as f32 / SR;
        let value = 1.4 * (std::f32::consts::TAU * 220.0 * time).sin();
        left[sample] = value;
        right[sample] = value;
    }
}

/// Steady-state rendering — including per-block parameter application, a
/// look-ahead change and a channel-link change — is allocation-free on the
/// audio thread.
///
/// A single test owns the whole measured region: the counting allocator is
/// process-wide, so concurrent allocator tests would cross-contaminate.
#[test]
fn process_is_allocation_free_on_the_audio_thread() {
    let params = LimiterParams {
        threshold_db: -6.0,
        ceiling_db: -1.0,
        ..Default::default()
    };
    let (_controller, mut processor) = LimiterController::new(BLOCK, SR, params);

    // Pre-allocate the audio buffers off the measured path and reuse them.
    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];

    // Warm up past the first-block force-apply of every setter and any lazy
    // one-time setup.
    for block in 0..16 {
        fill_tone(&mut left, &mut right, block);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &params);
    }

    // 1. Steady state with fixed parameters: no allocation.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block in 16..80 {
        fill_tone(&mut left, &mut right, block);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &params);
    }
    let steady = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(steady, 0, "steady-state render allocated {steady} time(s)");

    // 2. Per-block continuous-parameter changes (threshold / ceiling / release)
    //    exercise the cheap in-place setters: no allocation.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block in 80..160 {
        let mut params = params;
        params.threshold_db = -6.0 - (block % 12) as f32;
        params.ceiling_db = -1.0 - (block % 4) as f32 * 0.5;
        params.release_ms = 50.0 + (block % 20) as f32 * 10.0;
        fill_tone(&mut left, &mut right, block);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &params);
    }
    let continuous = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        continuous, 0,
        "continuous-param render allocated {continuous} time(s)"
    );

    // 3. Look-ahead and channel-link changes reset engine state but reuse the
    //    max-sized buffers → still no allocation.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block in 160..240 {
        let mut params = params;
        params.lookahead_ms = if block % 2 == 0 { 5.0 } else { 1.5 };
        params.link = if block % 3 == 0 {
            LinkChoice::MidSide
        } else {
            LinkChoice::Linked
        };
        fill_tone(&mut left, &mut right, block);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &params);
    }
    let structural = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        structural, 0,
        "look-ahead / link change allocated {structural} time(s)"
    );

    // 4. Mono rendering (single channel via the internal scratch pair) is also
    //    allocation-free.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block in 240..288 {
        fill_tone(&mut left, &mut right, block);
        let mut channels: [&mut [f32]; 1] = [&mut left];
        processor.process(&mut channels, &params);
    }
    let mono = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(mono, 0, "mono render allocated {mono} time(s)");
}
