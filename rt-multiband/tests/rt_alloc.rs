//! RT-safety allocation audit for [`MultibandProcessor`] (Roadmap-11 Phase 4:
//! "Zero allocations in `process()`").
//!
//! Installs a **counting global allocator** and asserts that steady-state
//! rendering — including per-block parameter application (threshold, ratio,
//! crossover frequency, mode and link changes, all of which are in-place
//! setters) — performs no heap allocation on the audio-thread path.  It lives
//! in its own test binary so the process-wide counting allocator is isolated
//! from the other test binaries.
//!
//! Unlike the sibling dynamics plugins there is no structural rebuild to keep
//! off this path: the multi-band engine allocates nothing after construction.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rt_multiband::{BandModeChoice, MultibandParams, MultibandProcessor, StereoLinkChoice};

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

/// Fill a stereo block with a hot 120 Hz sine so several bands are actively
/// reducing gain.
fn fill_tone(left: &mut [f32], right: &mut [f32], block_index: usize) {
    for sample in 0..left.len() {
        let time = (block_index * BLOCK + sample) as f32 / SR;
        let value = 0.9 * (std::f32::consts::TAU * 120.0 * time).sin();
        left[sample] = value;
        right[sample] = value;
    }
}

/// Steady-state rendering — including per-block parameter application — is
/// allocation-free on the audio thread.
///
/// A single test owns the whole measured region: the counting allocator is
/// process-wide, so concurrent allocator tests would cross-contaminate.
#[test]
fn process_is_allocation_free_on_the_audio_thread() {
    let params = MultibandParams::default();
    let mut processor = MultibandProcessor::new(SR, &params);

    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];

    // Warm up past the first-block force-apply of every setter.
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

    // 2. Per-block continuous + discrete parameter changes exercise the cheap
    //    in-place setters (threshold, ratio, crossover retune, mode, link): no
    //    allocation.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block in 80..200 {
        let mut params = params;
        params.bands[0].threshold_db = -30.0 - (block % 10) as f32;
        params.bands[0].ratio = 2.0 + (block % 8) as f32;
        params.bands[1].mode = if block % 2 == 0 {
            BandModeChoice::Expander
        } else {
            BandModeChoice::Compressor
        };
        params.link = if block % 3 == 0 {
            StereoLinkChoice::Independent
        } else {
            StereoLinkChoice::Linked
        };
        params.crossover_hz[1] = 800.0 + (block % 20) as f32 * 20.0;
        fill_tone(&mut left, &mut right, block);
        let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
        processor.process(&mut channels, &params);
    }
    let changing = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        changing, 0,
        "parameter-change render allocated {changing} time(s)"
    );

    // 3. Mono rendering (single channel paired with itself) is also
    //    allocation-free.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for block in 200..248 {
        fill_tone(&mut left, &mut right, block);
        let mut channels: [&mut [f32]; 1] = [&mut left];
        processor.process(&mut channels, &params);
    }
    let mono = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(mono, 0, "mono render allocated {mono} time(s)");
}
