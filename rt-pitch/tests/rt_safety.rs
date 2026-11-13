//! Real-time-safety audit for the `rt-pitch` engine (Roadmap-03 Phase 4).
//!
//! Covers the non-allocation checks of the audit: the denormal guard flushes
//! near-silent spectral tails to true zero, and the latency the processor
//! reports matches the controller's build (and the shifter warm-up).  The
//! zero-allocation check lives in its own test binary (`rt_alloc.rs`) because
//! it installs a global allocator.

use rt_pitch::{PitchController, PitchParams, Quality};

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK: usize = 256;

/// Feeding silence must yield exact-zero output once warmed up: the shifters'
/// `DenormalGuard` flushes the near-silent spectral bins to zero rather than
/// leaving denormal dust that would spike CPU.
#[test]
fn silence_flushes_to_true_zero() {
    // A non-unison ratio keeps the whole spectral path active.
    let params = PitchParams {
        current_preset: 1,
        mix: 1.0,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (_controller, mut processor) = PitchController::new(BLOCK, SAMPLE_RATE, 1, &params);

    let mut last = vec![0.0f32; BLOCK];
    for _ in 0..200 {
        let mut silence = vec![0.0f32; BLOCK];
        {
            let mut channels: [&mut [f32]; 1] = [&mut silence];
            processor.process(&mut channels, &params);
        }
        last = silence;
    }
    assert!(
        last.iter().all(|&s| s == 0.0),
        "denormal guard inactive — silent output not flushed to zero: {last:?}"
    );
}

/// The processor reports exactly the controller's built latency, and it
/// changes to match when the quality (FFT size) is rebuilt.
#[test]
fn reported_latency_matches_build() {
    let params = PitchParams {
        quality: Quality::Standard,
        dry_delay_compensation: false,
        ..Default::default()
    };
    let (mut controller, mut processor) = PitchController::new(BLOCK, SAMPLE_RATE, 1, &params);

    // Standard/Smooth phase-vocoder latency = frame − hop = 1024 − 256.
    assert_eq!(controller.latency(), 768);

    let mut block = vec![0.1f32; BLOCK];
    {
        let mut channels: [&mut [f32]; 1] = [&mut block];
        processor.process(&mut channels, &params);
    }
    assert_eq!(processor.reported_latency(), controller.latency());

    // Rebuild at Studio quality (2048): latency grows and both agree again.
    let studio = PitchParams {
        quality: Quality::Studio,
        ..params
    };
    assert!(controller.apply_structural(&studio));
    assert_eq!(controller.latency(), 2048 - 512);

    let mut block = vec![0.1f32; BLOCK];
    {
        let mut channels: [&mut [f32]; 1] = [&mut block];
        processor.process(&mut channels, &studio);
    }
    assert_eq!(processor.reported_latency(), controller.latency());
}
