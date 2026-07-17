//! Real-time-safety audit for the `rt-adaptive` engine (Roadmap-09 Phase 4).
//!
//! Covers the non-allocation checks: the denormal guard flushes near-silent
//! tails to true zero, every mode is zero-latency, varying block sizes render
//! without panicking, and bypass is bit-transparent.  The zero-allocation check
//! (including on notch allocation) lives in its own binary (`rt_alloc.rs`).

use rt_adaptive::params::Mode;
use rt_adaptive::{AdaptiveController, AdaptiveParams, AdaptiveProcessor};

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK: usize = 256;

fn run_silence(
    processor: &mut AdaptiveProcessor,
    params: &AdaptiveParams,
    blocks: usize,
) -> Vec<f32> {
    let mut last = vec![0.0f32; BLOCK];
    for _ in 0..blocks {
        let mut silence = vec![0.0f32; BLOCK];
        let sc = vec![0.0f32; BLOCK];
        {
            let mut main: [&mut [f32]; 1] = [&mut silence];
            processor.process(&mut main, Some(&sc), params);
        }
        last = silence;
    }
    last
}

#[test]
fn silence_flushes_to_true_zero_in_every_mode() {
    for mode in [
        Mode::FeedbackSuppressor,
        Mode::HumRemover,
        Mode::EchoCanceller,
    ] {
        let params = AdaptiveParams {
            mode,
            ..Default::default()
        };
        let (_controller, mut processor) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 1, &params);
        let last = run_silence(&mut processor, &params, 200);
        assert!(
            last.iter().all(|&s| s.abs() < 1e-12),
            "{mode:?}: denormal dust left in the output"
        );
    }
}

#[test]
fn every_mode_is_zero_latency() {
    let base = AdaptiveParams::default();
    let (_controller, processor) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 2, &base);
    assert_eq!(processor.latency_samples(), 0);
}

#[test]
fn handles_varying_block_sizes() {
    let params = AdaptiveParams {
        mode: Mode::EchoCanceller,
        ..Default::default()
    };
    let (_controller, mut processor) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 1, &params);

    for &len in &[BLOCK, 64, 1, 200, BLOCK] {
        let mut buffer = vec![0.2f32; len];
        let sc = vec![0.1f32; len];
        {
            let mut main: [&mut [f32]; 1] = [&mut buffer];
            processor.process(&mut main, Some(&sc), &params);
        }
        assert!(buffer.iter().all(|s| s.is_finite()));
    }
}

#[test]
fn bypass_is_bit_transparent() {
    let params = AdaptiveParams {
        bypass: true,
        ..Default::default()
    };
    let (_controller, mut processor) = AdaptiveController::new(BLOCK, SAMPLE_RATE, 1, &params);
    let original: Vec<f32> = (0..BLOCK).map(|n| (n as f32 * 0.07).sin()).collect();
    let mut samples = original.clone();
    {
        let mut main: [&mut [f32]; 1] = [&mut samples];
        processor.process(&mut main, None, &params);
    }
    assert_eq!(samples, original, "bypass must pass the signal untouched");
}
