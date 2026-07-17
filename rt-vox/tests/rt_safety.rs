//! Real-time-safety audit for the `rt-vox` engine (Roadmap-08 Phase 4).
//!
//! Covers the non-allocation checks: the denormal guard flushes near-silent
//! spectral tails to true zero, the reported latency matches each mode's build,
//! and the processor tolerates varying block sizes without panicking.  The
//! zero-allocation check lives in its own binary (`rt_alloc.rs`).

use rt_vox::params::Mode;
use rt_vox::{VoxController, VoxParams, VoxProcessor};

const SAMPLE_RATE: f32 = 48_000.0;
const BLOCK: usize = 256;

fn run_silence(processor: &mut VoxProcessor, params: &VoxParams, blocks: usize) -> Vec<f32> {
    let mut last = vec![0.0f32; BLOCK];
    for _ in 0..blocks {
        let mut silence = vec![0.0f32; BLOCK];
        {
            let mut main: [&mut [f32]; 1] = [&mut silence];
            processor.process(&mut main, None, params);
        }
        last = silence;
    }
    last
}

#[test]
fn silence_flushes_to_true_zero_in_every_mode() {
    for mode in [Mode::Vocoder, Mode::Harmoniser, Mode::FormantShifter] {
        let params = VoxParams {
            mode,
            mix: 1.0,
            dry_delay_compensation: false,
            ..Default::default()
        };
        let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);
        let last = run_silence(&mut processor, &params, 200);
        assert!(
            last.iter().all(|&s| s.abs() < 1e-12),
            "{mode:?}: denormal dust left in the output"
        );
    }
}

#[test]
fn reported_latency_matches_mode() {
    let base = VoxParams::default();
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 2, &base);

    for mode in [Mode::Vocoder, Mode::Harmoniser, Mode::FormantShifter] {
        let params = VoxParams { mode, ..base };
        let mut left = vec![0.1f32; BLOCK];
        let mut right = vec![0.1f32; BLOCK];
        {
            let mut main: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut main, None, &params);
        }
        assert_eq!(
            processor.reported_latency(),
            processor.latency_samples(mode),
            "{mode:?}: reported latency disagrees with the advertised value"
        );
    }
}

#[test]
fn handles_varying_block_sizes() {
    // The processor is built for BLOCK; smaller blocks must render without
    // panicking (a host is free to send short buffers).
    let params = VoxParams {
        mode: Mode::Harmoniser,
        ..Default::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);

    for &len in &[BLOCK, 64, 1, 200, BLOCK] {
        let mut buffer = vec![0.2f32; len];
        {
            let mut main: [&mut [f32]; 1] = [&mut buffer];
            processor.process(&mut main, None, &params);
        }
        assert!(buffer.iter().all(|s| s.is_finite()));
    }
}

#[test]
fn bypass_is_bit_transparent() {
    let params = VoxParams {
        bypass: true,
        ..Default::default()
    };
    let (_controller, mut processor) = VoxController::new(BLOCK, SAMPLE_RATE, 1, &params);
    let original: Vec<f32> = (0..BLOCK).map(|n| (n as f32 * 0.07).sin()).collect();
    let mut samples = original.clone();
    {
        let mut main: [&mut [f32]; 1] = [&mut samples];
        processor.process(&mut main, None, &params);
    }
    assert_eq!(samples, original, "bypass must pass the signal untouched");
}
