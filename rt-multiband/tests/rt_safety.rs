//! RT-safety behavioural audit for [`MultibandProcessor`] (Roadmap-11 Phase 4:
//! denormal handling, zero-latency reporting, and finite/bounded output on the
//! audio path).
//!
//! The zero-allocation guarantee is covered separately in `rt_alloc.rs` (which
//! needs its own process-wide counting allocator); the `DenormalGuard` FTZ/DAZ
//! wrap is exercised end-to-end here by driving a silent tail and checking the
//! output decays to a sub-audible floor (rather than ringing or running away).

use rt_multiband::{BandModeChoice, MultibandParams, MultibandProcessor};

const SR: f32 = 48_000.0;
const BLOCK: usize = 512;

/// Render `blocks` blocks of a mono generator, returning the concatenated
/// output.
fn run_mono(
    processor: &mut MultibandProcessor,
    params: &MultibandParams,
    make: impl Fn(usize) -> f32,
    blocks: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks * BLOCK);
    for block_index in 0..blocks {
        let mut samples: Vec<f32> = (0..BLOCK).map(|n| make(block_index * BLOCK + n)).collect();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, params);
        }
        output.extend_from_slice(&samples);
    }
    output
}

#[test]
fn latency_is_zero_in_v1() {
    let params = MultibandParams::default();
    let processor = MultibandProcessor::new(SR, &params);
    assert_eq!(
        processor.latency_samples(),
        0,
        "v1 IIR crossover must report zero latency"
    );
}

#[test]
fn hot_signal_stays_finite_and_bounded() {
    // Drive every band hard and check the output never goes non-finite or wild.
    let mut params = MultibandParams::default();
    for band in &mut params.bands {
        band.threshold_db = -40.0;
        band.ratio = 12.0;
        band.makeup_db = 12.0;
    }
    let mut processor = MultibandProcessor::new(SR, &params);
    let output = run_mono(
        &mut processor,
        &params,
        |n| 1.5 * (std::f32::consts::TAU * 0.02 * n as f32).sin(),
        16,
    );
    assert!(output.iter().all(|s| s.is_finite()), "non-finite output");
    assert!(output.iter().all(|s| s.abs() < 8.0), "output ran away");
}

#[test]
fn silent_tail_decays_to_sub_audible_floor() {
    // With the DenormalGuard active (installed inside `process`), a decaying
    // tail must not sustain audible energy or park on CPU-eating denormals.
    // The LR4 crossover settles to a harmless sub-−600 dBFS limit cycle (a
    // *normal* float ~1e-36, which FTZ correctly leaves alone — it is not a
    // denormal), well below anything audible; the point is it decays and stays
    // finite rather than ringing.
    let mut params = MultibandParams::default();
    for band in &mut params.bands {
        band.mode = BandModeChoice::Bypass;
    }
    let mut processor = MultibandProcessor::new(SR, &params);

    run_mono(&mut processor, &params, |n| (n as f32 * 0.05).sin(), 4);
    let tail = run_mono(&mut processor, &params, |_| 0.0, 48);

    let last_block = &tail[tail.len() - BLOCK..];
    let residual = last_block.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!(residual < 1e-20, "silent tail did not decay: {residual:e}");
    assert!(last_block.iter().all(|s| s.is_finite()));
}

#[test]
fn extra_channels_are_left_untouched() {
    // A 3-channel buffer: the first two are the stereo pair, the third passes
    // through untouched (defensive — the plugin only declares mono/stereo).
    let params = MultibandParams::default();
    let mut processor = MultibandProcessor::new(SR, &params);
    let mut left = vec![0.2f32; BLOCK];
    let mut right = vec![0.2f32; BLOCK];
    let mut third = vec![0.3f32; BLOCK];
    {
        let mut channels: [&mut [f32]; 3] = [&mut left, &mut right, &mut third];
        processor.process(&mut channels, &params);
    }
    assert!(
        third.iter().all(|&s| s == 0.3),
        "extra channel was modified"
    );
}

#[test]
fn empty_and_zero_length_blocks_are_safe() {
    let params = MultibandParams::default();
    let mut processor = MultibandProcessor::new(SR, &params);
    // Zero-length block.
    let mut empty: Vec<f32> = Vec::new();
    {
        let mut channels: [&mut [f32]; 1] = [&mut empty];
        processor.process(&mut channels, &params);
    }
    assert_eq!(processor.last_process_samples(), 0);
    // No channels at all.
    {
        let mut channels: [&mut [f32]; 0] = [];
        processor.process(&mut channels, &params);
    }
    assert_eq!(processor.last_process_samples(), 0);
}
