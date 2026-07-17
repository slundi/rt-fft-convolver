//! End-to-end integration tests for the `rt-multiband` plugin engine
//! (Roadmap-11 Phase 4), driving [`MultibandProcessor`] the way the `nih-plug`
//! wrapper does: build once, then render blocks with a parameter snapshot.

use rt_multiband::{BandModeChoice, MultibandParams, MultibandProcessor, StereoLinkChoice};

const SR: f32 = 48_000.0;
const BLOCK: usize = 512;

fn rms(signal: &[f32]) -> f32 {
    (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Render a mono sine through `blocks` blocks.
fn run_tone(
    processor: &mut MultibandProcessor,
    params: &MultibandParams,
    freq: f32,
    amplitude: f32,
    blocks: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks * BLOCK);
    for block_index in 0..blocks {
        let mut samples: Vec<f32> = (0..BLOCK)
            .map(|n| {
                let time = (block_index * BLOCK + n) as f32 / SR;
                amplitude * (std::f32::consts::TAU * freq * time).sin()
            })
            .collect();
        {
            let mut channels: [&mut [f32]; 1] = [&mut samples];
            processor.process(&mut channels, params);
        }
        output.extend_from_slice(&samples);
    }
    output
}

fn all_bypass() -> MultibandParams {
    let mut params = MultibandParams::default();
    for band in &mut params.bands {
        band.mode = BandModeChoice::Bypass;
    }
    params
}

#[test]
fn bypass_flag_is_bit_transparent() {
    let params = MultibandParams {
        bypass: true,
        ..Default::default()
    };
    let mut processor = MultibandProcessor::new(SR, &params);
    let original: Vec<f32> = (0..BLOCK).map(|n| (n as f32 * 0.07).sin()).collect();
    let mut samples = original.clone();
    {
        let mut channels: [&mut [f32]; 1] = [&mut samples];
        processor.process(&mut channels, &params);
    }
    assert_eq!(samples, original, "global bypass must be transparent");
}

#[test]
fn bypassed_bands_reconstruct_the_input_magnitude() {
    // Every band in Bypass mode + full wet: the crossover sum is magnitude-flat.
    let params = all_bypass();
    let mut processor = MultibandProcessor::new(SR, &params);
    for &freq in &[80.0, 500.0, 3_000.0, 10_000.0] {
        let input = run_tone(&mut processor, &params, freq, 0.4, 6);
        let settled = &input[BLOCK * 3..];
        // Compare to the pure input RMS (0.4 amplitude sine → 0.283).
        let expected = 0.4 / std::f32::consts::SQRT_2;
        let err_db = 20.0 * (rms(settled) / expected).log10();
        assert!(
            err_db.abs() < 0.3,
            "at {freq} Hz reconstruction {err_db} dB"
        );
    }
}

#[test]
fn compressor_band_reduces_its_own_frequency_range() {
    // Compress the high band hard; a 12 kHz tone (band3) is reduced, a 60 Hz
    // tone (band0) passes untouched.
    let mut params = all_bypass();
    let top = params.bands.len() - 1;
    params.bands[top].mode = BandModeChoice::Compressor;
    params.bands[top].threshold_db = -35.0;
    params.bands[top].ratio = 10.0;
    params.bands[top].makeup_db = 0.0;

    let mut high = MultibandProcessor::new(SR, &params);
    let high_out = run_tone(&mut high, &params, 12_000.0, 0.6, 8);
    let high_ratio = rms(&high_out[BLOCK * 4..]) / (0.6 / std::f32::consts::SQRT_2);

    let mut low = MultibandProcessor::new(SR, &params);
    let low_out = run_tone(&mut low, &params, 60.0, 0.6, 8);
    let low_ratio = rms(&low_out[BLOCK * 4..]) / (0.6 / std::f32::consts::SQRT_2);

    assert!(high_ratio < 0.7, "high band not compressed: {high_ratio}");
    assert!(low_ratio > 0.9, "low tone should be untouched: {low_ratio}");
    assert!(high.band_gain_reduction_db(top) < -3.0, "GR not metered");
}

#[test]
fn solo_isolates_and_mute_removes_a_band() {
    // Solo the high band → a low tone is silenced.
    let mut soloed = all_bypass();
    let top = soloed.bands.len() - 1;
    soloed.bands[top].solo = true;
    let mut processor = MultibandProcessor::new(SR, &soloed);
    let out = run_tone(&mut processor, &soloed, 60.0, 0.5, 6);
    assert!(
        rms(&out[BLOCK * 3..]) < 0.1 * (0.5 / std::f32::consts::SQRT_2),
        "solo leaked the low band"
    );

    // Mute the low band → a low tone is (largely) removed.
    let mut muted = all_bypass();
    muted.bands[0].mute = true;
    let mut processor = MultibandProcessor::new(SR, &muted);
    let out = run_tone(&mut processor, &muted, 60.0, 0.5, 6);
    assert!(
        rms(&out[BLOCK * 3..]) < 0.2 * (0.5 / std::f32::consts::SQRT_2),
        "mute kept the low band"
    );
}

#[test]
fn independent_link_ducks_channels_separately() {
    // A loud left / quiet right into an independent-linked compressor: the
    // right channel keeps far more of its level than the left.
    let mut params = MultibandParams {
        link: StereoLinkChoice::Independent,
        ..Default::default()
    };
    for band in &mut params.bands {
        band.mode = BandModeChoice::Compressor;
        band.threshold_db = -24.0;
        band.ratio = 8.0;
    }
    let mut processor = MultibandProcessor::new(SR, &params);

    let mut last = (0.0f32, 0.0f32);
    for block_index in 0..12 {
        let mut left: Vec<f32> = (0..BLOCK)
            .map(|n| {
                let time = (block_index * BLOCK + n) as f32 / SR;
                0.9 * (std::f32::consts::TAU * 300.0 * time).sin()
            })
            .collect();
        let mut right = vec![0.01f32; BLOCK];
        {
            let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
            processor.process(&mut channels, &params);
        }
        last = (rms(&left), rms(&right) / 0.01);
    }
    // The quiet right channel is nearly untouched (gain ≈ 1); the loud left is
    // reduced well below its 0.9 amplitude RMS.
    assert!(
        last.1 > 0.9,
        "independent ducked the quiet channel: {}",
        last.1
    );
    assert!(
        last.0 < 0.9 / std::f32::consts::SQRT_2,
        "left not compressed"
    );
}

#[test]
fn crossover_retune_between_blocks_is_stable() {
    // Moving the crossover frequencies mid-stream must stay finite and keep
    // reconstructing (state-preserving retune).
    let mut params = all_bypass();
    let mut processor = MultibandProcessor::new(SR, &params);
    run_tone(&mut processor, &params, 1_000.0, 0.4, 4);
    params.crossover_hz = [80.0, 700.0, 4_000.0];
    let out = run_tone(&mut processor, &params, 1_000.0, 0.4, 6);
    assert!(out.iter().all(|s| s.is_finite()));
    let err_db = 20.0 * (rms(&out[BLOCK * 3..]) / (0.4 / std::f32::consts::SQRT_2)).log10();
    assert!(err_db.abs() < 0.4, "post-retune reconstruction {err_db} dB");
}
