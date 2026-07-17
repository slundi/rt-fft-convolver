//! Real-time-safety audit for the `rt-loudness-meter` engine (Roadmap-10
//! Phase 4).
//!
//! Covers the non-allocation checks: the meter is **analysis-only** so the
//! audio must pass through bit-for-bit; the denormal guard keeps near-silent
//! input from poisoning the readings; metering is zero-latency; and varying
//! block sizes render without panicking.  The zero-allocation check lives in
//! its own binary (`rt_alloc.rs`).

use std::sync::Arc;

use rt_loudness_core::MeterBroadcast;
use rt_loudness_meter::{LoudnessParams, MeterProcessor};

const SAMPLE_RATE: u32 = 48_000;
const BLOCK: usize = 256;
const LIGHT: (f64, f64) = (80.0, 0.1);

fn processor(channels: usize, max_block: usize) -> MeterProcessor {
    MeterProcessor::with_true_peak_design(
        channels,
        SAMPLE_RATE,
        max_block,
        Arc::new(MeterBroadcast::new()),
        LIGHT.0,
        LIGHT.1,
    )
}

#[test]
fn audio_passes_through_bit_for_bit() {
    // The meter must never modify the signal (analysis-only guarantee).
    let mut processor = processor(2, BLOCK);
    let params = LoudnessParams::default();

    let original: Vec<f32> = (0..BLOCK)
        .map(|n| (std::f32::consts::TAU * 1_000.0 * n as f32 / SAMPLE_RATE as f32).sin() * 0.7)
        .collect();
    let mut left = original.clone();
    let mut right = original.clone();
    let mut channels: [&mut [f32]; 2] = [&mut left, &mut right];
    processor.process_buffer(&mut channels, &params);

    assert_eq!(left, original, "left channel was modified");
    assert_eq!(right, original, "right channel was modified");
}

#[test]
fn near_silence_keeps_readings_finite_and_silent() {
    // A very quiet input (denormal-scale) must not drive the K-weighting
    // biquads into NaN/denormal territory or leave garbage in the readings.
    let mut processor = processor(1, BLOCK);
    let params = LoudnessParams::default();

    let tiny = vec![1e-30f32; BLOCK];
    for _ in 0..200 {
        processor.process(&[&tiny], &params);
    }
    let values = processor.values();
    assert!(values.momentary_lufs.is_finite() || values.momentary_lufs == f32::NEG_INFINITY);
    // Essentially silent → far below any real programme level.
    assert!(
        values.momentary_lufs < -100.0,
        "near-silence read as {} LUFS",
        values.momentary_lufs
    );
    assert!(!values.integrated_lufs.is_nan());
}

#[test]
fn metering_is_zero_latency() {
    assert_eq!(processor(2, BLOCK).latency_samples(), 0);
}

#[test]
fn handles_varying_block_sizes() {
    // A host may hand us short final blocks; anything up to max_block is fine.
    let mut processor = processor(2, BLOCK);
    let params = LoudnessParams::default();
    for size in [1usize, 7, 64, 200, BLOCK] {
        let left = vec![0.25f32; size];
        let right = vec![0.25f32; size];
        processor.process(&[&left, &right], &params);
    }
    assert!(processor.values().sample_peak_dbfs.is_finite());
}

#[test]
fn empty_block_is_a_noop() {
    let mut processor = processor(2, BLOCK);
    let params = LoudnessParams::default();
    let empty: [f32; 0] = [];
    processor.process(&[&empty, &empty], &params);
    assert_eq!(processor.values().momentary_lufs, f32::NEG_INFINITY);
}
