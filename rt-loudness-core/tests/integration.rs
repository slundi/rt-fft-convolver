//! End-to-end `rt-loudness-core` checks: the public [`LoudnessMeter`] chain
//! against known reference scenarios from ITU-R BS.1770-4 / EBU R128.

use std::f32::consts::TAU;

use rt_loudness_core::{LoudnessMeter, MeterBroadcast};

const SR: u32 = 48_000;
/// A light true-peak half-band design keeps the impulse-latency probe cheap;
/// it does not affect loudness accuracy (loudness never touches the resampler).
const LIGHT: (f64, f64) = (80.0, 0.1);

/// A `samples`-long sine at `freq` Hz with peak amplitude `amp`.
fn sine(freq: f32, samples: usize, amp: f32) -> Vec<f32> {
    (0..samples)
        .map(|index| amp * (TAU * freq * index as f32 / SR as f32).sin())
        .collect()
}

/// Feed a stereo (dual-mono) signal for `seconds`, in 4800-sample (100 ms)
/// blocks, and return the meter.
fn measure_dual_mono(freq: f32, amp: f32, seconds: f32) -> LoudnessMeter {
    let block = 4_800;
    let signal = sine(freq, block, amp);
    let mut meter = LoudnessMeter::with_true_peak_design(2, SR, block, LIGHT.0, LIGHT.1);
    let blocks = (seconds * 10.0) as usize; // 10 × 100 ms blocks per second
    for _ in 0..blocks {
        meter.process_block(&[&signal, &signal]);
    }
    meter
}

/// BS.1770-4 calibration: a full-scale (0 dBFS peak) 997/1000 Hz sine, summed
/// over an identical stereo pair, measures ≈ 0 LUFS.  (The +3 LU from summing
/// two channels and the −3 LU sine crest cancel against the −0.691 offset and
/// the ~0 dB K-weighting at 1 kHz.)
#[test]
fn full_scale_stereo_sine_is_near_zero_lufs() {
    let meter = measure_dual_mono(1_000.0, 1.0, 2.0);
    let integrated = meter.values().integrated_lufs;
    assert!(
        integrated.abs() < 1.0,
        "full-scale 1 kHz dual-mono sine integrated to {integrated} LUFS, expected ~0"
    );
}

/// Halving the amplitude (−6.02 dB) drops every loudness window by ~6 LU.
#[test]
fn halving_amplitude_drops_loudness_by_six_lu() {
    let loud = measure_dual_mono(1_000.0, 0.5, 2.0).values();
    let quiet = measure_dual_mono(1_000.0, 0.25, 2.0).values();
    let delta = loud.integrated_lufs - quiet.integrated_lufs;
    assert!(
        (delta - 6.02).abs() < 0.2,
        "−6 dB in amplitude should drop integrated loudness ~6 LU, got {delta}"
    );
}

/// Digital silence never registers on any window, and the absolute gate keeps
/// it out of the integrated result.
#[test]
fn silence_measures_as_no_loudness() {
    let meter = measure_dual_mono(1_000.0, 0.0, 2.0);
    let values = meter.values();
    assert_eq!(values.momentary_lufs, f32::NEG_INFINITY);
    assert_eq!(values.short_term_lufs, f32::NEG_INFINITY);
    assert_eq!(values.integrated_lufs, f32::NEG_INFINITY);
    assert_eq!(values.true_peak_dbtp, f32::NEG_INFINITY);
    assert_eq!(values.lra_lu, 0.0);
}

/// A steady tone has essentially no loudness range: its short-term loudness
/// never varies, so LRA collapses toward zero.
#[test]
fn steady_tone_has_negligible_lra() {
    // Need > 3 s so short-term blocks exist to populate the LRA histogram.
    let meter = measure_dual_mono(1_000.0, 0.5, 5.0);
    let lra = meter.values().lra_lu;
    assert!(lra < 1.0, "steady tone LRA {lra} LU should be ~0");
}

/// The lock-free broadcast carries exactly what the meter reports.
#[test]
fn broadcast_reflects_meter_values() {
    let meter = measure_dual_mono(1_000.0, 0.5, 2.0);
    let broadcast = MeterBroadcast::new();
    meter.publish(&broadcast);
    assert_eq!(broadcast.load(), meter.values());
}

/// A sine whose samples straddle its crest reconstructs above full sample
/// scale — true peak must exceed sample peak (the inter-sample-peak case).
#[test]
fn true_peak_catches_inter_sample_peak() {
    // 0.25·fs at 45°: samples all land at ±0.707, crest is between them.
    let block = 4_800;
    let signal: Vec<f32> = (0..block)
        .map(|n| (TAU * 0.25 * n as f32 + std::f32::consts::FRAC_PI_4).sin())
        .collect();
    let mut meter = LoudnessMeter::with_true_peak_design(1, SR, block, LIGHT.0, LIGHT.1);
    meter.process_block(&[&signal]);
    let values = meter.values();
    assert!(
        values.true_peak_dbtp > values.sample_peak_dbfs,
        "true peak {} dBTP should exceed sample peak {} dBFS",
        values.true_peak_dbtp,
        values.sample_peak_dbfs
    );
}
