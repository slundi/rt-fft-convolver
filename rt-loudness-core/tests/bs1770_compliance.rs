//! ITU-R BS.1770-4 / EBU R128 acceptance suite — the **primary quality gate**
//! (Roadmap-10 Phase 4).  These exercise the public [`LoudnessMeter`] against
//! reference scenarios with the standard's ±0.1 LU tolerances where they apply.
//!
//! The measurement path never touches the true-peak resampler, so the loudness
//! tests use the light half-band design for speed; the true-peak-accuracy test
//! uses the meter's **default** high-quality design (the one shipped).

use std::f32::consts::TAU;

use rt_loudness_core::{KWeightingFilter, LoudnessMeter};

const LIGHT: (f64, f64) = (80.0, 0.1);

/// A `samples`-long sine at `freq` Hz / `sample_rate` Hz, peak amplitude `amp`.
fn sine(freq: f32, sample_rate: u32, samples: usize, amp: f32) -> Vec<f32> {
    (0..samples)
        .map(|index| amp * (TAU * freq * index as f32 / sample_rate as f32).sin())
        .collect()
}

/// Peak amplitude for a given dBFS-peak level (`A = 10^(dBFS / 20)`).
///
/// The BS.1770 calibration is elegant here: for a dual-mono stereo 1 kHz sine
/// the two channel powers sum (+3.01 LU), the sine's crest (−3.01 dB) cancels
/// it, **and** the −0.691 dB loudness offset cancels the K-weighting's +0.69 dB
/// gain at 1 kHz.  Net: a dual-mono 1 kHz sine reads its dBFS-peak level
/// directly in LUFS — so a −23 dBFS tone measures −23.0 LUFS.
fn amplitude_for_dbfs(dbfs: f32) -> f32 {
    10.0f32.powf(dbfs / 20.0)
}

/// Drive a dual-mono stereo tone through a fresh meter for `seconds`.
fn measure_stereo(freq: f32, sample_rate: u32, amp: f32, seconds: f32) -> LoudnessMeter {
    let block = (sample_rate / 10) as usize; // 100 ms
    let signal = sine(freq, sample_rate, block, amp);
    let mut meter = LoudnessMeter::with_true_peak_design(2, sample_rate, block, LIGHT.0, LIGHT.1);
    let blocks = (seconds * 10.0) as usize;
    for _ in 0..blocks {
        meter.process_block(&[&signal, &signal]);
    }
    meter
}

/// EBU R128 calibration (Tech 3341 tests 1–2): a stereo 1 kHz sine set to a
/// target level reads that target on momentary, short-term and integrated,
/// within the standard's ±0.1 LU tolerance.
#[test]
fn calibration_stereo_sine_reads_target() {
    for target in [-23.0f32, -33.0] {
        let meter = measure_stereo(1_000.0, 48_000, amplitude_for_dbfs(target), 4.0);
        let values = meter.values();
        for (name, got) in [
            ("momentary", values.momentary_lufs),
            ("short-term", values.short_term_lufs),
            ("integrated", values.integrated_lufs),
        ] {
            assert!(
                (got - target).abs() <= 0.1,
                "{name} {got} LUFS off target {target} by > 0.1 LU"
            );
        }
    }
}

/// The calibration holds at 44.1 kHz too (sample-rate-independent K-weighting).
#[test]
fn calibration_holds_at_44100() {
    let meter = measure_stereo(1_000.0, 44_100, amplitude_for_dbfs(-23.0), 4.0);
    let integrated = meter.values().integrated_lufs;
    assert!(
        (integrated - -23.0).abs() <= 0.1,
        "44.1 kHz integrated {integrated} off -23 by > 0.1 LU"
    );
}

/// True-peak of a **full-scale 5 kHz sine at 44.1 kHz** is recovered to within
/// 0.1 dBTP of the true continuous peak (0 dBTP), using the shipped 4×
/// oversampling design (BS.1770-4 Annex 2 true-peak requirement).
#[test]
fn true_peak_5khz_full_scale_44k_within_0_1_dbtp() {
    let sample_rate = 44_100u32;
    let block = 4_410usize; // 100 ms
    let signal = sine(5_000.0, sample_rate, block, 1.0); // 0 dBFS peak

    // Default (high-quality) half-band design — the one the plugin ships.
    let mut meter = LoudnessMeter::new(1, sample_rate, block);
    for _ in 0..10 {
        meter.process_block(&[&signal]);
    }
    let values = meter.values();

    // The continuous peak of a unit sine is exactly 1.0 → 0 dBTP.
    assert!(
        (values.true_peak_dbtp - 0.0).abs() <= 0.1,
        "true peak {} dBTP not within 0.1 of 0.0",
        values.true_peak_dbtp
    );
    // Sample peak misses the inter-sample crest, so it reads lower.
    assert!(
        values.sample_peak_dbfs < values.true_peak_dbtp,
        "sample peak {} should be below true peak {}",
        values.sample_peak_dbfs,
        values.true_peak_dbtp
    );
}

/// Integrated gating — **absolute gate**: digital silence appended to a
/// programme is excluded, so the integrated value tracks the programme only
/// (per the −70 LUFS absolute gate).
#[test]
fn integrated_excludes_silence() {
    let sample_rate = 48_000u32;
    let block = 4_800usize;
    let programme = sine(1_000.0, sample_rate, block, amplitude_for_dbfs(-23.0));
    let silence = vec![0.0f32; block];

    let mut meter = LoudnessMeter::with_true_peak_design(2, sample_rate, block, LIGHT.0, LIGHT.1);
    for _ in 0..40 {
        meter.process_block(&[&programme, &programme]);
    }
    let before = meter.values().integrated_lufs;
    for _ in 0..40 {
        meter.process_block(&[&silence, &silence]);
    }
    let after = meter.values().integrated_lufs;

    assert!(
        (before - -23.0).abs() <= 0.1,
        "programme integrated {before}"
    );
    // Silence is absolute-gated out: the integrated value stays finite and near
    // −23 rather than collapsing toward −∞.  The small residual shift is from
    // the 400 ms gating blocks straddling the programme→silence cut (real,
    // per-standard), not from silent blocks leaking past the gate.
    assert!(after.is_finite(), "silence dragged integrated to -inf");
    assert!(
        (after - before).abs() <= 0.3,
        "silence shifted integrated from {before} to {after}"
    );
}

/// Integrated gating — **relative gate**: a passage more than 10 LU below the
/// loud programme is excluded, so the integrated value stays near the loud
/// part rather than the arithmetic average.
#[test]
fn integrated_excludes_quiet_passage_via_relative_gate() {
    let sample_rate = 48_000u32;
    let block = 4_800usize;
    let loud = sine(1_000.0, sample_rate, block, amplitude_for_dbfs(-18.0));
    let quiet = sine(1_000.0, sample_rate, block, amplitude_for_dbfs(-40.0));

    let mut meter = LoudnessMeter::with_true_peak_design(2, sample_rate, block, LIGHT.0, LIGHT.1);
    for _ in 0..40 {
        meter.process_block(&[&loud, &loud]);
    }
    for _ in 0..40 {
        meter.process_block(&[&quiet, &quiet]);
    }
    let integrated = meter.values().integrated_lufs;
    assert!(
        (integrated - -18.0).abs() <= 0.3,
        "relative gate failed: integrated {integrated} should stay near -18"
    );
}

/// K-weighting frequency response matches the standard curve shape: ~0 dB in
/// the 1 kHz reference region, the high-shelf plateau of ≈ +4 dB, and strong
/// RLB rejection of sub-bass (BS.1770-4 §2 / Tables 1–2; exact 48 kHz
/// coefficients are asserted in the `kweight` unit tests).
#[test]
fn kweighting_curve_matches_standard() {
    let sample_rate = 48_000u32;

    let gain_db = |freq: f32| -> f32 {
        let mut filter = KWeightingFilter::new(sample_rate);
        let total = sample_rate as usize;
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;
        for index in 0..total {
            let input = (TAU * freq * index as f32 / sample_rate as f32).sin();
            let output = filter.process_sample(input);
            if index > total * 3 / 4 {
                sum_sq += (output as f64).powi(2);
                count += 1;
            }
        }
        let out_rms = (sum_sq / count as f64).sqrt();
        // Input sine RMS is 1/√2; report gain relative to it.
        (20.0 * (out_rms / (1.0 / 2.0_f64.sqrt())).log10()) as f32
    };

    // 1 kHz reference: the K-weighting shelf contributes ≈ +0.69 dB here —
    // exactly what the −0.691 loudness offset is designed to cancel, so a
    // 1 kHz tone reads its dBFS level directly (see `amplitude_for_dbfs`).
    assert!(
        (gain_db(1_000.0) - 0.69).abs() < 0.3,
        "1 kHz gain {} not ~+0.69 dB",
        gain_db(1_000.0)
    );
    // High-shelf plateau: ≈ +4 dB by the top of the band.
    let high = gain_db(12_000.0);
    assert!((3.0..=4.5).contains(&high), "shelf plateau {high} dB");
    // RLB high-pass: sub-bass strongly rejected.
    assert!(gain_db(20.0) < -10.0, "20 Hz gain {}", gain_db(20.0));
    // Monotonic-ish rise from low-mid to high.
    assert!(
        gain_db(10_000.0) > gain_db(1_000.0),
        "high should be boosted over 1 kHz"
    );
}
