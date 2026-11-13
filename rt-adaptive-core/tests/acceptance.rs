//! Phase-4 acceptance suite for `rt-adaptive-core` (Roadmap-09 Phase 4
//! "Unit tests" bullets), exercised through the crate's public API.
//!
//! - [`nlms_converges_to_null_within_documented_bound`] — LMS/NLMS convergence.
//! - [`notch_bank_attenuates_1khz_by_over_20db`] — the notch-bank criterion.
//! - [`hum_tracker_identifies_fundamental_within_half_hz`] — the hum-tracker
//!   criterion.
//! - [`feedback_detector_locates_howl_frequency`] — the detector criterion.

use rt_adaptive_core::{AdaptiveFilter, FeedbackDetector, HumTracker, NotchBank};

const SR: f32 = 48_000.0;

fn sine(freq: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| (std::f32::consts::TAU * freq * n as f32 / SR).sin())
        .collect()
}

/// Deterministic pseudo-white noise (xorshift) — no `rand` dependency.
fn noise(len: usize, amplitude: f32) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * amplitude
        })
        .collect()
}

fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Goertzel power of `signal` at `freq` — energy in one narrow band.
fn power_at(signal: &[f32], freq: f32) -> f64 {
    let omega = std::f64::consts::TAU * freq as f64 / SR as f64;
    let coeff = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &sample in signal {
        let s0 = sample as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

#[test]
fn nlms_converges_to_null_within_documented_bound() {
    // System identification: reference = white noise, desired = its echo through
    // a known FIR path.  NLMS must drive the residual to < −60 dBFS (rms 1e-3)
    // within the documented bound of 15 000 samples (~0.31 s at 48 kHz).
    const ECHO_PATH: [f32; 4] = [0.6, -0.3, 0.15, 0.05];
    const BOUND: usize = 15_000;

    let mut filter = AdaptiveFilter::<4>::nlms(0.5);
    let reference = noise(20_000, 0.5);
    let mut history = [0.0f32; 4];
    let mut residuals = Vec::with_capacity(reference.len());
    for &sample in &reference {
        history.copy_within(0..3, 1);
        history[0] = sample;
        let desired: f32 = ECHO_PATH
            .iter()
            .zip(history.iter())
            .map(|(h, x)| h * x)
            .sum();
        let estimate = filter.predict(sample);
        let error = desired - estimate;
        filter.adapt(error);
        residuals.push(error);
    }
    let converged = rms(&residuals[BOUND..]);
    assert!(
        converged < 1e-3,
        "residual {converged} above the 1e-3 bound"
    );
}

#[test]
fn notch_bank_attenuates_1khz_by_over_20db() {
    // Roadmap bar: a 1 kHz notch at Q = 10 attenuates a 1 kHz sine by > 20 dB.
    let mut bank = NotchBank::new(SR as u32);
    assert!(bank.allocate(1_000.0, 10.0).is_some());

    let mut buf = sine(1_000.0, 48_000);
    let before = power_at(&buf[24_000..], 1_000.0);
    bank.process_block(&mut buf);
    let after = power_at(&buf[24_000..], 1_000.0);

    let attenuation_db = 10.0 * (before / after.max(1e-30)).log10();
    assert!(
        attenuation_db > 20.0,
        "only {attenuation_db:.1} dB of rejection"
    );
}

#[test]
fn hum_tracker_identifies_fundamental_within_half_hz() {
    // A 59.7 Hz mains hum (+ harmonics) drifted off the 60 Hz nominal must be
    // identified within ± 0.5 Hz by the adaptive tracker.
    let mut tracker = HumTracker::new(SR as u32, 60.0);
    tracker.set_max_drift(1.0);

    let fundamental = 59.7;
    let f1 = sine(fundamental, SR as usize * 3);
    let f2 = sine(fundamental * 2.0, SR as usize * 3);
    let f3 = sine(fundamental * 3.0, SR as usize * 3);
    let signal: Vec<f32> = (0..f1.len())
        .map(|n| f1[n] + 0.5 * f2[n] + 0.25 * f3[n])
        .collect();

    for chunk in signal.chunks(2048) {
        let mut buf = chunk.to_vec();
        tracker.process_block(&mut buf);
    }
    assert!(
        (tracker.tracked_hz() - fundamental).abs() < 0.5,
        "tracked {} Hz, off by {}",
        tracker.tracked_hz(),
        (tracker.tracked_hz() - fundamental).abs()
    );
}

#[test]
fn feedback_detector_locates_howl_frequency() {
    // A sustained 2 kHz howl over a quiet floor is located within ± 100 Hz.
    let mut detector = FeedbackDetector::new(SR, 512);
    let tone = sine(2_000.0, 512 * 8);
    let floor = noise(tone.len(), 0.02);
    let mixed: Vec<f32> = tone.iter().zip(&floor).map(|(t, n)| t + n).collect();

    let mut found = None;
    for chunk in mixed.chunks(512) {
        if let Some(freq) = detector.process(chunk) {
            found = Some(freq);
            break;
        }
    }
    let freq = found.expect("howl not detected");
    assert!((freq - 2_000.0).abs() < 100.0, "located {freq} Hz");
}
