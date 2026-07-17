//! Phase-4 acceptance suite for `rt-multiband-core` (Roadmap-11 Phase 4).
//!
//! Each test maps to a bullet in the roadmap's "Unit tests" / "Sidechain" /
//! "RT-safety" lists, exercised through the public API at the real engine
//! level (the whole [`BandProcessor`] chain — detector → envelope → gain
//! computer — not just the stateless curve, which the module unit tests
//! already cover):
//!
//! - [`crossover_reconstruction_is_magnitude_flat`] — bypassed bands sum flat.
//! - [`compressor_20db_over_2to1_hardknee_is_minus_10db`] — the headline
//!   compressor acceptance, measured as *applied* gain reduction.
//! - [`expander_10db_under_2to1_is_minus_10db`] — downward expansion.
//! - [`attack_time_constant_within_ten_percent`] /
//!   [`release_time_constant_within_ten_percent`] — ballistics timing.
//! - [`internal_sidechain_hpf_gates_the_detector`] — the v1 (internal)
//!   sidechain path routes and tilts the detector.
//! - [`zero_bulk_latency_impulse_starts_at_sample_zero`] — the v1 IIR crossover
//!   adds no reported latency.
//!
//! # A note on "bit-identical" reconstruction
//!
//! The roadmap's unit-test bullet says bypassed bands sum "bit-identical
//! (IIR)".  A Linkwitz-Riley crossover sum is an **all-pass** — magnitude-flat
//! but phase-shifted — so the honest, achievable guarantee is *magnitude-flat*
//! reconstruction (< 0.1 dB), not sample-for-sample identity.  Bit-identity
//! would require a subtractive crossover, which sacrifices the 24 dB/oct
//! high-band slope; Phase 0 chose LR4, so this suite asserts flat magnitude.

use std::f32::consts::TAU;

use rt_multiband_core::{
    BandMode, BandProcessor, Crossover, EnvelopeFollower, MultibandEngine, NUM_CROSSOVERS,
};

const SR: f32 = 48_000.0;
const FREQS: [f32; NUM_CROSSOVERS] = [200.0, 1_000.0, 5_000.0];

fn make_sine(freq: f32, amplitude: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| amplitude * (TAU * freq * n as f32 / SR).sin())
        .collect()
}

fn rms(signal: &[f32]) -> f32 {
    (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

/// A compressor band with fast attack + a long hold so the peak detector's
/// envelope settles near the signal peak (steady-state gain reduction).
fn steady_band(mode: BandMode, threshold_db: f32, ratio: f32) -> BandProcessor {
    BandProcessor::new(
        SR,
        1.0,   // 1 ms attack — snaps onto the level
        200.0, // 200 ms release — holds near the peak between cycles
        2.0,
        1.0, // peak detector (blend = 1)
        mode,
        threshold_db,
        ratio,
        0.0, // hard knee
    )
}

/// Feed a sustained tone through a band and return the deepest (most negative)
/// and shallowest (closest to 0) gain reduction over the settled tail.
fn settled_gain_reduction(band: &mut BandProcessor, freq: f32, amplitude: f32) -> (f32, f32) {
    let tone = make_sine(freq, amplitude, 24_000);
    let mut deepest = 0.0f32;
    let mut shallowest = f32::NEG_INFINITY;
    for (n, &sample) in tone.iter().enumerate() {
        band.process(sample, sample);
        if n > 18_000 {
            let gr = band.gain_reduction_db();
            deepest = deepest.min(gr);
            shallowest = shallowest.max(gr);
        }
    }
    (deepest, shallowest)
}

// ── Crossover reconstruction ─────────────────────────────────────────────────

#[test]
fn crossover_reconstruction_is_magnitude_flat() {
    let mut crossover = Crossover::new(SR, FREQS);
    // Full-band probe tones, including the crossover edges, reconstruct flat.
    for &freq in &[40.0, 200.0, 700.0, 1_000.0, 5_000.0, 14_000.0] {
        let mut fresh = Crossover::new(SR, FREQS);
        let input = make_sine(freq, 0.5, 24_000);
        let summed: Vec<f32> = input.iter().map(|&x| fresh.split(x).iter().sum()).collect();
        let err_db = 20.0 * (rms(&summed[12_000..]) / rms(&input[12_000..])).log10();
        assert!(
            err_db.abs() < 0.2,
            "at {freq} Hz reconstruction {err_db} dB"
        );
    }
    // And a broadband impulse response sums flat too.
    let mut sum_energy = 0.0f32;
    let mut in_energy = 0.0f32;
    for n in 0..8_000 {
        let x = if n == 0 { 1.0 } else { 0.0 };
        let summed: f32 = crossover.split(x).iter().sum();
        sum_energy += summed * summed;
        in_energy += x * x;
    }
    let err_db = 10.0 * (sum_energy / in_energy).log10();
    assert!(err_db.abs() < 0.1, "impulse reconstruction {err_db} dB");
}

// ── Compressor / expander static acceptance (end-to-end) ─────────────────────

#[test]
fn compressor_20db_over_2to1_hardknee_is_minus_10db() {
    // A 0 dBFS tone into a −20 dB threshold is 20 dB over; at 2:1 the output is
    // 10 dB over → 10 dB of gain reduction.  Measured as the applied reduction.
    let mut band = steady_band(BandMode::Compressor, -20.0, 2.0);
    let (deepest, _) = settled_gain_reduction(&mut band, 1_000.0, 1.0);
    assert!(
        (deepest - -10.0).abs() < 1.0,
        "compressor GR {deepest} dB, want −10"
    );
}

#[test]
fn expander_10db_under_2to1_is_minus_10db() {
    // A −30 dBFS tone is 10 dB below a −20 dB threshold; a 2:1 expander pushes
    // it to −40 → 10 dB of downward expansion at the signal peak.
    let mut band = steady_band(BandMode::Expander, -20.0, 2.0);
    let amplitude = rt_fft_convolver::db_to_linear(-30.0);
    let (_, shallowest) = settled_gain_reduction(&mut band, 1_000.0, amplitude);
    assert!(
        (shallowest - -10.0).abs() < 1.5,
        "expander GR {shallowest} dB, want −10"
    );
}

// ── Ballistics timing ────────────────────────────────────────────────────────

#[test]
fn attack_time_constant_within_ten_percent() {
    // The envelope reaches 1 − 1/e of a unit step in one attack time constant.
    let attack_ms = 10.0;
    let mut follower = EnvelopeFollower::new(SR, attack_ms, 100.0);
    let target = 1.0 - std::f32::consts::E.recip();
    let mut samples = 0;
    while follower.process(1.0) < target {
        samples += 1;
        assert!(samples < SR as usize, "attack never reached target");
    }
    let expected = attack_ms * 1e-3 * SR;
    let error = (samples as f32 - expected).abs() / expected;
    assert!(error < 0.1, "attack τ off by {}%", error * 100.0);
}

#[test]
fn release_time_constant_within_ten_percent() {
    // From full charge, the envelope decays to 1/e in one release time constant.
    let release_ms = 80.0;
    let mut follower = EnvelopeFollower::new(SR, 0.01, release_ms);
    for _ in 0..SR as usize {
        follower.process(1.0);
    }
    let target = std::f32::consts::E.recip();
    let mut samples = 0;
    while follower.process(0.0) > target {
        samples += 1;
        assert!(samples < SR as usize, "release never reached target");
    }
    let expected = release_ms * 1e-3 * SR;
    let error = (samples as f32 - expected).abs() / expected;
    assert!(error < 0.1, "release τ off by {}%", error * 100.0);
}

// ── Sidechain (internal, v1) ─────────────────────────────────────────────────

#[test]
fn internal_sidechain_hpf_gates_the_detector() {
    // A low tone that a high SC-HPF filters out of the detector should trigger
    // far less compression than the same tone with the SC filter open.
    let low_freq = 100.0;

    let mut open = steady_band(BandMode::Compressor, -30.0, 8.0);
    let (open_gr, _) = settled_gain_reduction(&mut open, low_freq, 0.8);

    let mut gated = steady_band(BandMode::Compressor, -30.0, 8.0);
    gated.sidechain_mut(0).set_hpf_hz(1_000.0);
    gated.sidechain_mut(1).set_hpf_hz(1_000.0);
    let (gated_gr, _) = settled_gain_reduction(&mut gated, low_freq, 0.8);

    assert!(open_gr < -6.0, "SC-open band should compress: {open_gr} dB");
    assert!(
        gated_gr > open_gr + 6.0,
        "SC HPF should gate the low tone (open {open_gr} dB vs gated {gated_gr} dB)"
    );
}

// ── RT-safety: reported latency ──────────────────────────────────────────────

#[test]
fn zero_bulk_latency_impulse_starts_at_sample_zero() {
    // The IIR crossover is minimum-phase / causal: an impulse produces a
    // non-zero output on the very first sample, so the engine adds no bulk
    // delay (v1 reports zero latency; linear-phase FIR latency is a v2 concern).
    let mut engine = MultibandEngine::new(SR, FREQS);
    for index in 0..engine.num_bands() {
        engine
            .band_mut(index)
            .gain_computer_mut()
            .set_mode(BandMode::Bypass);
    }
    let (left0, right0) = engine.process(1.0, 1.0);
    assert!(
        left0.abs() > 1e-6,
        "no output at sample 0 (unexpected delay)"
    );
    assert!(
        right0.abs() > 1e-6,
        "no output at sample 0 (unexpected delay)"
    );
}
