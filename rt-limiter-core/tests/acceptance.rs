//! Phase-4 acceptance suite for `rt-limiter-core` (Roadmap-07 Phase 4).
//!
//! Each test maps to a bullet in the roadmap's "Unit tests" / "RT-safety"
//! lists, exercised through the public [`Limiter`] API at the real (default
//! 96 dB) resampler design:
//!
//! - [`output_true_peak_never_exceeds_ceiling`] — the brick-wall guarantee for
//!   an aliasing-prone signal at every tested oversampling factor.
//! - [`sub_threshold_is_bit_identical_to_delayed_dry`] — bypass equivalence.
//! - [`attack_reaches_target_within_lookahead`] — attack timing, ± 1 sample.
//! - [`release_returns_to_unity_within_specified_time`] — release timing, ± 10 %.
//! - [`reported_latency_is_sample_accurate`] — reported latency == real delay.
//! - [`denormal_guard_flushes_the_tail`] — FTZ guard neutralises subnormals.

use rt_fft_convolver::{DenormalGuard, db_to_linear};
use rt_limiter_core::{ChannelLink, Limiter, ReleaseCurve, TruePeakDetector};
use std::f32::consts::TAU;

const SR: f32 = 48_000.0;

/// Process one sample (fed to both channels) and return `(output, gain)` where
/// `gain` is the peak reduction applied to that sample.
fn step(limiter: &mut Limiter, sample: f32) -> (f32, f32) {
    let mut left = [sample];
    let mut right = [sample];
    limiter.process_block(&mut left, &mut right);
    (left[0], limiter.min_gain())
}

// ── Ceiling guarantee ────────────────────────────────────────────────────────

/// The output true-peak (measured at the same factor the limiter controls)
/// never exceeds the ceiling, for a hot near-Nyquist signal engineered to have
/// large inter-sample peaks — at every tested oversampling factor.
#[test]
fn output_true_peak_never_exceeds_ceiling() {
    const MAX_BLOCK: usize = 512;
    for factor in [2usize, 4, 8, 16] {
        for ceiling_db in [-1.0f32, -6.0] {
            // threshold == ceiling: the limiter pulls the true-peak to it.
            let mut limiter = Limiter::new(
                SR,
                MAX_BLOCK,
                factor,
                ceiling_db,
                ceiling_db,
                50.0,
                ReleaseCurve::Exponential,
                ChannelLink::Linked,
            );

            let total = MAX_BLOCK * 12;
            let make = |n: usize| 1.9 * (TAU * 0.23 * n as f32 + 0.7).sin();
            let mut collected = Vec::with_capacity(total);
            for start in (0..total).step_by(MAX_BLOCK) {
                let mut left: Vec<f32> = (start..start + MAX_BLOCK).map(make).collect();
                let mut right = left.clone();
                limiter.process_block(&mut left, &mut right);
                collected.extend_from_slice(&left);
            }

            let mut probe = TruePeakDetector::new(factor, MAX_BLOCK);
            let mut peaks = vec![0.0f32; MAX_BLOCK];
            let ceiling = db_to_linear(ceiling_db);
            let mut worst = 0.0f32;
            for (block_index, chunk) in collected.chunks(MAX_BLOCK).enumerate() {
                probe.detect_block(chunk, &mut peaks);
                if block_index >= 3 {
                    worst = worst.max(peaks.iter().copied().fold(0.0, f32::max));
                }
            }
            assert!(
                worst <= ceiling * 1.02,
                "factor {factor}, ceiling {ceiling_db} dB: output true-peak {worst} \
                 exceeds ceiling {ceiling}"
            );
        }
    }
}

// ── Bypass equivalence ─────────────────────────────────────────────────────

/// A signal that never exceeds the threshold comes out bit-identical to the
/// delayed dry signal (no gain change, no clipping, no dither).
#[test]
fn sub_threshold_is_bit_identical_to_delayed_dry() {
    const MAX_BLOCK: usize = 256;
    let mut limiter = Limiter::new(
        SR,
        MAX_BLOCK,
        4,
        0.0, // threshold 0 dBFS
        0.0, // ceiling 0 dBFS
        50.0,
        ReleaseCurve::Exponential,
        ChannelLink::Linked,
    );
    let latency = limiter.latency_samples();

    let total = MAX_BLOCK * 8;
    // 0.3 peak → always sub-threshold, so gain stays exactly 1.0.
    let source: Vec<f32> = (0..total)
        .map(|n| 0.3 * (TAU * 0.01 * n as f32).sin())
        .collect();

    let mut output = Vec::with_capacity(total);
    for chunk in source.chunks(MAX_BLOCK) {
        let mut left = chunk.to_vec();
        let mut right = chunk.to_vec();
        limiter.process_block(&mut left, &mut right);
        output.extend_from_slice(&left);
    }

    for index in latency..total {
        assert_eq!(
            output[index],
            source[index - latency],
            "sample {index} not bit-identical to the delayed dry signal"
        );
    }
}

// ── Attack timing (± 1 sample) ─────────────────────────────────────────────

/// Gain reduction reaches the target exactly `look-ahead` samples before the
/// offending audio leaves the delay line (± 1 sample).
///
/// Uses factor 1×, whose detector is a zero-latency `|x|` envelope with no
/// linear-phase pre-ringing, so the timing is unambiguous.
#[test]
fn attack_reaches_target_within_lookahead() {
    let mut limiter = Limiter::new(
        SR,
        1, // per-sample processing
        1,
        -6.0,
        0.0,
        50.0,
        ReleaseCurve::Exponential,
        ChannelLink::Linked,
    );
    let lookahead = limiter.lookahead_samples();
    let latency = limiter.latency_samples();
    let threshold = db_to_linear(-6.0);

    let onset = 300usize; // input index where the loud region begins
    let total = onset + latency + 400;

    let mut gr_hit: Option<usize> = None;
    let mut loud_out: Option<usize> = None;
    for index in 0..total {
        let input = if index >= onset { 1.0 } else { 0.0 };
        let (output, gain) = step(&mut limiter, input);
        if gr_hit.is_none() && gain <= threshold * 1.02 {
            gr_hit = Some(index);
        }
        if loud_out.is_none() && output.abs() >= threshold * 0.5 {
            loud_out = Some(index);
        }
    }

    let gr_hit = gr_hit.expect("gain reduction never reached the target");
    let loud_out = loud_out.expect("loud audio never reached the output");

    // Reduction engages as the peak enters the look-ahead window (≈ onset)...
    assert!(
        (gr_hit as isize - onset as isize).abs() <= 1,
        "GR engaged at {gr_hit}, expected ≈ {onset}"
    );
    // ...exactly `look-ahead` samples before the loud audio is output (± 1).
    let anticipation = loud_out as isize - gr_hit as isize;
    assert!(
        (anticipation - lookahead as isize).abs() <= 1,
        "anticipation {anticipation} samples, expected look-ahead {lookahead} ± 1"
    );
}

// ── Release timing (± 10 %) ────────────────────────────────────────────────

/// With the linear curve the gain returns to unity within the specified
/// release time (± 10 %), measured from the moment recovery actually begins
/// (after the peak has cleared the look-ahead window).
#[test]
fn release_returns_to_unity_within_specified_time() {
    let release_ms = 50.0;
    let mut limiter = Limiter::new(
        SR,
        1,
        1,
        -6.0,
        0.0,
        release_ms,
        ReleaseCurve::Linear,
        ChannelLink::Linked,
    );
    let lookahead = limiter.lookahead_samples();

    // Drive with full scale so the gain settles at threshold/peak.
    let mut settled_gain = 1.0;
    for _ in 0..(lookahead + 200) {
        let (_out, gain) = step(&mut limiter, 1.0);
        settled_gain = gain;
    }
    assert!(settled_gain < 0.9, "limiter did not engage: {settled_gain}");

    // Switch to silence and time the recovery to unity.
    let mut release_start: Option<usize> = None;
    let mut reached_unity: Option<usize> = None;
    for index in 0..20_000 {
        let (_out, gain) = step(&mut limiter, 0.0);
        if release_start.is_none() && gain > settled_gain + 1e-4 {
            release_start = Some(index);
        }
        if release_start.is_some() && gain >= 1.0 - 1e-4 {
            reached_unity = Some(index);
            break;
        }
    }

    let release_start = release_start.expect("release never started");
    let reached_unity = reached_unity.expect("gain never returned to unity");
    let measured = (reached_unity - release_start) as f32;

    // Linear release crosses the full 0→1 range in `release_ms`; from
    // `settled_gain` it covers `(1 − settled_gain)` of that span.
    let expected = (1.0 - settled_gain) * release_ms * 1e-3 * SR;
    let ratio = measured / expected;
    assert!(
        (0.9..=1.1).contains(&ratio),
        "release took {measured} samples, expected ≈ {expected} (ratio {ratio:.3})"
    );
}

// ── Reported latency accuracy ──────────────────────────────────────────────

/// The reported latency equals the actual sample-accurate delay: a
/// sub-threshold impulse emerges delayed by exactly `latency_samples()`, for
/// several look-ahead settings.
#[test]
fn reported_latency_is_sample_accurate() {
    const MAX_BLOCK: usize = 256;
    for lookahead_ms in [0.1f32, 2.0, 5.0, 20.0] {
        let mut limiter = Limiter::new(
            SR,
            MAX_BLOCK,
            4,
            0.0,
            0.0,
            50.0,
            ReleaseCurve::Exponential,
            ChannelLink::Linked,
        );
        limiter.set_lookahead_ms(lookahead_ms);
        let latency = limiter.latency_samples();

        let total = MAX_BLOCK * 6;
        let mut source = vec![0.0f32; total];
        source[0] = 0.25; // small → sub-threshold → linear path

        let mut output = Vec::with_capacity(total);
        for chunk in source.chunks(MAX_BLOCK) {
            let mut left = chunk.to_vec();
            let mut right = chunk.to_vec();
            limiter.process_block(&mut left, &mut right);
            output.extend_from_slice(&left);
        }

        let (peak_index, _) =
            output
                .iter()
                .enumerate()
                .fold((0usize, 0.0f32), |best, (index, &value)| {
                    if value.abs() > best.1 {
                        (index, value.abs())
                    } else {
                        best
                    }
                });
        assert_eq!(
            peak_index, latency,
            "look-ahead {lookahead_ms} ms: impulse delayed by {peak_index}, reported {latency}"
        );
    }
}

// ── Denormal guard ─────────────────────────────────────────────────────────

/// With a [`DenormalGuard`] active for the render, subnormal inputs never
/// survive into the output (FTZ/DAZ flushes them), so the ring-down never
/// carries the subnormals that would spike CPU.
#[test]
fn denormal_guard_flushes_the_tail() {
    const MAX_BLOCK: usize = 128;
    let mut limiter = Limiter::new(
        SR,
        MAX_BLOCK,
        4,
        0.0,
        0.0,
        50.0,
        ReleaseCurve::Exponential,
        ChannelLink::Linked,
    );

    let _guard = DenormalGuard::new();
    // A block of subnormal magnitudes (below f32::MIN_POSITIVE ≈ 1.18e-38).
    let subnormal = 1.0e-40f32;
    for _ in 0..8 {
        let mut left = vec![subnormal; MAX_BLOCK];
        let mut right = vec![subnormal; MAX_BLOCK];
        limiter.process_block(&mut left, &mut right);
        for &sample in left.iter().chain(right.iter()) {
            // Under FTZ/DAZ, every product is flushed: outputs are 0 or normal.
            assert!(
                sample == 0.0 || sample.is_normal(),
                "subnormal {sample:e} survived the guarded render"
            );
        }
    }
}
