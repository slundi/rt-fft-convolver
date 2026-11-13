//! End-to-end integration tests for the public `rt-limiter-core` API.
//!
//! These exercise [`Limiter`] through its published surface (default 96 dB
//! resampler design) rather than the light designs the unit tests inject, so
//! they double as a smoke test of the real construction path.

use rt_fft_convolver::db_to_linear;
use rt_limiter_core::{ChannelLink, Limiter, ReleaseCurve, TruePeakDetector};
use std::f32::consts::TAU;

const SAMPLE_RATE: f32 = 48_000.0;
const MAX_BLOCK: usize = 512;

/// Run a mono source (fed to both channels) through the limiter in blocks and
/// return the concatenated left output.
fn run(limiter: &mut Limiter, source: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(source.len());
    for chunk in source.chunks(MAX_BLOCK) {
        let mut left = chunk.to_vec();
        let mut right = chunk.to_vec();
        limiter.process_block(&mut left, &mut right);
        output.extend_from_slice(&left);
    }
    output
}

/// A hot, aliasing-prone signal is held at or below the ceiling at 4× — the
/// headline brick-wall guarantee, through the real (default-design) engine.
#[test]
fn brickwall_holds_at_default_design() {
    let ceiling_db = -1.0;
    let mut limiter = Limiter::new(
        SAMPLE_RATE,
        MAX_BLOCK,
        4,
        ceiling_db,
        ceiling_db,
        50.0,
        ReleaseCurve::Exponential,
        ChannelLink::Linked,
    );

    let total = MAX_BLOCK * 12;
    // Near-Nyquist, off-grid, over-unity: maximal inter-sample peaks.
    let source: Vec<f32> = (0..total)
        .map(|n| 1.9 * (TAU * 0.23 * n as f32 + 0.7).sin())
        .collect();
    let output = run(&mut limiter, &source);

    // Measure the settled output's true-peak at the same 4× the limiter used.
    let mut probe = TruePeakDetector::new(4, MAX_BLOCK);
    let mut peaks = vec![0.0f32; MAX_BLOCK];
    let ceiling_lin = db_to_linear(ceiling_db);
    let mut worst = 0.0f32;
    for (block_index, chunk) in output.chunks(MAX_BLOCK).enumerate() {
        probe.detect_block(chunk, &mut peaks);
        if block_index >= 3 {
            worst = worst.max(peaks.iter().copied().fold(0.0, f32::max));
        }
    }
    assert!(
        worst <= ceiling_lin * 1.02,
        "true-peak {worst} exceeds ceiling {ceiling_lin}"
    );
}

/// Latency reported to the host equals the actual sample delay: a quiet
/// impulse comes out delayed by exactly `latency_samples()`.
#[test]
fn reported_latency_is_sample_accurate() {
    let mut limiter = Limiter::new(
        SAMPLE_RATE,
        MAX_BLOCK,
        4,
        0.0,
        0.0,
        50.0,
        ReleaseCurve::Exponential,
        ChannelLink::Linked,
    );
    let latency = limiter.latency_samples();

    // A small impulse stays sub-threshold → passes through the linear path.
    let total = MAX_BLOCK * 4;
    let mut source = vec![0.0f32; total];
    source[0] = 0.25;
    let output = run(&mut limiter, &source);

    let (peak_index, _) = output
        .iter()
        .enumerate()
        .fold((0usize, 0.0f32), |best, (i, &s)| {
            if s.abs() > best.1 { (i, s.abs()) } else { best }
        });
    assert_eq!(
        peak_index, latency,
        "impulse delayed by {peak_index}, reported {latency}"
    );
}

/// Processing is deterministic: two runs of an identical stream from the same
/// initial state produce identical output (no hidden global RNG / time state).
#[test]
fn processing_is_deterministic() {
    let build = || {
        Limiter::new(
            SAMPLE_RATE,
            MAX_BLOCK,
            4,
            -3.0,
            -1.0,
            50.0,
            ReleaseCurve::Auto,
            ChannelLink::MidSide,
        )
    };
    let source: Vec<f32> = (0..MAX_BLOCK * 6)
        .map(|n| 1.2 * (TAU * 0.08 * n as f32).sin())
        .collect();

    let mut first = build();
    let mut second = build();
    assert_eq!(run(&mut first, &source), run(&mut second, &source));
}
