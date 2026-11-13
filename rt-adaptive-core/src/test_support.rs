//! Shared test helpers (compiled only under `cfg(test)`).

use std::f32::consts::TAU;

/// Generate `samples` of a unit-amplitude sine at `freq` Hz.
pub fn make_sine(freq: f32, sample_rate: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| (TAU * freq * n as f32 / sample_rate).sin())
        .collect()
}

/// Root-mean-square level of a signal — handy for energy comparisons.
pub fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum: f32 = signal.iter().map(|&s| s * s).sum();
    (sum / signal.len() as f32).sqrt()
}
