//! Shared `#[cfg(test)]` signal helpers for the loudness-core unit tests.

use std::f32::consts::TAU;

/// A `samples`-long sine at `freq` Hz, `sample_rate` Hz, peak amplitude `amp`.
pub(crate) fn make_sine(freq: f32, sample_rate: u32, samples: usize, amp: f32) -> Vec<f32> {
    (0..samples)
        .map(|index| amp * (TAU * freq * index as f32 / sample_rate as f32).sin())
        .collect()
}

/// Root-mean-square of a buffer (linear).
#[allow(dead_code)]
pub(crate) fn rms(buffer: &[f32]) -> f32 {
    if buffer.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = buffer.iter().map(|&sample| (sample as f64).powi(2)).sum();
    (sum_sq / buffer.len() as f64).sqrt() as f32
}
