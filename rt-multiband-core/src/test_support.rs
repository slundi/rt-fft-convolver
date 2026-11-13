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
    let sum: f32 = signal.iter().map(|&sample| sample * sample).sum();
    (sum / signal.len() as f32).sqrt()
}

/// A deterministic, zero-mean pseudo-random broadband signal in `[-1, 1)`.
///
/// A tiny xorshift PRNG so the reconstruction / bypass tests have full-band
/// excitation without pulling in a dependency or `Math.random`-style
/// nondeterminism.
pub fn make_noise(samples: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..samples)
        .map(|_| {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            // Map the top 24 bits to [-1, 1).
            ((bits >> 40) as f32 / (1u32 << 23) as f32) - 1.0
        })
        .collect()
}
