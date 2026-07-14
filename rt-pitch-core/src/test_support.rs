//! Shared test helpers (compiled only under `cfg(test)`).
//!
//! Every shifter is validated the same way: feed a pure sine, measure the
//! dominant output frequency, and check it landed where the ratio predicts.

use realfft::num_complex::Complex;
use rt_fft_convolver::FftHandler;
use std::f32::consts::TAU;

/// Generate `samples` of a unit-amplitude sine at `freq` Hz.
pub fn make_sine(freq: f32, sample_rate: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| (TAU * freq * n as f32 / sample_rate).sin())
        .collect()
}

/// Generate a naïve sawtooth at `freq` Hz — harmonically rich with a clear
/// once-per-period discontinuity, which is what the pitch-synchronous PSOLA
/// path is designed for (a pure sine has no such structure).  The fundamental
/// still carries the most energy, so a dominant-frequency probe finds it.
pub fn make_sawtooth(freq: f32, sample_rate: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|n| {
            let phase = freq * n as f32 / sample_rate;
            2.0 * (phase - (phase + 0.5).floor())
        })
        .collect()
}

/// Dominant spectral frequency (Hz) of the steady-state tail of `signal`,
/// measured with a 4096-point Hann-windowed DFT.
pub fn dominant_frequency_hz(signal: &[f32], sample_rate: f32) -> f32 {
    let len = 4096;
    assert!(signal.len() >= len, "signal too short to analyse");
    let start = signal.len() - len;
    let mut fft = FftHandler::new(len);
    let mut spectrum = vec![Complex::default(); len / 2 + 1];
    let windowed: Vec<f32> = signal[start..]
        .iter()
        .enumerate()
        .map(|(index, &sample)| {
            let w = 0.5 - 0.5 * (TAU * index as f32 / len as f32).cos();
            sample * w
        })
        .collect();
    fft.forward(&windowed, &mut spectrum);

    let mut best_bin = 0;
    let mut best_mag = 0.0;
    for (bin, value) in spectrum.iter().enumerate().skip(1) {
        let magnitude = value.norm();
        if magnitude > best_mag {
            best_mag = magnitude;
            best_bin = bin;
        }
    }
    best_bin as f32 * sample_rate / len as f32
}
