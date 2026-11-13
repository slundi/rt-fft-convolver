//! Schroeder all-pass diffusers and a series chain of them.
//!
//! # Role: input diffusion
//!
//! Fed straight into the FDN, a sharp transient (a pick attack) produces a
//! handful of discrete, audible early echoes before the network fills in.  A
//! short cascade of all-pass filters ahead of the network smears that transient
//! into a dense burst *without colouring the steady-state spectrum* — that is
//! the whole point of an all-pass: unit magnitude at every frequency, only the
//! phase (hence the time smearing) changes.  The `Diffusion` control sets the
//! shared feedback gain `g`.
//!
//! # Difference equation
//!
//! For a delay of `M` samples and coefficient `g` (Schroeder / Moorer form):
//!
//! ```text
//! d      = z^−M of the delay input           (the M-sample-old delayed value)
//! u[n]   = x[n] + g·d                         (value pushed into the delay)
//! y[n]   = d − g·u[n]                          (all-pass output)
//! ```
//!
//! which realises `H(z) = (z^−M − g) / (1 − g·z^−M)`, an all-pass
//! (`|H(e^{jω})| = 1` for all ω) as long as `|g| < 1`.
//!
//! # Real-time safety
//!
//! [`AllPass::process`] / [`AllPassChain::process`] allocate nothing, and each
//! delay write is flushed to zero to keep the recursive path denormal-free.

use rt_fft_convolver::flush_to_zero;

/// Largest usable magnitude for the all-pass coefficient; `|g| → 1` makes the
/// filter ring indefinitely, so we cap short of it for stability.
const MAX_GAIN: f32 = 0.9;

/// A single Schroeder all-pass section with an integer delay of `M` samples.
#[derive(Debug)]
pub struct AllPass {
    /// Delay storage of exactly `M` samples (the all-pass delay length).
    buf: Vec<f32>,
    /// Circular read/write cursor.
    idx: usize,
    /// Feedback / feed-forward coefficient, `|g| ≤ MAX_GAIN`.
    gain: f32,
}

impl AllPass {
    /// Create an all-pass with an `delay_samples`-sample delay and coefficient
    /// `gain` (clamped to `±MAX_GAIN`).
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(delay_samples: usize, gain: f32) -> Self {
        let mut allpass = Self {
            buf: vec![0.0; delay_samples.max(1)],
            idx: 0,
            gain: 0.0,
        };
        allpass.set_gain(gain);
        allpass
    }

    /// Set the coefficient, clamped to `±MAX_GAIN` for guaranteed stability.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(-MAX_GAIN, MAX_GAIN);
    }

    /// Clear the delay memory.
    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|sample| *sample = 0.0);
        self.idx = 0;
    }

    /// Process one sample.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buf[self.idx];
        let stored = input + self.gain * delayed;
        self.buf[self.idx] = flush_to_zero(stored);
        self.idx += 1;
        if self.idx == self.buf.len() {
            self.idx = 0;
        }
        delayed - self.gain * stored
    }
}

/// A series cascade of [`AllPass`] sections sharing one diffusion coefficient.
#[derive(Debug)]
pub struct AllPassChain {
    stages: Vec<AllPass>,
}

impl AllPassChain {
    /// Build a chain with the given per-stage delay lengths, all set to `gain`.
    ///
    /// **Allocates** — construct off the render thread.
    pub fn new(delays: &[usize], gain: f32) -> Self {
        Self {
            stages: delays
                .iter()
                .map(|&delay| AllPass::new(delay, gain))
                .collect(),
        }
    }

    /// Set the shared diffusion coefficient on every stage.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn set_gain(&mut self, gain: f32) {
        for stage in &mut self.stages {
            stage.set_gain(gain);
        }
    }

    /// Clear every stage's delay memory.
    pub fn reset(&mut self) {
        for stage in &mut self.stages {
            stage.reset();
        }
    }

    /// Process one sample through the whole cascade.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let mut sample = input;
        for stage in &mut self.stages {
            sample = stage.process(sample);
        }
        sample
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Feed a sine through the all-pass and confirm the steady-state magnitude
    /// is unity (that is the defining property of an all-pass).
    fn steady_magnitude(mut allpass: AllPass, freq: f32, sample_rate: f32) -> f32 {
        let mut peak = 0.0f32;
        let total = 8_192;
        for n in 0..total {
            let x = (TAU * freq * n as f32 / sample_rate).sin();
            let y = allpass.process(x);
            if n > total * 3 / 4 {
                peak = peak.max(y.abs());
            }
        }
        peak
    }

    #[test]
    fn allpass_has_unit_magnitude() {
        // A few frequencies should all pass at ~unity gain.
        for &freq in &[100.0, 1_000.0, 5_000.0] {
            let mag = steady_magnitude(AllPass::new(97, 0.7), freq, 48_000.0);
            assert!(
                (mag - 1.0).abs() < 0.05,
                "all-pass magnitude at {freq} Hz was {mag}, not ~1.0"
            );
        }
    }

    #[test]
    fn gain_is_clamped_for_stability() {
        let mut allpass = AllPass::new(13, 5.0); // absurd request
        allpass.process(1.0); // impulse in
        // Run a while; the clamped coefficient must keep it bounded.
        for _ in 0..10_000 {
            let value = allpass.process(0.0);
            assert!(value.is_finite() && value.abs() < 10.0);
        }
    }

    #[test]
    fn zero_gain_is_pure_delay() {
        let mut allpass = AllPass::new(4, 0.0);
        let mut out = Vec::new();
        for n in 0..10 {
            out.push(allpass.process(if n == 0 { 1.0 } else { 0.0 }));
        }
        // g = 0 → y[n] = x[n−M], a pure 4-sample delay.
        assert!((out[4] - 1.0).abs() < 1e-6, "{out:?}");
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn chain_preserves_energy_broadband() {
        // A chain of all-passes is still all-pass: white-ish input energy in ≈
        // energy out (allowing for the transient fill).
        let mut chain = AllPassChain::new(&[142, 107, 379, 277], 0.7);
        let mut energy_in = 0.0f32;
        let mut energy_out = 0.0f32;
        for n in 0..20_000 {
            let x = (n as f32 * 0.3).sin() * (n as f32 * 0.017).cos();
            let y = chain.process(x);
            if n > 2_000 {
                energy_in += x * x;
                energy_out += y * y;
            }
        }
        let ratio = energy_out / energy_in;
        assert!((ratio - 1.0).abs() < 0.05, "energy ratio {ratio} not ~1.0");
    }

    #[test]
    fn reset_clears_chain() {
        let mut chain = AllPassChain::new(&[7, 11], 0.6);
        chain.process(1.0);
        chain.reset();
        for _ in 0..40 {
            assert!(chain.process(0.0).abs() < 1e-6);
        }
    }
}
