//! First-order all-pass fractional-delay filter — the sub-sample tuning element
//! of a Karplus-Strong string.
//!
//! # Why an all-pass
//!
//! A string plays in tune only if the *total* loop delay equals `fs / f0`
//! samples exactly.  The ring buffer can only supply an integer number of
//! samples, so the leftover fraction `D ∈ [0, 1)` is realised by an all-pass
//!
//! ```text
//! A(z) = (a + z⁻¹) / (1 + a·z⁻¹),   a = (1 − D) / (1 + D)
//! ```
//!
//! whose magnitude response is unity at every frequency (so it does not colour
//! the string) while its **phase delay** at low frequencies is `≈ D` samples.
//! Choosing `a` from `D` this way places the delay where we need it without any
//! amplitude loss — see the Phase 0 tuning example in
//! `docs/Roadmap-06-physical-modeling.md`.
//!
//! # Conditioning
//!
//! The coefficient is well-behaved for `D` roughly in `[0.1, 1)`.  Very small
//! `D` pushes `a` toward 1 and makes the transient ring; the string tuning code
//! ([`crate::string`]) borrows one integer sample to keep `D` in that range, so
//! this type only clamps defensively.
//!
//! # Real-time safety
//!
//! [`process`](TuningAllpass::process) is two multiply-adds; `set_delay` is one
//! divide.  No allocation, no locking.  The state is flushed to zero on write
//! so a decaying loop stays out of the denormal slow path.

use rt_fft_convolver::flush_to_zero;

/// A first-order all-pass used purely as a fractional-sample delay.
#[derive(Debug, Clone, Copy)]
pub struct TuningAllpass {
    /// All-pass coefficient `a = (1 − D) / (1 + D)`.
    coeff: f32,
    /// Previous input `x[n−1]`.
    x1: f32,
    /// Previous output `y[n−1]`.
    y1: f32,
}

impl TuningAllpass {
    /// Create an all-pass approximating a fractional delay of `frac` samples.
    pub fn new(frac: f32) -> Self {
        let mut allpass = Self {
            coeff: 0.0,
            x1: 0.0,
            y1: 0.0,
        };
        allpass.set_delay(frac);
        allpass
    }

    /// Retune to a fractional delay of `frac` samples (state preserved).
    ///
    /// **Real-time safe.**  `frac` is clamped to `[0.01, 0.999]` to keep the
    /// coefficient finite and the filter well conditioned.
    #[inline]
    pub fn set_delay(&mut self, frac: f32) {
        let delay = frac.clamp(0.01, 0.999);
        self.coeff = (1.0 - delay) / (1.0 + delay);
    }

    /// The current all-pass coefficient (mostly for tests / introspection).
    #[inline]
    pub fn coeff(&self) -> f32 {
        self.coeff
    }

    /// Clear the filter memory.
    #[inline]
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }

    /// Filter one sample: `y[n] = a·x[n] + x[n−1] − a·y[n−1]`.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.coeff * input + self.x1 - self.coeff * self.y1;
        self.x1 = input;
        self.y1 = flush_to_zero(output);
        self.y1
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// The all-pass must not change the magnitude of a steady tone (|H| = 1).
    #[test]
    fn magnitude_is_unity() {
        const SR: f32 = 48_000.0;
        for &freq in &[100.0, 1_000.0, 5_000.0] {
            let mut allpass = TuningAllpass::new(0.5);
            let mut sum_sq = 0.0f32;
            let mut count = 0u32;
            for n in 0..48_000 {
                let x = (TAU * freq * n as f32 / SR).sin();
                let y = allpass.process(x);
                if n > 24_000 {
                    sum_sq += y * y;
                    count += 1;
                }
            }
            let magnitude = (sum_sq / count as f32).sqrt() * std::f32::consts::SQRT_2;
            assert!(
                (magnitude - 1.0).abs() < 1e-2,
                "all-pass altered magnitude at {freq} Hz: {magnitude}"
            );
        }
    }

    /// Larger fractional delay → smaller coefficient (monotone mapping).
    #[test]
    fn coefficient_decreases_with_delay() {
        let small = TuningAllpass::new(0.1).coeff();
        let large = TuningAllpass::new(0.9).coeff();
        assert!(small > large, "coeff should shrink as delay grows");
        // D = 0.5 gives a = (1-0.5)/(1+0.5) = 1/3.
        assert!((TuningAllpass::new(0.5).coeff() - 1.0 / 3.0).abs() < 1e-6);
    }

    /// Phase delay at a low frequency approximates the requested fraction.
    #[test]
    fn low_frequency_phase_delay_matches_request() {
        const SR: f32 = 48_000.0;
        let freq = 100.0; // low, so the linear-phase approximation holds well
        let requested = 0.5;
        let mut allpass = TuningAllpass::new(requested);

        // Record input and delayed output, then compare sub-sample-interpolated
        // positive-going zero crossings — the delay is a fraction of a sample,
        // so integer crossing indices would be far too coarse.
        let omega = TAU * freq / SR;
        let mut input = Vec::with_capacity(20_000);
        let mut output = Vec::with_capacity(20_000);
        for n in 0..20_000 {
            let x = (omega * n as f32).sin();
            input.push(x);
            output.push(allpass.process(x));
        }
        let first_crossing = |signal: &[f32], from: usize| -> f32 {
            for index in from..signal.len() - 1 {
                if signal[index] <= 0.0 && signal[index + 1] > 0.0 {
                    return index as f32 - signal[index] / (signal[index + 1] - signal[index]);
                }
            }
            panic!("no zero crossing found");
        };
        let in_zc = first_crossing(&input, 9_995);
        let out_zc = first_crossing(&output, 9_993);
        let measured = out_zc - in_zc;
        assert!(
            (measured - requested).abs() < 0.05,
            "phase delay {measured} not ≈ {requested}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut allpass = TuningAllpass::new(0.5);
        allpass.process(1.0);
        allpass.reset();
        // First output after reset is a·x with zero history.
        assert!((allpass.process(1.0) - allpass.coeff()).abs() < 1e-9);
    }
}
