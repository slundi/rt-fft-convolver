//! A second-order (biquad) IIR section — the building block for the
//! Linkwitz-Riley crossover and the sidechain filter.
//!
//! Uses the RBJ "Audio EQ Cookbook" coefficients and a **transposed
//! direct-form II** difference equation, which is numerically well-behaved for
//! `f32` audio and needs only two state words:
//!
//! ```text
//! y[n]  = b0·x[n] + z1
//! z1    = b1·x[n] − a1·y[n] + z2
//! z2    = b2·x[n] − a2·y[n]
//! ```
//!
//! Coefficients are pre-normalised by `a0` at construction, so the render loop
//! is pure multiply-add — no division, no allocation, RT-safe.
//!
//! (This mirrors `rt_spectral_core::Biquad`, which only exposes a *notch*
//! response with private coefficients.  A local copy carrying the lowpass /
//! highpass / allpass / high-shelf responses the crossover and sidechain need
//! keeps `rt-multiband-core` self-contained and avoids widening another
//! crate's public API for one consumer — the same "duplicate the tiny biquad"
//! choice the rest of the workspace already makes.)

use std::f32::consts::PI;

/// A second-order IIR filter section (transposed direct-form II).
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    // Feed-forward coefficients (already divided by a0).
    b0: f32,
    b1: f32,
    b2: f32,
    // Feed-back coefficients (already divided by a0).
    a1: f32,
    a2: f32,
    // Filter state.
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// An all-pass-through filter (`y = x`) — the neutral starting point.
    pub fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// A 2nd-order lowpass at `cutoff_hz` with quality `q`.
    pub fn lowpass(sample_rate: f32, cutoff_hz: f32, q: f32) -> Self {
        let mut filter = Self::identity();
        filter.set_lowpass(sample_rate, cutoff_hz, q);
        filter
    }

    /// A 2nd-order highpass at `cutoff_hz` with quality `q`.
    pub fn highpass(sample_rate: f32, cutoff_hz: f32, q: f32) -> Self {
        let mut filter = Self::identity();
        filter.set_highpass(sample_rate, cutoff_hz, q);
        filter
    }

    /// A 2nd-order allpass at `centre_hz` with quality `q` (flat magnitude, the
    /// crossover phase-alignment correction — see [`crate::crossover`]).
    pub fn allpass(sample_rate: f32, centre_hz: f32, q: f32) -> Self {
        let mut filter = Self::identity();
        filter.set_allpass(sample_rate, centre_hz, q);
        filter
    }

    /// A high-shelf boosting/cutting by `gain_db` above `corner_hz`.
    pub fn high_shelf(sample_rate: f32, corner_hz: f32, gain_db: f32, q: f32) -> Self {
        let mut filter = Self::identity();
        filter.set_high_shelf(sample_rate, corner_hz, gain_db, q);
        filter
    }

    /// Shared angular-frequency / clamp preamble for every RBJ setter.
    ///
    /// Returns `(cos ω0, sin ω0)` for the clamped cutoff.
    #[inline]
    fn omega(sample_rate: f32, freq_hz: f32) -> (f32, f32) {
        let nyquist = sample_rate * 0.5;
        let freq = freq_hz.clamp(1.0, nyquist * 0.999);
        let w0 = 2.0 * PI * freq / sample_rate;
        (w0.cos(), w0.sin())
    }

    /// Store coefficients from an un-normalised `(b0,b1,b2,a0,a1,a2)` tuple,
    /// dividing through by `a0`.  **Preserves state** (no click on retune).
    #[inline]
    fn set_coeffs(&mut self, b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) {
        let inv_a0 = 1.0 / a0;
        self.b0 = b0 * inv_a0;
        self.b1 = b1 * inv_a0;
        self.b2 = b2 * inv_a0;
        self.a1 = a1 * inv_a0;
        self.a2 = a2 * inv_a0;
    }

    /// Recompute lowpass coefficients in place, preserving state.  RT-safe.
    pub fn set_lowpass(&mut self, sample_rate: f32, cutoff_hz: f32, q: f32) {
        let (cos_w0, sin_w0) = Self::omega(sample_rate, cutoff_hz);
        let alpha = sin_w0 / (2.0 * q.max(1e-4));
        let b1 = 1.0 - cos_w0;
        let b0 = b1 * 0.5;
        self.set_coeffs(b0, b1, b0, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha);
    }

    /// Recompute highpass coefficients in place, preserving state.  RT-safe.
    pub fn set_highpass(&mut self, sample_rate: f32, cutoff_hz: f32, q: f32) {
        let (cos_w0, sin_w0) = Self::omega(sample_rate, cutoff_hz);
        let alpha = sin_w0 / (2.0 * q.max(1e-4));
        let b0 = (1.0 + cos_w0) * 0.5;
        let b1 = -(1.0 + cos_w0);
        self.set_coeffs(b0, b1, b0, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha);
    }

    /// Recompute allpass coefficients in place, preserving state.  RT-safe.
    pub fn set_allpass(&mut self, sample_rate: f32, centre_hz: f32, q: f32) {
        let (cos_w0, sin_w0) = Self::omega(sample_rate, centre_hz);
        let alpha = sin_w0 / (2.0 * q.max(1e-4));
        self.set_coeffs(
            1.0 - alpha,
            -2.0 * cos_w0,
            1.0 + alpha,
            1.0 + alpha,
            -2.0 * cos_w0,
            1.0 - alpha,
        );
    }

    /// Recompute high-shelf coefficients in place, preserving state.  RT-safe.
    pub fn set_high_shelf(&mut self, sample_rate: f32, corner_hz: f32, gain_db: f32, q: f32) {
        let (cos_w0, sin_w0) = Self::omega(sample_rate, corner_hz);
        let amp = 10.0_f32.powf(gain_db / 40.0);
        let alpha = sin_w0 / (2.0 * q.max(1e-4));
        let two_sqrt_a_alpha = 2.0 * amp.sqrt() * alpha;
        let ap1 = amp + 1.0;
        let am1 = amp - 1.0;
        self.set_coeffs(
            amp * (ap1 + am1 * cos_w0 + two_sqrt_a_alpha),
            -2.0 * amp * (am1 + ap1 * cos_w0),
            amp * (ap1 + am1 * cos_w0 - two_sqrt_a_alpha),
            ap1 - am1 * cos_w0 + two_sqrt_a_alpha,
            2.0 * (am1 - ap1 * cos_w0),
            ap1 - am1 * cos_w0 - two_sqrt_a_alpha,
        );
    }

    /// Clear the filter memory without touching the coefficients.
    #[inline]
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Filter one sample.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

impl Default for Biquad {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_sine;

    const SR: f32 = 48_000.0;

    /// Settled peak magnitude of a sustained sine at `freq` after warm-up.
    fn tone_response(mut filter: Biquad, freq: f32) -> f32 {
        let buf = make_sine(freq, SR, 48_000);
        let mut peak = 0.0f32;
        for (n, &x) in buf.iter().enumerate() {
            let y = filter.process_sample(x);
            if n > 36_000 {
                peak = peak.max(y.abs());
            }
        }
        peak
    }

    #[test]
    fn identity_passes_signal_unchanged() {
        let mut filter = Biquad::identity();
        for &x in &[0.3, -0.7, 1.0, -1.0, 0.0] {
            assert!((filter.process_sample(x) - x).abs() < 1e-7);
        }
    }

    #[test]
    fn lowpass_passes_below_and_rejects_above() {
        let cutoff = 1_000.0;
        let q = std::f32::consts::FRAC_1_SQRT_2;
        assert!(tone_response(Biquad::lowpass(SR, cutoff, q), 100.0) > 0.9);
        assert!(tone_response(Biquad::lowpass(SR, cutoff, q), 10_000.0) < 0.05);
    }

    #[test]
    fn highpass_rejects_below_and_passes_above() {
        let cutoff = 1_000.0;
        let q = std::f32::consts::FRAC_1_SQRT_2;
        assert!(tone_response(Biquad::highpass(SR, cutoff, q), 100.0) < 0.05);
        assert!(tone_response(Biquad::highpass(SR, cutoff, q), 10_000.0) > 0.9);
    }

    #[test]
    fn allpass_has_flat_magnitude() {
        let q = std::f32::consts::FRAC_1_SQRT_2;
        for &freq in &[80.0, 500.0, 1_000.0, 5_000.0, 12_000.0] {
            let gain = tone_response(Biquad::allpass(SR, 1_000.0, q), freq);
            assert!(
                (gain - 1.0).abs() < 0.02,
                "allpass |H| at {freq} Hz = {gain}"
            );
        }
    }

    #[test]
    fn high_shelf_boosts_highs_leaves_lows() {
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let shelf = Biquad::high_shelf(SR, 4_000.0, 6.0, q);
        // A well-below-corner tone is untouched…
        assert!((tone_response(shelf, 100.0) - 1.0).abs() < 0.05);
        // …a well-above-corner tone is lifted toward +6 dB (×1.995).
        let high = tone_response(shelf, 15_000.0);
        assert!(high > 1.7, "high-shelf gain {high} not near +6 dB");
    }

    #[test]
    fn cutoff_above_nyquist_is_clamped_not_nan() {
        let mut filter = Biquad::lowpass(SR, 30_000.0, 0.7);
        assert!(filter.process_sample(0.5).is_finite());
    }
}
