//! A second-order (12 dB/oct) biquad filter — the cab's low-cut and high-cut.
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
//! A single biquad is a 2-pole filter → 12 dB/octave, matching the roadmap's
//! `Low Cut` (high-pass) and `High Cut` (low-pass) slopes.

use std::f32::consts::PI;

/// Butterworth quality factor (maximally flat passband) for a lone biquad:
/// `1/√2`.
pub const BUTTERWORTH_Q: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// A second-order IIR filter section.
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    // Feed-forward coefficients (already divided by a0).
    b0: f32,
    b1: f32,
    b2: f32,
    // Feed-back coefficients (already divided by a0).
    a1: f32,
    a2: f32,
    // Filter state (transposed direct-form II).
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

    /// A high-pass (low-cut) section at `cutoff_hz`.
    pub fn high_pass(sample_rate: u32, cutoff_hz: f32, q: f32) -> Self {
        let mut f = Self::identity();
        f.set_high_pass(sample_rate, cutoff_hz, q);
        f
    }

    /// A low-pass (high-cut) section at `cutoff_hz`.
    pub fn low_pass(sample_rate: u32, cutoff_hz: f32, q: f32) -> Self {
        let mut f = Self::identity();
        f.set_low_pass(sample_rate, cutoff_hz, q);
        f
    }

    /// Recompute high-pass coefficients in place, **preserving filter state**
    /// so a live cutoff change does not click.
    ///
    /// **Real-time safe** — no allocation.
    pub fn set_high_pass(&mut self, sample_rate: u32, cutoff_hz: f32, q: f32) {
        let (cos_w0, alpha) = Self::intermediates(sample_rate, cutoff_hz, q);
        // RBJ high-pass numerator.
        let b0 = (1.0 + cos_w0) * 0.5;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) * 0.5;
        self.set_coeffs(b0, b1, b2, cos_w0, alpha);
    }

    /// Recompute low-pass coefficients in place, preserving filter state.
    ///
    /// **Real-time safe** — no allocation.
    pub fn set_low_pass(&mut self, sample_rate: u32, cutoff_hz: f32, q: f32) {
        let (cos_w0, alpha) = Self::intermediates(sample_rate, cutoff_hz, q);
        // RBJ low-pass numerator.
        let b0 = (1.0 - cos_w0) * 0.5;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) * 0.5;
        self.set_coeffs(b0, b1, b2, cos_w0, alpha);
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

    /// Filter a block in place.
    #[inline]
    pub fn process_block(&mut self, buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = self.process_sample(*x);
        }
    }

    /// Shared `cos(ω₀)` and `α = sin(ω₀)/(2Q)` for the RBJ formulas.
    ///
    /// The cutoff is clamped below Nyquist so `tan`/`sin` stay finite.
    fn intermediates(sample_rate: u32, cutoff_hz: f32, q: f32) -> (f32, f32) {
        let nyquist = sample_rate as f32 * 0.5;
        // Keep strictly inside (0, Nyquist) to avoid degenerate coefficients.
        let cutoff = cutoff_hz.clamp(1.0, nyquist * 0.999);
        let q = q.max(1e-4);
        let w0 = 2.0 * PI * cutoff / sample_rate as f32;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        (cos_w0, alpha)
    }

    /// Normalise `(b0, b1, b2)` and the standard RBJ denominator by `a0` and
    /// store them; state (`z1`, `z2`) is left untouched.
    fn set_coeffs(&mut self, b0: f32, b1: f32, b2: f32, cos_w0: f32, alpha: f32) {
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let inv_a0 = 1.0 / a0;
        self.b0 = b0 * inv_a0;
        self.b1 = b1 * inv_a0;
        self.b2 = b2 * inv_a0;
        self.a1 = a1 * inv_a0;
        self.a2 = a2 * inv_a0;
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

    const SR: u32 = 48_000;

    /// Run a constant (DC) signal through the filter until it settles and
    /// return the steady-state output.
    fn dc_response(mut f: Biquad) -> f32 {
        let mut y = 0.0;
        for _ in 0..4_000 {
            y = f.process_sample(1.0);
        }
        y
    }

    /// Run a full-scale Nyquist signal (+1, −1, +1, …) and return the settled
    /// output magnitude.
    fn nyquist_response(mut f: Biquad) -> f32 {
        let mut y = 0.0;
        for n in 0..4_000 {
            let x = if n % 2 == 0 { 1.0 } else { -1.0 };
            y = f.process_sample(x);
        }
        y.abs()
    }

    #[test]
    fn identity_passes_signal_unchanged() {
        let mut f = Biquad::identity();
        for &x in &[0.3, -0.7, 1.0, -1.0, 0.0] {
            assert!((f.process_sample(x) - x).abs() < 1e-7);
        }
    }

    #[test]
    fn high_pass_blocks_dc_and_passes_highs() {
        let f = Biquad::high_pass(SR, 200.0, BUTTERWORTH_Q);
        // DC is rejected → ~0.
        assert!(dc_response(f).abs() < 1e-3, "DC leaked through high-pass");
        // A high frequency (Nyquist) passes ~unity.
        let hi = nyquist_response(f);
        assert!((hi - 1.0).abs() < 1e-2, "Nyquist gain {hi} not ~1.0");
    }

    #[test]
    fn low_pass_passes_dc_and_blocks_highs() {
        let f = Biquad::low_pass(SR, 5_000.0, BUTTERWORTH_Q);
        // DC passes ~unity.
        let dc = dc_response(f);
        assert!((dc - 1.0).abs() < 1e-2, "DC gain {dc} not ~1.0");
        // Nyquist is rejected → ~0.
        assert!(
            nyquist_response(f) < 1e-2,
            "Nyquist leaked through low-pass"
        );
    }

    #[test]
    fn passband_is_near_unity_in_the_middle() {
        // A 1 kHz tone sits in the passband of a 20 Hz HP + 20 kHz LP chain.
        let mut hp = Biquad::high_pass(SR, 20.0, BUTTERWORTH_Q);
        let mut lp = Biquad::low_pass(SR, 20_000.0, BUTTERWORTH_Q);
        let freq = 1_000.0f32;
        let mut peak = 0.0f32;
        for n in 0..8_000 {
            let x = (2.0 * PI * freq * n as f32 / SR as f32).sin();
            let y = lp.process_sample(hp.process_sample(x));
            if n > 4_000 {
                peak = peak.max(y.abs());
            }
        }
        assert!(
            (peak - 1.0).abs() < 0.05,
            "1 kHz passband peak {peak} not ~1.0"
        );
    }

    #[test]
    fn set_cutoff_preserves_state_field() {
        // Retuning must not reset the delay line (click-free automation).
        let mut f = Biquad::high_pass(SR, 200.0, BUTTERWORTH_Q);
        f.process_sample(1.0);
        let (z1, z2) = (f.z1, f.z2);
        f.set_high_pass(SR, 300.0, BUTTERWORTH_Q);
        assert_eq!((f.z1, f.z2), (z1, z2));
    }

    #[test]
    fn cutoff_above_nyquist_is_clamped_not_nan() {
        // 30 kHz cut at 48 kHz SR is above Nyquist; must stay finite.
        let mut f = Biquad::low_pass(SR, 30_000.0, BUTTERWORTH_Q);
        let y = f.process_sample(0.5);
        assert!(y.is_finite());
    }
}
