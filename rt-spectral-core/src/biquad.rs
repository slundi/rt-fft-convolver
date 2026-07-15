//! A second-order (biquad) IIR section, used here as the **notch** building
//! block for hum removal.
//!
//! Uses the RBJ "Audio EQ Cookbook" notch coefficients and a **transposed
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
//! (This mirrors the convolver's own biquad; it is duplicated here rather than
//! shared to keep `rt-spectral-core` free of a plugin-crate dependency.  See
//! the Roadmap-04 shared-crate extraction note.)

use std::f32::consts::PI;

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

    /// A band-reject (notch) section centred at `centre_hz` with quality `q`.
    pub fn notch(sample_rate: u32, centre_hz: f32, q: f32) -> Self {
        let mut f = Self::identity();
        f.set_notch(sample_rate, centre_hz, q);
        f
    }

    /// Recompute notch coefficients in place, **preserving filter state** so a
    /// live frequency change does not click.
    ///
    /// **Real-time safe** — no allocation.
    pub fn set_notch(&mut self, sample_rate: u32, centre_hz: f32, q: f32) {
        let nyquist = sample_rate as f32 * 0.5;
        let centre = centre_hz.clamp(1.0, nyquist * 0.999);
        let q = q.max(1e-4);
        let w0 = 2.0 * PI * centre / sample_rate as f32;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);

        // RBJ notch numerator: b0 = 1, b1 = −2cos, b2 = 1.
        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
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

    /// Settled magnitude of a sustained sine at `freq` after warm-up.
    ///
    /// A high-`Q` notch rings for `~Q / (π·f0)` seconds, so we run a full
    /// second (48 k samples) and only measure the final quarter.
    fn tone_response(mut f: Biquad, freq: f32) -> f32 {
        let mut peak = 0.0f32;
        for n in 0..48_000 {
            let x = (2.0 * PI * freq * n as f32 / SR as f32).sin();
            let y = f.process_sample(x);
            if n > 36_000 {
                peak = peak.max(y.abs());
            }
        }
        peak
    }

    #[test]
    fn identity_passes_signal_unchanged() {
        let mut f = Biquad::identity();
        for &x in &[0.3, -0.7, 1.0, -1.0, 0.0] {
            assert!((f.process_sample(x) - x).abs() < 1e-7);
        }
    }

    #[test]
    fn notch_rejects_centre_frequency() {
        let f = Biquad::notch(SR, 60.0, 30.0);
        // The 60 Hz centre is deeply attenuated…
        assert!(tone_response(f, 60.0) < 0.1, "notch did not reject centre");
    }

    #[test]
    fn notch_passes_far_frequencies() {
        let f = Biquad::notch(SR, 60.0, 30.0);
        // …while a distant tone passes near unity.
        let pass = tone_response(f, 1_000.0);
        assert!((pass - 1.0).abs() < 0.05, "off-centre gain {pass} not ~1.0");
    }

    #[test]
    fn set_notch_preserves_state() {
        let mut f = Biquad::notch(SR, 60.0, 30.0);
        f.process_sample(1.0);
        let (z1, z2) = (f.z1, f.z2);
        f.set_notch(SR, 120.0, 30.0);
        assert_eq!((f.z1, f.z2), (z1, z2));
    }

    #[test]
    fn centre_above_nyquist_is_clamped_not_nan() {
        let mut f = Biquad::notch(SR, 30_000.0, 30.0);
        assert!(f.process_sample(0.5).is_finite());
    }
}
