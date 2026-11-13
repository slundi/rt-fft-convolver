//! A general Direct-Form-I biquad, and the two shapes the engine needs from it:
//! the **modal resonator** (a mode of the modal bank) and the **body filter**
//! (a peaking colour applied to the plucked-string output).
//!
//! # Modal resonator
//!
//! A single mode is a two-pole resonator
//!
//! ```text
//!            b0
//! H(z) = ─────────────────────────,   θ = 2π·f/fs,   R = 10^(−3 / (T60·fs))
//!        1 − 2R·cosθ·z⁻¹ + R²·z⁻²
//! ```
//!
//! whose impulse response is a sinusoid at `f` decaying with time-to-−60 dB
//! `T60` — literally "excitation impulse → decaying sinusoid", the Phase 0
//! modal example.  The numerator gain is set to `amp·sinθ` so every mode peaks
//! at roughly `amp` regardless of its frequency (the raw resonator gain scales
//! as `1/sinθ`).
//!
//! # Body filter
//!
//! A [`Biquad::peaking`] bell colours the string output to imitate an
//! instrument body resonance without paying for a convolution.  (An IR-based
//! body via `rt-fft-convolver` is the optional Phase 1 upgrade noted in the
//! roadmap prerequisites.)
//!
//! # Real-time safety
//!
//! `process` is five multiply-adds; the constructors run a little trig and are
//! called off the render thread (on note-on / preset change).  State is flushed
//! to zero so a ringing resonator stays out of the denormal slow path.

use rt_fft_convolver::flush_to_zero;
use std::f32::consts::TAU;

/// A Direct-Form-I biquad: `y = b0·x + b1·x₋₁ + b2·x₋₂ − a1·y₋₁ − a2·y₋₂`
/// (coefficients pre-normalised by `a0`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    /// A two-pole resonator at `freq` Hz decaying with `t60` seconds, peaking
    /// near `amp`.  This is one mode of the modal bank.
    pub fn resonator(freq: f32, t60: f32, amp: f32, sample_rate: f32) -> Self {
        let nyquist = sample_rate * 0.5;
        let freq = freq.clamp(1.0, nyquist * 0.999);
        let theta = TAU * freq / sample_rate;
        // R from T60: R^(T60·fs) = 10^(−3)  ⇒  R = 10^(−3 / (T60·fs)).
        let radius = if t60 > 0.0 {
            10.0f32
                .powf(-3.0 / (t60 * sample_rate))
                .clamp(0.0, 0.999_999)
        } else {
            0.0
        };
        Self {
            // Normalise the peak: raw resonator gain ≈ 1/sinθ, so scale by sinθ.
            b0: amp * theta.sin(),
            a1: -2.0 * radius * theta.cos(),
            a2: radius * radius,
            ..Self::default()
        }
    }

    /// A peaking (bell) EQ: `+gain_db` at `freq` with quality `q`.  Used as the
    /// body-resonance colour on the plucked-string output.
    pub fn peaking(freq: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let nyquist = sample_rate * 0.5;
        let freq = freq.clamp(1.0, nyquist * 0.999);
        let q = q.max(1e-3);
        let omega = TAU * freq / sample_rate;
        let (sin, cos) = (omega.sin(), omega.cos());
        let alpha = sin / (2.0 * q);
        let gain = 10.0f32.powf(gain_db / 40.0); // amplitude, √ of power

        // RBJ cookbook peaking-EQ coefficients.
        let b0 = 1.0 + alpha * gain;
        let b1 = -2.0 * cos;
        let b2 = 1.0 - alpha * gain;
        let a0 = 1.0 + alpha / gain;
        let a1 = -2.0 * cos;
        let a2 = 1.0 - alpha / gain;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Self::default()
        }
    }

    /// Clear the filter memory (keep the coefficients).
    #[inline]
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Filter one sample.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = flush_to_zero(output);
        self.y1
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// Estimate the dominant frequency of a signal from its positive-going
    /// zero crossings.
    fn frequency_from_zero_crossings(signal: &[f32]) -> f32 {
        let mut crossings = Vec::new();
        for pair in signal.windows(2).enumerate() {
            let (index, window) = pair;
            if window[0] <= 0.0 && window[1] > 0.0 {
                // Linear-interpolate the sub-sample crossing instant.
                let frac = -window[0] / (window[1] - window[0]);
                crossings.push(index as f32 + frac);
            }
        }
        assert!(crossings.len() >= 2, "not enough zero crossings");
        let periods: f32 = crossings
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .sum::<f32>()
            / (crossings.len() - 1) as f32;
        SR / periods
    }

    /// An excited resonator rings at its design frequency (Phase 0 modal
    /// example) — checked here well within ±5 cents.
    #[test]
    fn resonator_rings_at_design_frequency() {
        let target = 440.0;
        let mut resonator = Biquad::resonator(target, 2.0, 1.0, SR);
        let mut signal = Vec::with_capacity(20_000);
        for n in 0..20_000 {
            signal.push(resonator.process(if n == 0 { 1.0 } else { 0.0 }));
        }
        let measured = frequency_from_zero_crossings(&signal[500..]);
        let cents = 1200.0 * (measured / target).log2();
        assert!(cents.abs() < 5.0, "off by {cents} cents ({measured} Hz)");
    }

    /// The resonator impulse response decays (energy in the tail < energy in the
    /// head) and stays finite.
    #[test]
    fn resonator_impulse_response_decays() {
        let mut resonator = Biquad::resonator(300.0, 1.0, 1.0, SR);
        let mut head = 0.0f32;
        let mut tail = 0.0f32;
        for n in 0..48_000 {
            let y = resonator.process(if n == 0 { 1.0 } else { 0.0 });
            assert!(y.is_finite());
            if n < 4_800 {
                head += y * y;
            } else if n >= 43_200 {
                tail += y * y;
            }
        }
        assert!(tail < head * 1e-3, "tail {tail} not decayed vs head {head}");
    }

    /// Longer T60 ⇒ pole radius closer to 1 ⇒ slower decay (more tail energy).
    #[test]
    fn longer_t60_decays_slower() {
        let energy = |t60: f32| {
            let mut resonator = Biquad::resonator(300.0, t60, 1.0, SR);
            let mut tail = 0.0f32;
            for n in 0..48_000 {
                let y = resonator.process(if n == 0 { 1.0 } else { 0.0 });
                if n >= 24_000 {
                    tail += y * y;
                }
            }
            tail
        };
        assert!(energy(3.0) > energy(0.5));
    }

    /// A peaking body filter boosts its centre band and leaves DC roughly alone.
    #[test]
    fn peaking_boosts_centre_band() {
        use std::f32::consts::TAU;
        let centre = 500.0;
        let magnitude = |freq: f32| {
            let mut body = Biquad::peaking(centre, 1.0, 9.0, SR);
            let mut sum_sq = 0.0f32;
            let mut count = 0u32;
            for n in 0..48_000 {
                let y = body.process((TAU * freq * n as f32 / SR).sin());
                if n > 24_000 {
                    sum_sq += y * y;
                    count += 1;
                }
            }
            (sum_sq / count as f32).sqrt() * std::f32::consts::SQRT_2
        };
        assert!(magnitude(centre) > 1.5, "no boost at centre");
        assert!(
            (magnitude(50.0) - 1.0).abs() < 0.2,
            "DC band wrongly altered"
        );
    }
}
