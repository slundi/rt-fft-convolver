//! Per-band level detector — a **peak / RMS blend** (Phase 0 decision).
//!
//! The detector turns a band's samples into an instantaneous *level* estimate
//! that the [`crate::EnvelopeFollower`] then shapes with attack/release
//! ballistics.  A single `blend` control crossfades between the two classic
//! detector characters:
//!
//! - **Peak** (`blend = 1`) — `|x|`; catches transients, right for high bands
//!   and de-essing.
//! - **RMS** (`blend = 0`) — the square-root of a one-pole mean-square average;
//!   musical, level-riding, right for low bands.
//!
//! ```text
//! ms[n] = (1 − c)·x[n]² + c·ms[n−1]     c = exp(−1 / (window_s · fs))
//! level = blend·|x| + (1 − blend)·√ms
//! ```
//!
//! One squared sample plus a one-pole per sample — RT-safe.

/// A peak / RMS-blend level detector for one band/channel.
#[derive(Debug, Clone, Copy)]
pub struct Detector {
    /// `0.0` = pure RMS, `1.0` = pure peak.
    blend: f32,
    /// One-pole averaging coefficient for the mean-square estimate.
    ms_coeff: f32,
    /// Running mean-square state.
    mean_square: f32,
}

impl Detector {
    /// Build a detector at `sample_rate` averaging RMS over `window_ms`, with
    /// the given peak/RMS `blend` (clamped to `0..=1`).
    pub fn new(sample_rate: f32, window_ms: f32, blend: f32) -> Self {
        let mut detector = Self {
            blend: blend.clamp(0.0, 1.0),
            ms_coeff: 0.0,
            mean_square: 0.0,
        };
        detector.set_window_ms(sample_rate, window_ms);
        detector
    }

    /// Recompute the RMS averaging window.  RT-safe.
    pub fn set_window_ms(&mut self, sample_rate: f32, window_ms: f32) {
        let seconds = (window_ms * 1e-3).max(1e-6);
        self.ms_coeff = (-1.0 / (seconds * sample_rate)).exp();
    }

    /// Set the peak/RMS blend (`0` = RMS, `1` = peak).  RT-safe.
    #[inline]
    pub fn set_blend(&mut self, blend: f32) {
        self.blend = blend.clamp(0.0, 1.0);
    }

    /// Clear the averaging state.
    #[inline]
    pub fn reset(&mut self) {
        self.mean_square = 0.0;
    }

    /// Feed one sample and return the blended level estimate (`≥ 0`).  RT-safe.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let squared = x * x;
        self.mean_square = (1.0 - self.ms_coeff) * squared + self.ms_coeff * self.mean_square;
        let peak = x.abs();
        let rms = self.mean_square.sqrt();
        self.blend * peak + (1.0 - self.blend) * rms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::make_sine;

    const SR: f32 = 48_000.0;

    /// Peak of the detector output over the settled second half of a sustained
    /// sine (the detector is instantaneous, so a *max* is the meaningful probe —
    /// the envelope follower does the holding downstream).
    fn settled_peak(mut detector: Detector, freq: f32) -> f32 {
        let buf = make_sine(freq, SR, 24_000);
        let mut peak = 0.0f32;
        for (n, &x) in buf.iter().enumerate() {
            let level = detector.process(x);
            if n > 12_000 {
                peak = peak.max(level);
            }
        }
        peak
    }

    #[test]
    fn pure_rms_of_unit_sine_is_root_half() {
        // RMS of a unit sine = 1/√2 ≈ 0.707.  Measure the mean of the settled
        // detector so the ripple around √½ averages out.
        let mut detector = Detector::new(SR, 5.0, 0.0);
        let buf = make_sine(500.0, SR, 24_000);
        for &x in &buf[..12_000] {
            detector.process(x);
        }
        let mean: f32 = buf[12_000..]
            .iter()
            .map(|&x| detector.process(x))
            .sum::<f32>()
            / 12_000.0;
        assert!(
            (mean - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02,
            "rms {mean}"
        );
    }

    #[test]
    fn pure_peak_of_unit_sine_approaches_one() {
        let level = settled_peak(Detector::new(SR, 5.0, 1.0), 500.0);
        assert!(level > 0.95, "peak level {level} not near 1.0");
    }

    #[test]
    fn blend_is_between_peak_and_rms() {
        let rms = settled_peak(Detector::new(SR, 5.0, 0.0), 500.0);
        let peak = settled_peak(Detector::new(SR, 5.0, 1.0), 500.0);
        let blended = settled_peak(Detector::new(SR, 5.0, 0.5), 500.0);
        assert!(
            blended > rms && blended < peak,
            "blend {blended} not between"
        );
    }

    #[test]
    fn silence_detects_zero() {
        let mut detector = Detector::new(SR, 5.0, 0.5);
        for _ in 0..1_000 {
            detector.process(0.0);
        }
        assert!(detector.process(0.0).abs() < 1e-9);
    }
}
