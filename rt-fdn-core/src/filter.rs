//! One-pole tone filters placed inside every FDN feedback path.
//!
//! Two jobs, both per branch (Phase 0 "damping" + Phase 1 "low-cut / high-cut
//! per feedback path", see `docs/Roadmap-05-fdn-reverb.md`):
//!
//! - [`OnePoleLowpass`] — the **damping / high-cut**.  Rolling off highs a
//!   little on every feedback pass makes them decay faster than lows, which is
//!   what real rooms do (air absorption) and is the single most important knob
//!   for a natural-sounding tail.
//! - [`OnePoleHighpass`] — the **low-cut**.  Bleeds sub energy out of the loop
//!   so long decays don't build up boomy mud.
//!
//! # Coefficient
//!
//! Both use the standard one-pole low-pass with per-sample coefficient
//!
//! ```text
//! a = 1 − exp(−2π·fc / fs)
//! ```
//!
//! which places the −3 dB point near `fc`.  The high-pass is realised as the
//! *complement* of a low-pass (`hp = x − lp`), so the low-cut and high-cut use
//! identical, well-behaved maths and a low-pass+high-pass pair sums back to the
//! input.
//!
//! # Real-time safety
//!
//! `process` is a couple of multiply-adds; `set_cutoff` calls `exp` once.  No
//! allocation, no locking.

use std::f32::consts::TAU;

/// Convert a cutoff frequency to the one-pole smoothing coefficient `a`,
/// clamped to a stable, useful `(0, 1]` and kept below Nyquist.
#[inline]
fn cutoff_coeff(cutoff_hz: f32, sample_rate: f32) -> f32 {
    if sample_rate <= 0.0 {
        return 1.0;
    }
    let nyquist = sample_rate * 0.5;
    let cutoff = cutoff_hz.clamp(1.0, nyquist * 0.999);
    (1.0 - (-TAU * cutoff / sample_rate).exp()).clamp(f32::MIN_POSITIVE, 1.0)
}

/// A one-pole low-pass: `y[n] = y[n−1] + a·(x[n] − y[n−1])`.
#[derive(Debug, Clone, Copy)]
pub struct OnePoleLowpass {
    /// Smoothing coefficient in `(0, 1]`.
    a: f32,
    /// Filter state (`y[n−1]`).
    state: f32,
}

impl OnePoleLowpass {
    /// Create a low-pass with the −3 dB point near `cutoff_hz`.
    pub fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        Self {
            a: cutoff_coeff(cutoff_hz, sample_rate),
            state: 0.0,
        }
    }

    /// Move the cutoff without disturbing the state (click-free live change).
    ///
    /// **Real-time safe** apart from a single `exp`.
    #[inline]
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        self.a = cutoff_coeff(cutoff_hz, sample_rate);
    }

    /// Clear the filter memory.
    #[inline]
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Filter one sample.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        self.state += self.a * (input - self.state);
        self.state
    }
}

/// A one-pole high-pass, realised as `x − lowpass(x)`.
#[derive(Debug, Clone, Copy)]
pub struct OnePoleHighpass {
    lowpass: OnePoleLowpass,
}

impl OnePoleHighpass {
    /// Create a high-pass with the −3 dB point near `cutoff_hz`.
    pub fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        Self {
            lowpass: OnePoleLowpass::new(cutoff_hz, sample_rate),
        }
    }

    /// Move the cutoff without disturbing the state.
    ///
    /// **Real-time safe** apart from a single `exp`.
    #[inline]
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        self.lowpass.set_cutoff(cutoff_hz, sample_rate);
    }

    /// Clear the filter memory.
    #[inline]
    pub fn reset(&mut self) {
        self.lowpass.reset();
    }

    /// Filter one sample.
    ///
    /// **Real-time safe.**
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        input - self.lowpass.process(input)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    const SR: f32 = 48_000.0;

    /// Settled magnitude of a sustained unit sine after warm-up, measured via
    /// RMS (peak sampling under-reads tones whose period divides the sample
    /// rate — e.g. 8 kHz at 48 kHz caps at sin 60° = 0.866).
    fn tone<F: FnMut(f32) -> f32>(mut process: F, freq: f32) -> f32 {
        let mut sum_sq = 0.0f32;
        let mut count = 0u32;
        for n in 0..48_000 {
            let x = (TAU * freq * n as f32 / SR).sin();
            let y = process(x);
            if n > 36_000 {
                sum_sq += y * y;
                count += 1;
            }
        }
        // magnitude = rms_out / rms_in, and rms of a unit sine is 1/√2.
        (sum_sq / count as f32).sqrt() * std::f32::consts::SQRT_2
    }

    #[test]
    fn lowpass_passes_lows_rejects_highs() {
        let mut low = OnePoleLowpass::new(1_000.0, SR);
        let pass = tone(|x| low.process(x), 100.0);
        let mut low2 = OnePoleLowpass::new(1_000.0, SR);
        let reject = tone(|x| low2.process(x), 15_000.0);
        assert!(pass > 0.9, "low-pass attenuated a low tone: {pass}");
        assert!(reject < 0.2, "low-pass passed a high tone: {reject}");
    }

    #[test]
    fn highpass_passes_highs_rejects_lows() {
        let mut high = OnePoleHighpass::new(500.0, SR);
        let pass = tone(|x| high.process(x), 8_000.0);
        let mut high2 = OnePoleHighpass::new(500.0, SR);
        let reject = tone(|x| high2.process(x), 30.0);
        assert!(pass > 0.9, "high-pass attenuated a high tone: {pass}");
        assert!(reject < 0.2, "high-pass passed a low tone: {reject}");
    }

    #[test]
    fn lowpass_dc_gain_is_unity() {
        let mut low = OnePoleLowpass::new(1_000.0, SR);
        let mut y = 0.0;
        for _ in 0..48_000 {
            y = low.process(1.0);
        }
        assert!((y - 1.0).abs() < 1e-3, "DC gain {y} not ~1.0");
    }

    #[test]
    fn cutoff_above_nyquist_is_clamped_not_nan() {
        let mut low = OnePoleLowpass::new(1_000_000.0, SR);
        assert!(low.process(0.5).is_finite());
    }

    #[test]
    fn reset_clears_state() {
        let mut low = OnePoleLowpass::new(1_000.0, SR);
        low.process(1.0);
        low.reset();
        // First output after reset should be a·x with zero history.
        let first = low.process(0.0);
        assert_eq!(first, 0.0);
    }
}
