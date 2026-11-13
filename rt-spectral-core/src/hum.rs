//! Time-domain hum removal: a cascade of RBJ biquad notches at a mains
//! fundamental (50 / 60 Hz or custom) and its harmonics.
//!
//! Phase 0 moved hum out of the per-bin STFT chain: mains hum is a handful of
//! *pure lines*, and surgical time-domain notches remove them with no added
//! latency and no spectral leakage into neighbouring bins.  A single biquad
//! notch is very narrow (high `Q`), so the audible signal either side of each
//! harmonic is essentially untouched.

use crate::biquad::Biquad;

/// Maximum number of harmonics the bank can notch (Roadmap-04 "Harmonics
/// (1–10)").  Fixed so the notch `Vec` is allocated once, off-RT.
pub const MAX_HARMONICS: usize = 10;

/// Hum-removal controls (Roadmap-04 Phase 2 "Hum controls").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HumParams {
    /// Whether the operator runs at all.
    pub enabled: bool,
    /// Fundamental frequency in Hz (typically 50 or 60).
    pub fundamental_hz: f32,
    /// Number of harmonics to notch, including the fundamental (`1..=MAX`).
    pub harmonics: usize,
    /// Notch quality factor — higher is narrower.
    pub q: f32,
}

impl Default for HumParams {
    fn default() -> Self {
        Self {
            enabled: false,
            fundamental_hz: 50.0,
            harmonics: 3,
            q: 30.0,
        }
    }
}

/// A bank of up to [`MAX_HARMONICS`] notch filters, reconfigured lazily when
/// the parameters change.
#[derive(Debug)]
pub struct HumNotchBank {
    sample_rate: u32,
    notches: Vec<Biquad>,
    /// How many leading notches are currently active (harmonics that fall
    /// below Nyquist).
    active: usize,
    /// Last parameters the coefficients were built for; `None` forces a
    /// rebuild on the first call.
    configured: Option<HumParams>,
}

impl HumNotchBank {
    /// Build a bank for `sample_rate`.  Pre-allocates all notch slots.
    ///
    /// **Allocates** — off-RT only.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            notches: vec![Biquad::identity(); MAX_HARMONICS],
            active: 0,
            configured: None,
        }
    }

    /// Clear all notch filter memories (call on a transport discontinuity).
    #[inline]
    pub fn reset(&mut self) {
        for notch in &mut self.notches {
            notch.reset();
        }
    }

    /// (Re)compute the notch coefficients for `params`.  Coefficient math only,
    /// no allocation — safe to call from the audio thread, though the engine
    /// only calls it when the parameters actually change.
    fn configure(&mut self, params: &HumParams) {
        let nyquist = self.sample_rate as f32 * 0.5;
        let harmonics = params.harmonics.clamp(1, MAX_HARMONICS);
        let mut active = 0;
        for (i, notch) in self.notches.iter_mut().enumerate().take(harmonics) {
            let freq = params.fundamental_hz * (i as f32 + 1.0);
            if freq >= nyquist {
                break; // higher harmonics fold past Nyquist — stop
            }
            notch.set_notch(self.sample_rate, freq, params.q);
            active = i + 1;
        }
        self.active = active;
        self.configured = Some(*params);
    }

    /// Notch the mains hum out of `buf` in place.
    ///
    /// Reconfigures first if `params` changed since the last call.  Filter
    /// state is preserved across a reconfigure so parameter automation does
    /// not click.
    ///
    /// # Real-time safety
    /// No allocation.
    pub fn process_block(&mut self, buf: &mut [f32], params: &HumParams) {
        if !params.enabled {
            return;
        }
        if self.configured.as_ref() != Some(params) {
            self.configure(params);
        }
        for notch in self.notches.iter_mut().take(self.active) {
            notch.process_block(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SR: u32 = 48_000;

    fn sine(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32).sqrt()
    }

    #[test]
    fn disabled_passes_through() {
        let mut bank = HumNotchBank::new(SR);
        let mut buf = sine(60.0, 1024);
        let before = buf.clone();
        bank.process_block(
            &mut buf,
            &HumParams {
                enabled: false,
                ..HumParams::default()
            },
        );
        assert_eq!(buf, before);
    }

    #[test]
    fn removes_fundamental_hum() {
        let mut bank = HumNotchBank::new(SR);
        let params = HumParams {
            enabled: true,
            fundamental_hz: 60.0,
            harmonics: 1,
            q: 30.0,
        };
        let mut buf = sine(60.0, 48_000);
        bank.process_block(&mut buf, &params);
        // Judge the settled tail (skip the notch's transient warm-up).
        let tail = &buf[24_000..];
        assert!(rms(tail) < 0.05, "60 Hz not notched: rms {}", rms(tail));
    }

    #[test]
    fn leaves_music_frequency_alone() {
        let mut bank = HumNotchBank::new(SR);
        let params = HumParams {
            enabled: true,
            fundamental_hz: 60.0,
            harmonics: 3,
            q: 30.0,
        };
        let mut buf = sine(1_000.0, 48_000);
        let before = rms(&buf[24_000..]);
        bank.process_block(&mut buf, &params);
        let after = rms(&buf[24_000..]);
        assert!(
            (after - before).abs() < 0.02,
            "1 kHz disturbed: {before}→{after}"
        );
    }

    #[test]
    fn harmonics_above_nyquist_are_skipped() {
        let mut bank = HumNotchBank::new(SR);
        let params = HumParams {
            enabled: true,
            fundamental_hz: 20_000.0, // 2nd harmonic (40 kHz) > Nyquist
            harmonics: 5,
            q: 30.0,
        };
        let mut buf = sine(1_000.0, 512);
        bank.process_block(&mut buf, &params);
        assert_eq!(bank.active, 1, "only the fundamental should be active");
    }
}
