//! True-peak (dBTP) and sample-peak (dBFS) metering.
//!
//! # Reuse, not re-implementation
//!
//! BS.1770-4 measures true peak by up-sampling ≥ 4× and taking the peak of the
//! reconstructed signal (an inter-sample peak can sit above every stored
//! sample — see [`rt_limiter_core::TruePeakDetector`]'s own docs).  That
//! detector — a Kaiser half-band polyphase oversampler wrapped as a per-sample
//! `|x|` envelope — already exists, tested, in the limiter core.  Phase 0
//! (Roadmap-10) decided to **reuse it** rather than grow a second copy: one
//! detector per channel at 4×, plus a peak-hold across the session.
//!
//! Sample peak (dBFS) is the plain per-sample `|x|` max — free to track
//! alongside, and useful for spotting the gap to true peak.
//!
//! # Real-time safety
//!
//! Detectors and the oversampled scratch are sized in [`TruePeakMeter::new`];
//! [`TruePeakMeter::process_block`] only reads/writes those and updates two
//! `max` holds — no allocation, no lock, no panic for blocks `<= max_block`.

use rt_fft_convolver::linear_to_db;
use rt_limiter_core::TruePeakDetector;

/// Oversampling factor for true-peak (BS.1770-4 mandates ≥ 4×).
const OVERSAMPLE_FACTOR: usize = 4;

/// Oversampled true-peak + sample-peak meter across all channels.
pub struct TruePeakMeter {
    /// One 4× detector per channel.
    detectors: Vec<TruePeakDetector>,
    /// Oversampled-peak scratch, one input block wide, reused per channel.
    scratch: Vec<f32>,
    /// Running max of the true (oversampled) peak, linear.
    true_peak_hold: f32,
    /// Running max of the raw sample peak, linear.
    sample_peak_hold: f32,
    max_block: usize,
}

impl TruePeakMeter {
    /// Build a meter for `channels` channels and a maximum block of
    /// `max_block` samples, using the detector's default (high-quality)
    /// half-band design.
    ///
    /// **Allocates** — call off the audio thread.  (The detector also measures
    /// its group delay with a unit impulse here, off-RT.)
    pub fn new(channels: usize, max_block: usize) -> Self {
        let detectors = (0..channels)
            .map(|_| TruePeakDetector::new(OVERSAMPLE_FACTOR, max_block))
            .collect();
        Self {
            detectors,
            scratch: vec![0.0; max_block],
            true_peak_hold: 0.0,
            sample_peak_hold: 0.0,
            max_block,
        }
    }

    /// Build with an explicit half-band design (lighter designs keep tests
    /// fast; see [`TruePeakDetector::with_design`]).
    ///
    /// **Allocates** — call off the audio thread.
    pub fn with_design(
        channels: usize,
        max_block: usize,
        atten_db: f64,
        transition_frac: f64,
    ) -> Self {
        let detectors = (0..channels)
            .map(|_| {
                TruePeakDetector::with_design(
                    OVERSAMPLE_FACTOR,
                    max_block,
                    atten_db,
                    transition_frac,
                )
            })
            .collect();
        Self {
            detectors,
            scratch: vec![0.0; max_block],
            true_peak_hold: 0.0,
            sample_peak_hold: 0.0,
            max_block,
        }
    }

    /// Update both peak holds from one block of every channel.  Channels beyond
    /// the configured count (or a longer-than-`max_block` block) are ignored.
    ///
    /// **Real-time safe.**
    pub fn process_block(&mut self, channels: &[&[f32]]) {
        for (detector, input) in self.detectors.iter_mut().zip(channels) {
            let block = input.len().min(self.max_block);
            let input = &input[..block];

            // Sample peak: raw |x| max.
            for &sample in input {
                self.sample_peak_hold = self.sample_peak_hold.max(sample.abs());
            }

            // True peak: oversampled |x| envelope max over the block.
            let peaks = &mut self.scratch[..block];
            detector.detect_block(input, peaks);
            for &peak in peaks.iter() {
                self.true_peak_hold = self.true_peak_hold.max(peak);
            }
        }
    }

    /// Held true peak in **dBTP** (`−∞` until any signal has passed).
    pub fn true_peak_dbtp(&self) -> f32 {
        linear_to_db(self.true_peak_hold)
    }

    /// Held sample peak in **dBFS**.
    pub fn sample_peak_dbfs(&self) -> f32 {
        linear_to_db(self.sample_peak_hold)
    }

    /// Clear both holds and every detector's delay line.
    pub fn reset(&mut self) {
        self.true_peak_hold = 0.0;
        self.sample_peak_hold = 0.0;
        for detector in self.detectors.iter_mut() {
            detector.reset();
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_4, TAU};

    /// A light half-band design keeps the impulse-latency probe cheap.
    const LIGHT: (f64, f64) = (80.0, 0.1);

    #[test]
    fn empty_holds_are_silence() {
        let meter = TruePeakMeter::with_design(2, 64, LIGHT.0, LIGHT.1);
        assert_eq!(meter.true_peak_dbtp(), f32::NEG_INFINITY);
        assert_eq!(meter.sample_peak_dbfs(), f32::NEG_INFINITY);
    }

    #[test]
    fn sample_peak_tracks_maximum() {
        let mut meter = TruePeakMeter::with_design(1, 8, LIGHT.0, LIGHT.1);
        let block = [0.1f32, -0.5, 0.25, -0.8, 0.0, 0.3, -0.2, 0.4];
        meter.process_block(&[&block]);
        // −0.8 → 20·log10(0.8) ≈ −1.94 dBFS.
        assert!((meter.sample_peak_dbfs() - -1.938).abs() < 0.01);
    }

    #[test]
    fn true_peak_exceeds_sample_peak_for_intersample_case() {
        // 0.25·fs sine at 45°: every sample lands at ±0.707 but the waveform
        // reconstructs to ≈ full scale between them — the canonical ISP case.
        let block = 256;
        let signal: Vec<f32> = (0..block)
            .map(|n| (TAU * 0.25 * n as f32 + FRAC_PI_4).sin())
            .collect();
        let mut meter = TruePeakMeter::with_design(1, block, LIGHT.0, LIGHT.1);
        meter.process_block(&[&signal]);
        assert!(
            meter.true_peak_dbtp() > meter.sample_peak_dbfs(),
            "true peak {} should exceed sample peak {}",
            meter.true_peak_dbtp(),
            meter.sample_peak_dbfs()
        );
    }

    #[test]
    fn reset_clears_holds() {
        let mut meter = TruePeakMeter::with_design(1, 8, LIGHT.0, LIGHT.1);
        meter.process_block(&[&[1.0f32; 8]]);
        meter.reset();
        assert_eq!(meter.true_peak_dbtp(), f32::NEG_INFINITY);
        assert_eq!(meter.sample_peak_dbfs(), f32::NEG_INFINITY);
    }
}
