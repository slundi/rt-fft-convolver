//! Adaptive mains-hum tracker.
//!
//! Mains hum is a fundamental (50 / 60 Hz) plus harmonics.  A fixed comb of
//! notches removes it, but the mains frequency drifts by up to ~± 0.5 Hz, so a
//! pinned notch slowly leaks the fundamental.  This tracker (Roadmap-09 Phase 0)
//! wraps two reused primitives:
//!
//! - [`HumNotchBank`](rt_spectral_core::HumNotchBank) — the comb of RBJ biquad
//!   notches that does the removal.
//! - [`PitchTracker`](rt_pitch_core::PitchTracker) — an autocorrelation
//!   fundamental estimator that, when `adaptive` is on, nudges the comb to
//!   follow the drift.
//!
//! The estimate is **confidence-gated and clamped** to `nominal ± max_drift`,
//! so the comb tracks mains wander but never chases music down into the bass.
//!
//! # Real-time safety
//!
//! Analysis and notching run from pre-allocated buffers; [`process_block`]
//! allocates nothing.  The near-silent notch tails are covered by a
//! [`DenormalGuard`].
//!
//! [`process_block`]: HumTracker::process_block

use rt_fft_convolver::DenormalGuard;
use rt_pitch_core::PitchTracker;
use rt_spectral_core::{HumNotchBank, HumParams};

/// Lowest mains frequency the tracker will search (Hz).
const TRACK_MIN_HZ: f32 = 40.0;
/// Highest mains frequency the tracker will search (Hz).
const TRACK_MAX_HZ: f32 = 70.0;
/// Confidence below which an estimate is ignored (keeps the comb where it is).
const CONFIDENCE_GATE: f32 = 0.85;
/// One-pole smoothing applied to accepted estimates (0 = frozen, 1 = instant).
const DRIFT_SMOOTHING: f32 = 0.25;

/// Tracks the mains fundamental and drives a [`HumNotchBank`].
pub struct HumTracker {
    tracker: PitchTracker,
    bank: HumNotchBank,
    /// Analysis frame (long enough for the tracker's lowest lag).
    frame: Vec<f32>,
    /// Fill level of `frame`; a fresh estimate runs each time it is full.
    filled: usize,
    /// Nominal mains frequency (the `Fundamental` control, 50 / 60 / custom).
    nominal_hz: f32,
    /// Current tracked fundamental fed to the comb.
    tracked_hz: f32,
    /// Maximum deviation from `nominal_hz` the tracker may follow.
    max_drift_hz: f32,
    /// Whether to track drift (`false` pins the comb at `nominal_hz`).
    adaptive: bool,
    /// Number of harmonics to notch (`1 ..= MAX_HARMONICS`).
    harmonics: usize,
    /// Notch quality factor.
    q: f32,
    /// Whether the whole operator runs.
    enabled: bool,
}

impl HumTracker {
    /// Build a tracker for `sample_rate` at `nominal_hz` (typically 50 or 60).
    ///
    /// **Allocates** — off-RT only.
    pub fn new(sample_rate: u32, nominal_hz: f32) -> Self {
        let tracker = PitchTracker::new(sample_rate as f32, TRACK_MIN_HZ, TRACK_MAX_HZ);
        let frame_len = tracker.min_frame_len();
        Self {
            tracker,
            bank: HumNotchBank::new(sample_rate),
            frame: vec![0.0; frame_len],
            filled: 0,
            nominal_hz,
            tracked_hz: nominal_hz,
            max_drift_hz: 1.0,
            adaptive: true,
            harmonics: 3,
            q: 30.0,
            enabled: true,
        }
    }

    /// Enable or disable the whole operator (disabled = bypass, no notching).
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the nominal fundamental (50 / 60 / custom).  Resets the tracked
    /// value so a mode change doesn't carry stale drift.
    pub fn set_fundamental(&mut self, nominal_hz: f32) {
        self.nominal_hz = nominal_hz;
        self.tracked_hz = nominal_hz;
    }

    /// Toggle drift tracking.  When turned off the comb snaps back to nominal.
    pub fn set_adaptive(&mut self, adaptive: bool) {
        self.adaptive = adaptive;
        if !adaptive {
            self.tracked_hz = self.nominal_hz;
        }
    }

    /// Set how many harmonics (including the fundamental) to notch.
    pub fn set_harmonics(&mut self, harmonics: usize) {
        self.harmonics = harmonics;
    }

    /// Set the notch quality factor (higher is narrower).
    pub fn set_q(&mut self, q: f32) {
        self.q = q;
    }

    /// Set the maximum drift (Hz) the tracker may follow away from nominal.
    pub fn set_max_drift(&mut self, max_drift_hz: f32) {
        self.max_drift_hz = max_drift_hz.max(0.0);
    }

    /// The fundamental currently driving the comb (for the Phase 3 display).
    #[inline]
    pub fn tracked_hz(&self) -> f32 {
        self.tracked_hz
    }

    /// Clear the notch memories and analysis buffer (transport discontinuity).
    pub fn reset(&mut self) {
        self.bank.reset();
        self.filled = 0;
        self.tracked_hz = self.nominal_hz;
    }

    /// Remove mains hum from `buf` in place, updating the tracked fundamental
    /// first when `adaptive` is on.
    ///
    /// # Real-time safety
    /// No allocation.
    pub fn process_block(&mut self, buf: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let _guard = DenormalGuard::new();

        if self.adaptive {
            self.accumulate_and_track(buf);
        }

        let params = HumParams {
            enabled: true,
            fundamental_hz: self.tracked_hz,
            harmonics: self.harmonics,
            q: self.q,
        };
        self.bank.process_block(buf, &params);
    }

    /// Copy `buf` into the analysis frame; whenever a full frame accumulates,
    /// run the pitch tracker and nudge `tracked_hz`.
    fn accumulate_and_track(&mut self, buf: &[f32]) {
        let frame_len = self.frame.len();
        let mut read = 0;
        while read < buf.len() {
            let take = (frame_len - self.filled).min(buf.len() - read);
            self.frame[self.filled..self.filled + take].copy_from_slice(&buf[read..read + take]);
            self.filled += take;
            read += take;

            if self.filled == frame_len {
                self.update_tracked();
                self.filled = 0;
            }
        }
    }

    /// Run one estimate over the full frame and update `tracked_hz`.
    fn update_tracked(&mut self) {
        if let Some(estimate) = self.tracker.analyse(&self.frame)
            && estimate.confidence >= CONFIDENCE_GATE
        {
            let lo = self.nominal_hz - self.max_drift_hz;
            let hi = self.nominal_hz + self.max_drift_hz;
            let target = estimate.frequency.clamp(lo, hi);
            self.tracked_hz += DRIFT_SMOOTHING * (target - self.tracked_hz);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_sine, rms};

    const SR: u32 = 48_000;

    /// A 60 Hz fundamental plus a few harmonics, the classic hum signature.
    fn hum(fundamental: f32, len: usize) -> Vec<f32> {
        let f1 = make_sine(fundamental, SR as f32, len);
        let f2 = make_sine(fundamental * 2.0, SR as f32, len);
        let f3 = make_sine(fundamental * 3.0, SR as f32, len);
        (0..len)
            .map(|n| f1[n] + 0.5 * f2[n] + 0.25 * f3[n])
            .collect()
    }

    #[test]
    fn removes_hum_and_leaves_music() {
        let mut tracker = HumTracker::new(SR, 60.0);
        tracker.set_adaptive(false); // static comb, exact 60 Hz
        let mut buf = hum(60.0, 48_000);
        let before = rms(&buf[24_000..]);
        tracker.process_block(&mut buf);
        let after = rms(&buf[24_000..]);
        assert!(after < before * 0.2, "hum not removed: {before}→{after}");
    }

    #[test]
    fn tracks_fundamental_within_half_hz() {
        // Phase 4 bar: a drifted 59.9 Hz hum is identified within ± 0.5 Hz.
        let mut tracker = HumTracker::new(SR, 60.0);
        tracker.set_max_drift(1.0);
        // Feed enough blocks for the smoother to settle on the true frequency.
        let signal = hum(59.9, 48_000 * 3);
        for chunk in signal.chunks(2048) {
            let mut buf = chunk.to_vec();
            tracker.process_block(&mut buf);
        }
        assert!(
            (tracker.tracked_hz() - 59.9).abs() < 0.5,
            "tracked {} Hz",
            tracker.tracked_hz()
        );
    }

    #[test]
    fn disabled_is_transparent() {
        let mut tracker = HumTracker::new(SR, 60.0);
        tracker.set_enabled(false);
        let mut buf = hum(60.0, 4096);
        let before = buf.clone();
        tracker.process_block(&mut buf);
        assert_eq!(buf, before);
    }

    #[test]
    fn adaptive_off_pins_to_nominal() {
        let mut tracker = HumTracker::new(SR, 50.0);
        tracker.set_adaptive(false);
        // Even fed a 60 Hz signal, the comb must stay at the nominal 50 Hz.
        let signal = hum(60.0, 48_000);
        for chunk in signal.chunks(2048) {
            let mut buf = chunk.to_vec();
            tracker.process_block(&mut buf);
        }
        assert_eq!(tracker.tracked_hz(), 50.0);
    }

    #[test]
    fn drift_is_clamped_to_max() {
        let mut tracker = HumTracker::new(SR, 60.0);
        tracker.set_max_drift(0.5);
        // A wildly off 66 Hz signal must not drag the comb past nominal + 0.5.
        let signal = hum(66.0, 48_000 * 3);
        for chunk in signal.chunks(2048) {
            let mut buf = chunk.to_vec();
            tracker.process_block(&mut buf);
        }
        assert!(
            tracker.tracked_hz() <= 60.5 + 1e-3,
            "{}",
            tracker.tracked_hz()
        );
    }
}
