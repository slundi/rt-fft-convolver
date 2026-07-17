//! Internal sidechain filter — the detector-path EQ (Phase 0: internal
//! sidechain in v1; external aux input deferred to v2).
//!
//! The filter shapes the signal the *detector* sees, not the audio that passes
//! through the band, so it tilts *when* and *how hard* a band reacts without
//! colouring the output:
//!
//! - **HPF** — a highpass so low-frequency energy stops triggering the band
//!   (e.g. keep kick drum out of a vocal band, de-ess without pumping on
//!   plosives).
//! - **High shelf** — lifts (or dips) the detector's sensitivity to highs,
//!   the classic de-esser tilt.
//!
//! Two biquads in series per channel — RT-safe.  With the HPF at its 20 Hz
//! floor and the shelf at 0 dB the filter is effectively transparent, so a band
//! that does not use its sidechain pays only two cheap biquads.

use crate::biquad::Biquad;

/// Default (effectively "off") sidechain highpass corner in Hz.
pub const DEFAULT_HPF_HZ: f32 = 20.0;

/// Default sidechain high-shelf corner in Hz.
pub const DEFAULT_SHELF_HZ: f32 = 6_000.0;

/// A one-channel internal sidechain filter: highpass → high-shelf.
#[derive(Debug, Clone, Copy)]
pub struct SidechainFilter {
    sample_rate: f32,
    hpf: Biquad,
    shelf: Biquad,
    shelf_hz: f32,
    shelf_gain_db: f32,
}

impl SidechainFilter {
    /// Quality factor for both sections (Butterworth-ish, gentle).
    const Q: f32 = std::f32::consts::FRAC_1_SQRT_2;

    /// Build a transparent sidechain filter at `sample_rate` (HPF at 20 Hz,
    /// shelf at 0 dB).
    ///
    /// **Allocates** — off-RT only (trivially; the biquads are `Copy`).
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            hpf: Biquad::highpass(sample_rate, DEFAULT_HPF_HZ, Self::Q),
            shelf: Biquad::high_shelf(sample_rate, DEFAULT_SHELF_HZ, 0.0, Self::Q),
            shelf_hz: DEFAULT_SHELF_HZ,
            shelf_gain_db: 0.0,
        }
    }

    /// Set the highpass corner in Hz, preserving state.  RT-safe.
    pub fn set_hpf_hz(&mut self, cutoff_hz: f32) {
        self.hpf.set_highpass(self.sample_rate, cutoff_hz, Self::Q);
    }

    /// Set the high-shelf corner and gain, preserving state.  RT-safe.
    pub fn set_shelf(&mut self, corner_hz: f32, gain_db: f32) {
        self.shelf_hz = corner_hz;
        self.shelf_gain_db = gain_db;
        self.shelf
            .set_high_shelf(self.sample_rate, corner_hz, gain_db, Self::Q);
    }

    /// Clear the filter memory.
    #[inline]
    pub fn reset(&mut self) {
        self.hpf.reset();
        self.shelf.reset();
    }

    /// Filter one detector-path sample.  RT-safe.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        self.shelf.process_sample(self.hpf.process_sample(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_sine, rms};

    const SR: f32 = 48_000.0;

    fn settled_gain(mut filter: SidechainFilter, freq: f32) -> f32 {
        let buf = make_sine(freq, SR, 24_000);
        let out: Vec<f32> = buf.iter().map(|&x| filter.process(x)).collect();
        rms(&out[12_000..]) / rms(&buf[12_000..])
    }

    #[test]
    fn default_is_near_transparent() {
        let filter = SidechainFilter::new(SR);
        // A mid tone passes essentially untouched (HPF at 20 Hz, shelf 0 dB).
        assert!((settled_gain(filter, 1_000.0) - 1.0).abs() < 0.02);
    }

    #[test]
    fn hpf_removes_lows_from_the_detector() {
        let mut filter = SidechainFilter::new(SR);
        filter.set_hpf_hz(500.0);
        assert!(settled_gain(filter, 60.0) < 0.2, "HPF did not remove lows");
        assert!(settled_gain(filter, 4_000.0) > 0.9, "HPF ate the highs");
    }

    #[test]
    fn shelf_boosts_the_detector_highs() {
        let mut filter = SidechainFilter::new(SR);
        filter.set_shelf(4_000.0, 12.0);
        // The de-esser tilt: highs are amplified into the detector.
        assert!(settled_gain(filter, 12_000.0) > 2.0, "shelf did not boost");
        assert!((settled_gain(filter, 200.0) - 1.0).abs() < 0.05);
    }
}
