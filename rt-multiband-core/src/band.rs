//! One band's dynamics processor — the Phase 1
//! `BandProcessor { detector, gain_computer, envelope_follower, makeup_gain }`.
//!
//! It owns **per-channel** detector / envelope / sidechain state (so
//! `StereoLink::Independent` is a real per-channel gain) plus one shared,
//! stateless [`GainComputer`].  The feed-forward chain per channel is:
//!
//! ```text
//!   in ─▶ [sidechain filter] ─▶ [detector] ─▶ [envelope] ─▶ level_db
//!                                                              │
//!   in ───────────────────────────────────── × 10^((gain_db+makeup)/20) ─▶ out
//! ```
//!
//! In [`StereoLink::Linked`] the two channels' detector outputs are combined
//! with `max` and drive a single envelope, so both channels get identical gain
//! (no stereo-image shift).  `Solo` / `Mute` are honoured by the
//! [`crate::MultibandEngine`] when it sums the bands; the band itself always
//! computes its processed output.

use rt_fft_convolver::db_to_linear;

use crate::detector::Detector;
use crate::envelope::EnvelopeFollower;
use crate::gain_computer::{BandMode, GainComputer, StereoLink};
use crate::sidechain::SidechainFilter;

/// Number of channels a band processes (stereo).
const CHANNELS: usize = 2;

/// A single band's stereo dynamics processor.
#[derive(Debug, Clone)]
pub struct BandProcessor {
    detectors: [Detector; CHANNELS],
    envelopes: [EnvelopeFollower; CHANNELS],
    sidechains: [SidechainFilter; CHANNELS],
    gain_computer: GainComputer,
    link: StereoLink,
    makeup_db: f32,
    /// Audition this band alone.
    pub solo: bool,
    /// Silence this band in the sum.
    pub mute: bool,
    /// Last applied gain reduction in dB (`≤ 0`), for metering (Phase 3).
    gain_reduction_db: f32,
}

impl BandProcessor {
    /// Build a band at `sample_rate`.
    ///
    /// - `attack_ms` / `release_ms` — envelope ballistics.
    /// - `rms_window_ms`, `blend` — detector (peak/RMS blend).
    /// - `mode`, `threshold_db`, `ratio`, `knee_db` — static curve.
    ///
    /// **Allocates** — off-RT only.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f32,
        attack_ms: f32,
        release_ms: f32,
        rms_window_ms: f32,
        blend: f32,
        mode: BandMode,
        threshold_db: f32,
        ratio: f32,
        knee_db: f32,
    ) -> Self {
        Self {
            detectors: [Detector::new(sample_rate, rms_window_ms, blend); CHANNELS],
            envelopes: [EnvelopeFollower::new(sample_rate, attack_ms, release_ms); CHANNELS],
            sidechains: [SidechainFilter::new(sample_rate); CHANNELS],
            gain_computer: GainComputer::new(mode, threshold_db, ratio, knee_db),
            link: StereoLink::Linked,
            makeup_db: 0.0,
            solo: false,
            mute: false,
            gain_reduction_db: 0.0,
        }
    }

    /// Mutable access to the static gain computer (threshold / ratio / knee /
    /// mode / range setters).  RT-safe.
    #[inline]
    pub fn gain_computer_mut(&mut self) -> &mut GainComputer {
        &mut self.gain_computer
    }

    /// Mutable access to a channel's sidechain filter.  RT-safe.
    #[inline]
    pub fn sidechain_mut(&mut self, channel: usize) -> &mut SidechainFilter {
        &mut self.sidechains[channel]
    }

    /// Set attack/release on both channels.  RT-safe.
    pub fn set_ballistics(&mut self, attack_ms: f32, release_ms: f32) {
        for envelope in &mut self.envelopes {
            envelope.set_attack_ms(attack_ms);
            envelope.set_release_ms(release_ms);
        }
    }

    /// Set the peak/RMS blend on both channels.  RT-safe.
    pub fn set_blend(&mut self, blend: f32) {
        for detector in &mut self.detectors {
            detector.set_blend(blend);
        }
    }

    /// Set the stereo detector linking.  RT-safe.
    #[inline]
    pub fn set_link(&mut self, link: StereoLink) {
        self.link = link;
    }

    /// Set the makeup gain in dB.  RT-safe.
    #[inline]
    pub fn set_makeup_db(&mut self, makeup_db: f32) {
        self.makeup_db = makeup_db;
    }

    /// The band's current dynamics mode.
    #[inline]
    pub fn mode(&self) -> BandMode {
        self.gain_computer.mode()
    }

    /// Last applied gain reduction in dB (`≤ 0`), for GR metering.
    #[inline]
    pub fn gain_reduction_db(&self) -> f32 {
        self.gain_reduction_db
    }

    /// Clear all per-channel state.
    pub fn reset(&mut self) {
        for channel in 0..CHANNELS {
            self.detectors[channel].reset();
            self.envelopes[channel].reset();
            self.sidechains[channel].reset();
        }
        self.gain_reduction_db = 0.0;
    }

    /// Map a shaped envelope level to the linear gain to apply (`≤ 1` for
    /// reduction, `> 1` only via makeup), caching the reduction for metering.
    #[inline]
    fn level_to_gain(&mut self, env_level: f32) -> f32 {
        let level_db = rt_fft_convolver::linear_to_db(env_level);
        let reduction_db = self.gain_computer.gain_db(level_db);
        // Track the deepest reduction seen this sample (most negative).
        self.gain_reduction_db = self.gain_reduction_db.min(reduction_db);
        db_to_linear(reduction_db + self.makeup_db)
    }

    /// Process one stereo sample `(left, right)` and return the band's
    /// `(left, right)` contribution.  RT-safe.
    #[inline]
    pub fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        // Reset the per-sample reduction tracker to "no reduction".
        self.gain_reduction_db = 0.0;

        // Sidechain-filter each channel into its detector.
        let key_left = self.detectors[0].process(self.sidechains[0].process(left));
        let key_right = self.detectors[1].process(self.sidechains[1].process(right));

        match self.link {
            StereoLink::Linked => {
                // Shared key drives one envelope; both channels get one gain.
                let key = self.envelopes[0].process(key_left.max(key_right));
                let gain = self.level_to_gain(key);
                (left * gain, right * gain)
            }
            StereoLink::Independent => {
                let env_left = self.envelopes[0].process(key_left);
                let env_right = self.envelopes[1].process(key_right);
                let gain_left = self.level_to_gain(env_left);
                let gain_right = self.level_to_gain(env_right);
                (left * gain_left, right * gain_right)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_sine, rms};

    const SR: f32 = 48_000.0;

    fn comp_band(threshold_db: f32, ratio: f32) -> BandProcessor {
        // Fast, hard-knee, peak detector so steady-state matches the static curve.
        BandProcessor::new(
            SR,
            1.0,
            50.0,
            2.0,
            1.0,
            BandMode::Compressor,
            threshold_db,
            ratio,
            0.0,
        )
    }

    #[test]
    fn bypass_mode_passes_signal_unchanged() {
        let mut band = comp_band(-20.0, 4.0);
        band.gain_computer_mut().set_mode(BandMode::Bypass);
        let (left, right) = band.process(0.5, -0.3);
        assert!((left - 0.5).abs() < 1e-6 && (right - -0.3).abs() < 1e-6);
    }

    #[test]
    fn compressor_reduces_a_loud_tone() {
        // A 0 dBFS tone into a −20 dB threshold, 4:1 compressor: the output must
        // be quieter than the input once the envelope settles.
        let mut band = comp_band(-20.0, 4.0);
        let tone = make_sine(1_000.0, SR, 24_000);
        let out: Vec<f32> = tone.iter().map(|&x| band.process(x, x).0).collect();
        let in_rms = rms(&tone[12_000..]);
        let out_rms = rms(&out[12_000..]);
        assert!(
            out_rms < in_rms * 0.7,
            "no compression: {out_rms} vs {in_rms}"
        );
        assert!(band.gain_reduction_db() < -3.0, "GR meter not tracking");
    }

    #[test]
    fn makeup_gain_lifts_the_output() {
        let mut band = comp_band(-20.0, 4.0);
        band.gain_computer_mut().set_mode(BandMode::Bypass);
        band.set_makeup_db(6.0);
        let (left, _) = band.process(0.5, 0.5);
        assert!((left - 0.5 * db_to_linear(6.0)).abs() < 1e-4);
    }

    #[test]
    fn linked_applies_equal_gain_to_both_channels() {
        // A loud left, silent right: linked mode ducks both by the same factor.
        let mut band = comp_band(-20.0, 4.0);
        band.set_link(StereoLink::Linked);
        let tone = make_sine(1_000.0, SR, 24_000);
        let mut last = (0.0, 0.0);
        for &x in &tone {
            last = band.process(x, 0.5); // right held at a constant 0.5
        }
        // Right-channel gain = last.1 / 0.5; left-channel input is `last of tone`.
        let right_gain = last.1 / 0.5;
        assert!(right_gain < 0.9, "linked did not duck the quiet channel");
    }

    #[test]
    fn independent_leaves_quiet_channel_alone() {
        let mut band = comp_band(-20.0, 4.0);
        band.set_link(StereoLink::Independent);
        let tone = make_sine(1_000.0, SR, 24_000);
        let mut last = (0.0, 0.0);
        for &x in &tone {
            last = band.process(x, 0.001); // right essentially silent
        }
        let right_gain = last.1 / 0.001;
        assert!(right_gain > 0.95, "independent ducked a silent channel");
    }

    #[test]
    fn silence_stays_silent() {
        let mut band = comp_band(-40.0, 8.0);
        for _ in 0..1_000 {
            let (left, right) = band.process(0.0, 0.0);
            assert_eq!((left, right), (0.0, 0.0));
        }
    }
}
