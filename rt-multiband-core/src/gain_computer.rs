//! The static gain computer — maps a detector level (dB) to a gain to apply
//! (dB), realising the compressor / expander / gate transfer curves with a
//! configurable **soft knee** (Phase 0: soft knee, `width = 0` ⇒ hard knee).
//!
//! The computer is **stateless** — pure transfer-curve math — so one instance
//! can drive both channels of a stereo-linked band.  Ballistics live in the
//! [`crate::EnvelopeFollower`]; makeup gain is applied by the
//! [`crate::BandProcessor`].
//!
//! # Curves (all return the gain to *add*, in dB, `≤ 0`)
//!
//! - **Compressor** — downward compression *above* threshold.  With hard knee,
//!   `over = level − threshold`; above threshold the output over-threshold is
//!   `over / ratio`, so the gain is `over · (1/ratio − 1)`.  The knee smooths
//!   the corner over `± width/2` dB with the standard quadratic interpolation
//!   (Reiss & McPherson).
//! - **Expander / Gate** — downward expansion *below* threshold: gain
//!   `= (level − threshold) · (ratio − 1)`, clamped to `−range` dB.  A gate is
//!   an expander with a high ratio and a deep range (chosen in Phase 2).

/// Per-band dynamics mode (Phase 0: each band independently one of these).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandMode {
    /// Downward compression above threshold.
    Compressor,
    /// Downward expansion below threshold.
    Expander,
    /// Expander with a high ratio + deep range (Phase 2 configures the values).
    Gate,
    /// No dynamics — unity gain (band still contributes to the sum).
    Bypass,
}

/// Stereo detector linking for a band (Phase 0: linked default + independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoLink {
    /// One shared gain driven by `max(|L|, |R|)` — no image shift.
    Linked,
    /// Independent per-channel gain reduction.
    Independent,
}

/// Stateless static transfer-curve evaluator.
#[derive(Debug, Clone, Copy)]
pub struct GainComputer {
    mode: BandMode,
    threshold_db: f32,
    ratio: f32,
    /// Soft-knee width in dB (`0` ⇒ hard knee).
    knee_db: f32,
    /// Maximum downward expansion for expander/gate, in dB (positive).
    range_db: f32,
}

impl GainComputer {
    /// Build a computer.  `ratio ≥ 1`; `knee_db ≥ 0`; `range_db ≥ 0`.
    pub fn new(mode: BandMode, threshold_db: f32, ratio: f32, knee_db: f32) -> Self {
        Self {
            mode,
            threshold_db,
            ratio: ratio.max(1.0),
            knee_db: knee_db.max(0.0),
            range_db: 60.0,
        }
    }

    /// Set the mode.  RT-safe.
    #[inline]
    pub fn set_mode(&mut self, mode: BandMode) {
        self.mode = mode;
    }

    /// Set the threshold in dBFS.  RT-safe.
    #[inline]
    pub fn set_threshold_db(&mut self, threshold_db: f32) {
        self.threshold_db = threshold_db;
    }

    /// Set the ratio (clamped to `≥ 1`).  RT-safe.
    #[inline]
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = ratio.max(1.0);
    }

    /// Set the soft-knee width in dB (`0` ⇒ hard knee).  RT-safe.
    #[inline]
    pub fn set_knee_db(&mut self, knee_db: f32) {
        self.knee_db = knee_db.max(0.0);
    }

    /// Set the expander/gate range (max downward expansion) in dB.  RT-safe.
    #[inline]
    pub fn set_range_db(&mut self, range_db: f32) {
        self.range_db = range_db.max(0.0);
    }

    /// The band mode.
    #[inline]
    pub fn mode(&self) -> BandMode {
        self.mode
    }

    /// Compute the gain to apply (dB, `≤ 0`) for a detector `level_db`.  RT-safe.
    #[inline]
    pub fn gain_db(&self, level_db: f32) -> f32 {
        match self.mode {
            BandMode::Bypass => 0.0,
            BandMode::Compressor => self.compressor_gain(level_db),
            BandMode::Expander | BandMode::Gate => self.expander_gain(level_db),
        }
    }

    /// Downward compression above threshold, with a soft knee.
    #[inline]
    fn compressor_gain(&self, level_db: f32) -> f32 {
        let over = level_db - self.threshold_db;
        let slope = 1.0 / self.ratio - 1.0; // ≤ 0
        let half_knee = self.knee_db * 0.5;
        if over <= -half_knee {
            // Below the knee: no reduction.
            0.0
        } else if over >= half_knee {
            // Above the knee: linear region.
            slope * over
        } else {
            // Inside the knee: quadratic interpolation.
            let knee_in = over + half_knee;
            slope * knee_in * knee_in / (2.0 * self.knee_db)
        }
    }

    /// Downward expansion below threshold, with a soft knee, clamped to range.
    #[inline]
    fn expander_gain(&self, level_db: f32) -> f32 {
        let over = level_db - self.threshold_db; // negative below threshold
        let slope = self.ratio - 1.0; // ≥ 0
        let half_knee = self.knee_db * 0.5;
        let gain = if over >= half_knee {
            // Well above threshold: no expansion.
            0.0
        } else if over <= -half_knee {
            // Well below threshold: linear expansion.
            slope * over
        } else {
            // Inside the knee: quadratic interpolation to the linear segment.
            let knee_in = over - half_knee;
            -slope * knee_in * knee_in / (2.0 * self.knee_db)
        };
        gain.max(-self.range_db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressor_hard_knee_matches_roadmap_acceptance() {
        // Phase 4 bar: 20 dB above threshold, 2:1, hard knee → −10 dB gain.
        let computer = GainComputer::new(BandMode::Compressor, -20.0, 2.0, 0.0);
        let gain = computer.gain_db(0.0); // level 20 dB above the −20 dB threshold
        assert!((gain - -10.0).abs() < 1e-4, "gain {gain} != −10 dB");
    }

    #[test]
    fn compressor_is_transparent_below_threshold() {
        let computer = GainComputer::new(BandMode::Compressor, -20.0, 4.0, 0.0);
        assert_eq!(computer.gain_db(-40.0), 0.0);
    }

    #[test]
    fn compressor_soft_knee_is_gentler_at_the_corner() {
        let hard = GainComputer::new(BandMode::Compressor, -20.0, 4.0, 0.0);
        let soft = GainComputer::new(BandMode::Compressor, -20.0, 4.0, 12.0);
        // Exactly at threshold the hard knee still gives 0 dB; the soft knee has
        // already begun reducing (its knee spans −26..−14 dB).
        assert_eq!(hard.gain_db(-20.0), 0.0);
        assert!(soft.gain_db(-20.0) < 0.0);
        // Far above the knee both converge to the same linear segment.
        assert!((hard.gain_db(10.0) - soft.gain_db(10.0)).abs() < 1e-3);
    }

    #[test]
    fn expander_downward_expansion_matches_ratio() {
        // 10 dB below a −20 dB threshold at 2:1 → −10 dB further reduction.
        let computer = GainComputer::new(BandMode::Expander, -20.0, 2.0, 0.0);
        let gain = computer.gain_db(-30.0);
        assert!((gain - -10.0).abs() < 1e-4, "gain {gain} != −10 dB");
    }

    #[test]
    fn expander_is_transparent_above_threshold() {
        let computer = GainComputer::new(BandMode::Expander, -20.0, 2.0, 0.0);
        assert_eq!(computer.gain_db(-5.0), 0.0);
    }

    #[test]
    fn expander_respects_range_clamp() {
        let mut computer = GainComputer::new(BandMode::Expander, -20.0, 4.0, 0.0);
        computer.set_range_db(12.0);
        // Deep below threshold the raw expansion would exceed 12 dB; clamp holds.
        assert_eq!(computer.gain_db(-80.0), -12.0);
    }

    #[test]
    fn bypass_is_always_unity() {
        let computer = GainComputer::new(BandMode::Bypass, -20.0, 4.0, 6.0);
        assert_eq!(computer.gain_db(20.0), 0.0);
        assert_eq!(computer.gain_db(-60.0), 0.0);
    }

    #[test]
    fn knee_is_continuous_at_both_edges() {
        let computer = GainComputer::new(BandMode::Compressor, 0.0, 3.0, 10.0);
        let slope = 1.0 / 3.0 - 1.0;
        // Lower knee edge (over = −5): joins the 0 dB segment.
        assert!(computer.gain_db(-5.0).abs() < 1e-4);
        // Upper knee edge (over = +5): joins the linear segment slope·over.
        assert!((computer.gain_db(5.0) - slope * 5.0).abs() < 1e-4);
    }
}
