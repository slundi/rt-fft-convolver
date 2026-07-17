//! Framework-agnostic parameter model for the `rt-loudness-meter` plugin.
//!
//! [`LoudnessParams`] mirrors the Phase-2 control list from
//! `docs/Roadmap-10-loudness-metering.md` as a plain `Copy` struct.  Keeping it
//! free of `nih-plug` (which lives behind the `nih` feature) lets the engine
//! and the target/threshold logic be unit-tested without the plugin framework.
//!
//! # One engine, preset targets (Phase 0)
//!
//! The plugin measures loudness *once* (BS.1770-4); a [`Target`] only chooses
//! the (LUFS, dBTP) pair the compliance check compares against.  Switching
//! target therefore never touches the measurement — it moves two threshold
//! numbers.  `Custom` exposes those two numbers directly.

/// Inclusive custom-target integrated-loudness range (LUFS).
pub const CUSTOM_LUFS_RANGE: (f32, f32) = (-36.0, 0.0);
/// Inclusive custom-target true-peak ceiling range (dBTP).
pub const CUSTOM_DBTP_RANGE: (f32, f32) = (-9.0, 0.0);

/// A delivery target: an integrated-loudness goal plus a true-peak ceiling.
///
/// The presets are the common streaming / broadcast specs; `Custom` defers to
/// [`LoudnessParams::custom_target_lufs`] / [`LoudnessParams::custom_target_dbtp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Target {
    /// EBU R128 broadcast: −23 LUFS, −1 dBTP.
    #[default]
    EbuR128,
    /// Spotify / YouTube / Tidal / Amazon: −14 LUFS, −1 dBTP.
    Streaming14,
    /// Apple Music / podcast: −16 LUFS, −1 dBTP.
    AppleMusic16,
    /// User-defined LUFS + dBTP.
    Custom,
}

impl Target {
    /// The (integrated-loudness target LUFS, true-peak ceiling dBTP) pair for a
    /// preset.  `Custom` returns `None` — the caller supplies the values from
    /// [`LoudnessParams`].
    #[inline]
    fn preset_thresholds(self) -> Option<(f32, f32)> {
        match self {
            Target::EbuR128 => Some((-23.0, -1.0)),
            Target::Streaming14 => Some((-14.0, -1.0)),
            Target::AppleMusic16 => Some((-16.0, -1.0)),
            Target::Custom => None,
        }
    }
}

/// A resolved compliance goal: the numbers the meter's readings are judged
/// against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Integrated-loudness target (LUFS).
    pub target_lufs: f32,
    /// True-peak ceiling (dBTP).
    pub ceiling_dbtp: f32,
}

/// The full parameter state of one `rt-loudness-meter` instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessParams {
    /// Delivery target for the compliance check.
    pub target: Target,
    /// Custom integrated-loudness target (used when `target == Custom`).
    pub custom_target_lufs: f32,
    /// Custom true-peak ceiling (used when `target == Custom`).
    pub custom_target_dbtp: f32,
    /// `Start/Stop`: when `false`, integrated + LRA freeze (momentary,
    /// short-term and peaks keep updating).
    pub running: bool,
    /// Rising edge clears integrated, LRA, history and the peak holds.
    pub reset: bool,
    /// Write meter readings to the log file (off-RT worker).
    pub log_enabled: bool,
    /// Host bypass — irrelevant to a meter (audio is never modified) but LV2
    /// requires the parameter to exist.
    pub bypass: bool,
}

impl Default for LoudnessParams {
    fn default() -> Self {
        Self {
            target: Target::default(),
            custom_target_lufs: -23.0,
            custom_target_dbtp: -1.0,
            running: true,
            reset: false,
            log_enabled: false,
            bypass: false,
        }
    }
}

impl LoudnessParams {
    /// Return a copy with the custom-target fields clamped into range.
    pub fn clamped(mut self) -> Self {
        self.custom_target_lufs = self
            .custom_target_lufs
            .clamp(CUSTOM_LUFS_RANGE.0, CUSTOM_LUFS_RANGE.1);
        self.custom_target_dbtp = self
            .custom_target_dbtp
            .clamp(CUSTOM_DBTP_RANGE.0, CUSTOM_DBTP_RANGE.1);
        self
    }

    /// Resolve the active compliance thresholds (preset or custom).
    pub fn thresholds(&self) -> Thresholds {
        match self.target.preset_thresholds() {
            Some((target_lufs, ceiling_dbtp)) => Thresholds {
                target_lufs,
                ceiling_dbtp,
            },
            None => Thresholds {
                target_lufs: self.custom_target_lufs,
                ceiling_dbtp: self.custom_target_dbtp,
            },
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_roadmap() {
        let params = LoudnessParams::default();
        assert_eq!(params.target, Target::EbuR128);
        assert!(params.running);
        assert!(!params.reset);
        assert!(!params.log_enabled);
        assert!(!params.bypass);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn presets_resolve_to_known_thresholds() {
        let mut params = LoudnessParams::default();
        params.target = Target::EbuR128;
        assert_eq!(params.thresholds().target_lufs, -23.0);
        params.target = Target::Streaming14;
        assert_eq!(params.thresholds().target_lufs, -14.0);
        params.target = Target::AppleMusic16;
        assert_eq!(params.thresholds().target_lufs, -16.0);
        // All presets share the −1 dBTP ceiling.
        assert_eq!(params.thresholds().ceiling_dbtp, -1.0);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn custom_target_uses_its_own_fields() {
        let mut params = LoudnessParams::default();
        params.target = Target::Custom;
        params.custom_target_lufs = -12.0;
        params.custom_target_dbtp = -2.0;
        let thresholds = params.thresholds();
        assert_eq!(thresholds.target_lufs, -12.0);
        assert_eq!(thresholds.ceiling_dbtp, -2.0);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn clamp_pulls_custom_target_into_range() {
        let mut params = LoudnessParams::default();
        params.custom_target_lufs = 50.0;
        params.custom_target_dbtp = 50.0;
        let params = params.clamped();
        assert_eq!(params.custom_target_lufs, CUSTOM_LUFS_RANGE.1);
        assert_eq!(params.custom_target_dbtp, CUSTOM_DBTP_RANGE.1);
    }

    #[test]
    fn clamp_is_idempotent_on_defaults() {
        let params = LoudnessParams::default();
        assert_eq!(params.clamped(), params);
    }
}
