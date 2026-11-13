//! Framework-agnostic parameter model for the `rt-drive` plugin.
//!
//! [`DriveParams`] mirrors the Phase-2 control list from
//! `docs/Roadmap-02-oversampled-drive.md` as a plain `Copy` struct with
//! documented ranges.  Keeping it free of `nih-plug` (which lives behind the
//! `nih` feature) lets the engine and ranges be unit-tested without the plugin
//! framework.
//!
//! # How the controls relate to a pedal preset
//!
//! A pedal preset ([`rt_drive_core::PedalModel`], selected by `model_index`)
//! supplies the *character*: the shaper curve, DC bias, reference gains and
//! default oversampling.  The continuous controls act **on top**:
//!
//! - `drive` `0..1` (default `0.5`) → a `±`[`DRIVE_SWING_DB`] dB offset on the
//!   pedal's input gain; `0.5` = the pedal as authored.
//! - `level_db` → an offset on the pedal's output gain.
//! - `bias` → an offset on the pedal's DC bias.
//! - `bass_db` / `mid_db` / `treble_db` → the plugin's own 3-band tone stack
//!   (independent of the pedal's authored voicing; `0 dB` = flat).
//! - `mix` `0..1` → dry / wet blend.
//! - `oversampling` → [`OversampleChoice::Preset`] uses the pedal's default.

/// Half-range of the `drive` control's input-gain offset, in dB.  At
/// `drive = 0.0` the offset is `−DRIVE_SWING_DB`; at `1.0`, `+DRIVE_SWING_DB`.
pub const DRIVE_SWING_DB: f32 = 24.0;

/// Inclusive `Level` range in dB (post-shaper make-up, offset on the pedal).
pub const LEVEL_DB_RANGE: (f32, f32) = (-24.0, 24.0);

/// Inclusive per-band tone range in dB.
pub const TONE_DB_RANGE: (f32, f32) = (-12.0, 12.0);

/// Inclusive pre-shaper bias range (offset on the pedal's bias).
pub const BIAS_RANGE: (f32, f32) = (-1.0, 1.0);

/// User-selectable oversampling factor.
///
/// [`OversampleChoice::Preset`] defers to the selected pedal's
/// `default_oversampling`; the explicit factors override it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OversampleChoice {
    /// Use the pedal preset's own `default_oversampling`.
    Preset,
    /// No oversampling.
    X1,
    /// 2× oversampling.
    X2,
    /// 4× oversampling.
    X4,
    /// 8× oversampling.
    X8,
    /// 16× oversampling.
    X16,
}

impl OversampleChoice {
    /// The concrete factor this choice resolves to, given the pedal's default.
    pub fn factor(self, pedal_default: u8) -> u8 {
        match self {
            OversampleChoice::Preset => pedal_default,
            OversampleChoice::X1 => 1,
            OversampleChoice::X2 => 2,
            OversampleChoice::X4 => 4,
            OversampleChoice::X8 => 8,
            OversampleChoice::X16 => 16,
        }
    }
}

/// The full parameter state of one `rt-drive` instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveParams {
    /// Index into the shipped pedal bank.
    pub model_index: usize,
    /// Drive amount `0.0..=1.0` (mapped to an input-gain offset).
    pub drive: f32,
    /// Output make-up level in dB.
    pub level_db: f32,
    /// Bass tone in dB.
    pub bass_db: f32,
    /// Mid tone in dB.
    pub mid_db: f32,
    /// Treble tone in dB.
    pub treble_db: f32,
    /// Dry / wet blend `0.0` = dry, `1.0` = wet.
    pub mix: f32,
    /// Oversampling factor choice.
    pub oversampling: OversampleChoice,
    /// Pre-shaper bias offset `-1.0..=1.0`.
    pub bias: f32,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
}

impl Default for DriveParams {
    fn default() -> Self {
        Self {
            model_index: 0,
            drive: 0.5,
            level_db: 0.0,
            bass_db: 0.0,
            mid_db: 0.0,
            treble_db: 0.0,
            mix: 1.0,
            oversampling: OversampleChoice::Preset,
            bias: 0.0,
            bypass: false,
        }
    }
}

impl DriveParams {
    /// Return a copy with every continuous field clamped into range.
    ///
    /// `model_index` is *not* clamped here (its valid range depends on the
    /// runtime bank size); the engine clamps it against the bank.
    pub fn clamped(mut self) -> Self {
        self.drive = self.drive.clamp(0.0, 1.0);
        self.level_db = self.level_db.clamp(LEVEL_DB_RANGE.0, LEVEL_DB_RANGE.1);
        self.bass_db = self.bass_db.clamp(TONE_DB_RANGE.0, TONE_DB_RANGE.1);
        self.mid_db = self.mid_db.clamp(TONE_DB_RANGE.0, TONE_DB_RANGE.1);
        self.treble_db = self.treble_db.clamp(TONE_DB_RANGE.0, TONE_DB_RANGE.1);
        self.mix = self.mix.clamp(0.0, 1.0);
        self.bias = self.bias.clamp(BIAS_RANGE.0, BIAS_RANGE.1);
        self
    }

    /// The input-gain offset in dB the `drive` control maps to.
    ///
    /// `drive = 0.5` → `0 dB` (pedal as authored); the ends reach
    /// `±`[`DRIVE_SWING_DB`].
    #[inline]
    pub fn drive_offset_db(self) -> f32 {
        (self.drive - 0.5) * 2.0 * DRIVE_SWING_DB
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_roadmap() {
        let params = DriveParams::default();
        assert_eq!(params.model_index, 0);
        assert_eq!(params.drive, 0.5);
        assert_eq!(params.level_db, 0.0);
        assert_eq!(params.mix, 1.0);
        assert_eq!(params.oversampling, OversampleChoice::Preset);
        assert!(!params.bypass);
    }

    #[test]
    fn drive_offset_is_zero_at_centre() {
        let params = DriveParams {
            drive: 0.5,
            ..Default::default()
        };
        assert!(params.drive_offset_db().abs() < 1e-6);
    }

    #[test]
    fn drive_offset_reaches_swing_at_extremes() {
        let hot = DriveParams {
            drive: 1.0,
            ..Default::default()
        };
        let cold = DriveParams {
            drive: 0.0,
            ..Default::default()
        };
        assert!((hot.drive_offset_db() - DRIVE_SWING_DB).abs() < 1e-6);
        assert!((cold.drive_offset_db() + DRIVE_SWING_DB).abs() < 1e-6);
    }

    #[test]
    fn preset_choice_defers_to_pedal_default() {
        assert_eq!(OversampleChoice::Preset.factor(8), 8);
        assert_eq!(OversampleChoice::X2.factor(8), 2);
        assert_eq!(OversampleChoice::X16.factor(2), 16);
    }

    #[test]
    fn clamp_pulls_continuous_fields_into_range() {
        let params = DriveParams {
            model_index: 0,
            drive: 5.0,
            level_db: 100.0,
            bass_db: -100.0,
            mid_db: 100.0,
            treble_db: -100.0,
            mix: 2.0,
            oversampling: OversampleChoice::X4,
            bias: -5.0,
            bypass: false,
        }
        .clamped();
        assert_eq!(params.drive, 1.0);
        assert_eq!(params.level_db, LEVEL_DB_RANGE.1);
        assert_eq!(params.bass_db, TONE_DB_RANGE.0);
        assert_eq!(params.mid_db, TONE_DB_RANGE.1);
        assert_eq!(params.mix, 1.0);
        assert_eq!(params.bias, BIAS_RANGE.0);
    }

    #[test]
    fn clamp_is_idempotent_on_defaults() {
        let params = DriveParams::default();
        assert_eq!(params.clamped(), params);
    }
}
