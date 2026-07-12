//! Plain, framework-agnostic parameter model for the neural amp plugin.
//!
//! [`NeuralAmpParams`] mirrors the Phase-2 control list from
//! `docs/Roadmap-01-neural-amp.md` as a simple `Copy` struct with documented
//! ranges and defaults.  Keeping the model free of `nih-plug` (which lives
//! behind the `nih` feature) lets us unit-test ranges and clamping without
//! pulling in the plugin framework.
//!
//! # Signal chain the params describe
//!
//! ```text
//! in ─▶ input_gain ─▶ [neural model, per-slot trim] ─▶ mix ─▶ output_gain ─▶ out
//!                                    ↑
//!                              current_slot
//! ```
//!
//! Under host bypass the whole chain is short-circuited: `out = in`.

/// Number of model slots per plugin instance (roadmap: 8).
pub const NUM_SLOTS: usize = 8;

/// Inclusive `(min, max)` range of the crossfade time in milliseconds.
pub const CROSSFADE_MS_RANGE: (f32, f32) = (5.0, 200.0);

/// Inclusive gain range in decibels for the input and output trims.
///
/// Neural amp models are trained at a specific input level (`~-18 dBFS RMS`
/// for NAM); the ±24 dB range covers the practical envelope for pickup
/// hotness differences and post-model make-up gain.
pub const GAIN_DB_RANGE: (f32, f32) = (-24.0, 24.0);

/// Inclusive per-slot model level trim in decibels.
///
/// A smaller range than the global input/output gain — this is a fixed
/// correction for the specific model's output level, not a performance control.
pub const MODEL_TRIM_DB_RANGE: (f32, f32) = (-12.0, 12.0);

/// The full parameter state of one neural amp instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuralAmpParams {
    /// Active model slot, `0..NUM_SLOTS`.
    pub current_slot: usize,
    /// Equal-power crossfade duration in ms when switching slots.
    pub crossfade_ms: f32,
    /// Input trim in dB, applied before the model.
    pub input_gain_db: f32,
    /// Output trim in dB, applied after the mix.
    pub output_gain_db: f32,
    /// Dry / wet blend, `0.0` = fully dry, `1.0` = fully wet (model).
    pub mix: f32,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
    /// Per-slot enable flag: disabled slots pass audio through dry.
    pub slot_enabled: [bool; NUM_SLOTS],
    /// Per-slot model level trim in dB.  Non-automatable — set once per
    /// model to correct its recorded loudness so switching preserves level.
    pub slot_trim_db: [f32; NUM_SLOTS],
}

impl Default for NeuralAmpParams {
    fn default() -> Self {
        Self {
            current_slot: 0,
            crossfade_ms: 20.0,
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            mix: 1.0,
            bypass: false,
            slot_enabled: [true; NUM_SLOTS],
            slot_trim_db: [0.0f32; NUM_SLOTS],
        }
    }
}

impl NeuralAmpParams {
    /// Return a copy of `self` with every field clamped into its documented
    /// range.  Useful when loading persisted state whose provenance is not
    /// trusted, or when a preset file exceeds the plugin's UI limits.
    pub fn clamped(mut self) -> Self {
        if self.current_slot >= NUM_SLOTS {
            self.current_slot = NUM_SLOTS - 1;
        }
        self.crossfade_ms = self
            .crossfade_ms
            .clamp(CROSSFADE_MS_RANGE.0, CROSSFADE_MS_RANGE.1);
        self.input_gain_db = self.input_gain_db.clamp(GAIN_DB_RANGE.0, GAIN_DB_RANGE.1);
        self.output_gain_db = self.output_gain_db.clamp(GAIN_DB_RANGE.0, GAIN_DB_RANGE.1);
        self.mix = self.mix.clamp(0.0, 1.0);
        for trim in self.slot_trim_db.iter_mut() {
            *trim = trim.clamp(MODEL_TRIM_DB_RANGE.0, MODEL_TRIM_DB_RANGE.1);
        }
        self
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Defaults match the roadmap's stated Phase-2 defaults.
    #[test]
    fn defaults_match_roadmap() {
        let params = NeuralAmpParams::default();
        assert_eq!(params.current_slot, 0);
        assert!((params.crossfade_ms - 20.0).abs() < 1e-6);
        assert_eq!(params.input_gain_db, 0.0);
        assert_eq!(params.output_gain_db, 0.0);
        assert_eq!(params.mix, 1.0);
        assert!(!params.bypass);
        assert_eq!(params.slot_enabled, [true; NUM_SLOTS]);
        assert_eq!(params.slot_trim_db, [0.0; NUM_SLOTS]);
    }

    /// Every field over its documented range is pulled back inside the range.
    #[test]
    fn clamp_pulls_every_field_into_range() {
        let params = NeuralAmpParams {
            current_slot: NUM_SLOTS + 3,
            crossfade_ms: 10_000.0,
            input_gain_db: 100.0,
            output_gain_db: -100.0,
            mix: -1.0,
            bypass: false,
            slot_enabled: [true; NUM_SLOTS],
            slot_trim_db: [100.0; NUM_SLOTS],
        };
        let clamped = params.clamped();

        assert_eq!(clamped.current_slot, NUM_SLOTS - 1);
        assert_eq!(clamped.crossfade_ms, CROSSFADE_MS_RANGE.1);
        assert_eq!(clamped.input_gain_db, GAIN_DB_RANGE.1);
        assert_eq!(clamped.output_gain_db, GAIN_DB_RANGE.0);
        assert_eq!(clamped.mix, 0.0);
        assert!(
            clamped
                .slot_trim_db
                .iter()
                .all(|value| *value == MODEL_TRIM_DB_RANGE.1)
        );
    }

    /// Clamp is a no-op when every field is already in range.
    #[test]
    fn clamp_is_idempotent_on_valid_input() {
        let params = NeuralAmpParams::default();
        assert_eq!(params.clamped(), params);
    }

    /// The number of slots matches the roadmap and rt-neural-core.
    #[test]
    fn slot_count_matches_core() {
        assert_eq!(NUM_SLOTS, rt_neural_core::SLOT_COUNT);
    }
}
