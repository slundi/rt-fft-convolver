//! Provenance and training metadata carried alongside every model.
//!
//! # What lives here
//!
//! [`ModelMetadata`] holds facts about a model that the code **cannot**
//! derive from the loaded weights alone:
//!
//! - The gear the model was captured from ("Fender Twin '65", "TS-9").
//! - The sample rate the training set was recorded at (48 kHz, 44.1 kHz).
//! - The input level the training set was recorded at (defaults to NAM's
//!   `-18 dBFS RMS` if unknown), used to place the "sweet spot" marker
//!   on the plugin's input meter.
//! - The source file format (`NAM`, `Proteus`, `GuitarML`).
//!
//! Fields the code **can** derive (architecture from the enum variant,
//! parameter count from summing weight lengths) stay on the model itself
//! rather than duplicating them here.
//!
//! # How it flows
//!
//! Metadata is populated by the (Phase-1.5) NAM / Proteus loader from the
//! JSON header, attached to the model via
//! [`LstmModel::with_metadata`](crate::prepared::LstmModel::with_metadata)
//! / [`WaveNetModel::with_metadata`](crate::prepared::WaveNetModel::with_metadata),
//! and displayed by the plugin editor.  Programmatic constructors (used
//! in tests today, and by users who assemble models by hand) can leave
//! every field at its `Default` — the editor renders "unknown" when a
//! field is `None`.

/// The recorded input level (dBFS RMS) most Neural Amp Modeler community
/// models are trained against.  Used as the default centre of the input
/// meter's "sweet spot" band when a model doesn't carry its own value.
pub const DEFAULT_SWEET_SPOT_DBFS: f32 = -18.0;

/// Provenance and training metadata attached to a prepared model.
///
/// Every field is optional so that programmatic constructors (tests,
/// tools building models on the fly) don't need to invent placeholder
/// values.  The plugin editor renders each `None` as "unknown".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelMetadata {
    /// The gear the model was captured from — e.g. "Fender Twin Reverb".
    pub source_gear_name: Option<String>,
    /// Sample rate the training dataset was recorded at, in Hz.
    pub training_sample_rate: Option<u32>,
    /// Input level (dBFS RMS) the model was trained at.  Overrides the
    /// plugin's default sweet-spot marker when `Some`.
    pub training_input_level_dbfs: Option<f32>,
    /// Name of the source file format — "NAM", "Proteus", "GuitarML".
    pub format_name: Option<String>,
}

impl ModelMetadata {
    /// Return the sweet-spot input level in dBFS.  Prefers the model's
    /// own value if `Some`, else falls back to [`DEFAULT_SWEET_SPOT_DBFS`].
    #[inline]
    pub fn sweet_spot_dbfs(&self) -> f32 {
        self.training_input_level_dbfs
            .unwrap_or(DEFAULT_SWEET_SPOT_DBFS)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, unpopulated metadata reports every field as unknown.
    #[test]
    fn default_is_all_unknown() {
        let meta = ModelMetadata::default();
        assert!(meta.source_gear_name.is_none());
        assert!(meta.training_sample_rate.is_none());
        assert!(meta.training_input_level_dbfs.is_none());
        assert!(meta.format_name.is_none());
    }

    /// Without a training-level override the sweet spot lands on the
    /// community default.
    #[test]
    fn sweet_spot_defaults_to_minus_eighteen_dbfs() {
        let meta = ModelMetadata::default();
        assert!((meta.sweet_spot_dbfs() - DEFAULT_SWEET_SPOT_DBFS).abs() < f32::EPSILON);
    }

    /// A per-model override wins over the default.
    #[test]
    fn model_supplied_level_overrides_default() {
        let meta = ModelMetadata {
            training_input_level_dbfs: Some(-12.0),
            ..ModelMetadata::default()
        };
        assert!((meta.sweet_spot_dbfs() - (-12.0)).abs() < f32::EPSILON);
    }
}
