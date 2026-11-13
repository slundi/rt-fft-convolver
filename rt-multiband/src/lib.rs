//! `rt-multiband` — multi-band compressor / limiter / expander plugin.
//!
//! # Layout
//!
//! Following the sibling `rt-limiter` / `rt-drive` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the [`MultibandParams`](params::MultibandParams) model, its
//!   documented control ranges / mappings, and the MIDI-CC decode helpers.
//! - [`engine`] — [`MultibandProcessor`](engine::MultibandProcessor), the
//!   audio-thread renderer that holds a resident
//!   [`rt_multiband_core::MultibandEngine`] and applies parameters in place.
//!   (v1 has no reallocating change, so there is no off-thread controller — see
//!   the module docs.)
//! - [`ui`] — framework-agnostic editor helpers (meter maths, readout
//!   formatters), `egui`-free and unit-tested.
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled only
//!   with the **`nih`** feature (`cargo build -p rt-multiband --features nih`).
//!   The Phase-3 editor (framework still a Phase-3 decision) will land behind a
//!   `gui` feature later.
//!
//! The Phase-2 controls (`docs/Roadmap-11-multiband-compressor.md`) map onto
//! the engine as documented on [`params`].

pub mod engine;
pub mod params;
pub mod spectrum;
pub mod ui;

pub use engine::MultibandProcessor;
pub use params::{
    ATTACK_MS_RANGE, BLEND_RANGE, BandModeChoice, BandParams, CROSSOVER_HZ_RANGE, CcTarget,
    DEFAULT_CROSSOVER_HZ, INPUT_GAIN_DB_RANGE, KNEE_DB_RANGE, MAKEUP_CC_BASE, MAKEUP_DB_RANGE,
    MultibandParams, OUTPUT_GAIN_DB_RANGE, RATIO_RANGE, RELEASE_MS_RANGE, SC_HPF_HZ_RANGE,
    StereoLinkChoice, THRESHOLD_CC_BASE, THRESHOLD_DB_RANGE, cc_value_to_range, decode_cc,
};
pub use spectrum::SpectrumProbe;
pub use ui::{
    DISPLAY_MAX_HZ, DISPLAY_MIN_HZ, MAX_GR_DB, METER_FLOOR_DBFS, SILENCE_DBFS, SPECTRUM_FLOOR_DB,
    peak_dbfs,
};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_multiband_core as multiband_core;

#[cfg(feature = "nih")]
mod plugin;
