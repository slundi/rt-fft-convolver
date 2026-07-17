//! `rt-adaptive` — real-time adaptive-filter plugin (feedback suppressor /
//! hum remover / echo canceller).
//!
//! # Layout
//!
//! Following the sibling `rt-vox` / `rt-limiter` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the [`AdaptiveParams`](params::AdaptiveParams) model with the
//!   Phase-2 control list ([`Mode`](params::Mode), the feedback / hum / echo
//!   controls, and the global gain / bypass) plus documented ranges.
//! - [`engine`] — [`AdaptiveController`](engine::AdaptiveController) (off-RT,
//!   rebuilds the echo bank when the AEC filter length changes) +
//!   [`AdaptiveProcessor`](engine::AdaptiveProcessor) (audio thread: runs the
//!   active mode, keeps the detector / notch banks / hum trackers resident),
//!   joined by the lock-free handoff from `rt-ir-plugin-core`.
//! - [`ui`] — pure, `egui`-free editor logic (frequency / Q / age formatting,
//!   spectrum downsampling + dB normalisation, notch-marker positioning),
//!   unit-testable without the plugin framework.
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled only
//!   with the **`nih`** feature (`cargo build -p rt-adaptive --features nih`).
//!   The `nih-plug-egui` editor (Phase 3) is behind the **`gui`** feature.
//!   Kept off the default build so `cargo test` / `clippy` stay fast and offline.

pub mod engine;
pub mod params;
pub mod ui;

pub use engine::{AdaptiveController, AdaptiveProcessor};
pub use params::{
    AdaptiveParams, CONVERGENCE_RANGE, DOUBLE_TALK_DB_RANGE, FilterLength, HUM_FUNDAMENTAL_RANGE,
    HUM_Q_RANGE, Mode, NOTCH_Q_RANGE, OUTPUT_GAIN_DB_RANGE, SENSITIVITY_RANGE,
};
pub use ui::SPECTRUM_DISPLAY_BINS;

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_adaptive_core as adaptive_core;

#[cfg(feature = "nih")]
mod plugin;
