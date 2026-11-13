//! `rt-drive` — oversampled nonlinear-drive plugin.
//!
//! # Layout
//!
//! Following the sibling `rt-neural-amp` / `rt-ir-cab` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is
//! an optional thin layer:
//!
//! - [`params`] — the [`DriveParams`](params::DriveParams) model and its
//!   documented ranges.
//! - [`engine`] — [`DriveController`](engine::DriveController) (off-RT) +
//!   [`DriveProcessor`](engine::DriveProcessor) (audio thread), joined by the
//!   lock-free handoff from `rt-ir-plugin-core`.
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled only
//!   with the **`nih`** feature (`cargo build -p rt-drive --features nih`).
//!   Kept off the default build so `cargo test` / `clippy` stay fast and
//!   offline.

pub mod engine;
pub mod params;
pub mod ui;

pub use engine::{DriveController, DriveProcessor, PreparedDrive};
pub use params::{
    BIAS_RANGE, DRIVE_SWING_DB, DriveParams, LEVEL_DB_RANGE, OversampleChoice, TONE_DB_RANGE,
};
pub use ui::{AbSlot, AbState, LevelMeter, clipboard_string, peak_dbfs, transfer_curve};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_drive_core as drive_core;

#[cfg(feature = "nih")]
mod plugin;
