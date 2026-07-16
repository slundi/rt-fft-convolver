//! `rt-limiter` — true-peak look-ahead brick-wall limiter / maximiser plugin.
//!
//! # Layout
//!
//! Following the sibling `rt-drive` / `rt-physical` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the [`LimiterParams`](params::LimiterParams) model and its
//!   documented control ranges / mappings.
//! - [`engine`] — [`LimiterController`](engine::LimiterController) (off-RT) +
//!   [`LimiterProcessor`](engine::LimiterProcessor) (audio thread), joined by
//!   the lock-free handoff from `rt-ir-plugin-core`.
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled only
//!   with the **`nih`** feature (`cargo build -p rt-limiter --features nih`).
//!   Kept off the default build so `cargo test` / `clippy` stay fast and
//!   offline.
//!
//! The Phase-2 controls (`docs/Roadmap-07-true-peak-limiter.md`) map onto the
//! engine as documented on [`params`]; the Phase-3 `nih-plug-egui` editor
//! (meters, GR history) is a later addition.

pub mod engine;
pub mod params;

pub use engine::{LimiterController, LimiterProcessor, PreparedLimiter};
pub use params::{
    CEILING_DB_RANGE, DitherChoice, LOOKAHEAD_MS_RANGE, LimiterParams, LinkChoice,
    OversampleChoice, RELEASE_MS_RANGE, Style, THRESHOLD_DB_RANGE,
};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_limiter_core as limiter_core;

#[cfg(feature = "nih")]
mod plugin;
