//! `rt-ir-cab` — guitar / bass cabinet simulator plugin.
//!
//! # Layout
//!
//! Following the Phase-0 convention, the DSP is **framework-agnostic** and
//! fully unit-testable; the `nih-plug` wrapper is an optional thin layer:
//!
//! - [`dsp`] — biquad EQ + gain helpers.
//! - [`params`] — the [`CabParams`](params::CabParams) model and EQ presets.
//! - [`engine`] — [`CabController`](engine::CabController) (off-RT) +
//!   [`CabProcessor`](engine::CabProcessor) (audio thread), joined by the
//!   Phase-0 lock-free handoff.
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled only
//!   with the **`nih`** feature (`cargo build -p rt-ir-cab --features nih`).
//!   Kept off the default build so `cargo test`/`clippy` stay fast and offline.

pub mod block;
pub mod dsp;
pub mod engine;
pub mod params;

pub use block::StereoBlockAdapter;
pub use engine::{CabController, CabProcessor, PreparedCab};
pub use params::{CabParams, CabPreset, NUM_SLOTS, PRESETS};

// Re-export the shared foundation for downstream convenience.
pub use rt_ir_plugin_core as core;

#[cfg(feature = "nih")]
mod plugin;
