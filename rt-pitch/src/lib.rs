//! `rt-pitch` — real-time pitch-shift / harmoniser plugin.
//!
//! # Layout
//!
//! Following the sibling `rt-drive` / `rt-neural-amp` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the [`PitchParams`](params::PitchParams) model and its
//!   documented ranges (`Algorithm`, `Quality`, the eight interval slots).
//! - [`engine`] — [`PitchController`](engine::PitchController) (off-RT, builds
//!   the [`Shifter`](rt_pitch_core::Shifter) voices) +
//!   [`PitchProcessor`](engine::PitchProcessor) (audio thread, two-voice
//!   equal-power crossfade), joined by the lock-free handoff from
//!   `rt-ir-plugin-core`.
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled only
//!   with the **`nih`** feature (`cargo build -p rt-pitch --features nih`).
//!   Kept off the default build so `cargo test` / `clippy` stay fast and
//!   offline.

pub mod engine;
pub mod params;

pub use engine::{PitchController, PitchProcessor, PreparedPitch};
pub use params::{
    Algorithm, CROSSFADE_MS_RANGE, MAX_PITCH_HZ, MIN_PITCH_HZ, PitchParams, Quality, SEMITONE_RANGE,
};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_pitch_core as pitch_core;

#[cfg(feature = "nih")]
mod plugin;
