//! `rt-vox` — real-time voice-processing plugin (vocoder / harmoniser /
//! formant shifter).
//!
//! # Layout
//!
//! Following the sibling `rt-pitch` / `rt-limiter` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the [`VoxParams`](params::VoxParams) model with the Phase-2
//!   control list ([`Mode`](params::Mode), [`Bands`](params::Bands), the four
//!   harmony voices, and the global mix / gain / bypass) plus documented ranges.
//! - [`engine`] — [`VoxController`](engine::VoxController) (off-RT, rebuilds the
//!   [`Vocoder`](rt_vox_core::Vocoder) when the band count changes) +
//!   [`VoxProcessor`](engine::VoxProcessor) (audio thread: runs the active mode,
//!   mixes dry/wet with latency compensation, routes MIDI to the harmoniser),
//!   joined by the lock-free handoff from `rt-ir-plugin-core`.
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled only
//!   with the **`nih`** feature (`cargo build -p rt-vox --features nih`).  The
//!   `nih-plug-egui` editor (Phase 3) is behind the **`gui`** feature.  Kept off
//!   the default build so `cargo test` / `clippy` stay fast and offline.

pub mod engine;
pub mod params;

pub use engine::{VoxController, VoxProcessor};
pub use params::{
    ATTACK_MS_RANGE, Bands, DETUNE_CENTS_RANGE, FORMANT_SHIFT_RANGE, Mode, OUTPUT_GAIN_DB_RANGE,
    PITCH_SHIFT_RANGE, RELEASE_MS_RANGE, VOICE_INTERVAL_RANGE, VOICE_LEVEL_RANGE, VoiceParams,
    VoxParams,
};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_vox_core as vox_core;

#[cfg(feature = "nih")]
mod plugin;
