//! `rt-spectral` — real-time spectral (STFT) processing plugin.
//!
//! Learn-seeded Wiener denoise, spectral gate / compressor, and time-domain
//! hum-notch / click-repair, wrapping the framework-agnostic
//! [`rt_spectral_core`] engine (see `docs/Roadmap-04-spectral-processing.md`,
//! Phase 2).
//!
//! # Layout
//!
//! Following the sibling `rt-pitch` / `rt-drive` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the automatable control ranges and the
//!   [`clamp_params`](params::clamp_params) helper that keeps a
//!   [`SpectralParams`] snapshot in range, unit-testable without the framework.
//! - [`engine`] — [`SpectralController`](engine::SpectralController) (off-RT:
//!   builds the per-channel [`SpectralEngine`]s, turns a captured noise clip
//!   into a [`NoiseProfile`](rt_spectral_core::NoiseProfile) and publishes it)
//!   plus [`SpectralProcessor`](engine::SpectralProcessor) (audio thread: the
//!   allocation-free render + the RT-safe Learn capture), joined by two
//!   lock-free hand-offs.
//! - `plugin` — the actual `nih-plug` `Plugin`, compiled only with the **`nih`**
//!   feature (`cargo build -p rt-spectral --features nih`).  Kept off the
//!   default build so `cargo test` / `clippy` stay fast and offline.  The
//!   `nih-plug-egui` editor (spectrogram, profile overlay) lands in Phase 3.

pub mod engine;
pub mod params;
pub mod ui;

pub use engine::{
    SpectralController, SpectralProcessor, SpectralTelemetry, TelemetryFrame, spectral_engine,
};
pub use params::{
    ATTACK_MS_RANGE, HARMONICS_RANGE, KNEE_DB_RANGE, LEARN_MS_RANGE, MAKEUP_DB_RANGE, MAX_LEARN_MS,
    MIX_RANGE, OUTPUT_GAIN_DB_RANGE, RATIO_RANGE, REDUCTION_DB_RANGE, RELEASE_MS_RANGE,
    SENSITIVITY_DB_RANGE, clamp_params, learn_samples,
};
pub use ui::{
    DISPLAY_CEIL_DB, DISPLAY_FLOOR_DB, Meter, SPECTROGRAM_BINS, SPECTROGRAM_COLUMNS, Spectrogram,
    db_to_intensity, downsample_log, format_db, format_latency_ms, latency_ms, magnitude_to_db,
};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_spectral_core as spectral_core;

#[cfg(feature = "nih")]
mod plugin;
