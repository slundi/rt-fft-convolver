//! `rt-neural-amp` — neural amp modelling plugin.
//!
//! # Layout
//!
//! Following the sibling `rt-ir-cab` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper
//! is an optional thin layer:
//!
//! - [`params`] — the [`NeuralAmpParams`](params::NeuralAmpParams) model
//!   and its documented ranges.
//! - [`engine`] — [`NeuralAmpController`](engine::NeuralAmpController)
//!   (off-RT) + [`NeuralAmpProcessor`](engine::NeuralAmpProcessor) (audio
//!   thread), joined by the lock-free handoff from `rt-ir-plugin-core`.
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled
//!   only with the **`nih`** feature
//!   (`cargo build -p rt-neural-amp --features nih`).  Kept off the
//!   default build so `cargo test` / `clippy` stay fast and offline.
//!
//! # Model loading
//!
//! The NAM / Proteus JSON loader is intentionally *not* wired into this
//! crate in Phase 2 — see [`rt_neural_core::loader`] for its stub.  Slots
//! are populated programmatically in tests and (Phase 1.5) will accept a
//! JSON-parsed [`PreparedModel`](rt_neural_core::PreparedModel).

pub mod block;
pub mod engine;
pub mod params;

pub use block::MonoBlockAdapter;
pub use engine::{NeuralAmpController, NeuralAmpProcessor, NeuralModelWrapper, PreparedNeuralAmp};
pub use params::{
    CROSSFADE_MS_RANGE, GAIN_DB_RANGE, MODEL_TRIM_DB_RANGE, NUM_SLOTS, NeuralAmpParams,
};

// Re-export the inference core so callers only depend on one crate.
pub use rt_neural_core::{
    DenormalGuard, LstmModel, ModelError, ModelSlotBank, PreparedModel, WaveNetModel,
};

// Re-export the shared plugin foundation for downstream convenience.
pub use rt_ir_plugin_core as core;

#[cfg(feature = "nih")]
mod plugin;
