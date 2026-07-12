//! Real-time neural amp modelling plugin.
//!
//! Phase 1 scaffold — see [`Roadmap-01-neural-amp.md`] for the plan.  The
//! plugin surface (`nih-plug` entry points, LV2 / CLAP / VST3 bundle
//! metadata, MIDI learn, slot UI) is filled in during Phase 2.
//!
//! [`Roadmap-01-neural-amp.md`]: ../../docs/Roadmap-01-neural-amp.md

// Re-export the inference core so callers only depend on one crate.
pub use rt_neural_core::{
    DenormalGuard, LstmModel, ModelError, ModelSlotBank, PreparedModel, SLOT_COUNT, WaveNetModel,
};
