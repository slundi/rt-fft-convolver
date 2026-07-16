//! `rt-physical` — physical-modelling instrument plugin.
//!
//! A polyphonic instrument wrapping the framework-agnostic
//! [`rt_physical_core`] engine (see `docs/Roadmap-06-physical-modeling.md`,
//! Phase 2): extended Karplus-Strong plucked strings and a modal resonator
//! bank, MIDI-driven (synth use) with an audio-input dry/wet blend (effect
//! use), body stereoisation, and quietest-then-oldest voice stealing.
//!
//! # Layout
//!
//! Following the sibling `rt-fdn-reverb` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the [`InstrumentParams`](params::InstrumentParams) model, the
//!   automatable control ranges, the model / preset / body dropdowns, and the
//!   control → engine-unit mappings, unit-testable without the framework.
//! - [`engine`] — [`PhysicalEngine`](engine::PhysicalEngine): the two pre-built
//!   voice banks, the body stereoiser and the global gain / mix / bypass bus,
//!   all allocation-free on the audio thread.
//! - [`ui`] — framework-agnostic display maths for the editor (the partial /
//!   mode spectrum), unit-testable without the framework.
//! - `plugin` — the actual `nih-plug` `Plugin`, compiled only with the **`nih`**
//!   feature (`cargo build -p rt-physical --features nih`).  Kept off the
//!   default build so `cargo test` / `clippy` stay fast and offline.  The
//!   Phase-3 `nih-plug-egui` editor is behind the additional **`gui`** feature
//!   (`--features gui`), which links X11/OpenGL.

pub mod engine;
pub mod params;
pub mod ui;

pub use engine::PhysicalEngine;
pub use params::{
    BODY_TYPES, BodyType, DECAY_S_RANGE, DECAY_SLOPE_RANGE, DETUNE_CENTS_RANGE, GAIN_DB_RANGE,
    InstrumentParams, MODELS, MODES_RANGE, POLYPHONY_RANGE, PRESETS, db_to_gain, model_from_index,
    model_label, model_to_index, preset_from_index, preset_label, preset_to_index,
    tune_mult_from_cents,
};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_physical_core as physical_core;

#[cfg(feature = "nih")]
mod plugin;
