//! Real-time neural amp modelling inference engine.
//!
//! # Crate scope
//!
//! `rt-neural-core` provides the **algorithmic core** for running pre-trained
//! neural amp / pedal models on an audio thread.  It is framework-agnostic —
//! no plugin format (`nih-plug`, LV2, CLAP, VST3) leaks into this crate.
//!
//! The two supported architectures are:
//!
//! - **LSTM** — small, ~1 % CPU, covers NAM and Proteus / GuitarML formats.
//! - **WaveNet** — dilated causal conv1d stack, higher fidelity for
//!   high-gain / complex distortion at higher CPU cost.
//!
//! Both are exposed through the same [`PreparedModel`] enum with a single
//! sample-by-sample entry point:
//!
//! ```rust,ignore
//! # use rt_neural_core::PreparedModel;
//! fn process(model: &mut PreparedModel, input: &[f32], output: &mut [f32]) {
//!     for (sample_in, sample_out) in input.iter().zip(output.iter_mut()) {
//!         *sample_out = model.process_sample(*sample_in);
//!     }
//! }
//! ```
//!
//! # Real-time safety
//!
//! Every `pub fn` on the render path (`process_sample`, `reset`) is
//! allocation-free, lock-free, and panic-free (release-build).  Weight
//! allocation, transposition and layout packing happen exclusively in the
//! loader, which is designed to be called from a background thread.
//!
//! # Module layout
//!
//! - [`activations`] — LUT-based tanh / sigmoid + branchless ReLU / GELU.
//! - [`dense`] — dense (matmul + bias) layer.
//! - [`conv1d`] — causal dilated Conv1D with a ring-buffered input history.
//! - [`lstm`] — LSTM cell (fused four-gate weights, gate-major layout).
//! - [`kernels`] — `pulp`-dispatched SIMD kernels backing the layers.
//! - [`prepared`] — the [`PreparedModel`] enum uniting LSTM and WaveNet.
//! - [`slot_bank`] — an 8-slot bank with lock-free hand-off from the loader
//!   thread to the audio thread.
//! - [`error`] — loader / validation errors.

#![warn(missing_docs)]

pub mod activations;
pub mod conv1d;
pub mod dense;
pub mod error;
pub mod kernels;
pub mod loader;
pub mod lstm;
pub mod prepared;
pub mod slot_bank;

pub use error::ModelError;
pub use loader::LoaderError;
pub use prepared::{LstmModel, PreparedModel, WaveNetModel};
pub use slot_bank::{ModelSlotBank, SLOT_COUNT};

// Re-export DenormalGuard so downstream plugins can pull it from a single
// place (they already depend on this crate for inference — no need to force
// them to also list `rt-fft-convolver`).
pub use rt_fft_convolver::DenormalGuard;
