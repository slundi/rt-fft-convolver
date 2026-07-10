//! `rt-ir-reverb` — convolution reverb plugin with true-stereo support.
//!
//! # Layout
//!
//! Same convention as `rt-ir-cab`: the DSP is **framework-agnostic** and fully
//! unit-testable; the `nih-plug` wrapper is an optional thin layer.
//!
//! - [`convolver`] — [`StereoConvolve`](convolver::StereoConvolve) abstraction
//!   over stereo / true-stereo IRs + a stereo equal-power crossfader.
//! - [`dsp`] — pre-delay, stereo width, and the dry/wet fader gain.
//! - [`params`] — the [`ReverbParams`](params::ReverbParams) model + presets.
//! - [`engine`] — [`ReverbController`](engine::ReverbController) (off-RT) +
//!   [`ReverbProcessor`](engine::ReverbProcessor) (audio thread).
//! - [`block`] — [`StereoBlockAdapter`](block::StereoBlockAdapter), fixed-block
//!   bridge with reported latency.
//! - `plugin` — the `nih-plug` `Plugin`, compiled only with the **`nih`**
//!   feature (`cargo build -p rt-ir-reverb --features nih`).
//!
//! The biquad EQ + gain helpers come from the shared
//! [`rt_ir_plugin_core::dsp`] module.

pub mod block;
pub mod convolver;
pub mod dsp;
pub mod engine;
pub mod params;

pub use block::StereoBlockAdapter;
pub use convolver::{ReverbConvolver, StereoConvolve, StereoSlotCrossfader};
pub use engine::{PreparedReverb, ReverbController, ReverbProcessor};
pub use params::{NUM_SLOTS, PRESETS, ReverbParams, ReverbPreset};

// Re-export the shared foundation for downstream convenience.
pub use rt_ir_plugin_core as core;

#[cfg(feature = "nih")]
mod plugin;
