mod dsp;
mod engine;
mod error;
mod utils;

// ─── Public API ───────────────────────────────────────────────────────────────
//
// Internal modules are kept private; only the types and functions that form the
// crate's stable interface are re-exported here.

pub use dsp::direct::DirectConvolver;

/// Partitioned-convolution engines.
pub use engine::fdl::FreqDomainDelayLine;
pub use engine::partition::UniformPartitionEngine;
pub use engine::stereo::{StereoConvolver, TrueStereoConvolver};

pub use error::LoadError;

/// IR loading and sample-rate conversion utilities.
pub use utils::resampler::{IrData, load_ir, resample};

/// Anti-denormal protection helpers for the audio render thread.
pub use utils::denormals::{DenormalGuard, flush_to_zero};
