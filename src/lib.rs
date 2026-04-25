mod dsp;
mod engine;
mod utils;

// ─── Public API ───────────────────────────────────────────────────────────────
//
// Internal modules are kept private; only the types and functions that form the
// crate's stable interface are re-exported here.

/// Partitioned-convolution engines.
pub use engine::partition::UniformPartitionEngine;
pub use engine::stereo::{StereoConvolver, TrueStereoConvolver};

/// IR loading and sample-rate conversion utilities.
pub use utils::resampler::{IrData, LoadError, load_ir, resample};

/// Anti-denormal protection helpers for the audio render thread.
pub use utils::denormals::{DenormalGuard, flush_to_zero};
