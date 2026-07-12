//! Loader / validation errors for `rt-neural-core`.
//!
//! Every failure that can occur off the real-time thread returns a
//! [`ModelError`].  Nothing on the render path (`process_sample`, `reset`)
//! can produce these — they are surfaced strictly at load time.

use thiserror::Error;

/// Failures that can occur while constructing or validating a model.
#[derive(Debug, Error)]
pub enum ModelError {
    /// The chosen architecture requires a hidden size ≥ 1.
    #[error("hidden size must be ≥ 1 (got {size})")]
    ZeroHiddenSize {
        /// The invalid hidden size that was requested.
        size: usize,
    },

    /// The layer received a weight matrix whose shape does not match the
    /// declared layer dimensions.
    #[error(
        "weight-shape mismatch: expected [{expected_rows}, {expected_cols}], got [{got_rows}, {got_cols}]"
    )]
    WeightShapeMismatch {
        /// Rows the layer expected.
        expected_rows: usize,
        /// Columns the layer expected.
        expected_cols: usize,
        /// Rows in the tensor supplied by the caller.
        got_rows: usize,
        /// Columns in the tensor supplied by the caller.
        got_cols: usize,
    },

    /// A bias vector length disagreed with the layer's output size.
    #[error("bias length mismatch: expected {expected}, got {got}")]
    BiasLengthMismatch {
        /// Output size the layer expected.
        expected: usize,
        /// Length of the bias vector supplied by the caller.
        got: usize,
    },

    /// A conv1d layer was configured with a non-positive kernel size or
    /// dilation.
    #[error("conv1d configuration invalid: kernel_size={kernel_size}, dilation={dilation}")]
    InvalidConv1dConfig {
        /// The kernel size that was requested.
        kernel_size: usize,
        /// The dilation factor that was requested.
        dilation: usize,
    },
}
