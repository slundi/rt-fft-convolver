//! Model file loading — **stub, filled in Phase 1.5**.
//!
//! # Status
//!
//! Phase 0 decided the two supported input formats:
//!
//! - **NAM** (`.nam` JSON, Steven Atkinson):
//!   <https://github.com/sdatkinson/NeuralAmpModelerCore>
//! - **Proteus / GuitarML** (`.json` from Aida / GuitarML training):
//!   <https://github.com/GuitarML/Proteus>
//!
//! Phase 1 built every layer primitive, the [`PreparedModel`] enum, the
//! slot bank, and the plugin control surface.  Actual JSON parsing is the
//! remaining piece and lands in Phase 1.5 (a focused follow-up that only
//! adds the two parsers plus their integration tests).
//!
//! Until then, downstream plugins populate slots programmatically — that
//! is exactly the API [`rt-neural-amp`](../../rt_neural_amp/index.html)
//! tests exercise.
//!
//! # Sketch of the target API
//!
//! ```rust,ignore
//! use rt_neural_core::{loader, PreparedModel};
//!
//! // NAM `.nam` (JSON) parsed to a PreparedModel.
//! let model_bytes = std::fs::read("cool_amp.nam")?;
//! let model: PreparedModel = loader::load_nam(&model_bytes)?;
//!
//! // Proteus / GuitarML JSON.
//! let model_bytes = std::fs::read("my_amp.json")?;
//! let model: PreparedModel = loader::load_proteus(&model_bytes)?;
//! ```
//!
//! Both entry points will:
//! 1. Parse the JSON header + config with `serde_json`.
//! 2. Validate architecture and dimensions against [`crate::error::ModelError`].
//! 3. Transpose weight tensors into the gate-major / channel-major layouts
//!    that [`crate::lstm::LstmCell`] / [`crate::conv1d::Conv1d`] expect
//!    (see Phase 0 decisions in `docs/Roadmap-01-neural-amp.md`).
//! 4. Return a fully-prepared [`PreparedModel`] ready for the audio thread.
//!
//! [`PreparedModel`]: crate::PreparedModel

use crate::PreparedModel;
use crate::error::ModelError;

/// Failures that can occur while parsing a model file.
///
/// A superset of [`ModelError`] — file-format errors sit at this layer, then
/// shape / config validation delegates to `ModelError` once parsed.
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    /// The model architecture named in the file is not one we support.
    #[error("unsupported architecture: {name}")]
    UnsupportedArchitecture {
        /// Architecture string as it appeared in the JSON.
        name: String,
    },

    /// The JSON parser rejected the payload.  Wrapped for context; downstream
    /// callers only need to `Display` it.
    #[error("JSON parse error: {message}")]
    ParseError {
        /// Human-readable parse error message.
        message: String,
    },

    /// Weight or config validation delegated to [`ModelError`] failed.
    #[error(transparent)]
    Model(#[from] ModelError),

    /// Loader is not yet implemented for the requested format.  Present so
    /// downstream plugins can wire up file properties before Phase 1.5 lands.
    #[error("loader for `{format}` is not implemented in Phase 1 (Phase 1.5 follow-up)")]
    NotImplemented {
        /// Human-readable format name (e.g., "NAM", "Proteus").
        format: &'static str,
    },
}

/// Parse a NAM `.nam` JSON payload into a [`PreparedModel`].
///
/// **Stub** — returns [`LoaderError::NotImplemented`] until Phase 1.5.
pub fn load_nam(_bytes: &[u8]) -> Result<PreparedModel, LoaderError> {
    Err(LoaderError::NotImplemented { format: "NAM" })
}

/// Parse a Proteus / GuitarML `.json` payload into a [`PreparedModel`].
///
/// **Stub** — returns [`LoaderError::NotImplemented`] until Phase 1.5.
pub fn load_proteus(_bytes: &[u8]) -> Result<PreparedModel, LoaderError> {
    Err(LoaderError::NotImplemented { format: "Proteus" })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nam_loader_stub_reports_not_implemented() {
        assert!(matches!(
            load_nam(&[]),
            Err(LoaderError::NotImplemented { format: "NAM" })
        ));
    }

    #[test]
    fn proteus_loader_stub_reports_not_implemented() {
        assert!(matches!(
            load_proteus(&[]),
            Err(LoaderError::NotImplemented { format: "Proteus" })
        ));
    }
}
