/// Errors that can occur while loading or resampling an impulse response.
#[derive(Debug)]
pub enum LoadError {
    /// Underlying I/O error (file not found, permissions, …).
    Io(std::io::Error),
    /// WAV parsing or encoding error from the `hound` crate.
    Hound(hound::Error),
    /// Failed to construct the resampler (bad rate pair, zero channels, …).
    Resample(rubato::ResamplerConstructionError),
    /// Runtime resampling error (buffer-size mismatch, …).
    ResampleProcess(rubato::ResampleError),
    /// The WAV file uses a bit depth or encoding that this loader does not
    /// support (e.g., 64-bit float PCM).
    UnsupportedFormat(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "I/O error: {e}"),
            LoadError::Hound(e) => write!(f, "WAV error: {e}"),
            LoadError::Resample(e) => write!(f, "resampler construction error: {e}"),
            LoadError::ResampleProcess(e) => write!(f, "resampling error: {e}"),
            LoadError::UnsupportedFormat(s) => write!(f, "unsupported WAV format: {s}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Io(e) => Some(e),
            LoadError::Hound(e) => Some(e),
            // rubato's error types do not expose a cause chain we can forward
            LoadError::Resample(_) | LoadError::ResampleProcess(_) => None,
            LoadError::UnsupportedFormat(_) => None,
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl From<hound::Error> for LoadError {
    fn from(e: hound::Error) -> Self {
        LoadError::Hound(e)
    }
}
