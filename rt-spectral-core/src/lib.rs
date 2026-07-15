//! Real-time spectral (STFT) processing primitives for the `rt-spectral`
//! plugin suite.
//!
//! # Crate scope
//!
//! `rt-spectral-core` is the **algorithmic core** for `rt-spectral` (see
//! `docs/Roadmap-04-spectral-processing.md`).  It is framework-agnostic — no
//! plugin format (`nih-plug`, LV2, CLAP, VST3) leaks into this crate — and
//! builds on the shared real-time primitives re-exported from the root
//! `rt-fft-convolver` crate (`Ring`, `FftHandler`, `DenormalGuard`, the
//! dB/linear gain helpers).
//!
//! # Signal path (Phase 0 decisions)
//!
//! ```text
//!   in ─▶ click repair ─▶ ┌─ STFT analysis ──────────────┐ ─▶ hum notch ─▶ out
//!        (time domain)    │  window · FFT                 │   (time domain)
//!                         │   └▶ denoise ▶ gate ▶ compress │
//!                         │      (per-bin, transient-gated)│
//!                         │  IFFT · window · overlap-add   │
//!                         └────────────────────────────────┘
//! ```
//!
//! - **Denoise** — [`Denoiser`]: per-bin Wiener gain driven by a
//!   decision-directed (Ephraim–Malah) a-priori SNR estimate, seeded from a
//!   learned [`NoiseProfile`].  The decision-directed smoothing is what keeps
//!   Wiener free of *musical noise*.
//! - **Spectral gate** — [`SpectralGate`]: per-bin downward expander with
//!   attack/release on the gain envelope.
//! - **Spectral compressor** — [`SpectralCompressor`]: per-bin static
//!   soft-knee compression curve.
//! - **Transient soft-bypass** — [`TransientDetector`]: spectral-flux onset
//!   detection that crossfades denoise/compress depth toward unity on attacks.
//! - **Hum removal** — [`HumNotchBank`]: cascaded RBJ biquad notches at a
//!   fundamental + harmonics (time domain, zero added latency).
//! - **Click repair** — [`ClickRepair`]: median-filter despiker with
//!   short-window interpolation (time domain, small fixed latency).
//!
//! [`Ring`]: rt_fft_convolver::Ring
//! [`FftHandler`]: rt_fft_convolver::FftHandler
//! [`DenormalGuard`]: rt_fft_convolver::DenormalGuard

pub mod biquad;
pub mod click;
pub mod compress;
pub mod denoise;
pub mod engine;
pub mod gate;
pub mod hum;
pub mod profile;
pub mod stft;
pub mod transient;
pub mod window;

pub use biquad::Biquad;
pub use click::{ClickParams, ClickRepair};
pub use compress::{CompressParams, SpectralCompressor};
pub use denoise::{DenoiseParams, Denoiser};
pub use engine::{SpectralEngine, SpectralParams};
pub use gate::{GateParams, SpectralGate};
pub use hum::{HumNotchBank, HumParams};
pub use profile::{NoiseProfile, ProfileReceiver, ProfileSender, learn_profile, profile_channel};
pub use stft::{Analyzer, Synthesizer};
pub use transient::{TransientDetector, TransientParams};
pub use window::{Window, WindowKind};

/// Complex sample type used throughout the spectrum buffers (re-exported from
/// `realfft` so downstream crates share one `num-complex` version).
pub use realfft::num_complex::Complex;
