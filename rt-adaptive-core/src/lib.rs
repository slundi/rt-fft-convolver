//! Real-time adaptive-filter DSP primitives for the `rt-adaptive` plugin.
//!
//! # Crate scope
//!
//! `rt-adaptive-core` is the **algorithmic core** for `rt-adaptive` (see
//! `docs/Roadmap-09-adaptive-filters.md`).  It is framework-agnostic — no
//! plugin format (`nih-plug`, LV2, CLAP, VST3) leaks into this crate — and
//! builds on the shared real-time primitives from the root `rt-fft-convolver`
//! crate (`Ring`, `FftHandler`, `DenormalGuard`) plus tested notch / pitch
//! primitives from the sibling `*-core` crates.
//!
//! # Phase 0 reuse decisions
//!
//! Every engine here favours *reuse* over a second parallel implementation:
//!
//! - The feedback notch bank and the hum comb reuse
//!   [`rt_spectral_core::Biquad`] / [`rt_spectral_core::HumNotchBank`] (the RBJ
//!   notch and its harmonic comb).
//! - The adaptive hum tracker reuses [`rt_pitch_core::PitchTracker`]
//!   (autocorrelation fundamental estimation) to follow mains drift.
//! - The feedback detector reuses [`rt_fft_convolver::FftHandler`] for its
//!   rolling analysis FFT.
//!
//! # Three modes, one plugin
//!
//! The `rt-adaptive` plugin exposes three modes over these engines:
//!
//! - **Feedback Suppressor** — the [`FeedbackDetector`] finds howling partials
//!   and the [`NotchBank`] notches them out (no per-sample adaptive filter, no
//!   added latency).
//! - **Hum Remover** — the [`HumTracker`] follows the mains fundamental and
//!   combs it (and its harmonics) away.
//! - **Echo Canceller** — the [`AdaptiveFilter`] (NLMS) models the echo path
//!   from a far-end reference and subtracts it from the near-end signal.
//!
//! The frequency-domain block-LMS long-filter AEC path is deferred to a later
//! version (Phase 0 decision); this crate ships the time-domain NLMS core.
//!
//! [`Ring`]: rt_fft_convolver::Ring
//! [`FftHandler`]: rt_fft_convolver::FftHandler
//! [`DenormalGuard`]: rt_fft_convolver::DenormalGuard

pub mod adaptive;
pub mod detector;
pub mod hum;
pub mod notch;

#[cfg(test)]
mod test_support;

pub use adaptive::AdaptiveFilter;
pub use detector::FeedbackDetector;
pub use hum::HumTracker;
pub use notch::{MAX_NOTCHES, NotchBank, NotchInfo};

// Re-export the reused primitives so downstream `rt-adaptive` code has one
// import path and does not need to name every sibling core crate directly.
pub use rt_spectral_core::{Biquad, HumParams};
