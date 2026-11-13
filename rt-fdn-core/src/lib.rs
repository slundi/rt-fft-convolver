//! Real-time algorithmic reverb engine for the `rt-fdn-reverb` plugin
//! (see `docs/Roadmap-05-fdn-reverb.md`).
//!
//! # Crate scope
//!
//! `rt-fdn-core` is the **algorithmic core**: a Feedback Delay Network reverb
//! with everything the render thread needs and nothing a plugin format needs.
//! It is framework-agnostic — no `nih-plug` / LV2 / CLAP / VST3 leaks in — and
//! reuses the root `rt-fft-convolver` crate only for the shared anti-denormal
//! helpers (`flush_to_zero`, re-exported [`DenormalGuard`]).
//!
//! # Architecture (Phase 0 decisions)
//!
//! - **8×8 FDN** ([`Fdn`]) with a lossless **fast Walsh–Hadamard** mixing
//!   matrix ([`hadamard_in_place`]) — dense diffusion, adds/subtracts only.
//! - **Coprime delay lengths** ([`BASE_DELAYS`]) for maximal modal density.
//! - **Per-delay LFO modulation** ([`SineLfo`], [`ModulatedDelay`]) with
//!   decorrelated phases for lush, chorus-y tail motion.
//! - **Per-branch damping + low-cut** ([`OnePoleLowpass`], [`OnePoleHighpass`])
//!   for frequency-dependent decay.
//! - **Schroeder input diffusion** ([`AllPassChain`]) so transients arrive
//!   dense.
//! - **Per-branch feedback gains** derived from the requested T60, so every
//!   branch decays at the same rate regardless of its length.
//! - **Parameter smoothing** ([`Smoother`], the Phase 1 prerequisite) on the
//!   size and modulation-depth glides to kill zipper noise.
//!
//! # Real-time safety
//!
//! Everything under [`Fdn::process_sample_stereo`] is allocation-free,
//! lock-free and panic-free in release builds, and flushes feedback writes to
//! zero so multi-second tails never fall into the denormal slow path.  Build
//! engines and presets off the render thread.
//!
//! # Example
//!
//! ```
//! use rt_fdn_core::{Fdn, PRESETS};
//!
//! let mut reverb = Fdn::<8>::new(48_000.0, PRESETS[1].params); // "Hall"
//! let (wet_l, wet_r) = reverb.process_sample_stereo(1.0, 1.0);
//! assert!(wet_l.is_finite() && wet_r.is_finite());
//! ```

pub mod allpass;
pub mod delay;
pub mod fdn;
pub mod filter;
pub mod lfo;
pub mod matrix;
pub mod preset;
pub mod smoother;

pub use allpass::{AllPass, AllPassChain};
pub use delay::ModulatedDelay;
pub use fdn::{Fdn, FdnParams, REFERENCE_SR};
pub use filter::{OnePoleHighpass, OnePoleLowpass};
pub use lfo::SineLfo;
pub use matrix::hadamard_in_place;
pub use preset::{BASE_DELAYS, PRESETS, Preset};
pub use smoother::Smoother;

/// Re-export the shared anti-denormal guard so the plugin layer can wrap its
/// render callback without depending on the root crate directly.
pub use rt_fft_convolver::DenormalGuard;
