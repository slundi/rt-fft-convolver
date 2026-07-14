//! Real-time DSP primitives for an oversampled nonlinear-drive plugin.
//!
//! # Crate scope
//!
//! `rt-drive-core` is the **algorithmic core** for `rt-drive` (see
//! `docs/Roadmap-02-oversampled-drive.md`).  It is framework-agnostic — no
//! plugin format (`nih-plug`, LV2, CLAP, VST3) leaks into this crate — and is
//! reusable by any other nonlinear-effect plugin.
//!
//! The signal chain a host plugin assembles from these primitives is:
//!
//! ```text
//!   in ─▶ [Oversampler ↑] ─▶ [Shaper] ─▶ [Oversampler ↓] ─▶ [DcBlocker] ─▶ [ToneStack] ─▶ out
//! ```
//!
//! - [`resampler`] — Kaiser-windowed **half-band polyphase** up/down-sampling,
//!   chained per octave for 2× / 4× / 8× / 16× oversampling.
//! - [`shaper`] — the [`Shaper`] curve bank (soft / hard / diode / tube / tape).
//! - [`dc_block`] — a 1-pole high-pass that removes the DC offset an
//!   asymmetric shaper injects.
//! - [`tone`] — [`Biquad`] building block plus [`Baxandall`] (2-band) and
//!   [`ThreeBand`] (bass/mid/treble) tone stacks.
//! - [`kernels`] — `pulp`-dispatched SIMD kernels (FIR dot product, soft-clip
//!   block) backing the resampler and shaper hot paths.
//! - [`stage`] — [`DriveStage`], the integrated engine wiring the above into
//!   the full `↑ → drive → shape → ↓ → DC-block → tone` path (Phase 1).
//! - [`pedal`] — [`PedalModel`], the data-driven preset descriptor, plus the
//!   shipped [`pedal::PEDALS`] bank.
//!
//! # Real-time safety
//!
//! Every method on the render path (`process`, `process_sample`,
//! `process_block`, `upsample`, `downsample`) is allocation-free, lock-free
//! and panic-free in release builds.  All heap allocation and coefficient
//! design happen in the `new()` / `set_*` constructors, which are meant to be
//! called off the audio thread.
//!
//! # Denormals
//!
//! Recursive filters here ([`DcBlocker`], [`Biquad`]) can drive their state
//! toward subnormal values on decay.  Wrap the whole render callback in
//! `rt_fft_convolver::DenormalGuard` (FTZ/DAZ) rather than paying a per-sample
//! software flush — the same convention the rest of the workspace uses.

#![warn(missing_docs)]

pub mod dc_block;
pub mod kernels;
pub mod pedal;
pub mod resampler;
pub mod shaper;
pub mod stage;
pub mod tone;

pub use dc_block::DcBlocker;
pub use kernels::Kernels;
pub use pedal::{PEDALS, PedalModel, ToneConfig};
pub use resampler::{Downsampler2x, Oversampler, Upsampler2x};
pub use shaper::Shaper;
pub use stage::DriveStage;
pub use tone::{Baxandall, Biquad, ThreeBand, ToneStack};
