//! Real-time DSP primitives for a multi-band compressor / limiter / expander.
//!
//! # Crate scope
//!
//! `rt-multiband-core` is the **algorithmic core** for `rt-multiband` (see
//! `docs/Roadmap-11-multiband-compressor.md`).  It is framework-agnostic — no
//! plugin format (`nih-plug`, LV2, CLAP, VST3) leaks in — and is the v1,
//! zero-latency, IIR, fixed-4-band core the Phase 0 decisions settled on.
//!
//! The signal path (Phase 0 topology) is assembled by [`MultibandEngine`]:
//!
//! ```text
//!   (l,r) ─▶ ×input ─▶ [Crossover] ─▶ 4 × [BandProcessor] ─▶ Σ ─▶ mix ─▶ ×output
//! ```
//!
//! - [`Crossover`] — Linkwitz-Riley 4th-order (24 dB/oct) four-band splitter
//!   with all-pass phase correction, so bypassed bands recombine
//!   magnitude-flat.
//! - [`BandProcessor`] — one band's stereo dynamics:
//!   [`SidechainFilter`] → [`Detector`] (peak/RMS blend) → [`EnvelopeFollower`]
//!   (analog attack/release) → [`GainComputer`] (soft-knee compressor /
//!   expander / gate) → makeup gain, with [`StereoLink`] linked / independent.
//! - [`Biquad`] — the shared RBJ second-order section the crossover and
//!   sidechain are built from.
//! - [`ParamSmoother`] — the "keep local" (Phase 0) one-pole ramp for
//!   click-free parameter changes.
//!
//! # Real-time safety
//!
//! Every render-path method (`process`, `process_block`, `split`) is
//! allocation-free, lock-free and panic-free.  All heap allocation and
//! coefficient design happen in the `new()` / `set_*` calls, meant to run off
//! the audio thread.  [`MultibandEngine::process_block`] installs a
//! `rt_fft_convolver::DenormalGuard` (FTZ/DAZ) so quiet passages never pay for
//! subnormal arithmetic — the convention the rest of the workspace uses.

#![warn(missing_docs)]

pub mod band;
pub mod biquad;
pub mod crossover;
pub mod detector;
pub mod engine;
pub mod envelope;
pub mod gain_computer;
pub mod sidechain;
pub mod smoother;

#[cfg(test)]
mod test_support;

pub use band::BandProcessor;
pub use biquad::Biquad;
pub use crossover::{Crossover, NUM_BANDS, NUM_CROSSOVERS};
pub use detector::Detector;
pub use engine::MultibandEngine;
pub use envelope::EnvelopeFollower;
pub use gain_computer::{BandMode, GainComputer, StereoLink};
pub use sidechain::SidechainFilter;
pub use smoother::ParamSmoother;
