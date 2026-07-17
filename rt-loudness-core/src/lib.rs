//! Real-time loudness-measurement DSP primitives for the `rt-loudness-meter`
//! plugin.
//!
//! # Crate scope
//!
//! `rt-loudness-core` is the **algorithmic core** for `rt-loudness-meter` (see
//! `docs/Roadmap-10-loudness-metering.md`).  It is framework-agnostic — no
//! plugin format (`nih-plug`, LV2, CLAP, VST3) leaks in — and implements the
//! ITU-R BS.1770-4 / EBU R128 measurement chain: analysis only, it never
//! modifies the signal.
//!
//! # Phase 0 findings realised here
//!
//! - **One measurement engine, presets on top.** BS.1770-4 is the *only*
//!   algorithm; EBU R128 and the streaming/broadcast targets differ only in
//!   the numbers a UI compares against, so no standard gets its own code path.
//! - **Reuse over re-implementation.** True peak reuses the tested
//!   [`rt_limiter_core::TruePeakDetector`] (4× Kaiser half-band polyphase),
//!   and `DenormalGuard` / the dB helpers come from the root
//!   `rt-fft-convolver` crate.
//! - **Bounded memory.** The gated integrated and LRA measurements use fixed
//!   [`LoudnessHistogram`]s (0.1 LU bins) and the time windows use a 100 ms
//!   sub-block ring, so a multi-hour session costs the same RAM as a few
//!   seconds — and every hot path is allocation-free.
//!
//! # The pieces
//!
//! - [`KWeightingFilter`] — the two-stage BS.1770-4 K-weighting biquad, per
//!   channel, sample-rate-independent.
//! - [`LoudnessIntegrator`] — momentary (400 ms), short-term (3 s), gated
//!   integrated (session) loudness and the LRA feed.
//! - [`LoudnessHistogram`] — the gated histogram behind integrated loudness
//!   and loudness range.
//! - [`TruePeakMeter`] — oversampled true peak (dBTP) and sample peak (dBFS).
//! - [`LoudnessMeter`] — the whole mono/stereo chain, producing [`MeterValues`].
//! - [`MeterBroadcast`] — lock-free publication of the latest readings to the
//!   editor / off-RT logger.

pub mod broadcast;
pub mod integrator;
pub mod kweight;
pub mod lra;
pub mod meter;
pub mod truepeak;

#[cfg(test)]
mod test_support;

pub use broadcast::{MeterBroadcast, MeterValues};
pub use integrator::LoudnessIntegrator;
pub use kweight::KWeightingFilter;
pub use lra::LoudnessHistogram;
pub use meter::LoudnessMeter;
pub use truepeak::TruePeakMeter;

// Re-export the reused true-peak primitive so downstream `rt-loudness-meter`
// code has one import path.
pub use rt_limiter_core::TruePeakDetector;
