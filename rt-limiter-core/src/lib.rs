//! Real-time DSP primitives for a true-peak look-ahead brick-wall limiter.
//!
//! # Crate scope
//!
//! `rt-limiter-core` is the **algorithmic core** for `rt-limiter` (see
//! `docs/Roadmap-07-true-peak-limiter.md`).  It is framework-agnostic — no
//! plugin format (`nih-plug`, LV2, CLAP, VST3) leaks in — and reusable by any
//! host that needs transparent true-peak limiting or loudness maximising.
//!
//! The full feed-forward signal chain (Phase 0 topology) is assembled by
//! [`Limiter`]:
//!
//! ```text
//!            ┌───────────────────── delay (look-ahead + detector) ─────────┐
//!   in ──────┤                                                             ▼
//!            └▶ [true-peak ↑] ─▶ [window-max] ─▶ [gain] ── g ─▶ ×g ─▶ [ceiling] ─▶ (dither) ─▶ out
//! ```
//!
//! - [`TruePeakDetector`] — oversampled inter-sample-peak envelope, reusing
//!   `rt-drive-core`'s Kaiser half-band polyphase resampler (Phase 0: depend on
//!   `rt-drive-core` directly, no shared crate).
//! - [`WindowMax`] — sliding-window maximum of the peak envelope; the
//!   look-ahead anticipation that makes the brick wall provable.
//! - [`GainComputer`] — feed-forward gain-reduction smoother with a selectable
//!   [`ReleaseCurve`] (`Linear` / `Exponential` / `Auto`).
//! - [`DelayLine`] — the dry-signal delay that aligns the reduction with the
//!   peak that caused it.
//! - [`CeilingClipper`] — the output brick-wall safety net.
//! - [`Dither`] / [`BitDepth`] — optional (`dither` feature) output bit-depth
//!   reducer with TPDF dither and noise shaping.
//! - [`Limiter`] — the integrated stereo engine wiring all of the above, with
//!   [`ChannelLink`] stereo-linked / mid-side modes.
//!
//! # Real-time safety
//!
//! Every render-path method (`process`, `process_block`, `push`,
//! `detect_block`) is allocation-free, lock-free and panic-free in release
//! builds.  All heap allocation and coefficient design happen in the `new()`
//! / `set_*` constructors, meant to run off the audio thread; buffers are
//! sized for the 20 ms maximum look-ahead so a live look-ahead change never
//! reallocates.
//!
//! # Denormals
//!
//! The gain-reduction envelope and the delay tails ring down toward subnormal
//! values on quiet passages.  Wrap the host render callback in
//! `rt_fft_convolver::DenormalGuard` (FTZ/DAZ) rather than paying a per-sample
//! software flush — the convention the rest of the workspace uses.

#![warn(missing_docs)]

pub mod clip;
pub mod delay;
pub mod detector;
pub mod gain;
pub mod limiter;
pub mod window;

#[cfg(feature = "dither")]
pub mod dither;

pub use clip::CeilingClipper;
pub use delay::DelayLine;
pub use detector::TruePeakDetector;
pub use gain::{GainComputer, ReleaseCurve};
pub use limiter::{ChannelLink, Limiter};
pub use window::WindowMax;

#[cfg(feature = "dither")]
pub use dither::{BitDepth, Dither};
