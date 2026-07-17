//! `rt-loudness-meter` — real-time EBU R128 / ITU-R BS.1770-4 loudness meter.
//!
//! # Layout
//!
//! Following the sibling `rt-limiter` / `rt-adaptive` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the [`LoudnessParams`](params::LoudnessParams) model with the
//!   Phase-2 control list ([`Target`](params::Target), the custom target,
//!   `Start/Stop`, `Reset`, `Log`, `Bypass`) plus documented ranges.  A
//!   [`Target`](params::Target) resolves to a (LUFS, dBTP) compliance pair —
//!   the Phase-0 "one engine, presets on top" decision.
//! - [`engine`] — [`MeterProcessor`](engine::MeterProcessor) (audio thread:
//!   feeds the [`LoudnessMeter`](rt_loudness_core::LoudnessMeter), gates
//!   integration on `Start/Stop`, edge-triggers `Reset`, publishes the
//!   lock-free broadcast) and [`MeterLogger`](engine::MeterLogger) (off-RT
//!   worker polling the broadcast into a CSV / JSON file).
//! - `plugin` — the actual `nih-plug` `Plugin` implementation, compiled only
//!   with the **`nih`** feature (`cargo build -p rt-loudness-meter --features
//!   nih`).  Analysis-only: audio passes through untouched.  The editor
//!   (Phase 3) will live behind the **`gui`** feature.
//!
//! # Why no controller / handoff
//!
//! Unlike the plugins that rebuild a filter bank on a parameter change, nothing
//! here reallocates when a control moves: the meter is sized once for the
//! channel count / sample rate in `initialize`, and every control (`Target`,
//! `Start/Stop`, `Reset`, `Log`) is applied in place per block.  So there is no
//! `*Controller` / `ir_handoff` — just a resident [`MeterProcessor`].

pub mod engine;
pub mod params;

pub use engine::{LogFormat, MeterLogger, MeterProcessor};
pub use params::{CUSTOM_DBTP_RANGE, CUSTOM_LUFS_RANGE, LoudnessParams, Target};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_loudness_core as loudness_core;

#[cfg(feature = "nih")]
mod plugin;
