//! `rt-fdn-reverb` — algorithmic Feedback Delay Network reverb plugin.
//!
//! Eight modulated true-stereo spaces with click-free preset crossfade,
//! pre-delay, wet-bus EQ, stereo width, freeze and MIDI recall, wrapping the
//! framework-agnostic [`rt_fdn_core`] engine (see
//! `docs/Roadmap-05-fdn-reverb.md`, Phase 2).
//!
//! # Layout
//!
//! Following the sibling `rt-ir-reverb` / `rt-spectral` convention, the DSP is
//! **framework-agnostic** and fully unit-testable; the `nih-plug` wrapper is an
//! optional thin layer:
//!
//! - [`params`] — the [`ReverbParams`](params::ReverbParams) model, the
//!   automatable control ranges, and the percent → engine-unit mappings,
//!   unit-testable without the framework.
//! - [`dsp`] — the wet-bus primitives ([`PreDelay`](dsp::PreDelay), stereo
//!   width, `−∞`-floored fader gain).
//! - [`engine`] — [`ReverbEngine`](engine::ReverbEngine): the pre-built FDN
//!   voice bank, the equal-power preset crossfade, and the wet bus, all
//!   allocation-free on the audio thread.
//! - `plugin` — the actual `nih-plug` `Plugin`, compiled only with the **`nih`**
//!   feature (`cargo build -p rt-fdn-reverb --features nih`).  Kept off the
//!   default build so `cargo test` / `clippy` stay fast and offline.  The
//!   `nih-plug-egui` editor (preset rack, IR-decay visualiser, freeze/MIDI
//!   LEDs) lands in Phase 3.

pub mod dsp;
pub mod engine;
pub mod params;

pub use dsp::{PreDelay, fader_gain, width_sample};
pub use engine::ReverbEngine;
pub use params::{
    CROSSFADE_MS_RANGE, DECAY_S_RANGE, FADER_DB_RANGE, HIGH_CUT_RANGE, LOW_CUT_RANGE,
    MOD_RATE_HZ_RANGE, NUM_PRESETS, NUM_SPACES, PREDELAY_MS_RANGE, ReverbParams, WIDTH_RANGE,
    damping_cutoff_hz, diffusion_gain, mod_depth_samples,
};

// Re-export the DSP core so downstream callers only depend on one crate.
pub use rt_fdn_core as fdn_core;

#[cfg(feature = "nih")]
mod plugin;
