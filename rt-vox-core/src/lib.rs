//! Real-time voice-processing DSP primitives for the `rt-vox` plugin.
//!
//! # Crate scope
//!
//! `rt-vox-core` is the **algorithmic core** for `rt-vox` (see
//! `docs/Roadmap-08-vocoder-harmonizer.md`).  It is framework-agnostic — no
//! plugin format (`nih-plug`, LV2, CLAP, VST3) leaks into this crate — and
//! builds on the shared real-time primitives from the root `rt-fft-convolver`
//! crate (`Ring`, `FftHandler`, `DenormalGuard`) plus the tested pitch-shift /
//! formant primitives of [`rt_pitch_core`].
//!
//! # Phase 0 reuse decisions
//!
//! Every engine here favours *reuse* over a second parallel implementation:
//!
//! - Pitch detection reuses [`rt_pitch_core::PitchTracker`] (autocorrelation).
//! - Formant analysis reuses [`rt_pitch_core::FormantEstimator`] (LPC default).
//! - The harmoniser's per-voice pitch shift reuses the off-RT-swappable
//!   [`rt_pitch_core::Shifter`] trait ([`PhaseVocoder`](rt_pitch_core::PhaseVocoder)).
//!
//! # Three engines, one plugin
//!
//! The `rt-vox` plugin exposes three modes over these engines:
//!
//! - [`Vocoder`] — FFT magnitude-envelope transfer: a modulator (voice) shapes
//!   a carrier through a bank of log-spaced bands.
//! - [`Harmoniser`] — up to [`MAX_VOICES`] pitch-shifted voices whose intervals
//!   come from fixed offsets, a diatonic scale, or held MIDI notes.
//! - [`FormantShifter`] — shifts the spectral envelope independently of pitch
//!   (and, optionally, pitch independently of formants).
//!
//! # Supporting modules
//!
//! - [`pitch`] — streaming framer around the autocorrelation tracker.
//! - [`envelope`] — per-band attack/release envelope follower.
//! - [`bands`] — log-spaced FFT-bin grouping for the vocoder.
//! - [`voicing`] — `SnapMode`, musical `Scale`, and MIDI-note helpers.
//!
//! [`Ring`]: rt_fft_convolver::Ring

pub mod bands;
pub mod envelope;
pub mod formant_shift;
pub mod harmoniser;
pub mod pitch;
pub mod spectral_probe;
pub mod vocoder;
pub mod voice;
pub mod voicing;

#[cfg(test)]
mod test_support;

pub use bands::LogBands;
pub use envelope::EnvelopeFollower;
pub use formant_shift::FormantShifter;
pub use harmoniser::{Harmoniser, MAX_VOICES};
pub use pitch::VoicePitchTracker;
pub use spectral_probe::SpectralProbe;
pub use vocoder::Vocoder;
pub use voice::Voice;
pub use voicing::{Scale, ScaleKind, SnapMode, midi_to_semitone_offset};

// Re-export the reused primitives so downstream `rt-vox` code has one import
// path and does not need to name `rt_pitch_core` directly.
pub use rt_pitch_core::{FormantEstimator, FormantMethod, PitchEstimate, semitones_to_ratio};
