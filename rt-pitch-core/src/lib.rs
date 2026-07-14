//! Real-time pitch-shifting DSP primitives for the `rt-pitch` plugin.
//!
//! # Crate scope
//!
//! `rt-pitch-core` is the **algorithmic core** for `rt-pitch` (see
//! `docs/Roadmap-03-pitch-shift.md`).  It is framework-agnostic — no plugin
//! format (`nih-plug`, LV2, CLAP, VST3) leaks into this crate — and builds on
//! the shared real-time primitives re-exported from the root `rt-fft-convolver`
//! crate (`Ring`, `FftHandler`, `DenormalGuard`).
//!
//! # Selectable shifters
//!
//! All three algorithms implement the [`Shifter`] trait so the host can swap
//! them off the audio thread (Roadmap-03 decision):
//!
//! - [`PhaseVocoder`] — "Smooth": STFT bin-remapping with Puckette
//!   peak-locking, transient detection, and optional formant preservation.
//!   Handles chords; the default.
//! - [`Wsola`] — "Tight/Mono": time-domain resample ∘ waveform-similarity
//!   overlap-add.  Low latency, punchy transients, best on monophonic leads.
//! - [`Psola`] — "Mono Voice": pitch-synchronous overlap-add driven by
//!   [`PitchTracker`]; bypasses on polyphonic input.
//!
//! # Supporting modules
//!
//! - [`window`] — Hann / Blackman-Harris windows with COLA normalisation.
//! - [`stft`] — streaming STFT analysis / overlap-add framing on [`Ring`].
//! - [`pitch`] — autocorrelation pitch tracking + polyphony guard.
//! - [`formant`] — cepstral / LPC / True-Envelope spectral-envelope estimation
//!   for formant-preserving shifts.
//! - [`preset`] — eight-slot interval bank with equal-power crossfade.
//! - [`shifter`] — the [`Shifter`] trait, ratio helpers, and the lock-free
//!   [`SharedRatio`] parameter cell.
//!
//! [`Ring`]: rt_fft_convolver::Ring

pub mod formant;
pub mod phase_vocoder;
pub mod pitch;
pub mod preset;
pub mod psola;
pub mod shifter;
pub mod stft;
pub mod window;
pub mod wsola;

#[cfg(test)]
mod test_support;

pub use formant::{FormantEstimator, FormantMethod};
pub use phase_vocoder::PhaseVocoder;
pub use pitch::{PitchEstimate, PitchTracker};
pub use preset::{NUM_PRESETS, PresetBank, PresetSlot};
pub use psola::Psola;
pub use shifter::{
    MAX_RATIO, MIN_RATIO, OnePoleSmoother, SharedRatio, Shifter, clamp_ratio, ratio_to_semitones,
    semitones_to_ratio,
};
pub use window::{Window, WindowKind};
pub use wsola::Wsola;
