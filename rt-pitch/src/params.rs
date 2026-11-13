//! Framework-agnostic parameter model for the `rt-pitch` plugin.
//!
//! [`PitchParams`] mirrors the Phase-2 control list from
//! `docs/Roadmap-03-pitch-shift.md` as a plain `Copy` struct with documented
//! ranges.  Keeping it free of `nih-plug` (which lives behind the `nih`
//! feature) lets the engine and ranges be unit-tested without the plugin
//! framework.
//!
//! # How the controls map onto `rt-pitch-core`
//!
//! - The eight [`intervals`](PitchParams::intervals) feed
//!   [`rt_pitch_core::PresetBank`] slots (semitones + enabled).
//! - `current_preset` / `voice2_preset` drive
//!   [`PresetBank::select`](rt_pitch_core::PresetBank::select); `dual_voice`
//!   routes `voice2_preset` to the second channel for harmoniser use.
//! - `algorithm` selects which [`rt_pitch_core::Shifter`] is built; `quality`
//!   sets its FFT / grain size.  Both are **structural** — a change rebuilds
//!   the voices off the audio thread.
//! - `formant_preservation` / `formant_method` are structural too (they build
//!   a [`rt_pitch_core::FormantEstimator`]); they apply only to the spectral
//!   ("Smooth") algorithm.
//! - `crossfade_ms`, `mix`, `dry_delay_compensation` and `bypass` are
//!   continuous — applied in place on the audio thread every block.

pub use rt_pitch_core::{FormantMethod, NUM_PRESETS, PresetSlot};

/// Inclusive interval range in semitones (± two octaves), matching the
/// shifters' `[MIN_RATIO, MAX_RATIO]` band.
pub const SEMITONE_RANGE: (f32, f32) = (-24.0, 24.0);

/// Inclusive `Crossfade Time` range in milliseconds.
pub const CROSSFADE_MS_RANGE: (f32, f32) = (5.0, 200.0);

/// Lowest fundamental the PSOLA ("Mono Voice") pitch tracker locks onto
/// (≈ A1 — covers low guitar and bass-baritone voice).
pub const MIN_PITCH_HZ: f32 = 55.0;

/// Highest fundamental the PSOLA pitch tracker tracks.
pub const MAX_PITCH_HZ: f32 = 1000.0;

/// Pitch-shift algorithm — selects which [`rt_pitch_core::Shifter`] is built.
///
/// The framework-agnostic twin of the plugin's `#[derive(Enum)]` control; the
/// mapping keeps `nih-plug` out of the engine and its tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// Phase vocoder — polyphonic / chords, arbitrary ratios.  The default.
    #[default]
    Smooth,
    /// WSOLA — time-domain, low latency, punchy on monophonic leads.
    TightMono,
    /// PSOLA — pitch-synchronous, monophonic voice; warns on polyphony.
    MonoVoice,
}

/// Latency / quality trade-off — sets the shifter's FFT (or grain) size.
///
/// At the fixed 4× overlap the algorithmic latency ≈ one frame, so a larger
/// frame trades latency for smoother spectral resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// 512-sample frame (~10.7 ms @ 48 kHz) — live monitoring.
    Live,
    /// 1024-sample frame (~21.3 ms) — the default balance.
    #[default]
    Standard,
    /// 2048-sample frame (~42.7 ms) — studio / offline smoothness.
    Studio,
}

impl Quality {
    /// FFT / grain size this quality maps to.  Divisible by the phase-vocoder
    /// overlap factor and even, so every shifter accepts it.
    #[inline]
    pub fn frame(self) -> usize {
        match self {
            Quality::Live => 512,
            Quality::Standard => 1024,
            Quality::Studio => 2048,
        }
    }
}

/// The full parameter state of one `rt-pitch` instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchParams {
    /// The eight harmony intervals (semitones + enabled) feeding the bank.
    pub intervals: [PresetSlot; NUM_PRESETS],
    /// Selected preset for voice 1 (all channels, or the left channel in
    /// dual-voice mode).  `0..NUM_PRESETS`.
    pub current_preset: usize,
    /// Selected preset for voice 2 (the right channel when `dual_voice`).
    pub voice2_preset: usize,
    /// Route `voice2_preset` to the second channel (independent harmony).
    pub dual_voice: bool,
    /// Preset crossfade time in milliseconds.
    pub crossfade_ms: f32,
    /// Which shifter algorithm is active.
    pub algorithm: Algorithm,
    /// FFT / grain size (latency trade-off).
    pub quality: Quality,
    /// Dry / wet blend `0.0` = dry, `1.0` = wet.
    pub mix: f32,
    /// Formant preservation (spectral / "Smooth" algorithm only).
    pub formant_preservation: bool,
    /// Spectral-envelope estimator used when `formant_preservation` is on.
    pub formant_method: FormantMethod,
    /// Delay the dry signal by the reported latency so mix stays phase-aligned.
    pub dry_delay_compensation: bool,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
}

impl Default for PitchParams {
    fn default() -> Self {
        // A musically useful default bank: unison, octaves, and common
        // intervals, ending in a gentle detune for a chorus voice (mirrors the
        // Roadmap-03 Phase-5 shipped preset list).
        let mut intervals = [PresetSlot::default(); NUM_PRESETS];
        let defaults = [
            0.0,   // unison
            12.0,  // octave up
            -12.0, // octave down
            7.0,   // perfect fifth
            5.0,   // perfect fourth
            3.0,   // minor third
            4.0,   // major third
            0.2,   // 20-cent detune (chorus)
        ];
        for (slot, semis) in intervals.iter_mut().zip(defaults) {
            slot.semitones = semis;
            slot.enabled = true;
        }

        Self {
            intervals,
            current_preset: 0,
            voice2_preset: 3, // fifth: a pleasant default harmony
            dual_voice: false,
            crossfade_ms: 20.0,
            algorithm: Algorithm::default(),
            quality: Quality::default(),
            mix: 1.0,
            formant_preservation: false,
            formant_method: FormantMethod::Cepstral,
            dry_delay_compensation: true,
            bypass: false,
        }
    }
}

impl PitchParams {
    /// Return a copy with every continuous field clamped into range.
    pub fn clamped(mut self) -> Self {
        for slot in &mut self.intervals {
            slot.semitones = slot.semitones.clamp(SEMITONE_RANGE.0, SEMITONE_RANGE.1);
        }
        self.current_preset = self.current_preset.min(NUM_PRESETS - 1);
        self.voice2_preset = self.voice2_preset.min(NUM_PRESETS - 1);
        self.crossfade_ms = self
            .crossfade_ms
            .clamp(CROSSFADE_MS_RANGE.0, CROSSFADE_MS_RANGE.1);
        self.mix = self.mix.clamp(0.0, 1.0);
        self
    }

    /// `true` when `other` differs in a **structural** field — one that forces
    /// the off-thread controller to rebuild the shifter voices (algorithm,
    /// quality, or the formant configuration).  The continuous fields
    /// (intervals, selection, crossfade, mix, bypass) never trigger a rebuild.
    #[inline]
    pub fn structural_changed(&self, other: &Self) -> bool {
        self.algorithm != other.algorithm
            || self.quality != other.quality
            || self.formant_preservation != other.formant_preservation
            || self.formant_method != other.formant_method
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_roadmap() {
        let params = PitchParams::default();
        assert_eq!(params.algorithm, Algorithm::Smooth);
        assert_eq!(params.quality, Quality::Standard);
        assert_eq!(params.mix, 1.0);
        assert!(!params.bypass);
        assert!(params.dry_delay_compensation);
        assert!((params.intervals[1].semitones - 12.0).abs() < 1e-6);
    }

    #[test]
    fn quality_maps_to_frame_sizes() {
        assert_eq!(Quality::Live.frame(), 512);
        assert_eq!(Quality::Standard.frame(), 1024);
        assert_eq!(Quality::Studio.frame(), 2048);
    }

    #[test]
    fn clamp_pulls_fields_into_range() {
        let mut params = PitchParams::default();
        params.intervals[0].semitones = 100.0;
        params.intervals[1].semitones = -100.0;
        params.current_preset = 99;
        params.voice2_preset = 99;
        params.crossfade_ms = 5000.0;
        params.mix = 2.0;
        let params = params.clamped();
        assert_eq!(params.intervals[0].semitones, SEMITONE_RANGE.1);
        assert_eq!(params.intervals[1].semitones, SEMITONE_RANGE.0);
        assert_eq!(params.current_preset, NUM_PRESETS - 1);
        assert_eq!(params.voice2_preset, NUM_PRESETS - 1);
        assert_eq!(params.crossfade_ms, CROSSFADE_MS_RANGE.1);
        assert_eq!(params.mix, 1.0);
    }

    #[test]
    fn clamp_is_idempotent_on_defaults() {
        let params = PitchParams::default();
        assert_eq!(params.clamped(), params);
    }

    #[test]
    fn structural_change_detects_only_rebuild_fields() {
        let base = PitchParams::default();

        // Continuous changes: no rebuild.
        let mut cont = base;
        cont.current_preset = 2;
        cont.mix = 0.3;
        cont.crossfade_ms = 100.0;
        cont.bypass = true;
        cont.intervals[0].semitones = 7.0;
        assert!(!base.structural_changed(&cont));

        // Structural changes: rebuild.
        let mut algo = base;
        algo.algorithm = Algorithm::TightMono;
        assert!(base.structural_changed(&algo));

        let mut qual = base;
        qual.quality = Quality::Studio;
        assert!(base.structural_changed(&qual));

        let mut formant = base;
        formant.formant_preservation = true;
        assert!(base.structural_changed(&formant));
    }
}
