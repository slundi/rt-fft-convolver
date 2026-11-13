//! Framework-agnostic parameter model for the `rt-vox` plugin.
//!
//! [`VoxParams`] mirrors the Phase-2 control list from
//! `docs/Roadmap-08-vocoder-harmonizer.md` as a plain `Copy` struct with
//! documented ranges.  Keeping it free of `nih-plug` (which lives behind the
//! `nih` feature) lets the engine and ranges be unit-tested without the plugin
//! framework.
//!
//! # How the controls map onto `rt-vox-core`
//!
//! - [`mode`](VoxParams::mode) selects which engine renders — the vocoder, the
//!   harmoniser, or the formant shifter.  It is **continuous**: all three
//!   engines are kept resident, so switching modes never rebuilds or drops MIDI
//!   state.
//! - [`bands`](VoxParams::bands) is the one **structural** control — changing
//!   the vocoder band count rebuilds the [`Vocoder`](rt_vox_core::Vocoder) off
//!   the audio thread.
//! - Everything else (attack/release, formant shift, sibilance, the per-voice
//!   settings, snap mode / key / scale, mix, output gain, bypass) is applied in
//!   place on the audio thread every block.

pub use rt_vox_core::{ScaleKind, SnapMode};

/// Which of the three engines is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Carrier shaped by a modulator (voice) — needs the sidechain input.
    #[default]
    Vocoder,
    /// One input note → a chord of pitch-shifted voices (MIDI-driven).
    Harmoniser,
    /// Shift formants independently of pitch (and optionally pitch too).
    FormantShifter,
}

/// Vocoder band count (the classic 8 / 16 / 24 / 32 channels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bands {
    /// 8 log-spaced bands — coarse, vintage.
    B8,
    /// 16 bands — the default.
    #[default]
    B16,
    /// 24 bands.
    B24,
    /// 32 bands — the most intelligible.
    B32,
}

impl Bands {
    /// The number of bands.
    #[inline]
    pub fn count(self) -> usize {
        match self {
            Bands::B8 => 8,
            Bands::B16 => 16,
            Bands::B24 => 24,
            Bands::B32 => 32,
        }
    }
}

/// Number of harmony voices (matches [`rt_vox_core::MAX_VOICES`]).
pub const NUM_VOICES: usize = rt_vox_core::MAX_VOICES;

/// Inclusive per-voice interval range in semitones (Roadmap-08: −12 … +12).
pub const VOICE_INTERVAL_RANGE: (f32, f32) = (-12.0, 12.0);
/// Inclusive per-voice output-level range (linear).
pub const VOICE_LEVEL_RANGE: (f32, f32) = (0.0, 1.5);
/// Inclusive per-voice detune range in cents.
pub const DETUNE_CENTS_RANGE: (f32, f32) = (-50.0, 50.0);
/// Inclusive formant-shift range in semitones (± one octave).
pub const FORMANT_SHIFT_RANGE: (f32, f32) = (-12.0, 12.0);
/// Inclusive formant-mode pitch-shift range in semitones.
pub const PITCH_SHIFT_RANGE: (f32, f32) = (-12.0, 12.0);
/// Inclusive envelope-follower attack range in milliseconds.
pub const ATTACK_MS_RANGE: (f32, f32) = (0.1, 100.0);
/// Inclusive envelope-follower release range in milliseconds.
pub const RELEASE_MS_RANGE: (f32, f32) = (1.0, 500.0);
/// Inclusive output-gain range in dB.
pub const OUTPUT_GAIN_DB_RANGE: (f32, f32) = (-24.0, 24.0);

/// One harmony voice's controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceParams {
    /// Interval from the input note, in semitones.
    pub interval: f32,
    /// Output level (linear).
    pub level: f32,
    /// Stereo pan `[-1, 1]`.
    pub pan: f32,
    /// Per-voice formant shift in semitones.
    pub formant_shift: f32,
    /// Fine detune in cents.
    pub detune_cents: f32,
    /// Whether the voice participates (Free / Scale modes; MIDI uses held notes).
    pub enabled: bool,
}

impl Default for VoiceParams {
    fn default() -> Self {
        Self {
            interval: 0.0,
            level: 1.0,
            pan: 0.0,
            formant_shift: 0.0,
            detune_cents: 0.0,
            enabled: false,
        }
    }
}

impl VoiceParams {
    /// Clamp every continuous field into range.
    fn clamped(mut self) -> Self {
        self.interval = self
            .interval
            .clamp(VOICE_INTERVAL_RANGE.0, VOICE_INTERVAL_RANGE.1);
        self.level = self.level.clamp(VOICE_LEVEL_RANGE.0, VOICE_LEVEL_RANGE.1);
        self.pan = self.pan.clamp(-1.0, 1.0);
        self.formant_shift = self
            .formant_shift
            .clamp(FORMANT_SHIFT_RANGE.0, FORMANT_SHIFT_RANGE.1);
        self.detune_cents = self
            .detune_cents
            .clamp(DETUNE_CENTS_RANGE.0, DETUNE_CENTS_RANGE.1);
        self
    }
}

/// The full parameter state of one `rt-vox` instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxParams {
    /// Active engine.
    pub mode: Mode,

    // ── Vocoder ──
    /// Band count (structural).
    pub bands: Bands,
    /// Envelope-follower attack in milliseconds.
    pub attack_ms: f32,
    /// Envelope-follower release in milliseconds.
    pub release_ms: f32,
    /// Vocoder formant shift in semitones.
    pub vocoder_formant_shift: f32,
    /// Route the modulator's highs directly (sibilance handling).
    pub sibilance: bool,
    /// Sibilance blend `[0, 1]`.
    pub sibilance_amount: f32,

    // ── Harmoniser ──
    /// The four harmony voices.
    pub voices: [VoiceParams; NUM_VOICES],
    /// Where voice intervals come from.
    pub snap_mode: SnapMode,
    /// Scale root pitch class `0..=11` (Scale mode).
    pub key: u8,
    /// Scale shape (Scale mode).
    pub scale_kind: ScaleKind,

    // ── Formant shifter ──
    /// Formant shift in semitones.
    pub formant_shift: f32,
    /// Independent pitch shift in semitones.
    pub formant_pitch_shift: f32,

    // ── Global ──
    /// Dry / wet blend `0.0` = dry, `1.0` = wet.
    pub mix: f32,
    /// Output gain in dB.
    pub output_gain_db: f32,
    /// Delay the dry signal by the reported latency so the mix stays aligned.
    pub dry_delay_compensation: bool,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
}

impl Default for VoxParams {
    fn default() -> Self {
        // A useful default harmony: an octave-up and a fifth-up voice, enabled.
        let mut voices = [VoiceParams::default(); NUM_VOICES];
        voices[0] = VoiceParams {
            interval: 12.0,
            enabled: true,
            ..VoiceParams::default()
        };
        voices[1] = VoiceParams {
            interval: 7.0,
            enabled: true,
            pan: 0.3,
            ..VoiceParams::default()
        };

        Self {
            mode: Mode::default(),
            bands: Bands::default(),
            attack_ms: 5.0,
            release_ms: 50.0,
            vocoder_formant_shift: 0.0,
            sibilance: false,
            sibilance_amount: 0.5,
            voices,
            snap_mode: SnapMode::default(),
            key: 0,
            scale_kind: ScaleKind::default(),
            formant_shift: 0.0,
            formant_pitch_shift: 0.0,
            mix: 1.0,
            output_gain_db: 0.0,
            dry_delay_compensation: true,
            bypass: false,
        }
    }
}

impl VoxParams {
    /// Return a copy with every continuous field clamped into range.
    pub fn clamped(mut self) -> Self {
        self.attack_ms = self.attack_ms.clamp(ATTACK_MS_RANGE.0, ATTACK_MS_RANGE.1);
        self.release_ms = self
            .release_ms
            .clamp(RELEASE_MS_RANGE.0, RELEASE_MS_RANGE.1);
        self.vocoder_formant_shift = self
            .vocoder_formant_shift
            .clamp(FORMANT_SHIFT_RANGE.0, FORMANT_SHIFT_RANGE.1);
        self.sibilance_amount = self.sibilance_amount.clamp(0.0, 1.0);
        for voice in &mut self.voices {
            *voice = voice.clamped();
        }
        self.key %= 12;
        self.formant_shift = self
            .formant_shift
            .clamp(FORMANT_SHIFT_RANGE.0, FORMANT_SHIFT_RANGE.1);
        self.formant_pitch_shift = self
            .formant_pitch_shift
            .clamp(PITCH_SHIFT_RANGE.0, PITCH_SHIFT_RANGE.1);
        self.mix = self.mix.clamp(0.0, 1.0);
        self.output_gain_db = self
            .output_gain_db
            .clamp(OUTPUT_GAIN_DB_RANGE.0, OUTPUT_GAIN_DB_RANGE.1);
        self
    }

    /// `true` when `other` differs in a **structural** field — one that forces
    /// the off-thread controller to rebuild the vocoder.  Only the band count
    /// qualifies; every other control is applied continuously.
    #[inline]
    pub fn structural_changed(&self, other: &Self) -> bool {
        self.bands != other.bands
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_roadmap() {
        let params = VoxParams::default();
        assert_eq!(params.mode, Mode::Vocoder);
        assert_eq!(params.bands, Bands::B16);
        assert_eq!(params.mix, 1.0);
        assert!(!params.bypass);
        assert!(params.dry_delay_compensation);
        assert!(params.voices[0].enabled);
        assert!((params.voices[0].interval - 12.0).abs() < 1e-6);
    }

    #[test]
    fn bands_map_to_counts() {
        assert_eq!(Bands::B8.count(), 8);
        assert_eq!(Bands::B16.count(), 16);
        assert_eq!(Bands::B24.count(), 24);
        assert_eq!(Bands::B32.count(), 32);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)] // exercising the clamp field by field
    fn clamp_pulls_fields_into_range() {
        let mut params = VoxParams::default();
        params.attack_ms = -5.0;
        params.release_ms = 5000.0;
        params.mix = 2.0;
        params.output_gain_db = 100.0;
        params.key = 25;
        params.voices[0].interval = 100.0;
        params.voices[0].pan = -3.0;
        params.formant_pitch_shift = 99.0;
        let params = params.clamped();
        assert_eq!(params.attack_ms, ATTACK_MS_RANGE.0);
        assert_eq!(params.release_ms, RELEASE_MS_RANGE.1);
        assert_eq!(params.mix, 1.0);
        assert_eq!(params.output_gain_db, OUTPUT_GAIN_DB_RANGE.1);
        assert_eq!(params.key, 1);
        assert_eq!(params.voices[0].interval, VOICE_INTERVAL_RANGE.1);
        assert_eq!(params.voices[0].pan, -1.0);
        assert_eq!(params.formant_pitch_shift, PITCH_SHIFT_RANGE.1);
    }

    #[test]
    fn clamp_is_idempotent_on_defaults() {
        let params = VoxParams::default();
        assert_eq!(params.clamped(), params);
    }

    #[test]
    fn only_band_count_is_structural() {
        let base = VoxParams::default();

        // Continuous changes: no rebuild.
        let mut cont = base;
        cont.mode = Mode::Harmoniser;
        cont.mix = 0.3;
        cont.attack_ms = 20.0;
        cont.voices[2].enabled = true;
        cont.bypass = true;
        assert!(!base.structural_changed(&cont));

        // Structural change: rebuild.
        let mut bands = base;
        bands.bands = Bands::B32;
        assert!(base.structural_changed(&bands));
    }
}
