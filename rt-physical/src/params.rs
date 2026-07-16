//! Plain, framework-agnostic parameter model for the physical-modelling
//! instrument (Phase 2 control list in `docs/Roadmap-06-physical-modeling.md`).
//!
//! The `nih-plug` wrapper (behind the `nih` feature) maps its automatable
//! parameters onto these fields; keeping the model framework-free lets us
//! unit-test ranges, clamping and the control → engine-unit mappings without
//! the plugin runtime — the same split the sibling `rt-fdn-reverb` uses.
//!
//! # v1 scope
//!
//! Phase 0 shipped two model families: [`Model::Plucked`] (extended
//! Karplus-Strong) and [`Model::Modal`] (biquad resonator bank).  The bowed /
//! struck / tube families in the roadmap's model-selector list are deferred, so
//! the [`Model`] dropdown exposes exactly the two implemented families.

use rt_physical_core::{MAX_MODES, MAX_VOICES, ModePreset, Model, PhysicalParams};

/// Detune range in cents (± one semitone).
pub const DETUNE_CENTS_RANGE: (f32, f32) = (-100.0, 100.0);
/// Sustain / decay (T60) range in seconds.
pub const DECAY_S_RANGE: (f32, f32) = (0.05, 30.0);
/// Modal decay-slope range (higher → bright modes die faster).
pub const DECAY_SLOPE_RANGE: (f32, f32) = (0.1, 3.0);
/// Output-gain range in dB.
pub const GAIN_DB_RANGE: (f32, f32) = (-60.0, 12.0);
/// Mode-count range for the modal bank.
pub const MODES_RANGE: (usize, usize) = (1, MAX_MODES);
/// Polyphony (voice-limit) range.
pub const POLYPHONY_RANGE: (usize, usize) = (1, MAX_VOICES);
/// Below this dB the output gain collapses to hard zero (a true `−∞` mute).
const SILENCE_DB: f32 = -59.5;

// ─── Model selector ─────────────────────────────────────────────────────────

/// The implemented model families, in dropdown order.
pub const MODELS: [Model; 2] = [Model::Plucked, Model::Modal];

/// Index of `model` in the [`MODELS`] dropdown.
pub fn model_to_index(model: Model) -> usize {
    MODELS.iter().position(|&m| m == model).unwrap_or(0)
}

/// Model family at dropdown `index` (clamped).
pub fn model_from_index(index: usize) -> Model {
    MODELS[index.min(MODELS.len() - 1)]
}

/// Human-readable label for a model family.
pub fn model_label(model: Model) -> &'static str {
    match model {
        Model::Plucked => "Plucked",
        Model::Modal => "Modal",
    }
}

// ─── Modal material preset ────────────────────────────────────────────────────

/// The modal timbre presets, in dropdown order.
pub const PRESETS: [ModePreset; 3] = [ModePreset::Metal, ModePreset::Wood, ModePreset::Glass];

/// Index of `preset` in the [`PRESETS`] dropdown.
pub fn preset_to_index(preset: ModePreset) -> usize {
    PRESETS.iter().position(|&p| p == preset).unwrap_or(0)
}

/// Modal preset at dropdown `index` (clamped).
pub fn preset_from_index(index: usize) -> ModePreset {
    PRESETS[index.min(PRESETS.len() - 1)]
}

/// Human-readable label for a modal preset.
pub fn preset_label(preset: ModePreset) -> &'static str {
    match preset {
        ModePreset::Metal => "Metal",
        ModePreset::Wood => "Wood",
        ModePreset::Glass => "Glass",
    }
}

// ─── Body type ────────────────────────────────────────────────────────────────

/// Instrument-body colour applied to the plucked-string output (Phase 2
/// `Body Type`).  A single peaking biquad per body; the custom-IR body noted in
/// the roadmap prerequisites is a later upgrade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BodyType {
    /// No body colour — the raw string.
    #[default]
    None,
    /// Small parlour / student guitar (higher, tighter air resonance).
    Small,
    /// Dreadnought steel-string (low, broad body boom).
    Dreadnought,
    /// Cello-like body (low, resonant).
    Cello,
}

/// Every body type, in dropdown order.
pub const BODY_TYPES: [BodyType; 4] = [
    BodyType::None,
    BodyType::Small,
    BodyType::Dreadnought,
    BodyType::Cello,
];

impl BodyType {
    /// Body resonance as a peaking-EQ spec `(freq_hz, q, gain_db)`, or `None`
    /// for the raw string.
    pub fn peak(self) -> Option<(f32, f32, f32)> {
        match self {
            BodyType::None => None,
            BodyType::Small => Some((190.0, 1.1, 7.0)),
            BodyType::Dreadnought => Some((110.0, 0.9, 8.0)),
            BodyType::Cello => Some((100.0, 0.8, 7.0)),
        }
    }

    /// Index in the [`BODY_TYPES`] dropdown.
    pub fn to_index(self) -> usize {
        BODY_TYPES.iter().position(|&b| b == self).unwrap_or(0)
    }

    /// Body type at dropdown `index` (clamped).
    pub fn from_index(index: usize) -> Self {
        BODY_TYPES[index.min(BODY_TYPES.len() - 1)]
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            BodyType::None => "None",
            BodyType::Small => "Small",
            BodyType::Dreadnought => "Dreadnought",
            BodyType::Cello => "Cello",
        }
    }
}

// ─── Parameter model ──────────────────────────────────────────────────────────

/// Full parameter state of one instrument instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstrumentParams {
    /// Active model family (switches which voice bank renders).
    pub model: Model,

    // Plucked-string controls.
    /// Pluck position along the string, `0..=1`.
    pub pluck_position: f32,
    /// Pick hardness, `0..=1` (soft → hard).
    pub pick_hardness: f32,
    /// String damping / tone, `0..=1` (bright → dark).
    pub damping: f32,
    /// Instrument body colour.
    pub body: BodyType,
    /// Detune in cents applied to every note.
    pub detune_cents: f32,

    // Modal controls.
    /// Number of modes to sound (`1..=32`).
    pub num_modes: usize,
    /// Modal material timbre.
    pub preset: ModePreset,
    /// Modal decay slope (higher → bright modes decay faster).
    pub decay_slope: f32,

    // Shared.
    /// Sustain / decay time (T60) in seconds — the fundamental's ring time.
    pub decay_s: f32,

    // Global.
    /// Output gain in dB.
    pub gain_db: f32,
    /// Dry/wet mix, `0..=1` (effect use blends the audio input; `1` = all synth).
    pub mix: f32,
    /// Voice limit, `1..=16`.
    pub polyphony: usize,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
}

impl Default for InstrumentParams {
    fn default() -> Self {
        Self {
            model: Model::Plucked,
            pluck_position: 0.25,
            pick_hardness: 0.6,
            damping: 0.3,
            body: BodyType::Small,
            detune_cents: 0.0,
            num_modes: 5,
            preset: ModePreset::Metal,
            decay_slope: 1.0,
            decay_s: 3.0,
            gain_db: 0.0,
            mix: 1.0,
            polyphony: MAX_VOICES,
            bypass: false,
        }
    }
}

impl InstrumentParams {
    /// Clamp every field to its documented range.
    pub fn clamped(mut self) -> Self {
        self.pluck_position = self.pluck_position.clamp(0.0, 1.0);
        self.pick_hardness = self.pick_hardness.clamp(0.0, 1.0);
        self.damping = self.damping.clamp(0.0, 1.0);
        self.detune_cents = self
            .detune_cents
            .clamp(DETUNE_CENTS_RANGE.0, DETUNE_CENTS_RANGE.1);
        self.num_modes = self.num_modes.clamp(MODES_RANGE.0, MODES_RANGE.1);
        self.decay_slope = self
            .decay_slope
            .clamp(DECAY_SLOPE_RANGE.0, DECAY_SLOPE_RANGE.1);
        self.decay_s = self.decay_s.clamp(DECAY_S_RANGE.0, DECAY_S_RANGE.1);
        self.gain_db = self.gain_db.clamp(GAIN_DB_RANGE.0, GAIN_DB_RANGE.1);
        self.mix = self.mix.clamp(0.0, 1.0);
        self.polyphony = self.polyphony.clamp(POLYPHONY_RANGE.0, POLYPHONY_RANGE.1);
        self
    }

    /// Linear output gain (`−∞` mute below [`SILENCE_DB`]).
    pub fn gain(&self) -> f32 {
        if self.gain_db <= SILENCE_DB {
            0.0
        } else {
            db_to_gain(self.gain_db)
        }
    }

    /// The per-note core parameters this state produces.
    pub fn core_params(&self) -> PhysicalParams {
        PhysicalParams {
            model: self.model,
            t60: self.decay_s,
            damping: self.damping,
            pick_hardness: self.pick_hardness,
            pluck_position: self.pluck_position,
            preset: self.preset,
            num_modes: self.num_modes,
            decay_slope: self.decay_slope,
            tune_mult: tune_mult_from_cents(self.detune_cents),
        }
    }
}

/// Convert dB to a linear amplitude factor.
#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert a detune in cents to a frequency multiplier `2^(cents / 1200)`.
#[inline]
pub fn tune_mult_from_cents(cents: f32) -> f32 {
    (cents / 1200.0).exp2()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_in_range_and_neutral() {
        let params = InstrumentParams::default();
        assert_eq!(params, params.clamped(), "defaults must survive a clamp");
        assert_eq!(params.detune_cents, 0.0);
        assert_eq!(params.mix, 1.0);
        assert!(!params.bypass);
        assert_eq!(params.polyphony, MAX_VOICES);
    }

    #[test]
    fn clamp_pulls_fields_into_range() {
        let wild = InstrumentParams {
            pluck_position: 5.0,
            pick_hardness: -2.0,
            damping: 9.0,
            detune_cents: 9_999.0,
            num_modes: 999,
            decay_slope: 99.0,
            decay_s: 1_000.0,
            gain_db: 99.0,
            mix: 5.0,
            polyphony: 999,
            ..InstrumentParams::default()
        }
        .clamped();
        assert_eq!(wild.pluck_position, 1.0);
        assert_eq!(wild.pick_hardness, 0.0);
        assert_eq!(wild.damping, 1.0);
        assert_eq!(wild.detune_cents, DETUNE_CENTS_RANGE.1);
        assert_eq!(wild.num_modes, MAX_MODES);
        assert_eq!(wild.decay_slope, DECAY_SLOPE_RANGE.1);
        assert_eq!(wild.decay_s, DECAY_S_RANGE.1);
        assert_eq!(wild.gain_db, GAIN_DB_RANGE.1);
        assert_eq!(wild.mix, 1.0);
        assert_eq!(wild.polyphony, MAX_VOICES);
    }

    #[test]
    fn gain_mutes_at_floor_and_is_unity_at_zero_db() {
        let with_gain = |gain_db| InstrumentParams {
            gain_db,
            ..InstrumentParams::default()
        };
        assert!((with_gain(0.0).gain() - 1.0).abs() < 1e-6);
        assert_eq!(
            with_gain(GAIN_DB_RANGE.0).gain(),
            0.0,
            "floor gain must be a hard mute"
        );
        let plus_six = with_gain(6.0).gain();
        assert!(plus_six > 1.9 && plus_six < 2.1, "+6 dB ≈ ×2");
    }

    #[test]
    fn tune_mult_is_an_octave_at_1200_cents() {
        assert!((tune_mult_from_cents(0.0) - 1.0).abs() < 1e-6);
        assert!((tune_mult_from_cents(1200.0) - 2.0).abs() < 1e-5);
        assert!((tune_mult_from_cents(-1200.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn core_params_carry_detune_and_decay() {
        let params = InstrumentParams {
            detune_cents: 1200.0,
            decay_s: 4.0,
            ..InstrumentParams::default()
        };
        let core = params.core_params();
        assert!((core.tune_mult - 2.0).abs() < 1e-5);
        assert_eq!(core.t60, 4.0);
    }

    #[test]
    fn model_dropdown_roundtrips() {
        for (index, &model) in MODELS.iter().enumerate() {
            assert_eq!(model_to_index(model), index);
            assert_eq!(model_from_index(index), model);
            assert!(!model_label(model).is_empty());
        }
        // Out-of-range index clamps rather than panicking.
        assert_eq!(model_from_index(99), MODELS[MODELS.len() - 1]);
    }

    #[test]
    fn preset_and_body_dropdowns_roundtrip() {
        for (index, &preset) in PRESETS.iter().enumerate() {
            assert_eq!(preset_to_index(preset), index);
            assert_eq!(preset_from_index(index), preset);
        }
        for (index, &body) in BODY_TYPES.iter().enumerate() {
            assert_eq!(body.to_index(), index);
            assert_eq!(BodyType::from_index(index), body);
        }
        assert_eq!(BodyType::from_index(99), BodyType::Cello);
        assert!(BodyType::None.peak().is_none());
        assert!(BodyType::Dreadnought.peak().is_some());
    }
}
