//! Plain, framework-agnostic parameter model for the cab plugin.
//!
//! [`CabParams`] mirrors the roadmap's control list (Phase 1) as a simple
//! `Copy` struct with documented ranges and defaults.  The `nih-plug` wrapper
//! (behind the `nih` feature) maps its automatable parameters onto these
//! fields; keeping the model free of `nih-plug` lets us unit-test ranges,
//! clamping, and presets without pulling in the plugin framework.

/// Number of IR slots per plugin (roadmap: 8).
pub const NUM_SLOTS: usize = 8;

/// Inclusive `(min, max)` range of the crossfade time in milliseconds.
pub const CROSSFADE_MS_RANGE: (f32, f32) = (5.0, 200.0);
/// Inclusive gain range in decibels for the input and output trims.
pub const GAIN_DB_RANGE: (f32, f32) = (-24.0, 24.0);
/// Inclusive low-cut (high-pass) cutoff range in hertz.
pub const LOW_CUT_RANGE: (f32, f32) = (20.0, 500.0);
/// Inclusive high-cut (low-pass) cutoff range in hertz.
pub const HIGH_CUT_RANGE: (f32, f32) = (2_000.0, 20_000.0);

/// The full parameter state of one cab instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CabParams {
    /// Active IR slot, `0..NUM_SLOTS`.
    pub current_slot: usize,
    /// Equal-power crossfade duration in ms when switching slots.
    pub crossfade_ms: f32,
    /// Input trim in dB, applied before the cab.
    pub input_gain_db: f32,
    /// Output trim in dB, applied after the mix.
    pub output_gain_db: f32,
    /// Dry/wet blend, `0.0` = fully dry, `1.0` = fully wet (cab).
    pub mix: f32,
    /// Low-cut (high-pass) cutoff in Hz, pre-convolution.
    pub low_cut_hz: f32,
    /// High-cut (low-pass) cutoff in Hz, post-convolution.
    pub high_cut_hz: f32,
    /// Normalise each IR to unit L2-norm (uses the crate's `Mixer` gain).
    pub ir_normalize: bool,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
    /// Per-slot enable flags; a disabled slot plays dry even if loaded.
    pub slot_enabled: [bool; NUM_SLOTS],
}

impl Default for CabParams {
    fn default() -> Self {
        Self {
            current_slot: 0,
            crossfade_ms: 20.0,
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            mix: 1.0, // 100 % wet by default (roadmap)
            low_cut_hz: LOW_CUT_RANGE.0,
            high_cut_hz: HIGH_CUT_RANGE.1,
            ir_normalize: false,
            bypass: false,
            slot_enabled: [true; NUM_SLOTS],
        }
    }
}

impl CabParams {
    /// Clamp every field to its documented range (defensive: hosts and
    /// presets can supply out-of-range values).
    pub fn clamped(mut self) -> Self {
        self.current_slot = self.current_slot.min(NUM_SLOTS - 1);
        self.crossfade_ms = self
            .crossfade_ms
            .clamp(CROSSFADE_MS_RANGE.0, CROSSFADE_MS_RANGE.1);
        self.input_gain_db = self.input_gain_db.clamp(GAIN_DB_RANGE.0, GAIN_DB_RANGE.1);
        self.output_gain_db = self.output_gain_db.clamp(GAIN_DB_RANGE.0, GAIN_DB_RANGE.1);
        self.mix = self.mix.clamp(0.0, 1.0);
        self.low_cut_hz = self.low_cut_hz.clamp(LOW_CUT_RANGE.0, LOW_CUT_RANGE.1);
        self.high_cut_hz = self.high_cut_hz.clamp(HIGH_CUT_RANGE.0, HIGH_CUT_RANGE.1);
        self
    }

    /// Apply a tone preset — **EQ only**.  Per the roadmap, presets never
    /// touch the IR selection, gains, or mix.
    pub fn apply_preset(&mut self, preset: &CabPreset) {
        self.low_cut_hz = preset.low_cut_hz;
        self.high_cut_hz = preset.high_cut_hz;
    }
}

/// A neutral tone-shaping preset that only moves the two EQ cutoffs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CabPreset {
    /// Display name.
    pub name: &'static str,
    /// Low-cut cutoff in Hz.
    pub low_cut_hz: f32,
    /// High-cut cutoff in Hz.
    pub high_cut_hz: f32,
}

/// Bundled EQ-only presets (roadmap: bright / dark / scooped).
///
/// With just a low-cut and a high-cut band these are approximations: "bright"
/// opens the top and trims mud, "dark" rolls off the top, "scooped" tightens
/// both ends.  A true mid-scoop would need a dedicated band (future work).
pub const PRESETS: &[CabPreset] = &[
    CabPreset {
        name: "Flat",
        low_cut_hz: LOW_CUT_RANGE.0,
        high_cut_hz: HIGH_CUT_RANGE.1,
    },
    CabPreset {
        name: "Bright",
        low_cut_hz: 90.0,
        high_cut_hz: HIGH_CUT_RANGE.1,
    },
    CabPreset {
        name: "Dark",
        low_cut_hz: LOW_CUT_RANGE.0,
        high_cut_hz: 5_000.0,
    },
    CabPreset {
        name: "Scooped",
        low_cut_hz: 120.0,
        high_cut_hz: 6_500.0,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_roadmap() {
        let p = CabParams::default();
        assert_eq!(p.current_slot, 0);
        assert_eq!(p.mix, 1.0); // 100 % wet
        assert_eq!(p.input_gain_db, 0.0);
        assert_eq!(p.output_gain_db, 0.0);
        assert!(!p.bypass);
        assert!(!p.ir_normalize);
        assert!(p.slot_enabled.iter().all(|&e| e));
        // Default EQ is wide open (no tone shaping).
        assert_eq!(p.low_cut_hz, LOW_CUT_RANGE.0);
        assert_eq!(p.high_cut_hz, HIGH_CUT_RANGE.1);
    }

    #[test]
    fn clamp_pulls_every_field_into_range() {
        let wild = CabParams {
            current_slot: 99,
            crossfade_ms: 10_000.0,
            input_gain_db: 100.0,
            output_gain_db: -100.0,
            mix: 5.0,
            low_cut_hz: 1.0,
            high_cut_hz: 100_000.0,
            ..CabParams::default()
        }
        .clamped();

        assert_eq!(wild.current_slot, NUM_SLOTS - 1);
        assert_eq!(wild.crossfade_ms, CROSSFADE_MS_RANGE.1);
        assert_eq!(wild.input_gain_db, GAIN_DB_RANGE.1);
        assert_eq!(wild.output_gain_db, GAIN_DB_RANGE.0);
        assert_eq!(wild.mix, 1.0);
        assert_eq!(wild.low_cut_hz, LOW_CUT_RANGE.0);
        assert_eq!(wild.high_cut_hz, HIGH_CUT_RANGE.1);
    }

    #[test]
    fn preset_touches_only_eq() {
        let mut p = CabParams {
            current_slot: 3,
            input_gain_db: 6.0,
            output_gain_db: -3.0,
            mix: 0.5,
            ..CabParams::default()
        };
        let before = p;
        let dark = PRESETS.iter().find(|p| p.name == "Dark").unwrap();
        p.apply_preset(dark);

        // EQ changed…
        assert_eq!(p.low_cut_hz, dark.low_cut_hz);
        assert_eq!(p.high_cut_hz, dark.high_cut_hz);
        // …everything else untouched.
        assert_eq!(p.current_slot, before.current_slot);
        assert_eq!(p.input_gain_db, before.input_gain_db);
        assert_eq!(p.output_gain_db, before.output_gain_db);
        assert_eq!(p.mix, before.mix);
        assert_eq!(p.slot_enabled, before.slot_enabled);
    }

    #[test]
    fn presets_stay_within_eq_ranges() {
        for preset in PRESETS {
            assert!(
                (LOW_CUT_RANGE.0..=LOW_CUT_RANGE.1).contains(&preset.low_cut_hz),
                "{} low cut out of range",
                preset.name
            );
            assert!(
                (HIGH_CUT_RANGE.0..=HIGH_CUT_RANGE.1).contains(&preset.high_cut_hz),
                "{} high cut out of range",
                preset.name
            );
        }
    }
}
