//! Plain, framework-agnostic parameter model for the reverb plugin.
//!
//! Mirrors the roadmap's Phase-2 control list.  The `nih-plug` wrapper maps its
//! automatable parameters onto these fields; keeping the model framework-free
//! lets us unit-test ranges, clamping, and presets without the plugin runtime.

/// Number of IR slots (roadmap: 8).
pub const NUM_SLOTS: usize = 8;

/// Crossfade time range in ms (longer than the cab — reverb tails need it).
pub const CROSSFADE_MS_RANGE: (f32, f32) = (5.0, 500.0);
/// Pre-delay range in ms.
pub const PREDELAY_MS_RANGE: (f32, f32) = (0.0, 500.0);
/// Dry / Wet fader range in dB (the low end behaves as `−∞`; see `fader_gain`).
pub const FADER_DB_RANGE: (f32, f32) = (-60.0, 6.0);
/// Stereo width range (`1.0` = 100 %, `2.0` = 200 %).
pub const WIDTH_RANGE: (f32, f32) = (0.0, 2.0);
/// Low-cut (high-pass) cutoff range in Hz (wet only).
pub const LOW_CUT_RANGE: (f32, f32) = (20.0, 500.0);
/// High-cut (low-pass) cutoff range in Hz (wet only).
pub const HIGH_CUT_RANGE: (f32, f32) = (2_000.0, 20_000.0);
/// IR length fraction range (`0.01` = 1 %, `1.0` = 100 %).
pub const IR_LENGTH_RANGE: (f32, f32) = (0.01, 1.0);

/// Full parameter state of one reverb instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReverbParams {
    /// Active IR slot, `0..NUM_SLOTS`.
    pub current_slot: usize,
    /// Equal-power crossfade duration in ms when switching slots.
    pub crossfade_ms: f32,
    /// Pre-delay before the reverb onset, in ms.
    pub predelay_ms: f32,
    /// Dry fader in dB (`−∞ … +6`); `−∞` mutes the dry path.
    pub dry_db: f32,
    /// Wet fader in dB (`−∞ … +6`).
    pub wet_db: f32,
    /// Stereo width applied to the wet signal (`0..=2`).
    pub width: f32,
    /// Wet-only low-cut (high-pass) cutoff in Hz.
    pub low_cut_hz: f32,
    /// Wet-only high-cut (low-pass) cutoff in Hz.
    pub high_cut_hz: f32,
    /// Reverse the IR (build-time).
    pub ir_reverse: bool,
    /// Fraction of the IR tail to keep (`0.01..=1.0`) — trims CPU.
    pub ir_length: f32,
    /// Host bypass — output equals the untouched input.
    pub bypass: bool,
    /// Per-slot enable flags; a disabled slot plays dry.
    pub slot_enabled: [bool; NUM_SLOTS],
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self {
            current_slot: 0,
            crossfade_ms: 100.0,
            predelay_ms: 0.0,
            dry_db: 0.0,
            wet_db: 0.0,
            width: 1.0,
            low_cut_hz: LOW_CUT_RANGE.0,
            high_cut_hz: HIGH_CUT_RANGE.1,
            ir_reverse: false,
            ir_length: 1.0,
            bypass: false,
            slot_enabled: [true; NUM_SLOTS],
        }
    }
}

impl ReverbParams {
    /// Clamp every field to its documented range.
    pub fn clamped(mut self) -> Self {
        self.current_slot = self.current_slot.min(NUM_SLOTS - 1);
        self.crossfade_ms = self
            .crossfade_ms
            .clamp(CROSSFADE_MS_RANGE.0, CROSSFADE_MS_RANGE.1);
        self.predelay_ms = self
            .predelay_ms
            .clamp(PREDELAY_MS_RANGE.0, PREDELAY_MS_RANGE.1);
        self.dry_db = self.dry_db.clamp(FADER_DB_RANGE.0, FADER_DB_RANGE.1);
        self.wet_db = self.wet_db.clamp(FADER_DB_RANGE.0, FADER_DB_RANGE.1);
        self.width = self.width.clamp(WIDTH_RANGE.0, WIDTH_RANGE.1);
        self.low_cut_hz = self.low_cut_hz.clamp(LOW_CUT_RANGE.0, LOW_CUT_RANGE.1);
        self.high_cut_hz = self.high_cut_hz.clamp(HIGH_CUT_RANGE.0, HIGH_CUT_RANGE.1);
        self.ir_length = self.ir_length.clamp(IR_LENGTH_RANGE.0, IR_LENGTH_RANGE.1);
        self
    }

    /// Apply a space/tone preset — leaves the IR selection, faders, and bypass
    /// untouched (the IR itself comes from the bundled pack; see [`PRESETS`]).
    pub fn apply_preset(&mut self, preset: &ReverbPreset) {
        self.predelay_ms = preset.predelay_ms;
        self.width = preset.width;
        self.low_cut_hz = preset.low_cut_hz;
        self.high_cut_hz = preset.high_cut_hz;
        self.ir_length = preset.ir_length;
    }
}

/// A space/tone preset.  These shape pre-delay, width, EQ, and tail length; the
/// matching room IR ships in the (Phase-5) bundled pack and is loaded into a
/// slot separately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReverbPreset {
    /// Display name.
    pub name: &'static str,
    /// Pre-delay in ms.
    pub predelay_ms: f32,
    /// Stereo width.
    pub width: f32,
    /// Low-cut cutoff in Hz.
    pub low_cut_hz: f32,
    /// High-cut cutoff in Hz.
    pub high_cut_hz: f32,
    /// IR length fraction.
    pub ir_length: f32,
}

/// Bundled space presets (roadmap: small room / large room / hall / plate /
/// spring).  Parameter shaping only — pair each with its room IR from the pack.
pub const PRESETS: &[ReverbPreset] = &[
    ReverbPreset {
        name: "Small Room",
        predelay_ms: 8.0,
        width: 0.8,
        low_cut_hz: 120.0,
        high_cut_hz: 12_000.0,
        ir_length: 0.5,
    },
    ReverbPreset {
        name: "Large Room",
        predelay_ms: 20.0,
        width: 1.0,
        low_cut_hz: 80.0,
        high_cut_hz: 14_000.0,
        ir_length: 0.8,
    },
    ReverbPreset {
        name: "Hall",
        predelay_ms: 35.0,
        width: 1.3,
        low_cut_hz: 60.0,
        high_cut_hz: 16_000.0,
        ir_length: 1.0,
    },
    ReverbPreset {
        name: "Plate",
        predelay_ms: 5.0,
        width: 1.1,
        low_cut_hz: 200.0,
        high_cut_hz: 18_000.0,
        ir_length: 0.7,
    },
    ReverbPreset {
        name: "Spring",
        predelay_ms: 0.0,
        width: 0.6,
        low_cut_hz: 300.0,
        high_cut_hz: 6_000.0,
        ir_length: 0.6,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_neutral() {
        let p = ReverbParams::default();
        assert_eq!(p.current_slot, 0);
        assert_eq!(p.dry_db, 0.0);
        assert_eq!(p.wet_db, 0.0);
        assert_eq!(p.width, 1.0);
        assert_eq!(p.predelay_ms, 0.0);
        assert_eq!(p.ir_length, 1.0);
        assert!(!p.ir_reverse);
        assert!(!p.bypass);
        assert!(p.slot_enabled.iter().all(|&e| e));
    }

    #[test]
    fn clamp_pulls_fields_into_range() {
        let wild = ReverbParams {
            current_slot: 99,
            crossfade_ms: 9_999.0,
            predelay_ms: -5.0,
            dry_db: 100.0,
            wet_db: -999.0,
            width: 9.0,
            low_cut_hz: 1.0,
            high_cut_hz: 99_999.0,
            ir_length: 5.0,
            ..ReverbParams::default()
        }
        .clamped();
        assert_eq!(wild.current_slot, NUM_SLOTS - 1);
        assert_eq!(wild.crossfade_ms, CROSSFADE_MS_RANGE.1);
        assert_eq!(wild.predelay_ms, PREDELAY_MS_RANGE.0);
        assert_eq!(wild.dry_db, FADER_DB_RANGE.1);
        assert_eq!(wild.wet_db, FADER_DB_RANGE.0);
        assert_eq!(wild.width, WIDTH_RANGE.1);
        assert_eq!(wild.low_cut_hz, LOW_CUT_RANGE.0);
        assert_eq!(wild.high_cut_hz, HIGH_CUT_RANGE.1);
        assert_eq!(wild.ir_length, IR_LENGTH_RANGE.1);
    }

    #[test]
    fn preset_leaves_ir_and_faders_untouched() {
        let mut p = ReverbParams {
            current_slot: 4,
            dry_db: -3.0,
            wet_db: -6.0,
            bypass: true,
            ir_reverse: true,
            ..ReverbParams::default()
        };
        let before = p;
        let hall = PRESETS.iter().find(|p| p.name == "Hall").unwrap();
        p.apply_preset(hall);

        // Space/tone params changed…
        assert_eq!(p.predelay_ms, hall.predelay_ms);
        assert_eq!(p.width, hall.width);
        assert_eq!(p.low_cut_hz, hall.low_cut_hz);
        assert_eq!(p.high_cut_hz, hall.high_cut_hz);
        assert_eq!(p.ir_length, hall.ir_length);
        // …IR selection, faders, bypass, reverse untouched.
        assert_eq!(p.current_slot, before.current_slot);
        assert_eq!(p.dry_db, before.dry_db);
        assert_eq!(p.wet_db, before.wet_db);
        assert_eq!(p.bypass, before.bypass);
        assert_eq!(p.ir_reverse, before.ir_reverse);
    }

    #[test]
    fn presets_are_within_ranges() {
        for p in PRESETS {
            assert!(
                (PREDELAY_MS_RANGE.0..=PREDELAY_MS_RANGE.1).contains(&p.predelay_ms),
                "{}",
                p.name
            );
            assert!(
                (WIDTH_RANGE.0..=WIDTH_RANGE.1).contains(&p.width),
                "{}",
                p.name
            );
            assert!(
                (LOW_CUT_RANGE.0..=LOW_CUT_RANGE.1).contains(&p.low_cut_hz),
                "{}",
                p.name
            );
            assert!(
                (HIGH_CUT_RANGE.0..=HIGH_CUT_RANGE.1).contains(&p.high_cut_hz),
                "{}",
                p.name
            );
            assert!(
                (IR_LENGTH_RANGE.0..=IR_LENGTH_RANGE.1).contains(&p.ir_length),
                "{}",
                p.name
            );
        }
    }
}
