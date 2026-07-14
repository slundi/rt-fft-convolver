//! Data-driven pedal presets — the "pedal model format" from the roadmap.
//!
//! A *pedal* carries **no DSP code**, only coefficients and enum selectors
//! over the shared [`crate::stage::DriveStage`] engine.  That is what lets new
//! pedals ship without touching the engine (the roadmap's `ShaperBank` goal).
//!
//! [`PedalModel`] is a plain `Copy` struct, so a bank of presets is a `const`
//! table baked into the binary — no allocation, no I/O.  A future
//! `serde`/`build.rs` authoring path (RON/TOML under `pedals/`) can populate
//! this same struct; that is deliberately deferred (see the roadmap).

use crate::shaper::Shaper;
use crate::tone::{Baxandall, ThreeBand, ToneStack};

/// Tone-stack *description* (pure data) that a pedal preset carries.
///
/// Built into a live, stateful [`ToneStack`] by [`ToneConfig::build`] when a
/// [`crate::stage::DriveStage`] is constructed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneConfig {
    /// No tone shaping.
    Bypass,
    /// 2-band Baxandall (bass / treble) shelving.
    Baxandall {
        /// Low-shelf gain (dB).
        bass_db: f32,
        /// High-shelf gain (dB).
        treble_db: f32,
    },
    /// 3-band bass / mid / treble.
    ThreeBand {
        /// Low-shelf gain (dB).
        bass_db: f32,
        /// Mid peak gain (dB).
        mid_db: f32,
        /// High-shelf gain (dB).
        treble_db: f32,
    },
}

impl ToneConfig {
    /// Build the stateful [`ToneStack`] for this description at `sample_rate`.
    ///
    /// **Allocates nothing**, but designs biquad coefficients (transcendental
    /// math) — call off the audio thread.
    pub fn build(self, sample_rate: f32) -> ToneStack {
        match self {
            ToneConfig::Bypass => ToneStack::Bypass,
            ToneConfig::Baxandall { bass_db, treble_db } => {
                ToneStack::Baxandall(Baxandall::new(sample_rate, bass_db, treble_db))
            }
            ToneConfig::ThreeBand {
                bass_db,
                mid_db,
                treble_db,
            } => ToneStack::ThreeBand(ThreeBand::new(sample_rate, bass_db, mid_db, treble_db)),
        }
    }
}

/// One "pedal" = coefficients + enum selectors over the shared engine.
///
/// The signal path a [`crate::stage::DriveStage`] applies is:
///
/// ```text
///  in ─▶ ×input_gain ─▶ +bias ─▶ shaper ─▶ ×output_gain ─▶ DC-block ─▶ tone ─▶ out
///          └──────── inside the oversampled region ───────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PedalModel {
    /// Display name — descriptive, not a trademark (see the roadmap's
    /// "Sourcing & legal" note).
    pub name: &'static str,
    /// Pre-shaper drive trim, in dB.  Higher → harder into the nonlinearity.
    pub input_gain_db: f32,
    /// The waveshaper curve.
    pub shaper: Shaper,
    /// DC offset injected pre-shaper for even-harmonic asymmetry.  The
    /// post-shaper DC blocker removes the resulting offset.
    pub bias: f32,
    /// Tone-stack description.
    pub tone: ToneConfig,
    /// Post-shaper make-up gain, in dB.
    pub output_gain_db: f32,
    /// Default oversampling factor (`1, 2, 4, 8, 16`).
    pub default_oversampling: u8,
}

/// The shipped preset bank ("ShaperBank"): a `const` table of pedal models.
///
/// Names are descriptive rather than trademarked (roadmap "Sourcing & legal").
/// This is the data the plugin's `Model` dropdown will enumerate; adding a
/// pedal here needs no engine change.
pub const PEDALS: &[PedalModel] = &[
    PedalModel {
        name: "Clean Boost",
        input_gain_db: 0.0,
        shaper: Shaper::Tanh,
        bias: 0.0,
        tone: ToneConfig::Bypass,
        output_gain_db: 0.0,
        default_oversampling: 2,
    },
    PedalModel {
        // Tube-Screamer-style mid-focused soft overdrive.
        name: "Green Overdrive",
        input_gain_db: 12.0,
        shaper: Shaper::Diode { drive: 1.5 },
        bias: 0.0,
        tone: ToneConfig::ThreeBand {
            bass_db: -3.0,
            mid_db: 4.0,
            treble_db: -1.0,
        },
        output_gain_db: -6.0,
        default_oversampling: 4,
    },
    PedalModel {
        // ProCo-Rat-style hard, buzzy distortion.
        name: "Rodent Distortion",
        input_gain_db: 20.0,
        shaper: Shaper::HardClip {
            threshold: 0.7,
            asymmetry: 0.2,
        },
        bias: 0.05,
        tone: ToneConfig::Baxandall {
            bass_db: -2.0,
            treble_db: 3.0,
        },
        output_gain_db: -10.0,
        default_oversampling: 8,
    },
    PedalModel {
        name: "Tube Warmth",
        input_gain_db: 6.0,
        shaper: Shaper::Tube,
        bias: 0.1,
        tone: ToneConfig::Baxandall {
            bass_db: 2.0,
            treble_db: 1.0,
        },
        output_gain_db: -3.0,
        default_oversampling: 4,
    },
    PedalModel {
        name: "Tape Saturator",
        input_gain_db: 3.0,
        shaper: Shaper::Tape { bias: 0.2 },
        bias: 0.0,
        tone: ToneConfig::Bypass,
        output_gain_db: 0.0,
        default_oversampling: 2,
    },
];

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped bank is non-empty and every name is unique.
    #[test]
    fn bank_names_are_unique_and_nonempty() {
        assert!(!PEDALS.is_empty());
        for (index, pedal) in PEDALS.iter().enumerate() {
            assert!(!pedal.name.is_empty(), "pedal {index} has empty name");
            for other in &PEDALS[index + 1..] {
                assert_ne!(
                    pedal.name, other.name,
                    "duplicate pedal name {}",
                    pedal.name
                );
            }
        }
    }

    /// Every shipped pedal specifies a valid power-of-two oversampling factor.
    #[test]
    fn bank_oversampling_factors_are_valid() {
        for pedal in PEDALS {
            assert!(
                matches!(pedal.default_oversampling, 1 | 2 | 4 | 8 | 16),
                "{}: bad oversampling {}",
                pedal.name,
                pedal.default_oversampling
            );
        }
    }

    /// `ToneConfig::build` maps each variant to the matching live stack.
    #[test]
    fn tone_config_builds_matching_stack() {
        assert!(matches!(
            ToneConfig::Bypass.build(48_000.0),
            ToneStack::Bypass
        ));
        assert!(matches!(
            ToneConfig::Baxandall {
                bass_db: 3.0,
                treble_db: -2.0
            }
            .build(48_000.0),
            ToneStack::Baxandall(_)
        ));
        assert!(matches!(
            ToneConfig::ThreeBand {
                bass_db: 1.0,
                mid_db: 2.0,
                treble_db: 3.0
            }
            .build(48_000.0),
            ToneStack::ThreeBand(_)
        ));
    }
}
