//! Data-driven pedal presets — the "pedal model format" from the roadmap.
//!
//! A *pedal* carries **no DSP code**, only coefficients and enum selectors
//! over the shared [`crate::stage::DriveStage`] engine.  That is what lets new
//! pedals ship without touching the engine (the roadmap's `ShaperBank` goal).
//!
//! # Authoring format ([decision]: TOML via `toml-span`)
//!
//! Pedals are authored as TOML.  The shipped bank lives in `pedals/*.toml`,
//! is `include_str!`-embedded into the binary and parsed once at start-up by
//! [`shipped_pedals`] — so it still "ships in-plugin" (no external files at
//! run time) while TOML stays the single source of truth.  User-authored
//! pedals go through the exact same parser, [`PedalModel::from_toml_str`].
//!
//! `toml-span` is used (not `serde`/`toml`) for its **byte-offset spans**: a
//! mistake in a hand-edited pedal file yields an IDE-style
//! [`PedalLoadError`] pointing at the offending bytes.  Parsing uses the
//! manual `ValueInner`-walking pattern.
//!
//! ## Schema
//!
//! ```toml
//! name = "Green Overdrive"
//! input_gain_db = 12.0        # optional, default 0
//! bias = 0.0                  # optional, default 0 (pre-shaper DC)
//! output_gain_db = -6.0       # optional, default 0
//! default_oversampling = 4    # optional, default 1; one of 1/2/4/8/16
//!
//! [shaper]
//! kind = "diode"              # tanh|reciprocal|cubic_soft|hard_clip|diode|tube|tape
//! drive = 1.5                 # per-kind fields (see below)
//!
//! [tone]
//! kind = "three_band"         # bypass|baxandall|three_band
//! bass_db = -3.0
//! mid_db = 4.0
//! treble_db = -1.0
//! ```
//!
//! Per-shaper fields: `hard_clip` → `threshold` (`> 0`), `asymmetry`
//! (`[-1, 1]`); `diode` → `drive` (`> 0`); `tape` → `bias`; the others take
//! no fields.

use crate::shaper::Shaper;
use crate::tone::{Baxandall, ThreeBand, ToneStack};
use toml_span::Span;
use toml_span::value::{Table, ValueInner};

/// Error returned when a pedal TOML document is malformed or out of range.
///
/// Every variant carries the byte-offset [`Span`] of the offending node so a
/// caller (editor, plugin UI) can highlight it.
#[derive(thiserror::Error, Debug)]
pub enum PedalLoadError {
    /// The document was not syntactically valid TOML.
    #[error("TOML syntax error: {0}")]
    Parse(String),
    /// A required key was absent.
    #[error("missing required key '{key}'")]
    MissingKey {
        /// Dotted key path.
        key: String,
        /// Span of the enclosing element (default when unavailable).
        span: Span,
    },
    /// A key had the wrong TOML type.
    #[error("invalid type for key '{key}': expected {expected}")]
    InvalidType {
        /// Dotted key path.
        key: String,
        /// The type that was expected.
        expected: &'static str,
        /// Span of the offending value.
        span: Span,
    },
    /// A key had a value outside its allowed range / set.
    #[error("invalid value for key '{key}': {message}")]
    InvalidValue {
        /// Dotted key path.
        key: String,
        /// Human-readable explanation.
        message: String,
        /// Span of the offending value.
        span: Span,
    },
}

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
///
/// This is a construction-time descriptor (owns its `name`), so it is `Clone`
/// but not `Copy`; the engine copies out the cheap `Copy` fields it needs.
#[derive(Debug, Clone, PartialEq)]
pub struct PedalModel {
    /// Display name — descriptive, not a trademark (see the roadmap's
    /// "Sourcing & legal" note).
    pub name: String,
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

impl PedalModel {
    /// Parse a pedal from a TOML document.
    ///
    /// Returns a span-carrying [`PedalLoadError`] on any syntax, type or
    /// range problem.  **Allocates** — call off the audio thread.
    pub fn from_toml_str(content: &str) -> Result<Self, PedalLoadError> {
        let value =
            toml_span::parse(content).map_err(|err| PedalLoadError::Parse(err.to_string()))?;
        let ValueInner::Table(root) = value.as_ref() else {
            return Err(PedalLoadError::InvalidType {
                key: "root".into(),
                expected: "table",
                span: value.span,
            });
        };
        parse_pedal(root)
    }
}

/// The TOML sources of the shipped pedal bank, embedded at compile time.
const SHIPPED_TOML: &[&str] = &[
    include_str!("../pedals/clean-boost.toml"),
    include_str!("../pedals/green-overdrive.toml"),
    include_str!("../pedals/rodent-distortion.toml"),
    include_str!("../pedals/tube-warmth.toml"),
    include_str!("../pedals/tape-saturator.toml"),
];

/// Parse the shipped pedal bank ("ShaperBank") from the embedded TOML.
///
/// Call once at start-up (off the audio thread).  A parse failure here is a
/// build-time authoring bug, not a runtime input error — the
/// `shipped_toml_all_parse` test guards it.
pub fn shipped_pedals() -> Result<Vec<PedalModel>, PedalLoadError> {
    SHIPPED_TOML
        .iter()
        .map(|content| PedalModel::from_toml_str(content))
        .collect()
}

// ─── toml-span extractors (manual ValueInner walking) ───────────────────────

/// Read an optional string.
fn get_str<'a>(table: &'a Table<'a>, key: &str) -> Result<Option<&'a str>, PedalLoadError> {
    match table.get(key) {
        Some(value) => match value.as_ref() {
            ValueInner::String(text) => Ok(Some(text.as_ref())),
            _ => Err(PedalLoadError::InvalidType {
                key: key.to_string(),
                expected: "string",
                span: value.span,
            }),
        },
        None => Ok(None),
    }
}

/// Read an optional float (an integer literal is accepted and widened).
fn get_float(table: &Table<'_>, key: &str) -> Result<Option<f32>, PedalLoadError> {
    match table.get(key) {
        Some(value) => match value.as_ref() {
            ValueInner::Float(number) => Ok(Some(*number as f32)),
            ValueInner::Integer(number) => Ok(Some(*number as f32)),
            _ => Err(PedalLoadError::InvalidType {
                key: key.to_string(),
                expected: "float",
                span: value.span,
            }),
        },
        None => Ok(None),
    }
}

/// Read an optional `u8`, mapping overflow/negatives to a span-carrying error.
fn get_u8(table: &Table<'_>, key: &str) -> Result<Option<u8>, PedalLoadError> {
    let Some(value) = table.get(key) else {
        return Ok(None);
    };
    let ValueInner::Integer(number) = value.as_ref() else {
        return Err(PedalLoadError::InvalidType {
            key: key.to_string(),
            expected: "integer",
            span: value.span,
        });
    };
    u8::try_from(*number)
        .map(Some)
        .map_err(|_| PedalLoadError::InvalidValue {
            key: key.to_string(),
            message: format!("value {number} is out of range for 0..=255"),
            span: value.span,
        })
}

/// Read an optional sub-table.
fn get_table<'a>(table: &'a Table<'a>, key: &str) -> Result<Option<&'a Table<'a>>, PedalLoadError> {
    match table.get(key) {
        Some(value) => match value.as_ref() {
            ValueInner::Table(inner) => Ok(Some(inner)),
            _ => Err(PedalLoadError::InvalidType {
                key: key.to_string(),
                expected: "table",
                span: value.span,
            }),
        },
        None => Ok(None),
    }
}

/// Read a required string, or `MissingKey`.
fn require_str<'a>(table: &'a Table<'a>, key: &str) -> Result<&'a str, PedalLoadError> {
    get_str(table, key)?.ok_or_else(|| PedalLoadError::MissingKey {
        key: key.to_string(),
        span: Span::default(),
    })
}

/// Read a required float, or `MissingKey`.
fn require_float(table: &Table<'_>, key: &str) -> Result<f32, PedalLoadError> {
    get_float(table, key)?.ok_or_else(|| PedalLoadError::MissingKey {
        key: key.to_string(),
        span: Span::default(),
    })
}

/// Span of `key` within `table`, or a default span if absent.
fn span_of(table: &Table<'_>, key: &str) -> Span {
    table.get(key).map(|value| value.span).unwrap_or_default()
}

// ─── Section parsers ─────────────────────────────────────────────────────────

fn parse_pedal(root: &Table<'_>) -> Result<PedalModel, PedalLoadError> {
    let name = require_str(root, "name")?.to_string();
    let input_gain_db = get_float(root, "input_gain_db")?.unwrap_or(0.0);
    let bias = get_float(root, "bias")?.unwrap_or(0.0);
    let output_gain_db = get_float(root, "output_gain_db")?.unwrap_or(0.0);

    let default_oversampling = get_u8(root, "default_oversampling")?.unwrap_or(1);
    if !matches!(default_oversampling, 1 | 2 | 4 | 8 | 16) {
        return Err(PedalLoadError::InvalidValue {
            key: "default_oversampling".into(),
            message: format!("{default_oversampling} is not one of 1, 2, 4, 8, 16"),
            span: span_of(root, "default_oversampling"),
        });
    }

    let shaper = match get_table(root, "shaper")? {
        Some(table) => parse_shaper(table)?,
        None => {
            return Err(PedalLoadError::MissingKey {
                key: "shaper".into(),
                span: Span::default(),
            });
        }
    };

    let tone = match get_table(root, "tone")? {
        Some(table) => parse_tone(table)?,
        None => ToneConfig::Bypass,
    };

    Ok(PedalModel {
        name,
        input_gain_db,
        shaper,
        bias,
        tone,
        output_gain_db,
        default_oversampling,
    })
}

fn parse_shaper(table: &Table<'_>) -> Result<Shaper, PedalLoadError> {
    let kind = require_str(table, "kind")?;
    let shaper = match kind {
        "tanh" => Shaper::Tanh,
        "reciprocal" => Shaper::Reciprocal,
        "cubic_soft" => Shaper::CubicSoft,
        "hard_clip" => {
            let threshold = require_float(table, "threshold")?;
            if threshold <= 0.0 {
                return Err(PedalLoadError::InvalidValue {
                    key: "shaper.threshold".into(),
                    message: format!("threshold must be > 0, got {threshold}"),
                    span: span_of(table, "threshold"),
                });
            }
            let asymmetry = get_float(table, "asymmetry")?.unwrap_or(0.0);
            if !(-1.0..=1.0).contains(&asymmetry) {
                return Err(PedalLoadError::InvalidValue {
                    key: "shaper.asymmetry".into(),
                    message: format!("asymmetry must be within [-1, 1], got {asymmetry}"),
                    span: span_of(table, "asymmetry"),
                });
            }
            Shaper::HardClip {
                threshold,
                asymmetry,
            }
        }
        "diode" => {
            let drive = require_float(table, "drive")?;
            if drive <= 0.0 {
                return Err(PedalLoadError::InvalidValue {
                    key: "shaper.drive".into(),
                    message: format!("drive must be > 0, got {drive}"),
                    span: span_of(table, "drive"),
                });
            }
            Shaper::Diode { drive }
        }
        "tube" => Shaper::Tube,
        "tape" => {
            let bias = get_float(table, "bias")?.unwrap_or(0.0);
            Shaper::Tape { bias }
        }
        other => {
            return Err(PedalLoadError::InvalidValue {
                key: "shaper.kind".into(),
                message: format!(
                    "\"{other}\" is not a known shaper \
                     (expected tanh, reciprocal, cubic_soft, hard_clip, diode, tube or tape)"
                ),
                span: span_of(table, "kind"),
            });
        }
    };
    Ok(shaper)
}

fn parse_tone(table: &Table<'_>) -> Result<ToneConfig, PedalLoadError> {
    let kind = require_str(table, "kind")?;
    let tone = match kind {
        "bypass" => ToneConfig::Bypass,
        "baxandall" => ToneConfig::Baxandall {
            bass_db: get_float(table, "bass_db")?.unwrap_or(0.0),
            treble_db: get_float(table, "treble_db")?.unwrap_or(0.0),
        },
        "three_band" => ToneConfig::ThreeBand {
            bass_db: get_float(table, "bass_db")?.unwrap_or(0.0),
            mid_db: get_float(table, "mid_db")?.unwrap_or(0.0),
            treble_db: get_float(table, "treble_db")?.unwrap_or(0.0),
        },
        other => {
            return Err(PedalLoadError::InvalidValue {
                key: "tone.kind".into(),
                message: format!(
                    "\"{other}\" is not a known tone stack \
                     (expected bypass, baxandall or three_band)"
                ),
                span: span_of(table, "kind"),
            });
        }
    };
    Ok(tone)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shipped pedal TOML parses, and their names are unique.
    #[test]
    fn shipped_toml_all_parse() {
        let pedals = shipped_pedals().expect("shipped pedals must parse");
        assert_eq!(pedals.len(), SHIPPED_TOML.len());
        for (index, pedal) in pedals.iter().enumerate() {
            assert!(!pedal.name.is_empty(), "pedal {index} has empty name");
            assert!(
                matches!(pedal.default_oversampling, 1 | 2 | 4 | 8 | 16),
                "{}: bad oversampling {}",
                pedal.name,
                pedal.default_oversampling
            );
            for other in &pedals[index + 1..] {
                assert_ne!(pedal.name, other.name, "duplicate name {}", pedal.name);
            }
        }
    }

    /// A full valid document parses into the expected model.
    #[test]
    fn parses_full_document() {
        let toml = r#"
            name = "Test Drive"
            input_gain_db = 9.0
            bias = 0.1
            output_gain_db = -3.0
            default_oversampling = 4

            [shaper]
            kind = "hard_clip"
            threshold = 0.6
            asymmetry = 0.25

            [tone]
            kind = "three_band"
            bass_db = 1.0
            mid_db = -2.0
            treble_db = 3.0
        "#;
        let pedal = PedalModel::from_toml_str(toml).unwrap();
        assert_eq!(pedal.name, "Test Drive");
        assert_eq!(pedal.input_gain_db, 9.0);
        assert_eq!(pedal.default_oversampling, 4);
        assert_eq!(
            pedal.shaper,
            Shaper::HardClip {
                threshold: 0.6,
                asymmetry: 0.25
            }
        );
        assert_eq!(
            pedal.tone,
            ToneConfig::ThreeBand {
                bass_db: 1.0,
                mid_db: -2.0,
                treble_db: 3.0
            }
        );
    }

    /// Optional fields default sensibly; `tone` defaults to bypass.
    #[test]
    fn optional_fields_default() {
        let toml = r#"
            name = "Minimal"
            [shaper]
            kind = "tanh"
        "#;
        let pedal = PedalModel::from_toml_str(toml).unwrap();
        assert_eq!(pedal.input_gain_db, 0.0);
        assert_eq!(pedal.bias, 0.0);
        assert_eq!(pedal.output_gain_db, 0.0);
        assert_eq!(pedal.default_oversampling, 1);
        assert_eq!(pedal.tone, ToneConfig::Bypass);
    }

    /// An integer literal is accepted where a float is expected.
    #[test]
    fn integer_accepted_as_float() {
        let toml = "name = \"Int\"\ninput_gain_db = 6\n[shaper]\nkind = \"tanh\"\n";
        let pedal = PedalModel::from_toml_str(toml).unwrap();
        assert_eq!(pedal.input_gain_db, 6.0);
    }

    /// A missing `name` is a `MissingKey` error.
    #[test]
    fn missing_name_is_error() {
        let toml = "[shaper]\nkind = \"tanh\"\n";
        assert!(matches!(
            PedalModel::from_toml_str(toml),
            Err(PedalLoadError::MissingKey { .. })
        ));
    }

    /// A missing `[shaper]` table is a `MissingKey` error.
    #[test]
    fn missing_shaper_is_error() {
        let toml = "name = \"No Shaper\"\n";
        assert!(matches!(
            PedalModel::from_toml_str(toml),
            Err(PedalLoadError::MissingKey { key, .. }) if key == "shaper"
        ));
    }

    /// An unknown shaper kind reports `InvalidValue` with a non-default span.
    #[test]
    fn unknown_shaper_kind_is_error_with_span() {
        let toml = "name = \"X\"\n[shaper]\nkind = \"fuzzface\"\n";
        let error = PedalModel::from_toml_str(toml).unwrap_err();
        match error {
            PedalLoadError::InvalidValue { key, span, .. } => {
                assert_eq!(key, "shaper.kind");
                // The span must point at the actual "fuzzface" bytes.
                assert!(span.end > span.start, "span should be non-empty");
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
    }

    /// An out-of-range oversampling factor is `InvalidValue`, not `InvalidType`.
    #[test]
    fn bad_oversampling_is_invalid_value() {
        let toml = "name = \"X\"\ndefault_oversampling = 3\n[shaper]\nkind = \"tanh\"\n";
        assert!(matches!(
            PedalModel::from_toml_str(toml),
            Err(PedalLoadError::InvalidValue { .. })
        ));
    }

    /// A non-positive diode drive is rejected.
    #[test]
    fn non_positive_diode_drive_is_error() {
        let toml = "name = \"X\"\n[shaper]\nkind = \"diode\"\ndrive = 0.0\n";
        assert!(matches!(
            PedalModel::from_toml_str(toml),
            Err(PedalLoadError::InvalidValue { key, .. }) if key == "shaper.drive"
        ));
    }

    /// A wrong-typed field is `InvalidType`.
    #[test]
    fn wrong_type_is_invalid_type() {
        let toml = "name = \"X\"\ninput_gain_db = \"loud\"\n[shaper]\nkind = \"tanh\"\n";
        assert!(matches!(
            PedalModel::from_toml_str(toml),
            Err(PedalLoadError::InvalidType { .. })
        ));
    }

    /// Malformed TOML surfaces as a `Parse` error.
    #[test]
    fn syntax_error_is_parse_error() {
        let toml = "name = \n";
        assert!(matches!(
            PedalModel::from_toml_str(toml),
            Err(PedalLoadError::Parse(_))
        ));
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
