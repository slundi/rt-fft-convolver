//! Framework-agnostic UI logic for the (Phase-3) `nih-plug-egui` editor.
//!
//! The egui drawing lives in the `gui`-gated `plugin` module; everything here
//! is pure, `egui`-free and therefore **unit-testable**:
//!
//! - [`hz_to_note_name`] / [`format_hz`] — the pitch-tracking indicator (Hz +
//!   note name).
//! - [`confidence_fraction`] — the pitch-confidence bar fill.
//! - [`spectral_heights`] — normalises a magnitude / envelope spectrum to
//!   `[0, 1]` draw heights for the formant display.
//! - [`format_interval`] / [`format_level`] / [`format_pan`] /
//!   [`format_detune`] — the harmoniser voice-grid cell labels.
//! - [`latency_ms`] / [`format_latency_ms`] — the latency badge.
//!
//! The MIDI-activity LED and the per-voice pad state reuse the shared
//! [`MidiActivity`](rt_ir_plugin_core::MidiActivity) /
//! [`pad_state`](rt_ir_plugin_core::pad_state) helpers, re-exported here for the
//! editor's convenience.

pub use rt_ir_plugin_core::{MidiActivity, MidiStatus, PadState, pad_state};

/// Number of log-spaced bins in the formant-display snapshot the audio thread
/// publishes to the editor.
pub const SPECTRUM_DISPLAY_BINS: usize = 48;

/// Twelve-tone pitch-class names (sharps).
const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Convert a frequency in Hz to a note name with octave, e.g. `440 → "A4"`.
///
/// Returns `"—"` for a non-positive / untracked frequency.  Uses the standard
/// MIDI mapping (A4 = 440 Hz = MIDI 69, octave 4).
pub fn hz_to_note_name(hz: f32) -> String {
    if hz <= 0.0 {
        return "—".to_string();
    }
    let midi = (69.0 + 12.0 * (hz / 440.0).log2()).round() as i32;
    let pitch_class = midi.rem_euclid(12) as usize;
    let octave = midi / 12 - 1;
    format!("{}{octave}", NOTE_NAMES[pitch_class])
}

/// Format a tracked frequency, e.g. `"220.0 Hz"`; `"— Hz"` when untracked.
pub fn format_hz(hz: f32) -> String {
    if hz <= 0.0 {
        "— Hz".to_string()
    } else {
        format!("{hz:.1} Hz")
    }
}

/// Clamp a pitch-tracker confidence to the `[0, 1]` bar-fill fraction.
#[inline]
pub fn confidence_fraction(confidence: f32) -> f32 {
    confidence.clamp(0.0, 1.0)
}

/// Convert a latency in samples to milliseconds at `sample_rate` (`0.0` for a
/// non-positive rate, before the host reports one).
#[inline]
pub fn latency_ms(samples: usize, sample_rate: f32) -> f32 {
    if sample_rate <= 0.0 {
        0.0
    } else {
        samples as f32 * 1000.0 / sample_rate
    }
}

/// Badge text for a latency in milliseconds, e.g. `"21.3 ms"`.
#[inline]
pub fn format_latency_ms(ms: f32) -> String {
    format!("{ms:.1} ms")
}

/// Normalise a magnitude spectrum to `[0, 1]` draw heights on a dB scale.
///
/// Each value is converted to dB relative to the spectrum's own peak, then
/// mapped so `floor_db` → `0.0` and the peak → `1.0`.  A silent spectrum maps
/// to all zeros.  `floor_db` should be negative (e.g. `-60.0`).  Allocates a
/// fresh `Vec` — editor thread only.
pub fn spectral_heights(magnitude: &[f32], floor_db: f32) -> Vec<f32> {
    let peak = magnitude.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 || floor_db >= 0.0 {
        return vec![0.0; magnitude.len()];
    }
    let inv_floor = 1.0 / floor_db; // floor_db < 0
    magnitude
        .iter()
        .map(|&value| {
            if value <= 0.0 {
                return 0.0;
            }
            // dB below the peak (≤ 0), then map [floor, 0] → [0, 1].
            let db = 20.0 * (value / peak).log10();
            (1.0 - db * inv_floor).clamp(0.0, 1.0)
        })
        .collect()
}

/// Format a harmony interval as a semitone + cents label, e.g. `"+7 st"`,
/// `"+20 ¢"`, `"-7 st -25 ¢"`, or `"unison"` at zero.
pub fn format_interval(semitones: f32) -> String {
    let total_cents = (semitones * 100.0).round() as i32;
    let st = total_cents / 100;
    let cents = total_cents % 100;
    match (st, cents) {
        (0, 0) => "unison".to_string(),
        (_, 0) => format!("{st:+} st"),
        (0, _) => format!("{cents:+} ¢"),
        _ => format!("{st:+} st {cents:+} ¢"),
    }
}

/// Format a linear output level as a percentage, e.g. `"100 %"`.
pub fn format_level(level: f32) -> String {
    format!("{:.0} %", level.max(0.0) * 100.0)
}

/// Format a pan `[-1, 1]` as `"L50"`, `"C"`, or `"R30"`.
pub fn format_pan(pan: f32) -> String {
    let clamped = pan.clamp(-1.0, 1.0);
    let percent = (clamped.abs() * 100.0).round() as i32;
    if percent == 0 {
        "C".to_string()
    } else if clamped < 0.0 {
        format!("L{percent}")
    } else {
        format!("R{percent}")
    }
}

/// Format a detune in cents, e.g. `"+5 ¢"` or `"0 ¢"`.
pub fn format_detune(cents: f32) -> String {
    format!("{:+} ¢", cents.round() as i32)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_names_map_standard_pitches() {
        assert_eq!(hz_to_note_name(440.0), "A4");
        assert_eq!(hz_to_note_name(261.63), "C4"); // middle C
        assert_eq!(hz_to_note_name(220.0), "A3");
        assert_eq!(hz_to_note_name(0.0), "—");
        assert_eq!(hz_to_note_name(-5.0), "—");
    }

    #[test]
    fn hz_formatting() {
        assert_eq!(format_hz(220.0), "220.0 Hz");
        assert_eq!(format_hz(0.0), "— Hz");
    }

    #[test]
    fn confidence_is_clamped() {
        assert_eq!(confidence_fraction(1.5), 1.0);
        assert_eq!(confidence_fraction(-0.2), 0.0);
        assert!((confidence_fraction(0.7) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn latency_conversion_and_badge() {
        let ms = latency_ms(1024, 48_000.0);
        assert!((ms - 21.333).abs() < 1e-2, "ms = {ms}");
        assert_eq!(format_latency_ms(ms), "21.3 ms");
        assert_eq!(latency_ms(1024, 0.0), 0.0);
    }

    #[test]
    fn spectral_heights_peak_is_one_floor_is_zero() {
        // Peak bin → 1.0; a bin at the floor (−60 dB ≈ 1/1000 of peak) → ~0.0.
        let magnitude = [1.0, 0.001, 0.0, 0.5];
        let heights = spectral_heights(&magnitude, -60.0);
        assert!((heights[0] - 1.0).abs() < 1e-4, "peak = {}", heights[0]);
        assert!(heights[1] < 0.02, "floor = {}", heights[1]);
        assert_eq!(heights[2], 0.0); // silent bin
        assert!(heights[3] > 0.5 && heights[3] < 1.0);
    }

    #[test]
    fn spectral_heights_of_silence_is_zero() {
        let heights = spectral_heights(&[0.0, 0.0, 0.0], -60.0);
        assert!(heights.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn interval_labels() {
        assert_eq!(format_interval(0.0), "unison");
        assert_eq!(format_interval(7.0), "+7 st");
        assert_eq!(format_interval(-12.0), "-12 st");
        assert_eq!(format_interval(0.2), "+20 ¢");
        assert_eq!(format_interval(7.25), "+7 st +25 ¢");
    }

    #[test]
    fn level_pan_detune_labels() {
        assert_eq!(format_level(1.0), "100 %");
        assert_eq!(format_level(0.0), "0 %");
        assert_eq!(format_pan(0.0), "C");
        assert_eq!(format_pan(-0.5), "L50");
        assert_eq!(format_pan(0.3), "R30");
        assert_eq!(format_detune(5.0), "+5 ¢");
        assert_eq!(format_detune(0.0), "+0 ¢");
        assert_eq!(format_detune(-12.0), "-12 ¢");
    }

    #[test]
    fn voice_pad_state_reuse() {
        // A harmony voice is always "loaded"; map enabled / active through.
        assert_eq!(pad_state(true, true, true), PadState::Active);
        assert_eq!(pad_state(true, true, false), PadState::Filled);
        assert_eq!(pad_state(true, false, false), PadState::Disabled);
    }
}
